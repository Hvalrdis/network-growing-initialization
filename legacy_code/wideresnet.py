# Wide Resnet, from https://www.kaggle.com/code/itslek/cifar-10-96-pytorch-wresnet-28x10-sf-v3-2
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ReLU1Fn(torch.autograd.Function):
    """
    ReLU with grad(f(0)) = 1.
    作用：对齐 TF 的 tf.math.maximum(x, 0) 在 0 点的梯度处理（GradMax 代码依赖这个性质）。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x.clamp_min(0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        # grad = 1 for x >= 0, 0 otherwise
        grad_input = grad_input * (x >= 0).to(dtype=grad_input.dtype)
        return (grad_input,)


class ReLU1(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ReLU1Fn.apply(x)


class BasicBlock(nn.Module):
    droprate = 0.0
    use_bn = True
    use_fixup = False
    fixup_l = 12

    # --- 新增：GradMax 风格开关（类变量，保持 NetworkBlock 的调用签名不变）---
    gradmax_style = False
    block_width_multiplier = 1.0
    normalization_type = "batchnorm"  # "batchnorm" | "layernorm" | "none"

    def __init__(self, in_planes, out_planes, stride):
        super(BasicBlock, self).__init__()

        self.in_planes = int(in_planes)
        self.out_planes = int(out_planes)
        self.stride = int(stride)

        assert self.use_fixup or self.use_bn, "Need to use at least one thing: Fixup or BatchNorm"

        # ============================
        # GradMax 风格 BasicBlock
        # BN -> ReLU -> Conv(hidden) -> (Norm?) -> ReLU1 -> Conv(out) -> add skip
        # skip: 只在 stride>1 用 1x1 conv，且作用在原始输入 x 上
        # ============================
        if bool(self.gradmax_style):
            if not bool(self.use_bn):
                raise ValueError("gradmax_style=True 需要 use_bn=True（GradMax/WRN 的配置就是这样）。")
            if bool(self.use_fixup):
                raise ValueError("gradmax_style=True 目前不支持 use_fixup=True；请保持 use_fixup=False。")

            self.bn0 = nn.BatchNorm2d(self.in_planes, eps=1e-5, momentum=0.1)
            self.relu0 = nn.ReLU(inplace=True)

            hidden = int(round(self.out_planes * float(self.block_width_multiplier)))
            hidden = max(1, hidden)
            self.hidden = hidden

            self.conv1 = nn.Conv2d(
                self.in_planes, hidden, kernel_size=3, stride=self.stride, padding=1, bias=False
            )

            nt = str(self.normalization_type).lower()
            if nt == "batchnorm":
                self.mid_norm: Optional[nn.Module] = nn.BatchNorm2d(hidden, eps=1e-5, momentum=0.1)
            elif nt == "layernorm":
                # Torch 没有完全等价 TF channels-last LayerNorm，这里用 GroupNorm(1, C) 做稳定替代
                self.mid_norm = nn.GroupNorm(1, hidden, eps=1e-5, affine=True)
            elif nt == "none":
                self.mid_norm = None
            else:
                raise ValueError(f"Unknown normalization_type: {self.normalization_type}")

            self.relu1 = ReLU1()
            self.conv2 = nn.Conv2d(hidden, self.out_planes, kernel_size=3, stride=1, padding=1, bias=False)

            # GradMax/TF: skip conv 仅在 stride>1
            if self.stride > 1:
                self.conv_res = nn.Conv2d(
                    self.in_planes, self.out_planes, kernel_size=1, stride=self.stride, padding=0, bias=False
                )
            else:
                # 为了严格对齐 GradMax 的 WRN 组网方式：stride==1 时通道不该变
                if self.in_planes != self.out_planes:
                    raise ValueError(
                        "gradmax_style 下，stride==1 时要求 in_planes==out_planes；"
                        "WRN 的组网也确实只在下采样(stride>1)时换通道。"
                    )
                self.conv_res = None

            self.equalInOut = self.in_planes == self.out_planes
            return

        # ============================
        # 原始 WideResNet BasicBlock（完全保留）
        # ============================
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.bn2 = nn.BatchNorm2d(self.out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            self.in_planes, self.out_planes, kernel_size=3, stride=self.stride, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(self.out_planes, self.out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.equalInOut = self.in_planes == self.out_planes
        self.conv_res = nn.Conv2d(
            self.in_planes, self.out_planes, kernel_size=1, stride=self.stride, padding=0, bias=False
        )
        self.conv_res = (not self.equalInOut) and self.conv_res or None

        if self.use_fixup:
            self.multiplicator = nn.Parameter(torch.ones(1, 1, 1, 1))
            self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, 1, 1))] * 4)

            k = self.conv1.kernel_size[0] * self.conv1.kernel_size[1] * self.conv1.out_channels
            self.conv1.weight.data.normal_(0, self.fixup_l ** (-0.5) * math.sqrt(2.0 / k))
            self.conv2.weight.data.zero_()

            if self.conv_res is not None:
                k = self.conv_res.kernel_size[0] * self.conv_res.kernel_size[1] * self.conv_res.out_channels
                self.conv_res.weight.data.normal_(0, math.sqrt(2.0 / k))

    def forward(self, x):
        # --- GradMax 风格 ---
        if bool(self.gradmax_style):
            y = self.relu0(self.bn0(x))
            y = self.conv1(y)
            if self.mid_norm is not None:
                y = self.mid_norm(y)
            y = self.relu1(y)
            if self.droprate > 0:
                y = F.dropout(y, p=self.droprate, training=self.training)
            y = self.conv2(y)

            skip = self.conv_res(x) if self.conv_res is not None else x
            return torch.add(skip, y)

        # --- 原始逻辑（完全保留）---
        if self.use_bn:
            x_out = self.relu(self.bn1(x))
            out = self.relu(self.bn2(self.conv1(x_out)))
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, training=self.training)
            out = self.conv2(out)
        else:
            x_out = self.relu(x + self.biases[0])
            out = self.conv1(x_out) + self.biases[1]
            out = self.relu(out) + self.biases[2]
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, training=self.training)
            out = self.multiplicator * self.conv2(out) + self.biases[3]

        if self.equalInOut:
            return torch.add(x, out)

        return torch.add(self.conv_res(x_out), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride):
        layers = []
        for i in range(int(nb_layers)):
            _in_planes = in_planes if i == 0 else out_planes
            _stride = stride if i == 0 else 1
            layers.append(block(_in_planes, out_planes, _stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(
        self,
        depth,
        num_classes,
        widen_factor=1,
        droprate=0.0,
        use_bn=True,
        use_fixup=False,
        # --- 新增：GradMax 风格参数（默认 False，不影响原行为）---
        gradmax_style: bool = False,
        block_width_multiplier: float = 1.0,
        normalization_type: str = "batchnorm",
    ):
        super(WideResNet, self).__init__()

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        assert (depth - 4) % 6 == 0, "You need to change the number of layers"
        n = (depth - 4) / 6

        BasicBlock.droprate = droprate
        BasicBlock.use_bn = use_bn
        BasicBlock.fixup_l = int(n * 3)
        BasicBlock.use_fixup = use_fixup

        # GradMax 风格开关与参数
        BasicBlock.gradmax_style = bool(gradmax_style)
        BasicBlock.block_width_multiplier = float(block_width_multiplier)
        BasicBlock.normalization_type = str(normalization_type)

        block = BasicBlock

        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2)

        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        # 初始化保持你原来的逻辑（避免无意改掉 baseline）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                k = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / k))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                if use_fixup:
                    m.weight.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
