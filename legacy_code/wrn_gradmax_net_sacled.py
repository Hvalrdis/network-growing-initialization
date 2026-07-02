import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import grow


class _ReLU1Fn(torch.autograd.Function):
    """ReLU with grad(f(0)) = 1 (matches TF tf.math.maximum(x,0) behavior used in GradMax code)."""

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


def _kaiming_like_(w: torch.Tensor, nonlinearity: str = "relu") -> None:
    # torch.nn.init.kaiming_normal_ uses fan_in by default
    nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity=nonlinearity)

def _scale_old_block_by_fanin(old_fan_in: int, new_fan_in: int) -> float:
    # 和你之前 imageclassificationnet_scale.py 一样的 sqrt(old/new) 思路
    if new_fan_in <= 0:
        return 1.0
    return math.sqrt(float(old_fan_in) / float(new_fan_in))

def _mean_filter_norm(layer: nn.Conv2d, eps: float = 1e-12) -> float:
    """Mean l2 norm of existing conv filters (over Cout dimension)."""
    W = layer.weight.detach()
    if W.numel() == 0:
        return 1.0
    # (Cout, Cin, kH, kW) -> (Cout, -1)
    norms = torch.linalg.norm(W.reshape(W.size(0), -1), ord=2, dim=1).clamp_min(eps)
    return float(norms.mean().item())


def _scale_new_filters(
    layer: nn.Conv2d,
    new_filters: torch.Tensor,
    scale: float,
    scale_method: str,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Scale new output filters to match mean filter norm (GradMax default)."""
    sm = (scale_method or "mean_norm").lower()
    if new_filters.numel() == 0:
        return new_filters

    if sm in ("fixed", "none", "he"):
        return new_filters * float(scale)
    if sm != "mean_norm":
        raise ValueError(f"Unknown scale_method: {scale_method}")

    target = _mean_filter_norm(layer, eps=eps) * float(scale)
    out = new_filters.clone()
    # Normalize each filter vector.
    flat = out.reshape(out.size(0), -1)
    norms = torch.linalg.norm(flat, ord=2, dim=1).clamp_min(eps)
    out = (out / norms.view(-1, 1, 1, 1)) * target
    return out


def _add_channels_groupnorm(gn: nn.GroupNorm, k: int) -> None:
    """Expand GroupNorm (used as a LayerNorm-ish alternative) by k channels."""
    if k <= 0:
        return
    dev = gn.weight.device
    new_weight = torch.cat([gn.weight, torch.ones(k, device=dev)], dim=0)
    new_bias = torch.cat([gn.bias, torch.zeros(k, device=dev)], dim=0)
    gn.weight = nn.Parameter(new_weight)
    gn.bias = nn.Parameter(new_bias)
    gn.num_channels += k


class WideResNetBasicBlock(nn.Module):
    """Pre-activation basic block: BN -> ReLU -> Conv -> (Norm?) -> ReLU1 -> Conv, then add skip."""

    def __init__(
        self,
        in_channels: int,
        filters: int,
        block_width_multiplier: float,
        stride: int,
        normalization_type: str,
        with_bias: bool = False,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.filters = int(filters)
        self.block_width_multiplier = float(block_width_multiplier)
        self.stride = int(stride)
        self.normalization_type = str(normalization_type).lower()

        self.bn0 = nn.BatchNorm2d(self.in_channels, eps=1e-5, momentum=0.1)
        self.relu0 = nn.ReLU(inplace=True)

        hidden = int(round(self.filters * self.block_width_multiplier))
        self.conv1 = nn.Conv2d(
            self.in_channels,
            hidden,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            bias=with_bias,
        )
        _kaiming_like_(self.conv1.weight)
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)

        # Optional normalization between convs
        if self.normalization_type == "batchnorm":
            self.mid_norm: Optional[nn.Module] = nn.BatchNorm2d(hidden, eps=1e-5, momentum=0.1)
        elif self.normalization_type == "layernorm":
            # TF LayerNormalization in channels-last is not identical to Torch's;
            # in GradMax official code WRN typically uses batchnorm.
            # We provide a stable fallback.
            self.mid_norm = nn.GroupNorm(1, hidden, eps=1e-5, affine=True)
        elif self.normalization_type == "none":
            self.mid_norm = None
        else:
            raise ValueError(f"Unknown normalization_type: {normalization_type}")

        self.relu1 = ReLU1()
        self.conv2 = nn.Conv2d(
            hidden,
            self.filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=with_bias,
        )
        _kaiming_like_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

        # Skip conv only when stride>1 (matches official TF code for WRN-28-1).
        self.skip: Optional[nn.Conv2d]
        if self.stride > 1:
            self.skip = nn.Conv2d(
                self.in_channels,
                self.filters,
                kernel_size=1,
                stride=self.stride,
                padding=0,
                bias=with_bias,
            )
            _kaiming_like_(self.skip.weight)
            if self.skip.bias is not None:
                nn.init.zeros_(self.skip.bias)
        else:
            self.skip = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bn0(x)
        y = self.relu0(y)
        y = self.conv1(y)
        if self.mid_norm is not None:
            y = self.mid_norm(y)
        y = self.relu1(y)
        y = self.conv2(y)
        skip = self.skip(x) if self.skip is not None else x
        return skip + y


class WideResNetGradMax(nn.Module):

    def __init__(
        self,
        num_classes: int = 10,
        depth: int = 28,
        width_multiplier: int = 1,
        block_width_multiplier: float = 1.0,
        normalization_type: str = "batchnorm",
        device: Optional[torch.device] = None,
        input_channels: int = 3,
        with_bias: bool = False,
        fc_bias: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("depth 应该满足 6n+4，例如 16/22/28/40")
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        self.num_classes = int(num_classes)
        self.depth = int(depth)
        self.width_multiplier = int(width_multiplier)
        self.block_width_multiplier = float(block_width_multiplier)
        self.normalization_type = str(normalization_type).lower()
        self.device = device if device is not None else torch.device("cpu")
        self.input_channels = int(input_channels)

        # GradMax hyper-params (和你当前 imageclassificationnet.py 的字段保持一致)
        self.gradmax_scale_method = "mean_norm"
        self.gradmax_init_scale = 1.0
        self.gradmax_epsilon = 0.0  # 对齐官方：incoming 连接 scale=epsilon；主实验常用 epsilon=0
        self.growRatio = 0.5
        # 用于 RoundRobin（不强制，你也可以在 main 里自己管理）
        self._rr_grow_counter = 0

        # Stem
        self.conv_stem = nn.Conv2d(self.input_channels, 16, kernel_size=3, stride=1, padding=1, bias=with_bias)
        _kaiming_like_(self.conv_stem.weight)
        if self.conv_stem.bias is not None:
            nn.init.zeros_(self.conv_stem.bias)

        num_blocks = (self.depth - 4) // 6
        # Groups: (filters, stride)
        group_specs = [(16, 1), (32, 2), (64, 2)]

        groups: List[nn.ModuleList] = []
        in_ch = 16
        for base_filters, group_stride in group_specs:
            filters = int(base_filters * self.width_multiplier)
            block_list = nn.ModuleList()
            for j in range(num_blocks):
                stride = group_stride if j == 0 else 1
                blk = WideResNetBasicBlock(
                    in_channels=in_ch,
                    filters=filters,
                    block_width_multiplier=self.block_width_multiplier,
                    stride=stride,
                    normalization_type=self.normalization_type,
                    with_bias=with_bias,
                )
                block_list.append(blk)
                in_ch = filters  # block output channels
            groups.append(block_list)
        self.groups = nn.ModuleList(groups)

        # Final layers
        self.final_bn = nn.BatchNorm2d(in_ch, eps=1e-5, momentum=0.1)

        self.final_relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(in_ch, self.num_classes, bias=fc_bias)
        _kaiming_like_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

        self.to(self.device)

    # --------------------------
    # 1) 普通 forward
    # --------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.conv_stem(x)
        for group in self.groups:
            for blk in group:
                x = blk(x)
        x = self.final_bn(x)
        x = self.final_relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    

    # --------------------------
    # 2) GradMax：候选 grow_layer_tuples（每个残差块一个 tuple）
    # --------------------------
    def get_grow_layer_tuples(self) -> List[List[nn.Module]]:
        """返回与官方 TF `grow_layer_tuples` 等价的结构。

        每个 tuple: [conv1, (mid_norm可选), conv2]
        - conv1：要增长输出通道
        - mid_norm：只做逐通道变换的层，增长时做 identity 扩维
        - conv2：消费新通道（增加输入通道）
        """
        tuples: List[List[nn.Module]] = []
        for group in self.groups:
            for blk in group:
                tpl: List[nn.Module] = [blk.conv1]
                if blk.mid_norm is not None:
                    tpl.append(blk.mid_norm)
                tpl.append(blk.conv2)
                tuples.append(tpl)
        return tuples

    # --------------------------
    # 3) GradMax：初始化每个 block 的 Waux
    # --------------------------
    def initForGradMax(self) -> None:
        """为每个残差块的 conv2 创建 aux 权重 Waux（不加入 optimizer）。"""
        dev = self.device
        for group in self.groups:
            for blk in group:
                conv1 = blk.conv1
                conv2 = blk.conv2

                k1h, k1w = conv1.weight.size(2), conv1.weight.size(3)
                k2h, k2w = conv2.weight.size(2), conv2.weight.size(3)
                kH, kW = k1h + k2h - 1, k1w + k2w - 1

                # Waux: (Cout(conv2), Cin(conv1), kH, kW)
                conv2.Waux = torch.zeros(
                    (conv2.out_channels, conv1.in_channels, kH, kW),
                    device=dev,
                    dtype=conv2.weight.dtype,
                    requires_grad=True,
                )
                conv2.Waux.retain_grad()

                # combined stride (官方公式)
                s1 = conv1.stride if isinstance(conv1.stride, tuple) else (conv1.stride, conv1.stride)
                s2 = conv2.stride if isinstance(conv2.stride, tuple) else (conv2.stride, conv2.stride)
                conv2.Waux_stride = tuple(
                    (a + b) if (a > 1 and b > 1) else (a + b - 1) for a, b in zip(s1, s2)
                )

    # --------------------------
    # 4) GradMax：forward with aux（loss.backward() 后 Waux.grad 有效）
    # --------------------------
    def forwardForGradMax(self, x: torch.Tensor) -> torch.Tensor:
        """前向中注入 aux 分支，保证 backward 后 conv2.Waux.grad 可用于 SVD。"""
        x = x.to(self.device)
        x = self.conv_stem(x)

        for group in self.groups:
            for blk in group:
                # pre-act
                y = blk.bn0(x)
                y = blk.relu0(y)
                x_pre = y  # 这是 conv1 的输入（对应官方 callback 的 inputs）

                y = blk.conv1(y)
                if blk.mid_norm is not None:
                    y = blk.mid_norm(y)
                y = blk.relu1(y)
                y = blk.conv2(y)

                # aux: add_h = aux_layer(x_pre), add to conv2 output
                if hasattr(blk.conv2, "Waux") and blk.conv2.Waux is not None:
                    kHaux, kWaux = blk.conv2.Waux.size(2), blk.conv2.Waux.size(3)
                    stride_aux = getattr(blk.conv2, "Waux_stride", (1, 1))
                    add_h = F.conv2d(
                        x_pre,
                        weight=blk.conv2.Waux,
                        bias=None,
                        stride=stride_aux,
                        padding=(kHaux // 2, kWaux // 2),
                    )
                    y = y + add_h

                skip = blk.skip(x) if blk.skip is not None else x
                x = skip + y

        x = self.final_bn(x)
        x = self.final_relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    # --------------------------
    # 5) GradMax：对某一个 block tuple 做一次增长（RoundRobin/AllAtOnce 由你的 main 控制）
    # --------------------------
    @torch.no_grad()
    def growGradMaxOneTuple(self, tuple_id: int, n_new: Optional[int] = None) -> None:
        """增长第 tuple_id 个残差块的 conv1 输出 + conv2 输入（不改变 conv2 输出，保证 skip-add 对齐）。"""
        tuples = self.get_grow_layer_tuples()
        if tuple_id < 0 or tuple_id >= len(tuples):
            raise IndexError(f"tuple_id 越界: {tuple_id} / {len(tuples)}")

        tpl = tuples[tuple_id]
        conv1: nn.Conv2d = tpl[0]  # type: ignore
        conv2: nn.Conv2d = tpl[-1]  # type: ignore
        mid_norm: Optional[nn.Module] = tpl[1] if (len(tpl) == 3) else None

        if n_new is None:
            n_new = int(conv1.out_channels * float(self.growRatio))
        n_new = int(n_new)
        if n_new <= 0:
            return

        if not hasattr(conv2, "Waux") or conv2.Waux is None or conv2.Waux.grad is None:
            raise RuntimeError(
                "growGradMaxOneTuple 之前必须先调用 initForGradMax()，"
                "并用 forwardForGradMax 跑过一个 batch 后再 loss.backward()。"
            )

        # Limit (对齐你当前链式实现)
        k2h, k2w = conv2.weight.size(2), conv2.weight.size(3)
        nbToGrowLimit = int(conv2.out_channels * k2h * k2w)
        if n_new > nbToGrowLimit:
            n_new = nbToGrowLimit

        # 1) 由 Waux.grad 生成 grow matrix A (转置版本)，然后 SVD 取 U 的前 k 列
        k1h, k1w = conv1.weight.size(2), conv1.weight.size(3)
        Cin = int(conv1.in_channels)
        Cout = int(conv2.out_channels)

        grad = conv2.Waux.grad
        if torch.isnan(grad).any():
            raise RuntimeError(f"Waux.grad 出现 NaN（tuple_id={tuple_id}），网络可能已经 dead")

        A = (
            grad.unfold(dimension=2, size=k2h, step=1)
            .unfold(dimension=3, size=k2w, step=1)
            .permute(0, 4, 5, 1, 2, 3)
            .reshape(Cout * k2h * k2w, Cin * k1h * k1w)
        )

        # svd(A): A = U S Vh；我们需要的是 U 的前 k 列（对应 TF 里 grow_matrix^T 的 right-singular）
        U, _, _ = torch.linalg.svd(A, full_matrices=False)
        to_add = (
            U[:, :n_new]
            .reshape(Cout, k2h, k2w, n_new)
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # (Cout, n_new, k2h, k2w)

        # 2) 先增长 conv1 输出通道（incoming 权重）
        if float(getattr(self, "gradmax_epsilon", 0.0)) == 0.0:
            grow.addFiltersZero(conv1, n_new)
        else:
            # random + mean_norm * epsilon（对齐官方 AddGradmax：scale=epsilon）
            W = conv1.weight
            newW = torch.randn((n_new, W.size(1), W.size(2), W.size(3)), device=W.device, dtype=W.dtype)
            newW = _scale_new_filters(
                conv1,
                newW,
                scale=float(self.gradmax_epsilon),
                scale_method=str(getattr(self, "gradmax_scale_method", "mean_norm")),
            )
            conv1.weight = nn.Parameter(torch.cat([conv1.weight, newW], dim=0))
            conv1.out_channels = conv1.weight.size(0)
            if conv1.bias is not None:
                conv1.bias = nn.Parameter(torch.cat([conv1.bias, torch.zeros(n_new, device=W.device, dtype=W.dtype)], dim=0))

        # 3) identity 扩维：mid_norm（如果有）
        if mid_norm is not None:
            if isinstance(mid_norm, nn.BatchNorm2d):
                grow.addChannelsBatchNorm(mid_norm, n_new)
            elif isinstance(mid_norm, nn.GroupNorm):
                _add_channels_groupnorm(mid_norm, n_new)

        # 4) 增长 conv2 输入通道（outgoing 权重）：scaleNewInputWeights + concat
        to_add = grow.scaleNewInputWeights(
            conv2,
            to_add.to(dtype=conv2.weight.dtype),
            scale=float(getattr(self, "gradmax_init_scale", 1.0)),
            scale_method=str(getattr(self, "gradmax_scale_method", "mean_norm")),
        )
        conv2.weight = nn.Parameter(torch.cat([conv2.weight, to_add], dim=1))
        conv2.in_channels = conv2.weight.size(1)

    @torch.no_grad()
    def growGradMax(self, tuple_id: Optional[int] = None, n_new: Optional[int] = None) -> None:
        """兼容你现有的 main：不传 tuple_id 时按 RoundRobin 增长 1 个 block。"""
        tuples = self.get_grow_layer_tuples()
        if not tuples:
            return
        if tuple_id is None:
            tuple_id = self._rr_grow_counter % len(tuples)
            self._rr_grow_counter += 1
        self.growGradMaxOneTuple(tuple_id=int(tuple_id), n_new=n_new)
        
    # --------------------------
    # 6) Mode1/2/3/4：用于你现有对比实验（只增长每个块的 conv1，并同步扩维 conv2 的输入）
    # --------------------------
    @torch.no_grad()
    def expand_block_conv1(self, tuple_id: int, nb_increase: int, mode: int) -> None:
        """对第 tuple_id 个残差块做“结构性增长”，复用你现有 mode1-4 的初始化逻辑。

        仅增长 conv1(out_channels) 并同步增长 conv2(in_channels)，不动 conv2(out_channels)，从而保持 skip-add 对齐。
        """
        tuples = self.get_grow_layer_tuples()
        if tuple_id < 0 or tuple_id >= len(tuples):
            raise IndexError(f"tuple_id 越界: {tuple_id} / {len(tuples)}")
        if nb_increase <= 0:
            return
        if mode not in (1, 2, 3, 4):
            raise ValueError("mode 只能是 1/2/3/4")

        tpl = tuples[tuple_id]
        conv1: nn.Conv2d = tpl[0]  # type: ignore
        conv2: nn.Conv2d = tpl[-1]  # type: ignore
        mid_norm: Optional[nn.Module] = tpl[1] if (len(tpl) == 3) else None

        # --- 1) grow conv1 out_channels ---
        W1_old = conv1.weight.data
        out_old, Cin, kH1, kW1 = W1_old.size()
        out_new = out_old + int(nb_increase)
        new_conv1 = nn.Conv2d(
            conv1.in_channels,
            out_new,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            dilation=conv1.dilation,
            groups=conv1.groups,
            bias=(conv1.bias is not None),
            device=W1_old.device,
        )
        new_conv1.weight.data[:out_old] = W1_old

        # 对齐 imageclassificationnet 的语义：
        # mode2 不能把 conv1 新滤波器置 0，否则会死通道。
        if mode in (1, 2, 3):
            fan_in = conv1.in_channels * kH1 * kW1
            std = math.sqrt(2.0 / fan_in)
            new_conv1.weight.data[out_old:].normal_(0, std)
        elif mode == 4:
            # === DEBUG: 定位 std NaN 来源 ===
            if not torch.isfinite(W1_old).all():
                bad = (~torch.isfinite(W1_old)).sum().item()
                wmin = W1_old.min().item()
                wmax = W1_old.max().item()
                print(f"[MODE4][NaN/Inf DETECTED] tuple_id={tuple_id} "
                    f"W1_old has {bad} non-finite; min={wmin}, max={wmax}, dtype={W1_old.dtype}")
            # ==================================

            std = float(W1_old.float().std(unbiased=False).item())

        if conv1.bias is not None:
            b_old = conv1.bias.data
            new_b = torch.zeros(out_new, device=b_old.device, dtype=b_old.dtype)
            new_b[:out_old] = b_old
            new_conv1.bias.data = new_b

        # 替换参数（保持模块引用：直接就地替换 weight/bias 和 meta 字段）
        conv1.weight = nn.Parameter(new_conv1.weight.data)
        if conv1.bias is not None:
            conv1.bias = nn.Parameter(new_conv1.bias.data)
        conv1.out_channels = out_new

        # --- 2) grow mid_norm channels (identity) ---
        if mid_norm is not None:
            if isinstance(mid_norm, nn.BatchNorm2d):
                grow.addChannelsBatchNorm(mid_norm, nb_increase)
            elif isinstance(mid_norm, nn.GroupNorm):
                _add_channels_groupnorm(mid_norm, nb_increase)

        # --- 3) grow conv2 in_channels ---
        W2_old = conv2.weight.data
        Cout2, Cin2_old, kH2, kW2 = W2_old.size()
        Cin2_new = Cin2_old + int(nb_increase)
        new_conv2 = nn.Conv2d(
            Cin2_new,
            conv2.out_channels,
            kernel_size=conv2.kernel_size,
            stride=conv2.stride,
            padding=conv2.padding,
            dilation=conv2.dilation,
            groups=conv2.groups,
            bias=(conv2.bias is not None),
            device=W2_old.device,
        )
        
        # === 新增：对“旧权重块”做 fan_in 缩放（只影响 mode1/2/3/4，因为本函数只允许 1-4）===
        old_fan_in2 = Cin2_old * kH2 * kW2
        new_fan_in2 = Cin2_new * kH2 * kW2
        scale_old2 = _scale_old_block_by_fanin(old_fan_in2, new_fan_in2)

        new_conv2.weight.data[:, :Cin2_old] = W2_old

        if mode in (1, 2):
            new_conv2.weight.data[:, Cin2_old:] = 0
        elif mode == 3:
            fan_in2 = Cin2_new * kH2 * kW2
            std2 = math.sqrt(2.0 / fan_in2)
            new_conv2.weight.data[:, Cin2_old:].normal_(0, std2)
        elif mode == 4:
            std2 = float(W2_old.std().item())
            new_conv2.weight.data[:, Cin2_old:].normal_(0, std2)

        if conv2.bias is not None:
            new_conv2.bias.data = conv2.bias.data.clone()

        conv2.weight = nn.Parameter(new_conv2.weight.data)
        if conv2.bias is not None:
            conv2.bias = nn.Parameter(new_conv2.bias.data)
        conv2.in_channels = Cin2_new


    @torch.no_grad()
    def expand_conv_layer_mode1(self, layer_id: int, nb_increase: int) -> None:
        self.expand_block_conv1(tuple_id=layer_id, nb_increase=nb_increase, mode=1)

    @torch.no_grad()
    def expand_conv_layer_mode2(self, layer_id: int, nb_increase: int) -> None:
        self.expand_block_conv1(tuple_id=layer_id, nb_increase=nb_increase, mode=2)

    @torch.no_grad()
    def expand_conv_layer_mode3(self, layer_id: int, nb_increase: int) -> None:
        self.expand_block_conv1(tuple_id=layer_id, nb_increase=nb_increase, mode=3)

    @torch.no_grad()
    def expand_conv_layer_mode4(self, layer_id: int, nb_increase: int) -> None:
        self.expand_block_conv1(tuple_id=layer_id, nb_increase=nb_increase, mode=4)

