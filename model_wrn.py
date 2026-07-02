"""WRN-28-1 construction and internal-width growth operations."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _growth as grow


class _ReLU1Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x.clamp_min(0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        grad_input = grad_input * (x >= 0).to(dtype=grad_input.dtype)
        return (grad_input,)


class ReLU1(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ReLU1Fn.apply(x)


def _kaiming_like_(w: torch.Tensor, nonlinearity: str = "relu") -> None:
    nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity=nonlinearity)


def _mean_filter_norm(layer: nn.Conv2d, eps: float = 1e-12) -> float:
    W = layer.weight.detach()
    if W.numel() == 0:
        return 1.0

    norms = torch.linalg.norm(W.reshape(W.size(0), -1), ord=2, dim=1).clamp_min(eps)
    return float(norms.mean().item())


def _scale_new_filters(
    layer: nn.Conv2d,
    new_filters: torch.Tensor,
    scale: float,
    scale_method: str,
    eps: float = 1e-12,
) -> torch.Tensor:
    sm = (scale_method or "mean_norm").lower()
    if new_filters.numel() == 0:
        return new_filters

    if sm in ("fixed", "none", "he"):
        return new_filters * float(scale)
    if sm != "mean_norm":
        raise ValueError(f"Unknown scale_method: {scale_method}")

    target = _mean_filter_norm(layer, eps=eps) * float(scale)
    out = new_filters.clone()

    flat = out.reshape(out.size(0), -1)
    norms = torch.linalg.norm(flat, ord=2, dim=1).clamp_min(eps)
    out = (out / norms.view(-1, 1, 1, 1)) * target
    return out


def _add_channels_groupnorm(gn: nn.GroupNorm, k: int) -> None:
    if k <= 0:
        return
    dev = gn.weight.device
    new_weight = torch.cat([gn.weight, torch.ones(k, device=dev)], dim=0)
    new_bias = torch.cat([gn.bias, torch.zeros(k, device=dev)], dim=0)
    gn.weight = nn.Parameter(new_weight)
    gn.bias = nn.Parameter(new_bias)
    gn.num_channels += k


class WideResNetBasicBlock(nn.Module):
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
        self.relu0 = ReLU1()

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

        if self.normalization_type == "batchnorm":
            self.mid_norm: Optional[nn.Module] = nn.BatchNorm2d(
                hidden, eps=1e-5, momentum=0.1
            )
        elif self.normalization_type == "layernorm":
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

        self.gradmax_scale_method = "mean_norm"
        self.gradmax_init_scale = 1.0
        self.gradmax_epsilon = 0.0
        self.growRatio = 0.5
        self._rr_grow_counter = 0

        self.conv_stem = nn.Conv2d(
            self.input_channels, 16, kernel_size=3, stride=1, padding=1, bias=with_bias
        )
        _kaiming_like_(self.conv_stem.weight)
        if self.conv_stem.bias is not None:
            nn.init.zeros_(self.conv_stem.bias)

        num_blocks = (self.depth - 4) // 6

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
                in_ch = filters
            groups.append(block_list)
        self.groups = nn.ModuleList(groups)

        self.final_bn = nn.BatchNorm2d(in_ch, eps=1e-5, momentum=0.1)
        self.final_relu = ReLU1()
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(in_ch, self.num_classes, bias=fc_bias)
        _kaiming_like_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

        self.to(self.device)

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

    def get_grow_layer_tuples(self) -> List[List[nn.Module]]:
        tuples: List[List[nn.Module]] = []
        for group in self.groups:
            for blk in group:
                tpl: List[nn.Module] = [blk.conv1]
                if blk.mid_norm is not None:
                    tpl.append(blk.mid_norm)
                tpl.append(blk.conv2)
                tuples.append(tpl)
        return tuples

    def initForGradMax(self) -> None:
        dev = self.device
        for group in self.groups:
            for blk in group:
                conv1 = blk.conv1
                conv2 = blk.conv2

                k1h, k1w = conv1.weight.size(2), conv1.weight.size(3)
                k2h, k2w = conv2.weight.size(2), conv2.weight.size(3)
                kH, kW = k1h + k2h - 1, k1w + k2w - 1

                conv2.Waux = torch.zeros(
                    (conv2.out_channels, conv1.in_channels, kH, kW),
                    device=dev,
                    dtype=conv2.weight.dtype,
                    requires_grad=True,
                )
                conv2.Waux.retain_grad()

                s1 = (
                    conv1.stride
                    if isinstance(conv1.stride, tuple)
                    else (conv1.stride, conv1.stride)
                )
                s2 = (
                    conv2.stride
                    if isinstance(conv2.stride, tuple)
                    else (conv2.stride, conv2.stride)
                )
                conv2.Waux_stride = tuple(
                    (a + b) if (a > 1 and b > 1) else (a + b - 1)
                    for a, b in zip(s1, s2)
                )

    def forwardForGradMax(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.conv_stem(x)

        for group in self.groups:
            for blk in group:
                y = blk.bn0(x)
                y = blk.relu0(y)
                x_pre = y

                y = blk.conv1(y)
                if blk.mid_norm is not None:
                    y = blk.mid_norm(y)
                y = blk.relu1(y)
                y = blk.conv2(y)

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

    @torch.no_grad()
    def growGradMaxOneTuple(self, tuple_id: int, n_new: Optional[int] = None) -> None:
        tuples = self.get_grow_layer_tuples()

        tpl = tuples[tuple_id]
        conv1: nn.Conv2d = tpl[0]
        conv2: nn.Conv2d = tpl[-1]
        mid_norm: Optional[nn.Module] = tpl[1] if (len(tpl) == 3) else None

        if n_new is None:
            n_new = int(conv1.out_channels * float(self.growRatio))
        n_new = int(n_new)
        if n_new <= 0:
            return

        k2h, k2w = conv2.weight.size(2), conv2.weight.size(3)
        nbToGrowLimit = int(conv2.out_channels * k2h * k2w)
        if n_new > nbToGrowLimit:
            n_new = nbToGrowLimit

        k1h, k1w = conv1.weight.size(2), conv1.weight.size(3)
        Cin = int(conv1.in_channels)
        Cout = int(conv2.out_channels)

        grad = conv2.Waux.grad

        A = (
            grad.unfold(dimension=2, size=k2h, step=1)
            .unfold(dimension=3, size=k2w, step=1)
            .permute(0, 4, 5, 1, 2, 3)
            .reshape(Cout * k2h * k2w, Cin * k1h * k1w)
        )

        U, _, _ = torch.linalg.svd(A, full_matrices=False)
        to_add = (
            U[:, :n_new].reshape(Cout, k2h, k2w, n_new).permute(0, 3, 1, 2).contiguous()
        )

        if float(getattr(self, "gradmax_epsilon", 0.0)) == 0.0:
            grow.add_zero_outputs(conv1, n_new)
        else:
            W = conv1.weight
            newW = torch.randn(
                (n_new, W.size(1), W.size(2), W.size(3)), device=W.device, dtype=W.dtype
            )
            newW = _scale_new_filters(
                conv1,
                newW,
                scale=float(self.gradmax_epsilon),
                scale_method=str(getattr(self, "gradmax_scale_method", "mean_norm")),
            )
            conv1.weight = nn.Parameter(torch.cat([conv1.weight, newW], dim=0))
            conv1.out_channels = conv1.weight.size(0)
            if conv1.bias is not None:
                conv1.bias = nn.Parameter(
                    torch.cat(
                        [
                            conv1.bias,
                            torch.zeros(n_new, device=W.device, dtype=W.dtype),
                        ],
                        dim=0,
                    )
                )

        if mid_norm is not None:
            if isinstance(mid_norm, nn.BatchNorm2d):
                grow.add_batchnorm_channels(mid_norm, n_new)
            elif isinstance(mid_norm, nn.GroupNorm):
                _add_channels_groupnorm(mid_norm, n_new)

        to_add = grow.scale_new_input_weights(
            conv2,
            to_add.to(dtype=conv2.weight.dtype),
            scale=float(getattr(self, "gradmax_init_scale", 1.0)),
            scale_method=str(getattr(self, "gradmax_scale_method", "mean_norm")),
        )
        conv2.weight = nn.Parameter(torch.cat([conv2.weight, to_add], dim=1))
        conv2.in_channels = conv2.weight.size(1)

    @torch.no_grad()
    def growGradMax(
        self, tuple_id: Optional[int] = None, n_new: Optional[int] = None
    ) -> None:
        tuples = self.get_grow_layer_tuples()
        if not tuples:
            return
        if tuple_id is None:
            tuple_id = self._rr_grow_counter % len(tuples)
            self._rr_grow_counter += 1
        self.growGradMaxOneTuple(tuple_id=int(tuple_id), n_new=n_new)

    @torch.no_grad()
    def expand_block_conv1(self, tuple_id: int, nb_increase: int, mode: int) -> None:
        tuples = self.get_grow_layer_tuples()
        nb_increase = int(nb_increase)
        mode = int(mode)

        if nb_increase <= 0:
            return
        if mode not in (1, 2, 3, 4, 5):
            raise ValueError(f"Unknown expansion mode: {mode}")

        tpl = tuples[int(tuple_id)]
        conv1: nn.Conv2d = tpl[0]
        conv2: nn.Conv2d = tpl[-1]
        mid_norm: Optional[nn.Module] = tpl[1] if (len(tpl) == 3) else None

        W1_old = conv1.weight.detach()
        out_old, _, kH1, kW1 = W1_old.size()
        out_new = out_old + nb_increase

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
            dtype=W1_old.dtype,
        )
        new_conv1.weight.data[:out_old].copy_(W1_old)

        if mode in (1, 2, 4):
            fan_in1 = conv1.in_channels * kH1 * kW1
            std1 = math.sqrt(2.0 / fan_in1)
            new_conv1.weight.data[out_old:].normal_(0.0, std1)
        elif mode == 3:
            new_conv1.weight.data[out_old:].zero_()
        elif mode == 5:
            std1 = float(W1_old.float().std(unbiased=False).item())
            new_conv1.weight.data[out_old:].normal_(0.0, std1)

        if conv1.bias is not None:
            b_old = conv1.bias.detach()
            new_conv1.bias.data.zero_()
            new_conv1.bias.data[:out_old].copy_(b_old)

        conv1.weight = nn.Parameter(new_conv1.weight.data)
        if conv1.bias is not None:
            conv1.bias = nn.Parameter(new_conv1.bias.data)
        conv1.out_channels = out_new

        if mid_norm is not None:
            if isinstance(mid_norm, nn.BatchNorm2d):
                grow.add_batchnorm_channels(mid_norm, nb_increase)
            elif isinstance(mid_norm, nn.GroupNorm):
                _add_channels_groupnorm(mid_norm, nb_increase)

        W2_old = conv2.weight.detach()
        _, Cin2_old, kH2, kW2 = W2_old.size()
        Cin2_new = Cin2_old + nb_increase

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
            dtype=W2_old.dtype,
        )
        new_conv2.weight.data[:, :Cin2_old].copy_(W2_old)

        if mode in (1, 2):
            new_conv2.weight.data[:, Cin2_old:].zero_()
        elif mode in (3, 4):
            fan_in2 = Cin2_new * kH2 * kW2
            std2 = math.sqrt(2.0 / fan_in2)
            new_conv2.weight.data[:, Cin2_old:].normal_(0.0, std2)
        elif mode == 5:
            std2 = float(W2_old.float().std(unbiased=False).item())
            new_conv2.weight.data[:, Cin2_old:].normal_(0.0, std2)

        if conv2.bias is not None:
            new_conv2.bias.data.copy_(conv2.bias.detach())

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

    @torch.no_grad()
    def expand_conv_layer_mode5(self, layer_id: int, nb_increase: int) -> None:
        self.expand_block_conv1(tuple_id=layer_id, nb_increase=nb_increase, mode=5)

    expand_conv_layer_mode_a = expand_conv_layer_mode1
    expand_conv_layer_mode_b = expand_conv_layer_mode2
    expand_conv_layer_mode_c = expand_conv_layer_mode3
    expand_conv_layer_mode_d = expand_conv_layer_mode4
    expand_conv_layer_mode_e = expand_conv_layer_mode5


MODE_SPECS = OrderedDict(
    [
        ("mode_a", "Mode A: Column-Zero Initialization"),
        ("mode_b", "Mode B: Row-First Column-Zero Initialization"),
        ("mode_c", "Mode C: Row-Zero Initialization (NZZ)"),
        ("mode_d", "Mode D: Homogeneous Initialization"),
        ("mode_e", "Mode E: Homogeneous Initialization with Empirical Variance"),
        ("gradmax", "GradMax"),
    ]
)


def build_model(
    num_classes: int,
    device: torch.device,
    width_multiplier: float,
    seed: int,
) -> WideResNetGradMax:
    model = WideResNetGradMax(
        num_classes=num_classes,
        depth=28,
        width_multiplier=1,
        block_width_multiplier=width_multiplier,
        normalization_type="batchnorm",
        device=device,
        with_bias=False,
        fc_bias=True,
        seed=seed,
    )
    model.gradmax_scale_method = "mean_norm"
    model.gradmax_init_scale = 0.5
    model.gradmax_epsilon = 0.0
    return model.to(device)


def preprocess(x: torch.Tensor) -> torch.Tensor:
    return x


def growth_plan(
    seed_model: WideResNetGradMax,
    big_model: WideResNetGradMax,
    grow_steps: int,
) -> list[list[int]]:
    differences = [
        int(big_tuple[0].out_channels) - int(seed_tuple[0].out_channels)
        for seed_tuple, big_tuple in zip(
            seed_model.get_grow_layer_tuples(),
            big_model.get_grow_layer_tuples(),
        )
    ]
    plan = [[0 for _ in differences] for _ in range(grow_steps)]
    for tuple_index, difference in enumerate(differences):
        base, remainder = divmod(int(difference), int(grow_steps))
        for grow_index in range(grow_steps):
            plan[grow_index][tuple_index] = base + (grow_index < remainder)
    return plan


def grow_model(
    model: WideResNetGradMax,
    mode: str,
    additions: Sequence[int],
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
    grow_batch_size: int,
) -> None:
    additions = [int(value) for value in additions]
    if mode == "gradmax":
        model.train()
        model.zero_grad(set_to_none=True)
        model.initForGradMax()
        logits = model.forwardForGradMax(x[:grow_batch_size])
        loss_fn(logits, y[:grow_batch_size]).backward()
        for tuple_index, amount in enumerate(additions):
            if amount > 0:
                model.growGradMax(tuple_id=tuple_index, n_new=amount)
        for group in model.groups:
            for block in group:
                if hasattr(block.conv2, "Waux"):
                    block.conv2.Waux = None
        return

    mode_ids = {
        "mode_a": 1,
        "mode_b": 2,
        "mode_c": 3,  # Mode C reduces to row-zero for internal-width growth.
        "mode_d": 4,
        "mode_e": 5,
    }
    if mode not in mode_ids:
        raise ValueError(f"Unknown WRN initialization mode: {mode}")
    for tuple_index, amount in enumerate(additions):
        if amount > 0:
            model.expand_block_conv1(tuple_index, amount, mode_ids[mode])
