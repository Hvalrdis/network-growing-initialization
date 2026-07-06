"""CvT-13 construction and stage-wise embedding-dimension growth.

The module includes the CvT-13 architecture, optional dense relative
localization, Initialization Modes A-E, Mixup/CutMix, and AdamW-state transfer.
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


MODE_SPECS = OrderedDict(
    [
        ("mode_a", "Mode A: Column-Zero Initialization"),
        ("mode_b", "Mode B: Row-First Column-Zero Initialization"),
        ("mode_c", "Mode C: Row-Zero Initialization"),
        ("mode_d", "Mode D: Homogeneous Initialization"),
        ("mode_e", "Mode E: Homogeneous Initialization with Empirical Variance"),
    ]
)

GROW_BEFORE_STEP = True

FINAL_DIMS = (64, 192, 384)
STAGE_HEADS = (1, 3, 6)
TensorCopier = Callable[[torch.Tensor, torch.Tensor], None]


class ChannelLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (variance.sqrt() + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = ChannelLayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DepthWiseConv2d(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel_size: int,
        padding: int,
        stride: int,
        bias: bool = True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_in,
                kernel_size=kernel_size,
                padding=padding,
                groups=dim_in,
                stride=stride,
                bias=bias,
            ),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        proj_kernel: int,
        kv_proj_stride: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        padding = proj_kernel // 2
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.to_q = DepthWiseConv2d(dim, inner_dim, proj_kernel, padding, 1, bias=False)
        self.to_kv = DepthWiseConv2d(
            dim, inner_dim * 2, proj_kernel, padding, kv_proj_stride, bias=False
        )
        self.to_out = nn.Sequential(nn.Conv2d(inner_dim, dim, 1), nn.Dropout(dropout))

    def _tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = tensor.shape
        return (
            tensor.view(batch, self.heads, self.dim_head, height * width)
            .permute(0, 1, 3, 2)
            .reshape(batch * self.heads, height * width, self.dim_head)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        q = self._tokens(self.to_q(x))
        k_map, v_map = self.to_kv(x).chunk(2, dim=1)
        k, v = self._tokens(k_map), self._tokens(v_map)
        attention = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * self.scale, dim=-1)
        out = torch.bmm(attention, v)
        out = (
            out.view(batch, self.heads, height * width, self.dim_head)
            .permute(0, 1, 3, 2)
            .reshape(batch, self.heads * self.dim_head, height, width)
        )
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        proj_kernel: int,
        kv_proj_stride: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                proj_kernel,
                                kv_proj_stride,
                                heads,
                                dim_head,
                                dropout,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_mult, dropout)),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attention, feed_forward in self.layers:
            x = attention(x) + x
            x = feed_forward(x) + x
        return x


class DenseRelativeLoc(nn.Module):
    """Dense relative localization auxiliary head."""

    def __init__(self, in_dim: int, sample_size: int = 32):
        super().__init__()
        self.in_dim = in_dim
        self.sample_size = sample_size
        self.layers = nn.Sequential(
            nn.Linear(in_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, height, width = x.shape
        first = torch.randint(height, (batch, self.sample_size, 2), device=x.device)
        second = torch.randint(height, (batch, self.sample_size, 2), device=x.device)
        flat = x.view(batch, channels, height * width)
        first_index = first[..., 0] * width + first[..., 1]
        second_index = second[..., 0] * width + second[..., 1]
        first_features = flat.gather(
            2, first_index.unsqueeze(1).expand(-1, channels, -1)
        )
        second_features = flat.gather(
            2, second_index.unsqueeze(1).expand(-1, channels, -1)
        )
        pairs = torch.cat([first_features, second_features], dim=1).transpose(1, 2)
        prediction = self.layers(pairs).reshape(-1, 2)
        delta = (first - second).float().reshape(-1, 2)
        return prediction, delta


@dataclass
class CvTWithDRLocOutput:
    logits: torch.Tensor
    drloc: torch.Tensor
    delta: torch.Tensor


class CvT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dims: tuple[int, int, int] = FINAL_DIMS,
        depths: tuple[int, int, int] = (1, 2, 10),
        heads: tuple[int, int, int] = STAGE_HEADS,
        dropout: float = 0.0,
        use_drloc: bool = False,
        sample_size: int = 32,
    ):
        super().__init__()
        kernels, strides = (7, 3, 3), (4, 2, 2)
        kv_strides = (2, 2, 2)
        input_dim = 3
        stages = []
        for dim, depth, stage_heads, kernel, stride, kv_stride in zip(
            dims, depths, heads, kernels, strides, kv_strides
        ):
            stages.append(
                nn.Sequential(
                    nn.Conv2d(
                        input_dim,
                        dim,
                        kernel_size=kernel,
                        padding=kernel // 2,
                        stride=stride,
                    ),
                    ChannelLayerNorm(dim),
                    Transformer(
                        dim,
                        proj_kernel=3,
                        kv_proj_stride=kv_stride,
                        depth=depth,
                        heads=stage_heads,
                        dim_head=dim // stage_heads,
                        mlp_mult=4,
                        dropout=dropout,
                    ),
                )
            )
            input_dim = dim
        self.layers = nn.Sequential(*stages)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[-1], num_classes)
        self.use_drloc = use_drloc
        self.drloc = DenseRelativeLoc(dims[-1], sample_size) if use_drloc else None
        self.arch = {
            "num_classes": num_classes,
            "dims": tuple(int(value) for value in dims),
            "depths": depths,
            "heads": heads,
            "dropout": dropout,
            "use_drloc": use_drloc,
            "sample_size": sample_size,
        }

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.head(self.avg_pool(features).flatten(1))

    def forward_with_drloc(self, x: torch.Tensor) -> CvTWithDRLocOutput:
        if self.drloc is None:
            raise RuntimeError("This CvT was built with use_drloc=False")
        features = self.forward_features(x)
        logits = self.head(self.avg_pool(features).flatten(1))
        prediction, delta = self.drloc(features)
        return CvTWithDRLocOutput(logits, prediction, delta)


def build_model(
    num_classes: int,
    device: torch.device,
    width_multiplier: float,
    seed: int,
) -> CvT:
    torch.manual_seed(int(seed))
    dims = []
    for final_dim, heads in zip(FINAL_DIMS, STAGE_HEADS):
        per_head = max(1, int(round(final_dim * width_multiplier / heads)))
        dims.append(per_head * heads)
    return CvT(num_classes=num_classes, dims=tuple(dims)).to(device)


def preprocess(x: torch.Tensor) -> torch.Tensor:
    return x


def growth_plan(seed_model: CvT, big_model: CvT, grow_steps: int) -> list[list[int]]:
    if grow_steps <= 0:
        raise ValueError("CvT grow_steps must be positive")
    per_head_differences = [
        big_model.arch["dims"][stage] // heads - seed_model.arch["dims"][stage] // heads
        for stage, heads in enumerate(STAGE_HEADS)
    ]
    if any(difference < 0 for difference in per_head_differences):
        raise ValueError("CvT target stage widths must not be smaller than seed widths")
    positive = [difference for difference in per_head_differences if difference > 0]
    if not positive:
        raise ValueError("CvT seed and target stage widths are identical")
    maximum = min(positive)
    if grow_steps > maximum:
        raise ValueError(
            f"CvT grow_steps={grow_steps} exceeds the smallest per-head stage "
            f"increase {per_head_differences}. Use at most {maximum}; every "
            "event must add at least one dimension to each attention head in "
            "every growable stage."
        )

    plan = [[0, 0, 0] for _ in range(grow_steps)]
    for stage, (heads, difference) in enumerate(zip(STAGE_HEADS, per_head_differences)):
        base, remainder = divmod(difference, grow_steps)
        for index in range(grow_steps):
            plan[index][stage] = (base + (index < remainder)) * heads
    return plan


def _safe_std(tensor: torch.Tensor) -> float:
    value = float(tensor.detach().float().std().item())
    return value if math.isfinite(value) and value > 0 else 1e-6


def _matrix_views(old, new, heads_out: int, heads_in: int):
    old_out, old_in = int(old.shape[0]), int(old.shape[1])
    new_out, new_in = int(new.shape[0]), int(new.shape[1])
    if any(
        value % heads != 0
        for value, heads in (
            (old_out, heads_out),
            (new_out, heads_out),
            (old_in, heads_in),
            (new_in, heads_in),
        )
    ):
        raise ValueError("CvT width is not divisible by the configured heads")
    old_ohd, new_ohd = old_out // heads_out, new_out // heads_out
    old_ihd, new_ihd = old_in // heads_in, new_in // heads_in
    extra = old.shape[2:]
    return (
        old.view(heads_out, old_ohd, heads_in, old_ihd, *extra),
        new.view(heads_out, new_ohd, heads_in, new_ihd, *extra),
        old_ohd,
        old_ihd,
    )


def _copy_matrix_state(old, new, heads_out: int, heads_in: int) -> None:
    new.zero_()
    old_view, new_view, old_ohd, old_ihd = _matrix_views(old, new, heads_out, heads_in)
    new_view[:, :old_ohd, :, :old_ihd].copy_(old_view)


def _kaiming_uniform_region(tensor: torch.Tensor, fan_in: int) -> None:
    if tensor.numel() > 0:
        bound = 1.0 / math.sqrt(max(1, fan_in))
        tensor.uniform_(-bound, bound)


def _copy_matrix_weight(old, new, heads_out: int, heads_in: int, mode: str) -> None:
    old_view, new_view, old_ohd, old_ihd = _matrix_views(old, new, heads_out, heads_in)
    if mode in {"mode_b", "mode_c"}:
        new_view.zero_()
    new_view[:, :old_ohd, :, :old_ihd].copy_(old_view)
    w_new1 = new_view[:, :old_ohd, :, old_ihd:]
    w_new2 = new_view[:, old_ohd:, :, :old_ihd]
    w_new3 = new_view[:, old_ohd:, :, old_ihd:]
    receptive = math.prod(old.shape[2:]) if old.ndim > 2 else 1
    fan_in_new = int(new.shape[1]) * receptive
    if mode == "mode_a":
        w_new1.zero_()
        _kaiming_uniform_region(w_new2, fan_in_new)
        _kaiming_uniform_region(w_new3, fan_in_new)
    elif mode == "mode_b":
        _kaiming_uniform_region(w_new2, int(old.shape[1]) * receptive)
    elif mode == "mode_c":
        _kaiming_uniform_region(w_new1, fan_in_new)
    elif mode == "mode_d":
        _kaiming_uniform_region(w_new1, fan_in_new)
        _kaiming_uniform_region(w_new2, fan_in_new)
        _kaiming_uniform_region(w_new3, fan_in_new)
    elif mode == "mode_e":
        std = _safe_std(old)
        if w_new1.numel() > 0:
            w_new1.normal_(0.0, std)
        new_rows = new_view[:, old_ohd:, :, :]
        if new_rows.numel() > 0:
            new_rows.normal_(0.0, std)
    else:
        raise ValueError(f"Unknown CvT mode: {mode}")


def _copy_vector_state(old, new, heads: int, axis: int = 0) -> None:
    new.zero_()
    old_size, new_size = int(old.shape[axis]), int(new.shape[axis])
    old_hd, new_hd = old_size // heads, new_size // heads
    old_shape, new_shape = list(old.shape), list(new.shape)
    old_shape[axis : axis + 1], new_shape[axis : axis + 1] = (
        [heads, old_hd],
        [heads, new_hd],
    )
    old_view, new_view = old.view(*old_shape), new.view(*new_shape)
    index = [slice(None)] * new_view.ndim
    index[axis + 1] = slice(0, old_hd)
    new_view[tuple(index)].copy_(old_view)


def _copy_bias_weight(old, new, heads: int, mode: str) -> None:
    old_size, new_size = int(old.shape[0]), int(new.shape[0])
    if old_size % heads or new_size % heads:
        if mode != "mode_d":
            new.zero_()
        new[:old_size].copy_(old)
        return

    old_hd, new_hd = old_size // heads, new_size // heads
    old_view = old.view(heads, old_hd)
    new_view = new.view(heads, new_hd)
    if mode != "mode_d":
        new_view.zero_()
    new_view[:, :old_hd].copy_(old_view)
    if mode == "mode_e" and new_hd > old_hd:
        for head in range(heads):
            new_view[head, old_hd:].normal_(0.0, _safe_std(old_view[head]))


def _copy_layernorm_weight(
    old, new, heads: int, *, is_gain: bool, mode: str
) -> None:
    if is_gain and mode != "mode_c":
        new.fill_(1.0)
    else:
        new.zero_()
    old_size, new_size = int(old.shape[1]), int(new.shape[1])
    if old_size % heads or new_size % heads:
        new[:, :old_size].copy_(old)
        return
    old_hd, new_hd = old_size // heads, new_size // heads
    old_view = old.view(1, heads, old_hd, 1, 1)
    new_view = new.view(1, heads, new_hd, 1, 1)
    new_view[:, :, :old_hd].copy_(old_view)


def _copy_depthwise_weight(old, new, heads: int, mode: str) -> None:
    old_size, new_size = int(old.shape[0]), int(new.shape[0])
    if old_size % heads or new_size % heads:
        new[:old_size].copy_(old)
        return
    old_hd, new_hd = old_size // heads, new_size // heads
    old_view = old.view(heads, old_hd, *old.shape[1:])
    new_view = new.view(heads, new_hd, *new.shape[1:])
    new_view[:, :old_hd].copy_(old_view)
    if mode == "mode_e" and new_hd > old_hd:
        for head in range(heads):
            new_view[head, old_hd:].normal_(0.0, _safe_std(old_view[head]))


def _copy_classifier_weight(old, new, heads: int, mode: str) -> None:
    old_out, old_in = int(old.shape[0]), int(old.shape[1])
    new_out, new_in = int(new.shape[0]), int(new.shape[1])
    if old_out != new_out:
        raise ValueError("CvT classifier output size changed during width growth")
    if old_in % heads or new_in % heads:
        if mode != "mode_d":
            new.zero_()
        new[:, :old_in].copy_(old)
        added = new[:, old_in:]
        if mode == "mode_c":
            _kaiming_uniform_region(added, new_in)
        elif mode == "mode_e" and added.numel() > 0:
            added.normal_(0.0, _safe_std(old))
        return

    old_hd, new_hd = old_in // heads, new_in // heads
    old_view = old.view(old_out, heads, old_hd)
    new_view = new.view(new_out, heads, new_hd)
    if mode != "mode_d":
        new_view.zero_()
    new_view[:, :, :old_hd].copy_(old_view)
    added = new_view[:, :, old_hd:]
    if mode == "mode_c":
        _kaiming_uniform_region(added, new_in)
    elif mode == "mode_e" and new_hd > old_hd:
        for head in range(heads):
            added[:, head].normal_(0.0, _safe_std(old_view[:, head]))


def _copy_input_weight(old, new, groups: int, mode: str) -> None:
    old_out, old_in = int(old.shape[0]), int(old.shape[1])
    new_out, new_in = int(new.shape[0]), int(new.shape[1])
    if old_out != new_out:
        raise ValueError("CvT auxiliary-head output size changed during width growth")
    if old_in % groups or new_in % groups:
        if mode != "mode_d":
            new.zero_()
        new[:, :old_in].copy_(old)
        added = new[:, old_in:]
    else:
        old_hd, new_hd = old_in // groups, new_in // groups
        old_view = old.view(old_out, groups, old_hd)
        new_view = new.view(new_out, groups, new_hd)
        if mode != "mode_d":
            new_view.zero_()
        new_view[:, :, :old_hd].copy_(old_view)
        added = new_view[:, :, old_hd:]
    if mode == "mode_c":
        _kaiming_uniform_region(added, new_in)
    elif mode == "mode_e" and added.numel() > 0:
        added.normal_(0.0, _safe_std(old))


def _matrix_copier(heads_out: int, heads_in: int) -> TensorCopier:
    return lambda old, new: _copy_matrix_state(old, new, heads_out, heads_in)


def _vector_copier(heads: int, axis: int = 0) -> TensorCopier:
    return lambda old, new: _copy_vector_state(old, new, heads, axis)


@torch.no_grad()
def _transfer_cvt(old_model: CvT, new_model: CvT, mode: str) -> None:
    registry: list[tuple[nn.Parameter, nn.Parameter, TensorCopier]] = []
    handled: set[nn.Parameter] = set()

    def register(old_p, new_p, copier: TensorCopier):
        registry.append((old_p, new_p, copier))
        handled.add(new_p)

    def matrix(old_p, new_p, heads_out: int, heads_in: int):
        _copy_matrix_weight(old_p.data, new_p.data, heads_out, heads_in, mode)
        register(old_p, new_p, _matrix_copier(heads_out, heads_in))

    def bias(old_p, new_p, heads: int):
        _copy_bias_weight(old_p.data, new_p.data, heads, mode)
        register(old_p, new_p, _vector_copier(heads))

    def layernorm(old_norm: ChannelLayerNorm, new_norm: ChannelLayerNorm, heads: int):
        _copy_layernorm_weight(
            old_norm.g.data, new_norm.g.data, heads, is_gain=True, mode=mode
        )
        _copy_layernorm_weight(
            old_norm.b.data, new_norm.b.data, heads, is_gain=False, mode=mode
        )
        register(old_norm.g, new_norm.g, _vector_copier(heads, axis=1))
        register(old_norm.b, new_norm.b, _vector_copier(heads, axis=1))

    def batchnorm(old_bn: nn.BatchNorm2d, new_bn: nn.BatchNorm2d, heads: int):
        new_bn.weight.fill_(1.0)
        new_bn.bias.zero_()
        for old_parameter, new_parameter in (
            (old_bn.weight, new_bn.weight),
            (old_bn.bias, new_bn.bias),
        ):
            new_parameter[: old_parameter.numel()].copy_(old_parameter)
            register(old_parameter, new_parameter, _vector_copier(heads))
        new_bn.running_mean.zero_()
        new_bn.running_var.fill_(1.0)
        new_bn.running_mean[: old_bn.num_features].copy_(old_bn.running_mean)
        new_bn.running_var[: old_bn.num_features].copy_(old_bn.running_var)

    def depthwise(
        old_dw: DepthWiseConv2d, new_dw: DepthWiseConv2d, heads_in: int, heads_out: int
    ):
        old_conv, new_conv = old_dw.net[0], new_dw.net[0]
        _copy_depthwise_weight(
            old_conv.weight.data, new_conv.weight.data, heads_in, mode
        )
        register(old_conv.weight, new_conv.weight, _vector_copier(heads_in))
        if old_conv.bias is not None:
            bias(old_conv.bias, new_conv.bias, heads_in)
        batchnorm(old_dw.net[1], new_dw.net[1], heads_in)
        matrix(old_dw.net[2].weight, new_dw.net[2].weight, heads_out, heads_in)
        if old_dw.net[2].bias is not None:
            bias(old_dw.net[2].bias, new_dw.net[2].bias, heads_out)

    for stage_index, (old_stage, new_stage) in enumerate(
        zip(old_model.layers, new_model.layers)
    ):
        heads_out = STAGE_HEADS[stage_index]
        heads_in = 1 if stage_index == 0 else STAGE_HEADS[stage_index - 1]
        matrix(old_stage[0].weight, new_stage[0].weight, heads_out, heads_in)
        bias(old_stage[0].bias, new_stage[0].bias, heads_out)
        layernorm(old_stage[1], new_stage[1], heads_out)

        for old_block, new_block in zip(old_stage[2].layers, new_stage[2].layers):
            old_attention_norm, old_ff_norm = old_block
            new_attention_norm, new_ff_norm = new_block
            for old_norm, new_norm in (
                (old_attention_norm.norm, new_attention_norm.norm),
                (old_ff_norm.norm, new_ff_norm.norm),
            ):
                layernorm(old_norm, new_norm, heads_out)

            old_attention, new_attention = old_attention_norm.fn, new_attention_norm.fn
            depthwise(old_attention.to_q, new_attention.to_q, heads_out, heads_out)
            depthwise(
                old_attention.to_kv, new_attention.to_kv, heads_out, 2 * heads_out
            )
            matrix(
                old_attention.to_out[0].weight,
                new_attention.to_out[0].weight,
                heads_out,
                heads_out,
            )
            bias(old_attention.to_out[0].bias, new_attention.to_out[0].bias, heads_out)

            old_ff, new_ff = old_ff_norm.fn, new_ff_norm.fn
            matrix(old_ff.net[0].weight, new_ff.net[0].weight, heads_out, heads_out)
            bias(old_ff.net[0].bias, new_ff.net[0].bias, heads_out)
            matrix(old_ff.net[3].weight, new_ff.net[3].weight, heads_out, heads_out)
            bias(old_ff.net[3].bias, new_ff.net[3].bias, heads_out)

    _copy_classifier_weight(
        old_model.head.weight.data,
        new_model.head.weight.data,
        STAGE_HEADS[-1],
        mode,
    )
    register(
        old_model.head.weight,
        new_model.head.weight,
        _matrix_copier(1, STAGE_HEADS[-1]),
    )
    new_model.head.bias.copy_(old_model.head.bias)
    register(old_model.head.bias, new_model.head.bias, lambda old, new: new.copy_(old))

    if old_model.drloc is not None and new_model.drloc is not None:
        old_input = old_model.drloc.layers[0]
        new_input = new_model.drloc.layers[0]
        groups = 2 * STAGE_HEADS[-1]
        _copy_input_weight(old_input.weight.data, new_input.weight.data, groups, mode)
        register(old_input.weight, new_input.weight, _matrix_copier(1, groups))
        new_input.bias.copy_(old_input.bias)
        register(old_input.bias, new_input.bias, lambda old, new: new.copy_(old))

    old_named = dict(old_model.named_parameters())
    for name, new_parameter in new_model.named_parameters():
        if new_parameter in handled or name not in old_named:
            continue
        old_parameter = old_named[name]
        if old_parameter.shape == new_parameter.shape:
            new_parameter.copy_(old_parameter)
            register(old_parameter, new_parameter, lambda old, new: new.copy_(old))

    new_model._growth_registry = registry


def grow_model(
    model: CvT,
    mode: str,
    additions: Sequence[int],
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
    grow_batch_size: int,
) -> CvT:
    del x, y, loss_fn, grow_batch_size
    target_dims = tuple(
        int(current) + (int(additions[index]) if index < len(additions) else 0)
        for index, current in enumerate(model.arch["dims"])
    )
    new_model = CvT(
        num_classes=model.arch["num_classes"],
        dims=target_dims,
        depths=model.arch["depths"],
        heads=model.arch["heads"],
        dropout=model.arch["dropout"],
        use_drloc=model.arch["use_drloc"],
        sample_size=model.arch["sample_size"],
    ).to(next(model.parameters()).device)
    _transfer_cvt(model, new_model, mode)
    return new_model


def _parameter_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim == 1 or name.endswith("bias") else decay).append(
            parameter
        )
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def make_optimizer_scheduler(
    model, config, total_steps: int, learning_rate: float | None = None
):
    optimizer = optim.AdamW(
        _parameter_groups(model, config.weight_decay),
        lr=config.learning_rate if learning_rate is None else learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    warmup_steps = int(config.warmup_epochs * total_steps / max(1, config.num_epochs))
    warmup_ratio = 1e-3
    min_ratio = 0.01

    def schedule(step: int) -> float:
        update = max(0, step - 1)
        if update < warmup_steps:
            return warmup_ratio + update * (1.0 - warmup_ratio) / max(
                1, warmup_steps
            )
        if update >= total_steps:
            return min_ratio
        return min_ratio + (1.0 - min_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * update / max(1, total_steps))
        )

    return optimizer, optim.lr_scheduler.LambdaLR(optimizer, schedule)


class _CvTLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        try:
            from timm.data import Mixup
            from timm.data.mixup import mixup_target
        except ImportError as error:
            raise RuntimeError(
                "CvT training requires timm; install requirements.txt before "
                "running CvT experiments."
            ) from error

        class DeviceAwareMixup(Mixup):
            def __call__(self, x, target):
                if len(x) % 2:
                    raise ValueError("CvT Mixup/CutMix requires an even batch size")
                if self.mode == "elem":
                    lam = self._mix_elem(x)
                elif self.mode == "pair":
                    lam = self._mix_pair(x)
                else:
                    lam = self._mix_batch(x)
                mixed_target = mixup_target(
                    target,
                    self.num_classes,
                    lam,
                    self.label_smoothing,
                    device=target.device,
                )
                return x, mixed_target

        num_classes = 100 if config.dataset == "cifar100" else 10
        self.mixup = DeviceAwareMixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=config.label_smoothing,
            num_classes=num_classes,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 1:
            return F.cross_entropy(logits, target)
        return torch.sum(-target * F.log_softmax(logits, dim=-1), dim=-1).mean()


def build_loss(config) -> nn.Module:
    return _CvTLoss(config)


def compute_training_loss(model, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module):
    if not hasattr(loss_fn, "mixup"):
        raise TypeError("CvT training loss must be created by build_loss()")
    mixed, soft_targets = loss_fn.mixup(x.clone(), y)
    logits = model(mixed)
    return logits, loss_fn(logits, soft_targets)


def _transfer_optimizer_state(
    old_optimizer, new_optimizer, old_model, new_model
) -> None:
    handled: set[nn.Parameter] = set()
    for old_parameter, new_parameter, copier in new_model._growth_registry:
        if old_parameter not in old_optimizer.state:
            continue
        for key, value in old_optimizer.state[old_parameter].items():
            if not torch.is_tensor(value) or value.ndim == 0:
                new_optimizer.state[new_parameter][key] = (
                    value.detach().clone()
                    if torch.is_tensor(value)
                    else copy.deepcopy(value)
                )
                continue
            buffer = torch.zeros(
                new_parameter.shape, device=new_parameter.device, dtype=value.dtype
            )
            copier(value.to(new_parameter.device), buffer)
            new_optimizer.state[new_parameter][key] = buffer
        handled.add(new_parameter)

    old_named = dict(old_model.named_parameters())
    for name, new_parameter in new_model.named_parameters():
        if new_parameter in handled or name not in old_named:
            continue
        old_parameter = old_named[name]
        if old_parameter not in old_optimizer.state:
            continue
        for key, value in old_optimizer.state[old_parameter].items():
            if not torch.is_tensor(value) or value.ndim == 0:
                new_optimizer.state[new_parameter][key] = (
                    value.detach().clone()
                    if torch.is_tensor(value)
                    else copy.deepcopy(value)
                )
            else:
                buffer = torch.zeros_like(new_parameter, dtype=value.dtype)
                slices = tuple(
                    slice(0, min(a, b)) for a, b in zip(value.shape, buffer.shape)
                )
                buffer[slices].copy_(value.to(buffer.device)[slices])
                new_optimizer.state[new_parameter][key] = buffer


def rebuild_optimizer_scheduler_after_growth(
    old_model,
    new_model,
    old_optimizer,
    old_scheduler,
    config,
    total_steps: int,
):
    new_optimizer, new_scheduler = make_optimizer_scheduler(
        new_model, config, total_steps
    )
    _transfer_optimizer_state(old_optimizer, new_optimizer, old_model, new_model)
    new_scheduler.load_state_dict(old_scheduler.state_dict())
    for new_group, old_group in zip(
        new_optimizer.param_groups, old_optimizer.param_groups
    ):
        new_group["lr"] = old_group["lr"]
    delattr(new_model, "_growth_registry")
    return new_optimizer, new_scheduler
