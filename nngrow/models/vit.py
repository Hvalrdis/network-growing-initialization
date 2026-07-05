"""Vision Transformer construction and embedding-dimension growth.

The module implements Initialization Modes A-E, head-wise parameter transfer,
AdamW-state transfer, CutMix, and the warmup-cosine schedule.
"""

from __future__ import annotations

import copy
import inspect
import math
from collections import OrderedDict
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import VisionTransformer


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

NUM_HEADS = 4
NUM_LAYERS = 8
FINAL_HIDDEN_DIM = 256
PATCH_SIZE = 2
MLP_RATIO = 4
DROP_PATH_RATE = 0.1

TensorCopier = Callable[[torch.Tensor, torch.Tensor], None]


def _safe_std(tensor: torch.Tensor) -> float:
    value = float(tensor.detach().float().std().item())
    return value if math.isfinite(value) and value > 0 else 1e-6


def _xavier_region(tensor: torch.Tensor, fan_in: int, fan_out: int) -> None:
    if tensor.numel() == 0:
        return
    limit = math.sqrt(6.0 / max(1, int(fan_in) + int(fan_out)))
    tensor.uniform_(-limit, limit)


def _matrix_views(
    old: torch.Tensor,
    new: torch.Tensor,
    heads_out: int,
    heads_in: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
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
        raise ValueError("A ViT growth dimension is not divisible by its head count")
    old_ohd, new_ohd = old_out // heads_out, new_out // heads_out
    old_ihd, new_ihd = old_in // heads_in, new_in // heads_in
    extra = old.shape[2:]
    old_view = old.view(heads_out, old_ohd, heads_in, old_ihd, *extra)
    new_view = new.view(heads_out, new_ohd, heads_in, new_ihd, *extra)
    return old_view, new_view, old_ohd, old_ihd


def _copy_matrix_state(
    old: torch.Tensor,
    new: torch.Tensor,
    heads_out: int,
    heads_in: int,
) -> None:
    new.zero_()
    old_view, new_view, old_ohd, old_ihd = _matrix_views(old, new, heads_out, heads_in)
    new_view[:, :old_ohd, :, :old_ihd].copy_(old_view)


def _copy_matrix_weight(
    old: torch.Tensor,
    new: torch.Tensor,
    heads_out: int,
    heads_in: int,
    mode: str,
) -> None:
    old_view, new_view, old_ohd, old_ihd = _matrix_views(old, new, heads_out, heads_in)
    if mode in {"mode_b", "mode_c"}:
        new_view.zero_()
    new_view[:, :old_ohd, :, :old_ihd].copy_(old_view)

    w_new1 = new_view[:, :old_ohd, :, old_ihd:]
    w_new2 = new_view[:, old_ohd:, :, :old_ihd]
    w_new3 = new_view[:, old_ohd:, :, old_ihd:]
    receptive = math.prod(old.shape[2:]) if old.ndim > 2 else 1
    fan_in = int(new.shape[1]) * receptive
    fan_out = int(new.shape[0]) * receptive

    if mode == "mode_a":
        w_new1.zero_()
    elif mode == "mode_b":
        _xavier_region(w_new2, int(old.shape[1]) * receptive, fan_out)
    elif mode == "mode_c":
        _xavier_region(w_new1, fan_in, fan_out)
    elif mode == "mode_d":
        # The expanded module already carries the default initialization.
        pass
    elif mode == "mode_e":
        std = _safe_std(old)
        for region in (w_new1, w_new2, w_new3):
            if region.numel() > 0:
                region.normal_(0.0, std)
    else:
        raise ValueError(f"Unknown ViT mode: {mode}")


def _copy_channel_state(
    old: torch.Tensor,
    new: torch.Tensor,
    heads: int,
    axis: int,
) -> None:
    new.zero_()
    old_channels, new_channels = int(old.shape[axis]), int(new.shape[axis])
    old_hd, new_hd = old_channels // heads, new_channels // heads
    old_shape = list(old.shape)
    new_shape = list(new.shape)
    old_shape[axis : axis + 1] = [heads, old_hd]
    new_shape[axis : axis + 1] = [heads, new_hd]
    old_view, new_view = old.view(*old_shape), new.view(*new_shape)
    index = [slice(None)] * new_view.ndim
    index[axis + 1] = slice(0, old_hd)
    new_view[tuple(index)].copy_(old_view)


def _copy_channel_weight(
    old: torch.Tensor,
    new: torch.Tensor,
    heads: int,
    axis: int,
    mode: str,
    zero_new: bool,
) -> None:
    if zero_new or mode == "mode_c":
        new.zero_()
    _copy_channel_state(old, new, heads, axis) if (
        zero_new or mode == "mode_c"
    ) else _copy_channel_overlap(old, new, heads, axis)
    if mode == "mode_e":
        old_channels, new_channels = int(old.shape[axis]), int(new.shape[axis])
        old_hd, new_hd = old_channels // heads, new_channels // heads
        shape = list(new.shape)
        shape[axis : axis + 1] = [heads, new_hd]
        view = new.view(*shape)
        index = [slice(None)] * view.ndim
        index[axis + 1] = slice(old_hd, None)
        region = view[tuple(index)]
        if region.numel() > 0:
            region.normal_(0.0, _safe_std(old))


def _copy_channel_overlap(
    old: torch.Tensor, new: torch.Tensor, heads: int, axis: int
) -> None:
    old_channels, new_channels = int(old.shape[axis]), int(new.shape[axis])
    old_hd, new_hd = old_channels // heads, new_channels // heads
    old_shape, new_shape = list(old.shape), list(new.shape)
    old_shape[axis : axis + 1] = [heads, old_hd]
    new_shape[axis : axis + 1] = [heads, new_hd]
    old_view, new_view = old.view(*old_shape), new.view(*new_shape)
    index = [slice(None)] * new_view.ndim
    index[axis + 1] = slice(0, old_hd)
    new_view[tuple(index)].copy_(old_view)


def _matrix_copier(heads_out: int, heads_in: int) -> TensorCopier:
    return lambda old, new: _copy_matrix_state(old, new, heads_out, heads_in)


def _channel_copier(heads: int, axis: int) -> TensorCopier:
    return lambda old, new: _copy_channel_state(old, new, heads, axis)


def _build_from_config(config: dict, device: torch.device) -> VisionTransformer:
    kwargs = dict(config)
    try:
        if "stochastic_depth_prob" in inspect.signature(VisionTransformer).parameters:
            kwargs["stochastic_depth_prob"] = DROP_PATH_RATE
    except (TypeError, ValueError):
        pass
    model = VisionTransformer(**kwargs).to(device)
    model.config = dict(config)
    return model


def build_model(
    num_classes: int,
    device: torch.device,
    width_multiplier: float,
    seed: int,
) -> VisionTransformer:
    torch.manual_seed(int(seed))
    head_dim = max(1, int(round((FINAL_HIDDEN_DIM * width_multiplier) / NUM_HEADS)))
    hidden_dim = NUM_HEADS * head_dim
    config = {
        "image_size": 32,
        "patch_size": PATCH_SIZE,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "hidden_dim": hidden_dim,
        "mlp_dim": MLP_RATIO * hidden_dim,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "num_classes": num_classes,
    }
    return _build_from_config(config, device)


def preprocess(x: torch.Tensor) -> torch.Tensor:
    return x


def growth_plan(
    seed_model: VisionTransformer, big_model: VisionTransformer, grow_steps: int
) -> list[list[int]]:
    old_head_dim = int(seed_model.config["hidden_dim"]) // NUM_HEADS
    final_head_dim = int(big_model.config["hidden_dim"]) // NUM_HEADS
    difference = final_head_dim - old_head_dim
    if grow_steps <= 0:
        raise ValueError("ViT grow_steps must be positive")
    if difference <= 0:
        raise ValueError(
            "ViT target per-head dimension must be larger than the seed "
            "per-head dimension"
        )
    if grow_steps > difference:
        raise ValueError(
            f"ViT grow_steps={grow_steps} exceeds the per-head dimension "
            f"increase ({old_head_dim} to {final_head_dim}). Use at most "
            f"{difference}; each event must add at least one dimension to "
            f"each of the {NUM_HEADS} attention heads."
        )
    plan = []
    for index in range(grow_steps):
        base, remainder = divmod(difference, grow_steps)
        plan.append([(base + (index < remainder)) * NUM_HEADS])
    return plan


@torch.no_grad()
def _transfer_vit(
    old_model: VisionTransformer, new_model: VisionTransformer, mode: str
) -> None:
    registry: list[tuple[nn.Parameter, nn.Parameter, TensorCopier]] = []

    def matrix(
        old_p: nn.Parameter, new_p: nn.Parameter, heads_out: int, heads_in: int
    ) -> None:
        _copy_matrix_weight(old_p.data, new_p.data, heads_out, heads_in, mode)
        registry.append((old_p, new_p, _matrix_copier(heads_out, heads_in)))

    def channel(
        old_p: nn.Parameter,
        new_p: nn.Parameter,
        axis: int,
        *,
        zero_new: bool,
    ) -> None:
        _copy_channel_weight(old_p.data, new_p.data, NUM_HEADS, axis, mode, zero_new)
        registry.append((old_p, new_p, _channel_copier(NUM_HEADS, axis)))

    matrix(old_model.conv_proj.weight, new_model.conv_proj.weight, NUM_HEADS, 1)
    channel(old_model.conv_proj.bias, new_model.conv_proj.bias, 0, zero_new=True)
    channel(
        old_model.class_token,
        new_model.class_token,
        2,
        zero_new=mode in {"mode_a", "mode_b", "mode_c"},
    )
    channel(
        old_model.encoder.pos_embedding,
        new_model.encoder.pos_embedding,
        2,
        zero_new=mode in {"mode_a", "mode_b", "mode_c"},
    )

    for old_block, new_block in zip(old_model.encoder.layers, new_model.encoder.layers):
        for old_ln, new_ln in (
            (old_block.ln_1, new_block.ln_1),
            (old_block.ln_2, new_block.ln_2),
        ):
            channel(old_ln.weight, new_ln.weight, 0, zero_new=mode == "mode_c")
            channel(old_ln.bias, new_ln.bias, 0, zero_new=True)

        old_attn, new_attn = old_block.self_attention, new_block.self_attention
        old_dim, new_dim = (
            int(old_model.config["hidden_dim"]),
            int(new_model.config["hidden_dim"]),
        )

        def qkv_weight_copy(
            old: torch.Tensor, new: torch.Tensor, weight_mode: str
        ) -> None:
            for part in range(3):
                _copy_matrix_weight(
                    old[part * old_dim : (part + 1) * old_dim],
                    new[part * new_dim : (part + 1) * new_dim],
                    NUM_HEADS,
                    NUM_HEADS,
                    weight_mode,
                )

        qkv_weight_copy(
            old_attn.in_proj_weight.data, new_attn.in_proj_weight.data, mode
        )

        def qkv_state(old: torch.Tensor, new: torch.Tensor) -> None:
            for part in range(3):
                _copy_matrix_state(
                    old[part * old_dim : (part + 1) * old_dim],
                    new[part * new_dim : (part + 1) * new_dim],
                    NUM_HEADS,
                    NUM_HEADS,
                )

        registry.append((old_attn.in_proj_weight, new_attn.in_proj_weight, qkv_state))

        if old_attn.in_proj_bias is not None:
            for part in range(3):
                _copy_channel_weight(
                    old_attn.in_proj_bias.data[part * old_dim : (part + 1) * old_dim],
                    new_attn.in_proj_bias.data[part * new_dim : (part + 1) * new_dim],
                    NUM_HEADS,
                    0,
                    mode,
                    True,
                )

            def qkv_bias_state(old: torch.Tensor, new: torch.Tensor) -> None:
                for part in range(3):
                    _copy_channel_state(
                        old[part * old_dim : (part + 1) * old_dim],
                        new[part * new_dim : (part + 1) * new_dim],
                        NUM_HEADS,
                        0,
                    )

            registry.append(
                (old_attn.in_proj_bias, new_attn.in_proj_bias, qkv_bias_state)
            )

        matrix(old_attn.out_proj.weight, new_attn.out_proj.weight, NUM_HEADS, NUM_HEADS)
        channel(old_attn.out_proj.bias, new_attn.out_proj.bias, 0, zero_new=True)
        matrix(old_block.mlp[0].weight, new_block.mlp[0].weight, NUM_HEADS, NUM_HEADS)
        channel(old_block.mlp[0].bias, new_block.mlp[0].bias, 0, zero_new=True)
        matrix(old_block.mlp[3].weight, new_block.mlp[3].weight, NUM_HEADS, NUM_HEADS)
        channel(old_block.mlp[3].bias, new_block.mlp[3].bias, 0, zero_new=True)

    channel(
        old_model.encoder.ln.weight,
        new_model.encoder.ln.weight,
        0,
        zero_new=mode == "mode_c",
    )
    channel(old_model.encoder.ln.bias, new_model.encoder.ln.bias, 0, zero_new=True)
    matrix(old_model.heads.head.weight, new_model.heads.head.weight, 1, NUM_HEADS)
    if old_model.heads.head.bias is not None:
        new_model.heads.head.bias.copy_(old_model.heads.head.bias)
        registry.append(
            (
                old_model.heads.head.bias,
                new_model.heads.head.bias,
                lambda old, new: new.copy_(old),
            )
        )

    new_model._growth_registry = registry


def grow_model(
    model: VisionTransformer,
    mode: str,
    additions: Sequence[int],
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
    grow_batch_size: int,
) -> VisionTransformer:
    del x, y, loss_fn, grow_batch_size
    amount = int(additions[0]) if additions else 0
    if amount <= 0:
        return model
    config = copy.deepcopy(model.config)
    config["hidden_dim"] = int(config["hidden_dim"]) + amount
    config["mlp_dim"] = MLP_RATIO * int(config["hidden_dim"])
    new_model = _build_from_config(config, next(model.parameters()).device)
    _transfer_vit(model, new_model, mode)
    return new_model


def _parameter_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if (
            parameter.ndim == 1
            or name.endswith("bias")
            or "norm" in name.lower()
            or "ln" in name.lower()
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
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
        betas=(0.9, 0.99),
        eps=1e-8,
    )
    warmup_steps = int(config.warmup_epochs * total_steps / max(1, config.num_epochs))

    def schedule(step: int) -> float:
        current = step + 1
        if current <= warmup_steps:
            return current / max(1, warmup_steps)
        progress = (current - warmup_steps) / max(1, total_steps - warmup_steps)
        min_ratio = 1e-6 / config.learning_rate
        return min_ratio + (1.0 - min_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return optimizer, optim.lr_scheduler.LambdaLR(optimizer, schedule)


def build_loss(config) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)


def compute_training_loss(model, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module):
    lam = float(np.random.beta(1.0, 1.0))
    permutation = torch.randperm(x.size(0), device=x.device)
    height, width = x.size(-2), x.size(-1)
    ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(height * ratio), int(width * ratio)
    cy, cx = int(np.random.randint(height)), int(np.random.randint(width))
    y1, y2 = max(0, cy - cut_h // 2), min(height, cy + cut_h // 2)
    x1, x2 = max(0, cx - cut_w // 2), min(width, cx + cut_w // 2)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[permutation, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(height * width))
    logits = model(mixed)
    loss = lam * loss_fn(logits, y) + (1.0 - lam) * loss_fn(logits, y[permutation])
    return logits, loss


def _transfer_optimizer_state(
    old_optimizer, new_optimizer, old_model, new_model
) -> None:
    handled: set[nn.Parameter] = set()
    for old_parameter, new_parameter, copier in new_model._growth_registry:
        if old_parameter not in old_optimizer.state:
            continue
        new_state = new_optimizer.state[new_parameter]
        for key, value in old_optimizer.state[old_parameter].items():
            if not torch.is_tensor(value) or value.ndim == 0:
                new_state[key] = (
                    value.detach().clone()
                    if torch.is_tensor(value)
                    else copy.deepcopy(value)
                )
                continue
            buffer = torch.zeros(
                new_parameter.shape, device=new_parameter.device, dtype=value.dtype
            )
            copier(value.to(new_parameter.device), buffer)
            new_state[key] = buffer
        handled.add(new_parameter)

    old_named, new_named = (
        dict(old_model.named_parameters()),
        dict(new_model.named_parameters()),
    )
    for name, new_parameter in new_named.items():
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
    for group, learning_rate in zip(
        new_optimizer.param_groups, old_optimizer.param_groups
    ):
        group["lr"] = learning_rate["lr"]
    delattr(new_model, "_growth_registry")
    return new_optimizer, new_scheduler
