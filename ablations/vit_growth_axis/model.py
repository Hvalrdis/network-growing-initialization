"""ViT adapter comparing growth of per-head dimension and head count.

Both variants start from H=4, d=16 (D=64) and end at D=256. Grow-d keeps
H=4, whereas Grow-H keeps d=16.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Callable, Sequence

import torch
import torch.nn as nn
from torchvision.models import VisionTransformer

from ...models import vit as base


MODE_SPECS = OrderedDict(
    [
        ("grow_d_b", "Grow-d (B)"),
        ("grow_d_d", "Grow-d (D)"),
        ("grow_h_b", "Grow-H (B)"),
        ("grow_h_d", "Grow-H (D)"),
    ]
)
GROW_BEFORE_STEP = True

TensorCopier = Callable[[torch.Tensor, torch.Tensor], None]


build_model = base.build_model
preprocess = base.preprocess
make_optimizer_scheduler = base.make_optimizer_scheduler
build_loss = base.build_loss
compute_training_loss = base.compute_training_loss


def growth_plan(
    seed_model: VisionTransformer,
    big_model: VisionTransformer,
    grow_steps: int,
) -> list[list[int]]:
    difference = int(big_model.config["hidden_dim"]) - int(
        seed_model.config["hidden_dim"]
    )
    quotient, remainder = divmod(difference, int(grow_steps))
    return [[quotient + (index < remainder)] for index in range(grow_steps)]


def _channel_mask(
    length: int,
    old_heads: int,
    new_heads: int,
    old_dim: int,
) -> torch.Tensor:
    if length % new_heads:
        raise ValueError(
            f"Channel dimension {length} is not divisible by {new_heads} heads"
        )
    new_dim = length // new_heads
    if old_heads > new_heads or old_dim > new_dim:
        raise ValueError("Growth cannot shrink the preserved head layout")
    mask = torch.zeros(new_heads, new_dim, dtype=torch.bool)
    mask[:old_heads, :old_dim] = True
    return mask.reshape(length)


def _new_row_mask(
    old: torch.Tensor,
    new: torch.Tensor,
    old_output_heads: int,
    new_output_heads: int,
    old_input_heads: int,
    new_input_heads: int,
) -> torch.Tensor:
    old_output_dim = int(old.shape[0]) // old_output_heads
    old_input_dim = int(old.shape[1]) // old_input_heads
    output_mask = _channel_mask(
        int(new.shape[0]),
        old_output_heads,
        new_output_heads,
        old_output_dim,
    ).to(new.device)
    input_mask = _channel_mask(
        int(new.shape[1]),
        old_input_heads,
        new_input_heads,
        old_input_dim,
    ).to(new.device)
    shape = (new.shape[0], new.shape[1], *([1] * (new.ndim - 2)))
    return ((~output_mask)[:, None] & input_mask[None, :]).view(shape)


def _copy_matrix_state(
    old: torch.Tensor,
    new: torch.Tensor,
    old_ho: int,
    new_ho: int,
    old_hi: int,
    new_hi: int,
) -> None:
    new.zero_()
    old_od, old_id = int(old.shape[0]) // old_ho, int(old.shape[1]) // old_hi
    new_od, new_id = int(new.shape[0]) // new_ho, int(new.shape[1]) // new_hi
    old_view = old.view(old_ho, old_od, old_hi, old_id, *old.shape[2:])
    new_view = new.view(new_ho, new_od, new_hi, new_id, *new.shape[2:])
    new_view[:old_ho, :old_od, :old_hi, :old_id].copy_(old_view)


def _copy_matrix_weight(
    old: torch.Tensor,
    new: torch.Tensor,
    old_ho: int,
    new_ho: int,
    old_hi: int,
    new_hi: int,
    mode: str,
) -> None:
    new_rows = _new_row_mask(old, new, old_ho, new_ho, old_hi, new_hi)
    if mode == "b":
        _copy_matrix_state(old, new, old_ho, new_ho, old_hi, new_hi)
    else:
        _copy_matrix_overlap(old, new, old_ho, new_ho, old_hi, new_hi)
    if mode == "b" and bool(new_rows.any()):
        receptive = math.prod(old.shape[2:]) if old.ndim > 2 else 1
        fan_in = int(old.shape[1]) * receptive
        fan_out = int(new.shape[0]) * receptive
        limit = math.sqrt(6.0 / max(1, fan_in + fan_out))
        random_values = torch.empty_like(new).uniform_(-limit, limit)
        new.copy_(torch.where(new_rows.expand_as(new), random_values, new))


def _copy_matrix_overlap(
    old: torch.Tensor,
    new: torch.Tensor,
    old_ho: int,
    new_ho: int,
    old_hi: int,
    new_hi: int,
) -> None:
    old_od, old_id = int(old.shape[0]) // old_ho, int(old.shape[1]) // old_hi
    new_od, new_id = int(new.shape[0]) // new_ho, int(new.shape[1]) // new_hi
    old_view = old.view(old_ho, old_od, old_hi, old_id, *old.shape[2:])
    new_view = new.view(new_ho, new_od, new_hi, new_id, *new.shape[2:])
    new_view[:old_ho, :old_od, :old_hi, :old_id].copy_(old_view)


def _copy_channel_state(
    old: torch.Tensor,
    new: torch.Tensor,
    old_heads: int,
    new_heads: int,
    axis: int,
) -> None:
    new.zero_()
    old_dim = int(old.shape[axis]) // old_heads
    new_dim = int(new.shape[axis]) // new_heads
    old_shape, new_shape = list(old.shape), list(new.shape)
    old_shape[axis : axis + 1] = [old_heads, old_dim]
    new_shape[axis : axis + 1] = [new_heads, new_dim]
    old_view, new_view = old.view(*old_shape), new.view(*new_shape)
    index = [slice(None)] * new_view.ndim
    index[axis] = slice(0, old_heads)
    index[axis + 1] = slice(0, old_dim)
    new_view[tuple(index)].copy_(old_view)


def _copy_channel_weight(
    old: torch.Tensor,
    new: torch.Tensor,
    old_heads: int,
    new_heads: int,
    axis: int,
    mode: str,
    zero_new: bool,
) -> None:
    if zero_new and mode == "b":
        _copy_channel_state(old, new, old_heads, new_heads, axis)
        return
    old_dim = int(old.shape[axis]) // old_heads
    new_dim = int(new.shape[axis]) // new_heads
    old_shape, new_shape = list(old.shape), list(new.shape)
    old_shape[axis : axis + 1] = [old_heads, old_dim]
    new_shape[axis : axis + 1] = [new_heads, new_dim]
    old_view, new_view = old.view(*old_shape), new.view(*new_shape)
    index = [slice(None)] * new_view.ndim
    index[axis] = slice(0, old_heads)
    index[axis + 1] = slice(0, old_dim)
    new_view[tuple(index)].copy_(old_view)


def _matrix_copier(
    old_ho: int,
    new_ho: int,
    old_hi: int,
    new_hi: int,
) -> TensorCopier:
    return lambda old, new: _copy_matrix_state(old, new, old_ho, new_ho, old_hi, new_hi)


def _channel_copier(old_heads: int, new_heads: int, axis: int) -> TensorCopier:
    return lambda old, new: _copy_channel_state(old, new, old_heads, new_heads, axis)


@torch.no_grad()
def _transfer(
    old_model: VisionTransformer, new_model: VisionTransformer, mode: str
) -> None:
    old_heads = int(old_model.config["num_heads"])
    new_heads = int(new_model.config["num_heads"])
    old_dim = int(old_model.config["hidden_dim"])
    new_dim = int(new_model.config["hidden_dim"])
    registry: list[tuple[nn.Parameter, nn.Parameter, TensorCopier]] = []

    def matrix(
        old_p: nn.Parameter,
        new_p: nn.Parameter,
        old_ho: int = old_heads,
        new_ho: int = new_heads,
        old_hi: int = old_heads,
        new_hi: int = new_heads,
    ) -> None:
        _copy_matrix_weight(
            old_p.data,
            new_p.data,
            old_ho,
            new_ho,
            old_hi,
            new_hi,
            mode,
        )
        registry.append((old_p, new_p, _matrix_copier(old_ho, new_ho, old_hi, new_hi)))

    def channel(
        old_p: nn.Parameter,
        new_p: nn.Parameter,
        axis: int,
        *,
        zero_new: bool,
    ) -> None:
        _copy_channel_weight(
            old_p.data,
            new_p.data,
            old_heads,
            new_heads,
            axis,
            mode,
            zero_new,
        )
        registry.append((old_p, new_p, _channel_copier(old_heads, new_heads, axis)))

    matrix(old_model.conv_proj.weight, new_model.conv_proj.weight, old_hi=1, new_hi=1)
    channel(old_model.conv_proj.bias, new_model.conv_proj.bias, 0, zero_new=True)
    channel(old_model.class_token, new_model.class_token, 2, zero_new=True)
    channel(
        old_model.encoder.pos_embedding,
        new_model.encoder.pos_embedding,
        2,
        zero_new=True,
    )

    for old_block, new_block in zip(old_model.encoder.layers, new_model.encoder.layers):
        for old_ln, new_ln in (
            (old_block.ln_1, new_block.ln_1),
            (old_block.ln_2, new_block.ln_2),
        ):
            channel(old_ln.weight, new_ln.weight, 0, zero_new=False)
            channel(old_ln.bias, new_ln.bias, 0, zero_new=True)

        old_attention, new_attention = (
            old_block.self_attention,
            new_block.self_attention,
        )
        for part in range(3):
            _copy_matrix_weight(
                old_attention.in_proj_weight.data[
                    part * old_dim : (part + 1) * old_dim
                ],
                new_attention.in_proj_weight.data[
                    part * new_dim : (part + 1) * new_dim
                ],
                old_heads,
                new_heads,
                old_heads,
                new_heads,
                mode,
            )

        def qkv_weight_state(old: torch.Tensor, new: torch.Tensor) -> None:
            for part in range(3):
                _copy_matrix_state(
                    old[part * old_dim : (part + 1) * old_dim],
                    new[part * new_dim : (part + 1) * new_dim],
                    old_heads,
                    new_heads,
                    old_heads,
                    new_heads,
                )

        registry.append(
            (
                old_attention.in_proj_weight,
                new_attention.in_proj_weight,
                qkv_weight_state,
            )
        )

        if old_attention.in_proj_bias is not None:
            for part in range(3):
                _copy_channel_weight(
                    old_attention.in_proj_bias.data[
                        part * old_dim : (part + 1) * old_dim
                    ],
                    new_attention.in_proj_bias.data[
                        part * new_dim : (part + 1) * new_dim
                    ],
                    old_heads,
                    new_heads,
                    0,
                    mode,
                    True,
                )

            def qkv_bias_state(old: torch.Tensor, new: torch.Tensor) -> None:
                for part in range(3):
                    _copy_channel_state(
                        old[part * old_dim : (part + 1) * old_dim],
                        new[part * new_dim : (part + 1) * new_dim],
                        old_heads,
                        new_heads,
                        0,
                    )

            registry.append(
                (
                    old_attention.in_proj_bias,
                    new_attention.in_proj_bias,
                    qkv_bias_state,
                )
            )

        matrix(old_attention.out_proj.weight, new_attention.out_proj.weight)
        channel(
            old_attention.out_proj.bias,
            new_attention.out_proj.bias,
            0,
            zero_new=True,
        )
        matrix(old_block.mlp[0].weight, new_block.mlp[0].weight)
        channel(old_block.mlp[0].bias, new_block.mlp[0].bias, 0, zero_new=True)
        matrix(old_block.mlp[3].weight, new_block.mlp[3].weight)
        channel(old_block.mlp[3].bias, new_block.mlp[3].bias, 0, zero_new=True)

    channel(
        old_model.encoder.ln.weight,
        new_model.encoder.ln.weight,
        0,
        zero_new=False,
    )
    channel(old_model.encoder.ln.bias, new_model.encoder.ln.bias, 0, zero_new=True)
    matrix(
        old_model.heads.head.weight,
        new_model.heads.head.weight,
        1,
        1,
        old_heads,
        new_heads,
    )
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
    variant: str,
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
    axis, mode = variant.split("_")[1:]
    config = dict(model.config)
    old_hidden = int(config["hidden_dim"])
    old_heads = int(config["num_heads"])
    new_hidden = old_hidden + amount
    if axis == "d":
        new_heads = old_heads
    elif axis == "h":
        head_dim = old_hidden // old_heads
        if new_hidden % head_dim:
            raise ValueError(
                "Grow-H requires each hidden-dimension increment to add whole heads"
            )
        new_heads = new_hidden // head_dim
    else:
        raise ValueError(f"Unknown ViT growth axis: {axis}")
    config.update(
        hidden_dim=new_hidden,
        mlp_dim=base.MLP_RATIO * new_hidden,
        num_heads=new_heads,
    )
    new_model = base._build_from_config(config, next(model.parameters()).device)
    _transfer(model, new_model, mode)
    return new_model


rebuild_optimizer_scheduler_after_growth = base.rebuild_optimizer_scheduler_after_growth
