import math
import random

import numpy as np
import torch
import torch.nn as nn

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _safe_std(x: torch.Tensor, fallback: float = 1e-6) -> float:
    s = float(x.detach().std().item())
    if not math.isfinite(s) or s <= 0.0:
        return float(fallback)
    return s


def _xavier_uniform_(tensor: torch.Tensor, fan_in: int, fan_out: int) -> None:
    if tensor is None or tensor.numel() == 0:
        return
    limit = math.sqrt(6.0 / float(max(1, int(fan_in) + int(fan_out))))
    nn.init.uniform_(tensor, -limit, +limit)


def _zero_(tensor: torch.Tensor) -> None:
    if tensor is not None and tensor.numel() > 0:
        nn.init.zeros_(tensor)


def _nzz_init_4d_head_matrix(
    new_w_view: torch.Tensor,
    old_w_view: torch.Tensor,
    old_out_dim: int,
    old_in_dim: int,
    *,
    fan_in: int,
    fan_out: int,
) -> None:
    """
    Head-wise NZZ for a matrix viewed as
        (heads_out, heads_in, out_dim_per_head, in_dim_per_head).

    Blocks:
      old block : old_out x old_in       -> copy
      W_new1    : old_out x new_in       -> random (N)
      W_new2    : new_out x old_in       -> zero   (Z)
      W_new3    : new_out x new_in       -> zero   (Z)
    """
    new_w_view.zero_()
    new_w_view[:, :, :old_out_dim, :old_in_dim].copy_(old_w_view)

    # W_new1 = N: old output part receiving the newly added input dimensions.
    _xavier_uniform_(
        new_w_view[:, :, :old_out_dim, old_in_dim:],
        fan_in=fan_in,
        fan_out=fan_out,
    )

    # W_new2 and W_new3 are already zero because new_w_view.zero_() was called.


def adjust_input_embedding(old_model, new_model, mode=None):  # D,C,P,P
    """
    Patch embedding has no previous hidden dimension to provide a W_new1 block.
    For NZZ, the newly added embedding dimensions are therefore kept inactive:
    new output rows are set to zero, old rows are copied.
    """
    old_conv_proj = old_model.conv_proj
    new_conv_proj = new_model.conv_proj
    old_embed_dim = old_conv_proj.out_channels
    new_embed_dim = new_conv_proj.out_channels
    C = old_conv_proj.in_channels
    P = old_conv_proj.weight.size(2)
    num_heads = int(old_model.config['num_heads'])
    old_head_dim = old_embed_dim // num_heads
    new_head_dim = new_embed_dim // num_heads

    old_w = old_conv_proj.weight.data.view(num_heads, old_head_dim, C, P, P)
    new_w = new_conv_proj.weight.data.view(num_heads, new_head_dim, C, P, P)

    with torch.no_grad():
        new_w.zero_()
        new_w[:, :old_head_dim, :, :, :].copy_(old_w)
        new_conv_proj.weight.data.copy_(new_w.view(new_embed_dim, C, P, P))

        if old_conv_proj.bias is not None and new_conv_proj.bias is not None:
            old_b = old_conv_proj.bias.data.view(num_heads, old_head_dim)
            new_b = new_conv_proj.bias.data.view(num_heads, new_head_dim)
            new_b.zero_()
            new_b[:, :old_head_dim].copy_(old_b)
            new_conv_proj.bias.data.copy_(new_b.view(new_embed_dim))


def adjust_position_embedding(old_model, new_model, mode=None):
    """Copy old token/position embeddings and keep new hidden dimensions at zero."""
    old_pos = old_model.encoder.pos_embedding
    new_pos = new_model.encoder.pos_embedding
    seq_len = old_pos.shape[1]
    old_embed_dim = old_pos.shape[2]
    new_embed_dim = new_model.config['hidden_dim']
    num_heads = int(old_model.config['num_heads'])
    old_head_dim = old_embed_dim // num_heads
    new_head_dim = new_embed_dim // num_heads

    old_pos_h = old_pos.data.view(1, seq_len, num_heads, old_head_dim)
    new_pos_h = new_pos.data.view(1, seq_len, num_heads, new_head_dim)
    old_cls_h = old_model.class_token.data.view(1, 1, num_heads, old_head_dim)
    new_cls_h = new_model.class_token.data.view(1, 1, num_heads, new_head_dim)

    with torch.no_grad():
        new_pos_h.zero_()
        new_pos_h[:, :, :, :old_head_dim].copy_(old_pos_h)
        new_model.encoder.pos_embedding.data.copy_(new_pos_h.view(1, seq_len, new_embed_dim))

        new_cls_h.zero_()
        new_cls_h[:, :, :, :old_head_dim].copy_(old_cls_h)
        new_model.class_token.data.copy_(new_cls_h.view(1, 1, new_embed_dim))


def adjust_classification_head(old_model, new_model, mode=None):  # class,D
    """
    Classification head is the next layer after the expanded representation.
    NZZ therefore makes the new input columns random (W_new1=N).
    """
    old_head = old_model.heads.head
    new_head = new_model.heads.head
    old_w = old_head.weight.data
    new_w = new_head.weight.data
    num_classes = old_w.shape[0]
    old_embed_dim = old_head.in_features
    new_embed_dim = new_head.in_features
    num_heads = int(old_model.config['num_heads'])
    old_head_dim = old_embed_dim // num_heads
    new_head_dim = new_embed_dim // num_heads

    old_w_h = old_w.view(num_classes, num_heads, old_head_dim)
    new_w_h = new_w.view(num_classes, num_heads, new_head_dim)

    with torch.no_grad():
        new_w_h.zero_()
        new_w_h[:, :, :old_head_dim].copy_(old_w_h)
        _xavier_uniform_(new_w_h[:, :, old_head_dim:], fan_in=new_embed_dim, fan_out=num_classes)
        new_model.heads.head.weight.data.copy_(new_w_h.view(num_classes, new_embed_dim))

        if old_head.bias is not None and new_head.bias is not None:
            new_head.bias.data.copy_(old_head.bias.data)


def adjust_transformer_mlp(old_block, new_block, num_heads, old_embed_dim, new_embed_dim, mode=None):
    old_head_dim = old_embed_dim // num_heads
    new_head_dim = new_embed_dim // num_heads

    old_linear1 = old_block.mlp[0]
    old_linear2 = old_block.mlp[3]
    new_linear1 = new_block.mlp[0]
    new_linear2 = new_block.mlp[3]

    old_mlp_dim = old_linear1.out_features
    new_mlp_dim = new_linear1.out_features
    old_mlp_head_dim = old_mlp_dim // num_heads
    new_mlp_head_dim = new_mlp_dim // num_heads

    old_w1 = old_linear1.weight.data.reshape(num_heads, old_mlp_head_dim, num_heads, old_head_dim).permute(0, 2, 1, 3)
    old_w2 = old_linear2.weight.data.reshape(num_heads, old_head_dim, num_heads, old_mlp_head_dim).permute(0, 2, 1, 3)
    new_w1 = new_linear1.weight.data.reshape(num_heads, new_mlp_head_dim, num_heads, new_head_dim).permute(0, 2, 1, 3)
    new_w2 = new_linear2.weight.data.reshape(num_heads, new_head_dim, num_heads, new_mlp_head_dim).permute(0, 2, 1, 3)

    with torch.no_grad():
        _nzz_init_4d_head_matrix(
            new_w1,
            old_w1,
            old_out_dim=old_mlp_head_dim,
            old_in_dim=old_head_dim,
            fan_in=new_embed_dim,
            fan_out=new_mlp_dim,
        )
        _nzz_init_4d_head_matrix(
            new_w2,
            old_w2,
            old_out_dim=old_head_dim,
            old_in_dim=old_mlp_head_dim,
            fan_in=new_mlp_dim,
            fan_out=new_embed_dim,
        )

        new_block.mlp[0].weight.data.copy_(new_w1.permute(0, 2, 1, 3).reshape(new_mlp_dim, new_embed_dim))
        new_block.mlp[3].weight.data.copy_(new_w2.permute(0, 2, 1, 3).reshape(new_embed_dim, new_mlp_dim))

        if new_linear1.bias is not None:
            old_b1 = old_linear1.bias.data.view(num_heads, old_mlp_head_dim)
            new_b1 = new_linear1.bias.data.view(num_heads, new_mlp_head_dim)
            new_b1.zero_()
            new_b1[:, :old_mlp_head_dim].copy_(old_b1)
            new_block.mlp[0].bias.data.copy_(new_b1.reshape(new_mlp_dim))

        if new_linear2.bias is not None:
            old_b2 = old_linear2.bias.data.view(num_heads, old_head_dim)
            new_b2 = new_linear2.bias.data.view(num_heads, new_head_dim)
            new_b2.zero_()
            new_b2[:, :old_head_dim].copy_(old_b2)
            new_block.mlp[3].bias.data.copy_(new_b2.reshape(new_embed_dim))


def adjust_transformer_ln(old_block, new_block, old_embed_dim, new_embed_dim, mode=None):
    old_attention = getattr(old_block, 'self_attention', None)
    if old_attention is None:
        raise RuntimeError('old_block has no self_attention; cannot infer num_heads for LN grow.')
    num_heads = int(getattr(old_attention, 'num_heads', old_embed_dim // old_attention.head_dim))
    old_head_dim = old_embed_dim // num_heads
    new_head_dim = new_embed_dim // num_heads

    for old_ln, new_ln in ((old_block.ln_1, new_block.ln_1), (old_block.ln_2, new_block.ln_2)):
        old_w = old_ln.weight.data.view(num_heads, old_head_dim)
        old_b = old_ln.bias.data.view(num_heads, old_head_dim)
        new_w = new_ln.weight.data.view(num_heads, new_head_dim)
        new_b = new_ln.bias.data.view(num_heads, new_head_dim)
        with torch.no_grad():
            new_w.zero_()
            new_b.zero_()
            new_w[:, :old_head_dim].copy_(old_w)
            new_b[:, :old_head_dim].copy_(old_b)


def adjust_transformer_blocks_fix_num_head(old_model, new_model, mode=None):
    for old_block, new_block in zip(old_model.encoder.layers, new_model.encoder.layers):
        old_attn = old_block.self_attention
        new_attn = new_block.self_attention
        old_head_dim = int(old_attn.head_dim)
        new_head_dim = int(new_attn.head_dim)
        old_embed_dim = int(old_model.config['hidden_dim'])
        new_embed_dim = int(new_model.config['hidden_dim'])
        num_heads = int(old_model.config['num_heads'])

        old_in_proj = old_attn.in_proj_weight.data
        new_in_proj = new_attn.in_proj_weight.data

        old_q = old_in_proj[:old_embed_dim, :old_embed_dim]
        old_k = old_in_proj[old_embed_dim:2 * old_embed_dim, :old_embed_dim]
        old_v = old_in_proj[2 * old_embed_dim:3 * old_embed_dim, :old_embed_dim]
        old_o = old_attn.out_proj.weight.data

        new_q = new_in_proj[:new_embed_dim, :new_embed_dim]
        new_k = new_in_proj[new_embed_dim:2 * new_embed_dim, :new_embed_dim]
        new_v = new_in_proj[2 * new_embed_dim:3 * new_embed_dim, :new_embed_dim]
        new_o = new_attn.out_proj.weight.data

        old_q_h = old_q.reshape(num_heads, old_head_dim, num_heads, old_head_dim).permute(0, 2, 1, 3)
        old_k_h = old_k.reshape(num_heads, old_head_dim, num_heads, old_head_dim).permute(0, 2, 1, 3)
        old_v_h = old_v.reshape(num_heads, old_head_dim, num_heads, old_head_dim).permute(0, 2, 1, 3)
        old_o_h = old_o.reshape(num_heads, old_head_dim, num_heads, old_head_dim).permute(0, 2, 1, 3)

        new_q_h = new_q.reshape(num_heads, new_head_dim, num_heads, new_head_dim).permute(0, 2, 1, 3)
        new_k_h = new_k.reshape(num_heads, new_head_dim, num_heads, new_head_dim).permute(0, 2, 1, 3)
        new_v_h = new_v.reshape(num_heads, new_head_dim, num_heads, new_head_dim).permute(0, 2, 1, 3)
        new_o_h = new_o.reshape(num_heads, new_head_dim, num_heads, new_head_dim).permute(0, 2, 1, 3)

        with torch.no_grad():
            for new_h, old_h in ((new_q_h, old_q_h), (new_k_h, old_k_h), (new_v_h, old_v_h), (new_o_h, old_o_h)):
                _nzz_init_4d_head_matrix(
                    new_h,
                    old_h,
                    old_out_dim=old_head_dim,
                    old_in_dim=old_head_dim,
                    fan_in=new_embed_dim,
                    fan_out=new_embed_dim,
                )

            q = new_q_h.permute(0, 2, 1, 3).reshape(new_embed_dim, new_embed_dim)
            k = new_k_h.permute(0, 2, 1, 3).reshape(new_embed_dim, new_embed_dim)
            v = new_v_h.permute(0, 2, 1, 3).reshape(new_embed_dim, new_embed_dim)
            o = new_o_h.permute(0, 2, 1, 3).reshape(new_embed_dim, new_embed_dim)
            new_attn.in_proj_weight.data.copy_(torch.cat([q, k, v], dim=0).to(new_in_proj.device))
            new_attn.out_proj.weight.data.copy_(o)

            if old_attn.in_proj_bias is not None and new_attn.in_proj_bias is not None:
                old_b = old_attn.in_proj_bias.data
                new_b = new_attn.in_proj_bias.data
                for offset_old, offset_new in ((0, 0), (old_embed_dim, new_embed_dim), (2 * old_embed_dim, 2 * new_embed_dim)):
                    old_part = old_b[offset_old:offset_old + old_embed_dim].view(num_heads, old_head_dim)
                    new_part = new_b[offset_new:offset_new + new_embed_dim].view(num_heads, new_head_dim)
                    new_part.zero_()
                    new_part[:, :old_head_dim].copy_(old_part)

            if old_attn.out_proj.bias is not None and new_attn.out_proj.bias is not None:
                old_b = old_attn.out_proj.bias.data.view(num_heads, old_head_dim)
                new_b = new_attn.out_proj.bias.data.view(num_heads, new_head_dim)
                new_b.zero_()
                new_b[:, :old_head_dim].copy_(old_b)
                new_attn.out_proj.bias.data.copy_(new_b.reshape(new_embed_dim))

        adjust_transformer_mlp(old_block, new_block, num_heads, old_embed_dim, new_embed_dim, mode=None)
        adjust_transformer_ln(old_block, new_block, old_embed_dim, new_embed_dim, mode=None)


def initialize_encoder_layernorm(old_model, new_model, mode=None):
    old_ln = old_model.encoder.ln
    new_ln = new_model.encoder.ln
    old_embed_dim = old_ln.normalized_shape[0]
    new_embed_dim = new_ln.normalized_shape[0]

    num_heads = None
    if hasattr(old_model, 'config') and isinstance(old_model.config, dict) and ('num_heads' in old_model.config):
        num_heads = int(old_model.config['num_heads'])
    if num_heads is None:
        first_attn = old_model.encoder.layers[0].self_attention
        num_heads = int(getattr(first_attn, 'num_heads', old_embed_dim // first_attn.head_dim))

    old_head_dim = old_embed_dim // num_heads
    new_head_dim = new_embed_dim // num_heads

    old_w = old_ln.weight.data.view(num_heads, old_head_dim)
    old_b = old_ln.bias.data.view(num_heads, old_head_dim)
    new_w = new_ln.weight.data.view(num_heads, new_head_dim)
    new_b = new_ln.bias.data.view(num_heads, new_head_dim)

    with torch.no_grad():
        new_w.zero_()
        new_b.zero_()
        new_w[:, :old_head_dim].copy_(old_w)
        new_b[:, :old_head_dim].copy_(old_b)
