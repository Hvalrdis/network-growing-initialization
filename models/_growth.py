"""Parameter-expansion utilities shared by the MLP, VGG, and WRN backbones."""

import torch


def add_zero_outputs(layer, count: int) -> None:
    """Append zero-initialized output channels or features."""
    k = int(count)
    if k <= 0:
        return

    W = layer.weight
    Cin = W.size(1)
    if isinstance(layer, torch.nn.Conv2d):
        kHeight, kWidth = W.size(2), W.size(3)
        newW = torch.zeros((k, Cin, kHeight, kWidth), device=W.device)
    else:
        newW = torch.zeros((k, Cin), device=W.device)
    layer.weight = torch.nn.Parameter(torch.concat((layer.weight, newW), dim=0))

    if layer.bias is not None:
        layer.bias = torch.nn.Parameter(
            torch.concat((layer.bias, torch.zeros(k, device=W.device)), dim=0)
        )

    if isinstance(layer, torch.nn.Conv2d):
        layer.out_channels += k
    else:
        layer.out_features += k


def add_batchnorm_channels(bn, count: int) -> None:
    """Append identity-initialized channels to a batch-normalization layer."""
    k = int(count)
    if k <= 0:
        return
    dev = bn.weight.device
    newWeight = torch.concat((bn.weight, torch.ones(k, device=dev)), dim=0)
    bn.weight = torch.nn.Parameter(newWeight)
    newBias = torch.concat((bn.bias, torch.zeros(k, device=dev)), dim=0)
    bn.bias = torch.nn.Parameter(newBias)
    bn.num_features += k
    bn.running_mean = torch.concat((bn.running_mean, torch.zeros(k, device=dev)), dim=0)
    bn.running_var = torch.concat((bn.running_var, torch.ones(k, device=dev)), dim=0)


def _l2_norm_flat(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.linalg.norm(x.reshape(-1), ord=2).clamp_min(eps)


def _mean_input_column_norm(layer) -> float:
    W = layer.weight.detach()
    if W.ndim == 4:
        norms = torch.linalg.norm(
            W.permute(1, 0, 2, 3).reshape(W.size(1), -1), ord=2, dim=1
        )
    else:
        norms = torch.linalg.norm(W, ord=2, dim=0)
    if norms.numel() == 0:
        return 1.0
    return float(norms.mean().item())


def _scale_tensor_by_method(
    t: torch.Tensor, target_norm: float, scale: float, scale_method: str
) -> torch.Tensor:
    sm = (scale_method or "mean_norm").lower()
    if sm in ("fixed", "none", "he"):
        return t * float(scale)
    if sm != "mean_norm":
        raise ValueError(f"Unknown scale_method: {scale_method}")
    return (t / _l2_norm_flat(t)) * (float(target_norm) * float(scale))


def scale_new_input_weights(
    layer,
    new_inputs: torch.Tensor | None,
    scale: float = 1.0,
    scale_method: str = "mean_norm",
) -> torch.Tensor | None:
    """Scale added input columns to the inherited mean column norm."""
    if new_inputs is None:
        return None
    if new_inputs.numel() == 0:
        return new_inputs

    target = _mean_input_column_norm(layer)

    if isinstance(layer, torch.nn.Conv2d):
        assert new_inputs.ndim == 4
        _, k, _, _ = new_inputs.shape
        out = new_inputs.clone()
        for j in range(k):
            col = out[:, j, :, :]
            out[:, j, :, :] = _scale_tensor_by_method(
                col, target_norm=target, scale=scale, scale_method=scale_method
            )
        return out

    assert new_inputs.ndim == 2
    _, k = new_inputs.shape
    out = new_inputs.clone()
    for j in range(k):
        col = out[:, j]
        out[:, j] = _scale_tensor_by_method(
            col, target_norm=target, scale=scale, scale_method=scale_method
        )
    return out
