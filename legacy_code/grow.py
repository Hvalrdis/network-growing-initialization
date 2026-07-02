import math
import torch
import torch.nn.functional as F

# Add k zero output channels/features to a conv/FC layer
def addFiltersZero(layer, k: int): # , kaiming_init=False):
    if k<=0:
        return

    W = layer.weight
    Cin = W.size(1)
    if isinstance(layer, torch.nn.Conv2d):
        kHeight, kWidth = W.size(2), W.size(3)
        newW = torch.zeros((k,Cin,kHeight,kWidth), device=W.device)
    else: # W.ndim==2:
        newW = torch.zeros((k,Cin), device=W.device)
    layer.weight = torch.nn.Parameter(torch.concat((layer.weight,newW),dim=0))

    if not layer.bias is None:
        layer.bias = torch.nn.Parameter(torch.concat((layer.bias,torch.zeros(k, device=W.device)),dim=0))

    if isinstance(layer, torch.nn.Conv2d):
        layer.out_channels+=k
    else:
        layer.out_features+=k

def addZeroInputs(layer, k: int):
    if k<=0:
        return

    W = layer.weight
    Cout = W.size(0)

    if W.ndim==4:
        kHeight, kWidth = W.size(2), W.size(3)
        newW = torch.zeros((Cout,k,kHeight,kWidth), device=W.device)
    else:
        newW = torch.zeros((Cout,k), device=W.device)

    newWeight = torch.concat((layer.weight,newW),dim=1)
    layer.weight = torch.nn.Parameter(newWeight)

    if isinstance(layer, torch.nn.Conv2d):
        layer.in_channels+=k
    else:
        layer.in_features+=k

# Add channels to BatchNorm (1D or 2D)
def addChannelsBatchNorm(bn, k):
    if k<=0:
        return
    # assert isinstance(bn, torch.nn.BatchNorm2d)
    dev = bn.weight.device
    newWeight = torch.concat((bn.weight, torch.ones(k, device=dev)), dim=0)
    bn.weight = torch.nn.Parameter(newWeight)
    newBias = torch.concat((bn.bias, torch.zeros(k, device=dev)), dim=0)
    bn.bias = torch.nn.Parameter(newBias)
    bn.num_features += k
    bn.running_mean = torch.concat((bn.running_mean, torch.zeros(k, device=dev)), dim=0)
    bn.running_var = torch.concat((bn.running_var, torch.ones(k, device=dev)), dim=0)
    
#--------------------------
def _l2_norm_flat(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.linalg.norm(x.reshape(-1), ord=2).clamp_min(eps)

def _mean_input_column_norm(layer) -> float:
    W = layer.weight.detach()
    if W.ndim == 4:
        norms = torch.linalg.norm(W.permute(1, 0, 2, 3).reshape(W.size(1), -1), ord=2, dim=1)
    else:
        norms = torch.linalg.norm(W, ord=2, dim=0)
    if norms.numel() == 0:
        return 1.0
    return float(norms.mean().item())


def _scale_tensor_by_method(t: torch.Tensor, target_norm: float, scale: float, scale_method: str) -> torch.Tensor:
    sm = (scale_method or "mean_norm").lower()
    if sm in ("fixed", "none", "he"):
        return t * float(scale)
    if sm != "mean_norm":
        raise ValueError(f"Unknown scale_method: {scale_method}")
    return (t / _l2_norm_flat(t)) * (float(target_norm) * float(scale))


def scaleNewInputWeights(layer, new_inputs: torch.Tensor, scale: float = 1.0, scale_method: str = "mean_norm") -> torch.Tensor:
    """
    Scale block that will be concatenated on input dimension.
    Conv2d: new_inputs (Cout, k, kH, kW)
    Linear: new_inputs (Cout, k)
    Each new input channel/feature is treated as a “column vector” and scaled to match mean norm.
    """
    if new_inputs is None:
        return None
    if new_inputs.numel() == 0:
        return new_inputs

    target = _mean_input_column_norm(layer)

    if isinstance(layer, torch.nn.Conv2d):
        assert new_inputs.ndim == 4
        Cout, k, kH, kW = new_inputs.shape
        out = new_inputs.clone()
        for j in range(k):
            col = out[:, j, :, :]
            out[:, j, :, :] = _scale_tensor_by_method(col, target_norm=target, scale=scale, scale_method=scale_method)
        return out

    assert new_inputs.ndim == 2
    Cout, k = new_inputs.shape
    out = new_inputs.clone()
    for j in range(k):
        col = out[:, j]
        out[:, j] = _scale_tensor_by_method(col, target_norm=target, scale=scale, scale_method=scale_method)
    return out