"""Three-stage convolutional network for the optimizer-state ablation.

The architecture matches Appendix B: each stage is
Conv(3x3)-BatchNorm-ReLU-MaxPool(2), followed by one linear classifier.
Growth appends channels and preserves the old parameter blocks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


INITIALIZATION_MODE_LABELS = {
    "a": "Mode A: Column-Zero Initialization",
    "b": "Mode B: Row-First Column-Zero Initialization",
    "d": "Mode D: Homogeneous Initialization",
    "e": "Mode E: Homogeneous Initialization with Empirical Variance",
}


class GrowingConvNet(nn.Module):
    """Lightweight VGG-style network used in Appendix B."""

    def __init__(self, widths: Sequence[int], num_classes: int = 10) -> None:
        super().__init__()
        if len(widths) != 3 or any(int(width) <= 0 for width in widths):
            raise ValueError("widths must contain three positive stage widths")
        self.widths = tuple(int(width) for width in widths)
        channels = (3, *self.widths)
        self.convs = nn.ModuleList(
            nn.Conv2d(channels[index], channels[index + 1], 3, padding=1, bias=True)
            for index in range(3)
        )
        self.norms = nn.ModuleList(nn.BatchNorm2d(width) for width in self.widths)
        for convolution in self.convs:
            nn.init.kaiming_normal_(convolution.weight, nonlinearity="relu")
        self.classifier = nn.Linear(
            self.widths[-1] * 4 * 4, int(num_classes), bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for convolution, normalization in zip(self.convs, self.norms):
            x = F.max_pool2d(F.relu(normalization(convolution(x)), inplace=True), 2)
        return self.classifier(x.flatten(1))


def _normal_(region: torch.Tensor, std: float) -> None:
    if region.numel():
        region.normal_(0.0, float(std))


@torch.no_grad()
def grow_model(
    model: GrowingConvNet, target_widths: Sequence[int], mode: str
) -> GrowingConvNet:
    """Expand all stages using Initialization Mode A, B, D, or E."""
    mode = mode.lower()
    if mode not in INITIALIZATION_MODE_LABELS:
        raise ValueError(f"Unknown initialization mode: {mode!r}")
    target = tuple(int(width) for width in target_widths)
    if len(target) != 3 or any(new < old for old, new in zip(model.widths, target)):
        raise ValueError(
            f"Target widths {target} must dominate seed widths {model.widths}"
        )

    device = next(model.parameters()).device
    grown = GrowingConvNet(target, model.classifier.out_features).to(device)

    for index, (old_layer, new_layer) in enumerate(zip(model.convs, grown.convs)):
        old_out, old_in = old_layer.out_channels, old_layer.in_channels
        new_in = new_layer.in_channels
        old_weight = old_layer.weight.detach()
        new_weight = new_layer.weight
        new_weight.zero_()
        new_weight[:old_out, :old_in].copy_(old_weight)

        if mode == "b":
            std = math.sqrt(2.0 / (old_in * 3 * 3))
            _normal_(new_weight[old_out:, :old_in], std)
        elif mode in {"a", "d"}:
            std = math.sqrt(2.0 / (new_in * 3 * 3))
            _normal_(new_weight[old_out:, :], std)
            if mode == "d":
                _normal_(new_weight[:old_out, old_in:], std)
        else:
            std = float(old_weight.float().std().item())
            _normal_(new_weight[old_out:, :], std)
            _normal_(new_weight[:old_out, old_in:], std)

        new_layer.bias.zero_()
        new_layer.bias[:old_out].copy_(old_layer.bias.detach())

        old_norm, new_norm = model.norms[index], grown.norms[index]
        new_norm.weight.fill_(1.0)
        new_norm.bias.zero_()
        new_norm.running_mean.zero_()
        new_norm.running_var.fill_(1.0)
        new_norm.weight[:old_out].copy_(old_norm.weight.detach())
        new_norm.bias[:old_out].copy_(old_norm.bias.detach())
        new_norm.running_mean[:old_out].copy_(old_norm.running_mean.detach())
        new_norm.running_var[:old_out].copy_(old_norm.running_var.detach())
        new_norm.num_batches_tracked.copy_(old_norm.num_batches_tracked)

    old_classifier, new_classifier = model.classifier, grown.classifier
    old_inputs = old_classifier.in_features
    new_classifier.weight.zero_()
    new_classifier.weight[:, :old_inputs].copy_(old_classifier.weight.detach())
    if mode == "d":
        _normal_(
            new_classifier.weight[:, old_inputs:],
            math.sqrt(2.0 / new_classifier.in_features),
        )
    elif mode == "e":
        _normal_(
            new_classifier.weight[:, old_inputs:],
            float(old_classifier.weight.detach().float().std().item()),
        )
    new_classifier.bias.copy_(old_classifier.bias.detach())
    return grown


def copy_state_overlap(source: torch.Tensor, destination: torch.Tensor) -> None:
    """Copy an old tensor-shaped optimizer state into the preserved prefix."""
    destination.zero_()
    if source.ndim == 0:
        destination.copy_(source)
        return
    slices = tuple(
        slice(0, min(old, new)) for old, new in zip(source.shape, destination.shape)
    )
    destination[slices].copy_(source[slices])
