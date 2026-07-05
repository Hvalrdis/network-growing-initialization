"""MLP construction and width-growth operations."""

from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _growth as grow


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        activation=nn.ReLU,
        init_type="kaiming",
        with_bias: bool = True,
    ):
        super(MLP, self).__init__()
        self.with_bias = with_bias
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        prev_size = input_size
        for h_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, h_size, bias=with_bias))
            self.activations.append(activation())
            prev_size = h_size

        self.output_layer = nn.Linear(prev_size, output_size, bias=with_bias)
        self.init_type = init_type
        self.activation_class = activation
        self._initialize_weights(init_type)

        self.growRatio = getattr(self, "growRatio", 0.25)
        self.gradMaxMode = getattr(self, "gradMaxMode", 1)
        self.gradmax_scale_method = getattr(self, "gradmax_scale_method", "mean_norm")
        self.gradmax_init_scale = getattr(self, "gradmax_init_scale", 1.0)
        self.isDead = getattr(self, "isDead", False)

        if hasattr(self, "_refresh_gradmax_indices"):
            self._refresh_gradmax_indices()

    def _initialize_weights(self, init_type):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self._init_layer_weights(layer, init_type)
        self._init_layer_weights(self.output_layer, init_type)

    def _init_layer_weights(self, layer, init_type):
        if init_type.lower() == "kaiming":
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        elif init_type.lower() == "xavier":
            nn.init.xavier_normal_(layer.weight)
        else:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        x = self.output_layer(x)
        return x

    def expand_layer_mode1(self, layer_index, num_new_neurons):
        old_layer = self.layers[layer_index]
        if not isinstance(old_layer, nn.Linear):
            raise TypeError(f"Layer {layer_index} is not nn.Linear")

        in_features = old_layer.in_features
        old_out_features = old_layer.out_features
        new_out_features = old_out_features + num_new_neurons

        old_weight = old_layer.weight.data

        new_weight = torch.empty(
            new_out_features,
            in_features,
            device=old_weight.device,
            dtype=old_weight.dtype,
        )

        new_weight[:old_out_features, :] = old_weight

        fan_in = in_features
        std = math.sqrt(2.0 / fan_in) if fan_in > 0 else 0.0
        new_weight[old_out_features:, :].normal_(0, std)

        expanded_layer = nn.Linear(in_features, new_out_features, bias=self.with_bias)
        expanded_layer.weight = nn.Parameter(new_weight)

        if self.with_bias:
            old_bias = old_layer.bias.data
            new_bias = torch.empty(
                new_out_features, device=old_weight.device, dtype=old_weight.dtype
            )
            new_bias[:old_out_features] = old_bias
            new_bias[old_out_features:] = 0.0
            expanded_layer.bias = nn.Parameter(new_bias)

        self.layers[layer_index] = expanded_layer

        if layer_index < len(self.layers) - 1:
            next_layer = self.layers[layer_index + 1]

            next_old_weight = next_layer.weight.data
            next_in_features = next_layer.in_features
            next_out_features = next_layer.out_features

            new_in_features = next_in_features + num_new_neurons
            new_weight_next = torch.empty(
                next_out_features,
                new_in_features,
                device=next_old_weight.device,
                dtype=next_old_weight.dtype,
            )

            new_weight_next[:, :next_in_features] = next_old_weight
            new_weight_next[:, next_in_features:] = 0.0

            expanded_next_layer = nn.Linear(
                new_in_features, next_out_features, bias=self.with_bias
            )
            expanded_next_layer.weight = nn.Parameter(new_weight_next)

            if self.with_bias:
                next_old_bias = next_layer.bias.data
                expanded_next_layer.bias = nn.Parameter(next_old_bias)

            self.layers[layer_index + 1] = expanded_next_layer
        else:
            out_old_weight = self.output_layer.weight.data
            out_old_bias = self.output_layer.bias.data if self.with_bias else None
            out_in_features = self.output_layer.in_features
            out_out_features = self.output_layer.out_features

            new_out_in_features = out_in_features + num_new_neurons
            new_weight_out = torch.empty(
                out_out_features,
                new_out_in_features,
                device=out_old_weight.device,
                dtype=out_old_weight.dtype,
            )

            new_weight_out[:, :out_in_features] = out_old_weight
            new_weight_out[:, out_in_features:] = 0.0

            expanded_output_layer = nn.Linear(
                new_out_in_features, out_out_features, bias=self.with_bias
            )
            expanded_output_layer.weight = nn.Parameter(new_weight_out)
            if self.with_bias:
                out_old_bias = self.output_layer.bias.data if self.with_bias else None
                expanded_output_layer.bias = nn.Parameter(out_old_bias)
            self.output_layer = expanded_output_layer

    def expand_layer_mode2(self, layer_index, num_new_neurons, last_nb_increase):
        old_layer = self.layers[layer_index]

        in_features = old_layer.in_features
        old_out_features = old_layer.out_features
        new_out_features = old_out_features + num_new_neurons

        last_old_in_features = in_features - last_nb_increase

        old_weight = old_layer.weight.data

        new_weight = torch.empty(
            new_out_features,
            in_features,
            device=old_weight.device,
            dtype=old_weight.dtype,
        )
        new_weight[:old_out_features, :] = old_weight

        if num_new_neurons > 0:
            fan_in = last_old_in_features
            std = math.sqrt(2.0 / fan_in) if fan_in > 0 else 0.0
            new_weight[old_out_features:, :last_old_in_features].normal_(0, std)

            new_weight[old_out_features:, last_old_in_features:] = 0.0

        expanded_layer = nn.Linear(in_features, new_out_features, bias=self.with_bias)
        expanded_layer.weight = nn.Parameter(new_weight)

        if self.with_bias:
            old_bias = old_layer.bias.data
            new_bias = torch.empty(
                new_out_features, device=old_weight.device, dtype=old_weight.dtype
            )
            new_bias[:old_out_features] = old_bias
            new_bias[old_out_features:] = 0.0
            expanded_layer.bias = nn.Parameter(new_bias)

        self.layers[layer_index] = expanded_layer

        if layer_index < len(self.layers) - 1:
            next_layer = self.layers[layer_index + 1]

            next_old_weight = next_layer.weight.data
            next_in_features = next_layer.in_features
            next_out_features = next_layer.out_features

            new_in_features = next_in_features + num_new_neurons
            new_weight_next = torch.empty(
                next_out_features,
                new_in_features,
                device=next_old_weight.device,
                dtype=next_old_weight.dtype,
            )
            new_weight_next[:, :next_in_features] = next_old_weight
            new_weight_next[:, next_in_features:] = 0.0

            expanded_next_layer = nn.Linear(
                new_in_features, next_out_features, bias=self.with_bias
            )
            expanded_next_layer.weight = nn.Parameter(new_weight_next)
            if self.with_bias:
                next_old_bias = next_layer.bias.data
                expanded_next_layer.bias = nn.Parameter(next_old_bias)
            self.layers[layer_index + 1] = expanded_next_layer
        else:
            out_old_weight = self.output_layer.weight.data
            out_old_bias = self.output_layer.bias.data if self.with_bias else None
            out_in_features = self.output_layer.in_features
            out_out_features = self.output_layer.out_features

            new_out_in_features = out_in_features + num_new_neurons
            new_weight_out = torch.empty(
                out_out_features,
                new_out_in_features,
                device=out_old_weight.device,
                dtype=out_old_weight.dtype,
            )
            new_weight_out[:, :out_in_features] = out_old_weight
            new_weight_out[:, out_in_features:] = 0.0

            expanded_output_layer = nn.Linear(
                new_out_in_features, out_out_features, bias=self.with_bias
            )
            expanded_output_layer.weight = nn.Parameter(new_weight_out)
            if self.with_bias:
                expanded_output_layer.bias = nn.Parameter(out_old_bias)
            self.output_layer = expanded_output_layer

    def expand_layer_mode3(self, layer_index, num_new_neurons):
        old_layer = self.layers[layer_index]

        in_features = old_layer.in_features
        old_out_features = old_layer.out_features
        new_out_features = old_out_features + num_new_neurons

        old_weight = old_layer.weight.data

        new_weight = torch.empty(
            new_out_features,
            in_features,
            device=old_weight.device,
            dtype=old_weight.dtype,
        )

        new_weight[:old_out_features, :] = old_weight

        fan_in = in_features
        std = math.sqrt(2.0 / fan_in) if fan_in > 0 else 0.0
        new_weight[old_out_features:, :].normal_(0, std)

        expanded_layer = nn.Linear(in_features, new_out_features, bias=self.with_bias)
        expanded_layer.weight = nn.Parameter(new_weight)

        if self.with_bias:
            old_bias = old_layer.bias.data
            new_bias = torch.empty(
                new_out_features, device=old_weight.device, dtype=old_weight.dtype
            )
            new_bias[:old_out_features] = old_bias
            new_bias[old_out_features:] = 0.0
            expanded_layer.bias = nn.Parameter(new_bias)

        self.layers[layer_index] = expanded_layer

        if layer_index < len(self.layers) - 1:
            next_layer = self.layers[layer_index + 1]

            next_old_weight = next_layer.weight.data
            next_in_features = next_layer.in_features
            next_out_features = next_layer.out_features

            new_in_features = next_in_features + num_new_neurons
            new_weight_next = torch.empty(
                next_out_features,
                new_in_features,
                device=next_old_weight.device,
                dtype=next_old_weight.dtype,
            )

            new_weight_next[:, :next_in_features] = next_old_weight

            fan_in2 = new_in_features
            std2 = math.sqrt(2.0 / fan_in2) if fan_in2 > 0 else 0.0
            new_weight_next[:, next_in_features:].normal_(0, std2)

            expanded_next_layer = nn.Linear(
                new_in_features, next_out_features, bias=self.with_bias
            )
            expanded_next_layer.weight = nn.Parameter(new_weight_next)
            if self.with_bias:
                next_old_bias = next_layer.bias.data
                expanded_next_layer.bias = nn.Parameter(next_old_bias)
            self.layers[layer_index + 1] = expanded_next_layer
        else:
            out_old_weight = self.output_layer.weight.data
            out_old_bias = self.output_layer.bias.data if self.with_bias else None
            out_in_features = self.output_layer.in_features
            out_out_features = self.output_layer.out_features

            new_out_in_features = out_in_features + num_new_neurons
            new_weight_out = torch.empty(
                out_out_features,
                new_out_in_features,
                device=out_old_weight.device,
                dtype=out_old_weight.dtype,
            )

            new_weight_out[:, :out_in_features] = out_old_weight

            fan_in3 = new_out_in_features
            std3 = math.sqrt(2.0 / fan_in3) if fan_in3 > 0 else 0.0
            new_weight_out[:, out_in_features:].normal_(0, std3)

            expanded_output_layer = nn.Linear(
                new_out_in_features, out_out_features, bias=self.with_bias
            )
            expanded_output_layer.weight = nn.Parameter(new_weight_out)
            if self.with_bias:
                expanded_output_layer.bias = nn.Parameter(out_old_bias)
            self.output_layer = expanded_output_layer

    def expand_layer_mode4(self, layer_index, num_new_neurons):
        old_layer = self.layers[layer_index]

        in_features = old_layer.in_features
        old_out_features = old_layer.out_features
        new_out_features = old_out_features + num_new_neurons

        old_weight = old_layer.weight.data

        new_weight = torch.empty(
            new_out_features,
            in_features,
            device=old_weight.device,
            dtype=old_weight.dtype,
        )

        new_weight[:old_out_features, :] = old_weight

        std = old_weight.std().item() if old_weight.numel() > 1 else 0.0
        new_weight[old_out_features:, :].normal_(0, std)

        expanded_layer = nn.Linear(in_features, new_out_features, bias=self.with_bias)
        expanded_layer.weight = nn.Parameter(new_weight)

        if self.with_bias:
            old_bias = old_layer.bias.data
            new_bias = torch.empty(
                new_out_features, device=old_weight.device, dtype=old_weight.dtype
            )
            new_bias[:old_out_features] = old_bias
            new_bias[old_out_features:] = 0.0
            expanded_layer.bias = nn.Parameter(new_bias)

        self.layers[layer_index] = expanded_layer

        if layer_index < len(self.layers) - 1:
            next_layer = self.layers[layer_index + 1]
            if not isinstance(next_layer, nn.Linear):
                raise TypeError(f"Layer {layer_index + 1} is not nn.Linear")

            next_old_weight = next_layer.weight.data
            next_in_features = next_layer.in_features
            next_out_features = next_layer.out_features

            new_in_features = next_in_features + num_new_neurons
            new_weight_next = torch.empty(
                next_out_features,
                new_in_features,
                device=next_old_weight.device,
                dtype=next_old_weight.dtype,
            )

            new_weight_next[:, :next_in_features] = next_old_weight

            std2 = next_old_weight.std().item() if next_old_weight.numel() > 1 else 0.0
            new_weight_next[:, next_in_features:].normal_(0, std2)

            expanded_next_layer = nn.Linear(
                new_in_features, next_out_features, bias=self.with_bias
            )
            expanded_next_layer.weight = nn.Parameter(new_weight_next)
            if self.with_bias:
                next_old_bias = next_layer.bias.data
                expanded_next_layer.bias = nn.Parameter(next_old_bias)
            self.layers[layer_index + 1] = expanded_next_layer
        else:
            out_old_weight = self.output_layer.weight.data
            out_old_bias = self.output_layer.bias.data if self.with_bias else None
            out_in_features = self.output_layer.in_features
            out_out_features = self.output_layer.out_features

            new_out_in_features = out_in_features + num_new_neurons
            new_weight_out = torch.empty(
                out_out_features,
                new_out_in_features,
                device=out_old_weight.device,
                dtype=out_old_weight.dtype,
            )

            new_weight_out[:, :out_in_features] = out_old_weight

            std3 = out_old_weight.std().item() if out_old_weight.numel() > 1 else 0.0
            new_weight_out[:, out_in_features:].normal_(0, std3)

            expanded_output_layer = nn.Linear(
                new_out_in_features, out_out_features, bias=self.with_bias
            )
            expanded_output_layer.weight = nn.Parameter(new_weight_out)
            if self.with_bias:
                expanded_output_layer.bias = nn.Parameter(out_old_bias)
            self.output_layer = expanded_output_layer

    def expand_layer_mode_c(self, layer_index, num_new_neurons, last_nb_increase=0):
        if int(num_new_neurons) <= 0:
            return

        if layer_index < 0 or layer_index >= len(self.layers):
            raise IndexError(
                f"layer_index={layer_index} is out of range for {len(self.layers)} hidden layers"
            )

        old_layer = self.layers[layer_index]
        if not isinstance(old_layer, nn.Linear):
            raise TypeError(f"Layer {layer_index} is not nn.Linear")

        num_new_neurons = int(num_new_neurons)
        in_features = int(old_layer.in_features)
        old_out_features = int(old_layer.out_features)
        new_out_features = old_out_features + num_new_neurons

        old_weight = old_layer.weight.data
        new_weight = torch.zeros(
            new_out_features,
            in_features,
            device=old_weight.device,
            dtype=old_weight.dtype,
        )
        new_weight[:old_out_features, :] = old_weight

        expanded_layer = nn.Linear(
            in_features, new_out_features, bias=self.with_bias
        ).to(old_weight.device)
        expanded_layer.weight = nn.Parameter(new_weight)

        if self.with_bias:
            old_bias = old_layer.bias.data
            new_bias = torch.zeros(
                new_out_features, device=old_bias.device, dtype=old_bias.dtype
            )
            new_bias[:old_out_features] = old_bias
            expanded_layer.bias = nn.Parameter(new_bias)

        self.layers[layer_index] = expanded_layer

        next_layer = (
            self.layers[layer_index + 1]
            if layer_index < len(self.layers) - 1
            else self.output_layer
        )
        if not isinstance(next_layer, nn.Linear):
            raise TypeError(f"The following layer after {layer_index} is not nn.Linear")

        next_old_weight = next_layer.weight.data
        next_old_bias = next_layer.bias.data if next_layer.bias is not None else None
        next_in_features = int(next_layer.in_features)
        next_out_features = int(next_layer.out_features)
        new_next_in_features = next_in_features + num_new_neurons

        new_weight_next = torch.empty(
            next_out_features,
            new_next_in_features,
            device=next_old_weight.device,
            dtype=next_old_weight.dtype,
        )
        new_weight_next[:, :next_in_features] = next_old_weight

        fan_in = max(1, new_next_in_features)
        std = math.sqrt(2.0 / fan_in)
        new_weight_next[:, next_in_features:].normal_(0.0, std)

        expanded_next_layer = nn.Linear(
            new_next_in_features,
            next_out_features,
            bias=(next_old_bias is not None),
        ).to(next_old_weight.device)
        expanded_next_layer.weight = nn.Parameter(new_weight_next)
        if next_old_bias is not None:
            expanded_next_layer.bias = nn.Parameter(next_old_bias.clone())

        if layer_index < len(self.layers) - 1:
            self.layers[layer_index + 1] = expanded_next_layer
        else:
            self.output_layer = expanded_next_layer

        self._refresh_gradmax_indices()

    def _get_all_linear_layers(self):
        return list(self.layers) + [self.output_layer]

    def _refresh_gradmax_indices(self):
        self.FCIdx = list(range(len(self._get_all_linear_layers())))

    def initForGradMax(self):
        dev = next(self.parameters()).device

        self._refresh_gradmax_indices()

        fc_layers = self._get_all_linear_layers()
        for layer_index in range(1, len(fc_layers)):
            previous_layer = fc_layers[layer_index - 1]
            layer = fc_layers[layer_index]
            layer.Waux = torch.zeros(
                (layer.out_features, previous_layer.in_features),
                requires_grad=True,
                device=dev,
                dtype=layer.weight.dtype,
            )
            layer.Waux.retain_grad()

    def forwardForGradMax(self, x):
        fc_layers = self._get_all_linear_layers()
        if len(fc_layers) >= 2:
            for layer_index in range(1, len(fc_layers)):
                if not hasattr(fc_layers[layer_index], "Waux"):
                    raise RuntimeError(
                        "Call initForGradMax() before forwardForGradMax() to initialize Waux"
                    )

        for i, (layer, act) in enumerate(zip(self.layers, self.activations)):
            layer.x = x.clone()

            if i > 0:
                prev_layer = self.layers[i - 1]
                xlm1 = prev_layer.x
                Waux_xlm1 = F.linear(xlm1, weight=layer.Waux, bias=None)
                y = layer(x)
                x = y + Waux_xlm1
            else:
                x = layer(x)

            x = act(x)

        self.output_layer.x = x.clone()
        if len(self.layers) > 0:
            prev_layer = self.layers[-1]
            xlm1 = prev_layer.x
            Waux_xlm1 = F.linear(xlm1, weight=self.output_layer.Waux, bias=None)
            y = self.output_layer(x)
            x = y + Waux_xlm1
        else:
            x = self.output_layer(x)

        return x

    def _add_zero_outputs(self, fc: nn.Linear, nb: int) -> None:
        if nb <= 0:
            return

        w = fc.weight.data
        zeros_w = torch.zeros((nb, w.size(1)), device=w.device, dtype=w.dtype)
        new_w = torch.cat([w, zeros_w], dim=0)
        fc.weight = nn.Parameter(new_w)
        fc.out_features = fc.weight.size(0)

        if fc.bias is not None:
            b = fc.bias.data
            zeros_b = torch.zeros((nb,), device=b.device, dtype=b.dtype)
            new_b = torch.cat([b, zeros_b], dim=0)
            fc.bias = nn.Parameter(new_b)

    def growGradMax(self, nbToGrow: list = None):
        self._refresh_gradmax_indices()

        fc_layers = self._get_all_linear_layers()
        if len(fc_layers) <= 1:
            return

        for layer_index in range(len(fc_layers) - 1):
            layer = fc_layers[layer_index]
            next_layer = fc_layers[layer_index + 1]

            if layer_index == 0:
                layer.inputsToAdd = None

            if nbToGrow is not None and layer_index < len(nbToGrow):
                layer.nbToGrow = int(nbToGrow[layer_index])
            else:
                layer.nbToGrow = int(layer.out_features * self.growRatio)

            if hasattr(self, "nbFeaturesOutFCMax"):
                max_list = getattr(self, "nbFeaturesOutFCMax")
                if isinstance(max_list, (list, tuple)):
                    if layer_index < len(max_list):
                        max_allowed = int(max_list[layer_index])
                    else:
                        max_allowed = int(max_list[-1])
                else:
                    max_allowed = int(max_list)

                if layer.out_features + layer.nbToGrow > max_allowed:
                    layer.nbToGrow = max_allowed - layer.out_features

            if layer.nbToGrow > 0:
                if torch.isnan(next_layer.Waux.grad).any().item() != 0:
                    warnings.warn(
                        f"GradMax stopped because layer {layer_index} produced a "
                        "NaN auxiliary gradient.",
                        RuntimeWarning,
                    )
                    self.isDead = True
                    return

                sv = torch.linalg.svd(next_layer.Waux.grad)
                U = sv[0]

                k = int(layer.nbToGrow)
                if k <= 0:
                    next_layer.inputsToAdd = None
                else:
                    if k <= U.size(1):
                        next_layer.inputsToAdd = U[:, :k]
                    else:
                        rep = (k + U.size(1) - 1) // U.size(1)
                        next_layer.inputsToAdd = U.repeat(1, rep)[:, :k]
            else:
                next_layer.inputsToAdd = None

        fc_layers[-1].nbToGrow = 0

        for layer in fc_layers:
            if getattr(layer, "inputsToAdd", None) is not None:
                to_add = layer.inputsToAdd
                to_add = grow.scale_new_input_weights(
                    layer,
                    to_add.to(layer.weight.dtype),
                    scale=getattr(self, "gradmax_init_scale", 1.0),
                    scale_method=getattr(self, "gradmax_scale_method", "mean_norm"),
                )

                W = layer.weight
                newWeight = torch.concat((W, to_add), dim=1)
                layer.weight = torch.nn.Parameter(newWeight)
                layer.in_features = layer.weight.size(1)

            if getattr(layer, "nbToGrow", 0) > 0:
                self._add_zero_outputs(layer, int(layer.nbToGrow))

        self._refresh_gradmax_indices()


MODE_SPECS = OrderedDict(
    [
        ("mode_a", "Mode A: Column-Zero Initialization"),
        ("mode_b", "Mode B: Row-First Column-Zero Initialization"),
        ("mode_c", "Mode C: Row-Zero Initialization"),
        ("mode_d", "Mode D: Homogeneous Initialization"),
        ("mode_e", "Mode E: Homogeneous Initialization with Empirical Variance"),
        ("gradmax", "GradMax"),
    ]
)

GRADMAX_USES_SEPARATE_BATCH = True


class _ReLU1Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x.clamp_min(0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        return grad_output * (x >= 0).to(dtype=grad_output.dtype)


class ReLU1(nn.Module):
    """ReLU with derivative one at zero, used by Mode C and GradMax."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ReLU1Fn.apply(x)


def _replace_relu(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, ReLU1())
        else:
            _replace_relu(child)


def _enforce_no_bias(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if name.endswith("bias"):
            with torch.no_grad():
                parameter.zero_()
            parameter.requires_grad_(False)


def build_model(
    num_classes: int,
    device: torch.device,
    width_multiplier: float,
    seed: int,
) -> MLP:
    if num_classes != 10:
        raise ValueError("The MLP configuration only supports MNIST (10 classes).")

    torch.manual_seed(int(seed))
    hidden_sizes = [
        max(1, int(512 * width_multiplier)),
        max(1, int(256 * width_multiplier)),
    ]
    model = MLP(
        input_size=28 * 28,
        hidden_sizes=hidden_sizes,
        output_size=num_classes,
        activation=nn.ReLU,
        init_type="kaiming",
        with_bias=False,
    )
    model.gradmax_scale_method = "mean_norm"
    model.gradmax_init_scale = 0.5
    _replace_relu(model)
    _enforce_no_bias(model)
    return model.to(device)


def preprocess(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(1) if x.ndim > 2 else x


def growth_plan(seed_model: MLP, big_model: MLP, grow_steps: int) -> list[list[int]]:
    diffs = [
        int(big.out_features) - int(seed.out_features)
        for seed, big in zip(seed_model.layers, big_model.layers)
    ]
    _validate_grow_steps(diffs, grow_steps)
    return _split_differences(diffs, grow_steps)


def _validate_grow_steps(differences: Sequence[int], grow_steps: int) -> None:
    if grow_steps <= 0:
        raise ValueError("MLP grow_steps must be positive")
    if any(difference < 0 for difference in differences):
        raise ValueError("MLP target widths must not be smaller than seed widths")
    positive = [difference for difference in differences if difference > 0]
    if not positive:
        raise ValueError(
            "MLP seed and target widths are identical; no growth is possible"
        )
    maximum = min(positive)
    if grow_steps > maximum:
        raise ValueError(
            f"MLP grow_steps={grow_steps} is too large for width differences "
            f"{list(differences)}. Use at most {maximum} so every growth event "
            "adds at least one unit to each growable hidden layer."
        )


def _split_differences(differences: Sequence[int], grow_steps: int) -> list[list[int]]:
    plan = [[0 for _ in differences] for _ in range(grow_steps)]
    for layer_index, difference in enumerate(differences):
        base, remainder = divmod(int(difference), int(grow_steps))
        for grow_index in range(grow_steps):
            plan[grow_index][layer_index] = base + (grow_index < remainder)
    return plan


def grow_model(
    model: MLP,
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
        logits = model.forwardForGradMax(preprocess(x[:grow_batch_size]))
        loss_fn(logits, y[:grow_batch_size]).backward()
        model.growGradMax(nbToGrow=additions)
        _enforce_no_bias(model)
        return

    previous_addition = 0
    for layer_index in range(len(model.layers)):
        amount = additions[layer_index] if layer_index < len(additions) else 0
        if amount <= 0:
            previous_addition = 0
            continue

        if mode == "mode_a":
            model.expand_layer_mode1(layer_index, amount)
        elif mode == "mode_b":
            model.expand_layer_mode2(layer_index, amount, previous_addition)
        elif mode == "mode_c":
            model.expand_layer_mode_c(layer_index, amount, previous_addition)
        elif mode == "mode_d":
            model.expand_layer_mode3(layer_index, amount)
        elif mode == "mode_e":
            model.expand_layer_mode4(layer_index, amount)
        else:
            raise ValueError(f"Unknown MLP initialization mode: {mode}")
        previous_addition = amount

    _enforce_no_bias(model)
