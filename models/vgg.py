"""VGG-11 construction and convolutional width-growth operations."""

from __future__ import annotations

import copy
import math
import warnings
from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _growth as grow


_INITIALIZATION_PATTERNS = {
    "mode_a": (False, True, True),
    "mode_b": (False, True, False),
    "mode_c": (True, False, False),
    "mode_d": (True, True, True),
}


class SimpleChainImageClassificationNetwork(torch.nn.Module):
    def __init__(self, input_size, input_channels, device=torch.device("cpu")):
        super().__init__()
        self.backbone = torch.nn.Sequential()
        self.classifier = torch.nn.Sequential()
        self.input_channels = input_channels
        self.input_size = input_size
        self.device = device

    def build(
        self,
        input_size=[],
        input_channels=[],
        backbone_config=[],
        classifier_config=[],
        with_batchnorm=True,
        with_maxpool=False,
        with_bias=True,
        with_relu=True,
        vgg_style=False,
        final_conv_out_channels=None,
        final_conv_stride=2,
        final_conv_add_bn_relu=False,
        bn_after_relu=False,
    ):
        assert input_channels == 1 or input_channels == 3

        if isinstance(input_size, (tuple, list)):
            assert len(input_size) == 2
            H, W = int(input_size[0]), int(input_size[1])
        else:
            H = W = int(input_size)

        backboneLayers = []
        in_channels = int(input_channels)

        if len(backbone_config) != 0:
            for seq_id, seq in enumerate(backbone_config):
                for iLayer, out_channels in enumerate(seq):
                    out_channels = int(out_channels)

                    if with_maxpool:
                        st = 1
                    else:
                        if vgg_style:
                            st = 2 if (seq_id > 0 and iLayer == 0) else 1
                        else:
                            st = 2 if (iLayer == len(seq) - 1) else 1

                    conv = torch.nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        stride=st,
                        bias=with_bias,
                    )
                    if with_relu:
                        torch.nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
                    backboneLayers.append(conv)

                    bn = torch.nn.BatchNorm2d(out_channels) if with_batchnorm else None
                    act = torch.nn.ReLU(inplace=True) if with_relu else None

                    if bn_after_relu:
                        if act is not None:
                            backboneLayers.append(act)
                        if bn is not None:
                            backboneLayers.append(bn)
                    else:
                        if bn is not None:
                            backboneLayers.append(bn)
                        if act is not None:
                            backboneLayers.append(act)

                    if with_maxpool and (iLayer == len(seq) - 1):
                        backboneLayers.append(
                            torch.nn.MaxPool2d(kernel_size=2, stride=2)
                        )

                    in_channels = out_channels

            if final_conv_out_channels is not None:
                final_conv_out_channels = int(final_conv_out_channels)
                logits_conv = torch.nn.Conv2d(
                    in_channels,
                    final_conv_out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=int(final_conv_stride),
                    bias=with_bias,
                )
                if with_relu:
                    torch.nn.init.kaiming_normal_(
                        logits_conv.weight, nonlinearity="relu"
                    )
                backboneLayers.append(logits_conv)
                in_channels = final_conv_out_channels

                if final_conv_add_bn_relu:
                    bn = (
                        torch.nn.BatchNorm2d(final_conv_out_channels)
                        if with_batchnorm
                        else None
                    )
                    act = torch.nn.ReLU(inplace=True) if with_relu else None

                    if bn_after_relu:
                        if act is not None:
                            backboneLayers.append(act)
                        if bn is not None:
                            backboneLayers.append(bn)
                    else:
                        if bn is not None:
                            backboneLayers.append(bn)
                        if act is not None:
                            backboneLayers.append(act)

            self.backbone = torch.nn.Sequential(*backboneLayers).to(device=self.device)

            with torch.no_grad():
                dummy = torch.zeros((1, int(input_channels), H, W), device=self.device)
                y = self.backbone(dummy)
            self.sizeBeforeFlatten = (int(y.size(2)), int(y.size(3)))
            in_features = int(y.size(1) * y.size(2) * y.size(3))

        else:
            assert isinstance(input_size, int)
            self.backbone = torch.nn.Sequential().to(device=self.device)
            self.sizeBeforeFlatten = (1, 1)
            in_features = int(in_channels * int(input_size))

        classifierLayers = []
        for iLayer in range(0, len(classifier_config)):
            out_features = int(classifier_config[iLayer])

            fc = torch.nn.Linear(in_features, out_features, bias=with_bias)
            if with_relu and iLayer < len(classifier_config) - 1:
                torch.nn.init.kaiming_normal_(fc.weight, nonlinearity="relu")
            classifierLayers.append(fc)

            if with_relu and iLayer < len(classifier_config) - 1:
                classifierLayers.append(torch.nn.ReLU(inplace=True))
            in_features = out_features

        self.classifier = torch.nn.Sequential(*classifierLayers).to(device=self.device)

        self.withBatchNorm = with_batchnorm
        self.withAvgPool = False
        self.gainDefault = 1.0
        if not hasattr(self, "growRatio"):
            self.growRatio = 0.0
        if not hasattr(self, "gradMaxMode"):
            self.gradMaxMode = 2
        self.isDead = False
        self._refresh_gradmax_indices()

        self.initial_state_dict = copy.deepcopy(self.state_dict())

    def forward(self, x):
        x1 = self.backbone(x)
        x1 = torch.flatten(x1, 1)

        if len(self.classifier) == 0:
            return x1

        expected_input_shape = self.classifier[0].in_features
        actual_input_shape = x1.size(1)

        assert actual_input_shape == expected_input_shape, (
            f"The output shape of the backbone ({actual_input_shape}) does not match "
            f"the input shape of the classifier's first layer ({expected_input_shape})"
        )

        return self.classifier(x1)

    def forward_activation(self, x):
        activations = {}
        x = x.to(self.device)

        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if isinstance(layer, torch.nn.ReLU):
                activations[i] = x

        return activations

    def _initialization_pattern(self, mode):
        if isinstance(mode, tuple) and len(mode) == 3:
            return tuple(bool(value) for value in mode)
        if mode not in _INITIALIZATION_PATTERNS:
            raise ValueError(f"Unknown initialization mode: {mode}")
        return _INITIALIZATION_PATTERNS[mode]

    def _initialize_region(self, tensor, sample, std):
        if tensor is None or tensor.numel() == 0:
            return
        if sample:
            tensor.normal_(0.0, float(std))
        else:
            tensor.zero_()

    def _conv_normal_std(self, in_channels, kernel_size):
        if isinstance(kernel_size, tuple):
            k = int(kernel_size[0]) * int(kernel_size[1])
        else:
            k = int(kernel_size) * int(kernel_size)
        fan_in = max(1, int(in_channels) * k)
        return math.sqrt(2.0 / fan_in)

    def _linear_normal_std(self, in_features):
        fan_in = max(1, int(in_features))
        return math.sqrt(2.0 / fan_in)

    def _find_next_conv_index(self, layer_index):
        next_conv_index = layer_index + 1
        while next_conv_index < len(self.backbone) and not isinstance(
            self.backbone[next_conv_index], torch.nn.Conv2d
        ):
            next_conv_index += 1
        if next_conv_index < len(self.backbone):
            return next_conv_index
        return None

    def _find_following_batchnorm_index(self, layer_index):
        idx = layer_index + 1
        while idx < len(self.backbone):
            if isinstance(self.backbone[idx], torch.nn.BatchNorm2d):
                return idx
            if isinstance(self.backbone[idx], torch.nn.Conv2d):
                break
            idx += 1
        return None

    def _expand_associated_batchnorm(
        self, layer_index, old_out_channels, new_out_channels
    ):
        bn_index = self._find_following_batchnorm_index(layer_index)
        if bn_index is None:
            return

        old_batchnorm = self.backbone[bn_index]
        new_batchnorm = torch.nn.BatchNorm2d(
            new_out_channels,
            eps=old_batchnorm.eps,
            momentum=old_batchnorm.momentum,
            affine=old_batchnorm.affine,
            track_running_stats=old_batchnorm.track_running_stats,
            device=self.device,
        )

        with torch.no_grad():
            if old_batchnorm.affine:
                new_batchnorm.weight.data[:old_out_channels] = old_batchnorm.weight.data
                new_batchnorm.bias.data[:old_out_channels] = old_batchnorm.bias.data
                new_batchnorm.weight.data[old_out_channels:] = 1
                new_batchnorm.bias.data[old_out_channels:] = 0

            if old_batchnorm.track_running_stats:
                new_batchnorm.running_mean[:old_out_channels] = (
                    old_batchnorm.running_mean
                )
                new_batchnorm.running_var[:old_out_channels] = old_batchnorm.running_var
                new_batchnorm.running_mean[old_out_channels:] = 0
                new_batchnorm.running_var[old_out_channels:] = 1
                new_batchnorm.num_batches_tracked.copy_(
                    old_batchnorm.num_batches_tracked
                )

        self.backbone[bn_index] = new_batchnorm.to(self.device)

    def _expand_current_conv_outputs(
        self, layer_index, nb_increase, last_nb_increase, mode
    ):
        layer = self.backbone[layer_index]
        if not isinstance(layer, torch.nn.Conv2d):
            raise ValueError(f"Layer {layer_index} is not a convolutional layer.")

        pattern = self._initialization_pattern(mode)
        _, sample_new2, sample_new3 = pattern

        with torch.no_grad():
            old_weight = layer.weight.data
            old_bias = layer.bias.data if layer.bias is not None else None
            old_out_channels = int(layer.out_channels)
            total_in_channels = int(layer.in_channels)
            prev_added_in_channels = int(
                max(0, min(int(last_nb_increase), total_in_channels))
            )
            old_input_channels = total_in_channels - prev_added_in_channels
            new_out_channels = old_out_channels + int(nb_increase)

            new_layer = torch.nn.Conv2d(
                total_in_channels,
                new_out_channels,
                layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=(old_bias is not None),
                device=self.device,
            )

            new_layer.weight.data[:old_out_channels, :, :, :] = old_weight

            if old_out_channels < new_out_channels:
                std_wnew2 = self._conv_normal_std(
                    old_input_channels
                    if pattern == _INITIALIZATION_PATTERNS["mode_b"]
                    else total_in_channels,
                    layer.kernel_size,
                )
                std_wnew3 = self._conv_normal_std(total_in_channels, layer.kernel_size)

                self._initialize_region(
                    new_layer.weight.data[old_out_channels:, :old_input_channels, :, :],
                    sample_new2,
                    std_wnew2,
                )
                self._initialize_region(
                    new_layer.weight.data[old_out_channels:, old_input_channels:, :, :],
                    sample_new3,
                    std_wnew3,
                )
            if old_bias is not None:
                new_bias = torch.zeros(
                    new_out_channels, device=self.device, dtype=old_bias.dtype
                )
                new_bias[:old_out_channels] = old_bias
                new_layer.bias.data = new_bias

            self.backbone[layer_index] = new_layer.to(self.device)
            self._expand_associated_batchnorm(
                layer_index, old_out_channels, new_out_channels
            )

    def _expand_next_conv_inputs(self, layer_index, nb_increase, mode):
        next_conv_index = self._find_next_conv_index(layer_index)
        if next_conv_index is None:
            return

        next_layer = self.backbone[next_conv_index]
        if not isinstance(next_layer, torch.nn.Conv2d):
            raise ValueError(f"Layer {next_conv_index} is not a convolutional layer.")

        sample_new1 = self._initialization_pattern(mode)[0]

        with torch.no_grad():
            old_weight_next_layer = next_layer.weight.data
            old_bias_next_layer = (
                next_layer.bias.data if next_layer.bias is not None else None
            )
            old_in_channels = int(next_layer.in_channels)
            new_in_channels = old_in_channels + int(nb_increase)

            new_next_layer = torch.nn.Conv2d(
                new_in_channels,
                next_layer.out_channels,
                next_layer.kernel_size,
                stride=next_layer.stride,
                padding=next_layer.padding,
                dilation=next_layer.dilation,
                groups=next_layer.groups,
                bias=(old_bias_next_layer is not None),
                device=self.device,
            )

            new_next_layer.weight.data[:, :old_in_channels, :, :] = (
                old_weight_next_layer
            )

            std = self._conv_normal_std(new_in_channels, next_layer.kernel_size)
            self._initialize_region(
                new_next_layer.weight.data[:, old_in_channels:, :, :],
                sample_new1,
                std,
            )
            if old_bias_next_layer is not None:
                new_next_layer.bias.data = old_bias_next_layer.clone()

            self.backbone[next_conv_index] = new_next_layer.to(self.device)

    def _infer_last_nb_increase_for_layer(self, layer_index):
        conv_indices = self._get_conv_indices()
        try:
            pos = conv_indices.index(layer_index)
        except ValueError:
            return 0

        if pos <= 0:
            return 0

        prev_conv_index = conv_indices[pos - 1]
        if getattr(self, "_mode_last_expanded_conv_index", None) != prev_conv_index:
            return 0

        return int(getattr(self, "_mode_last_nb_increase", 0))

    def _expand_conv_layer_for_mode(
        self, layer_index, nb_increase, mode, last_nb_increase=None
    ):
        if int(nb_increase) <= 0:
            return

        pattern = self._initialization_pattern(mode)
        if last_nb_increase is None:
            last_nb_increase = self._infer_last_nb_increase_for_layer(layer_index)

        self._expand_current_conv_outputs(
            layer_index=layer_index,
            nb_increase=int(nb_increase),
            last_nb_increase=int(last_nb_increase),
            mode=pattern,
        )
        self._expand_next_conv_inputs(
            layer_index=layer_index,
            nb_increase=int(nb_increase),
            mode=pattern,
        )

        self._mode_last_expanded_conv_index = int(layer_index)
        self._mode_last_nb_increase = int(nb_increase)

        if self.is_last_conv_layer(layer_index):
            output_size = self.update_output_size()
            self.adjust_classifier(output_size, mode=pattern)

    def is_last_conv_layer(self, layer_index):
        next_conv_index = layer_index + 1
        while next_conv_index < len(self.backbone):
            if isinstance(self.backbone[next_conv_index], torch.nn.Conv2d):
                return False
            next_conv_index += 1
        return True

    def update_output_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.input_channels, *self.input_size)).to(self.device)
            x = self.backbone(x)
            x = torch.flatten(x, 1)
            output_size = x.size(1)
        return output_size

    def adjust_classifier(self, new_in_features, mode):
        if len(self.classifier) == 0:
            return

        first_layer = self.classifier[0]
        if not isinstance(first_layer, torch.nn.Linear):
            raise TypeError("The first layer of the classifier is not a linear layer.")

        sample_added = self._initialization_pattern(mode)[0]

        old_weight = first_layer.weight.data
        old_bias = first_layer.bias.data if first_layer.bias is not None else None

        spatial = int(self.sizeBeforeFlatten[0] * self.sizeBeforeFlatten[1])
        old_in_channels = old_weight.size(1) // spatial
        weight_4d = old_weight.reshape(
            old_weight.size(0),
            old_in_channels,
            self.sizeBeforeFlatten[0],
            self.sizeBeforeFlatten[1],
        )

        added_features = int(new_in_features) - int(old_weight.size(1))
        if added_features < 0 or added_features % spatial != 0:
            raise ValueError(
                f"Invalid classifier expansion: old_in_features={old_weight.size(1)}, "
                f"new_in_features={new_in_features}, spatial={spatial}"
            )

        added_channels = added_features // spatial
        init_shape = (
            weight_4d.size(0),
            added_channels,
            self.sizeBeforeFlatten[0],
            self.sizeBeforeFlatten[1],
        )

        init_weight = torch.empty(
            init_shape, device=self.device, dtype=old_weight.dtype
        )
        std = self._linear_normal_std(new_in_features)
        self._initialize_region(init_weight, sample_added, std)
        new_weight_4d = torch.cat((weight_4d, init_weight), dim=1)
        new_weight = new_weight_4d.reshape(old_weight.size(0), int(new_in_features))

        new_first_layer = torch.nn.Linear(
            int(new_in_features),
            first_layer.out_features,
            bias=(old_bias is not None),
            device=self.device,
        )
        new_first_layer.weight.data = new_weight
        if old_bias is not None:
            new_first_layer.bias.data = old_bias.clone()

        layers = list(self.classifier.children())[1:]
        new_classifier = torch.nn.Sequential(new_first_layer, *layers)
        self.classifier = new_classifier.to(self.device)

    def _get_conv_indices(self):
        return [
            i
            for i, layer in enumerate(self.backbone)
            if isinstance(layer, torch.nn.Conv2d)
        ]

    def _get_fc_indices(self):
        return [
            i
            for i, layer in enumerate(self.classifier)
            if isinstance(layer, torch.nn.Linear)
        ]

    def _refresh_gradmax_indices(self):
        self.convIdx = self._get_conv_indices()
        self.FCIdx = self._get_fc_indices()

    def initForGradMax(self):
        dev = self.device

        for layer_index in range(1, len(self.convIdx)):
            convPrev = self.backbone[self.convIdx[layer_index - 1]]
            conv = self.backbone[self.convIdx[layer_index]]
            assert isinstance(convPrev, torch.nn.Conv2d) and isinstance(
                conv, torch.nn.Conv2d
            )
            W = conv.weight
            kernelHeight2, kernelWidth2 = W.size(2), W.size(3)
            W = convPrev.weight
            kernelHeight1, kernelWidth1 = W.size(2), W.size(3)
            kernelHeight, kernelWidth = (
                kernelHeight1 + kernelHeight2 - 1,
                kernelWidth1 + kernelWidth2 - 1,
            )
            conv.Waux = torch.zeros(
                (conv.out_channels, convPrev.in_channels, kernelHeight, kernelWidth),
                requires_grad=True,
            ).to(device=dev)
            conv.Waux.retain_grad()

            s1 = (
                convPrev.stride
                if isinstance(convPrev.stride, tuple)
                else (convPrev.stride, convPrev.stride)
            )
            s2 = (
                conv.stride
                if isinstance(conv.stride, tuple)
                else (conv.stride, conv.stride)
            )
            conv.Waux_stride = tuple(
                (a + b) if (a > 1 and b > 1) else (a + b - 1) for a, b in zip(s1, s2)
            )

        if len(self.convIdx) != 0 and len(self.FCIdx) != 0:
            fc = self.classifier[self.FCIdx[0]]
            convPrev = self.backbone[self.convIdx[-1]]
            height = self.sizeBeforeFlatten[0]
            width = self.sizeBeforeFlatten[1]

            fc.Waux = torch.zeros(
                (fc.out_features, convPrev.in_channels, height * 2, width * 2),
                requires_grad=True,
            ).to(device=dev)
            fc.Waux.retain_grad()

        for layer_index in range(1, len(self.FCIdx)):
            fcPrev = self.classifier[self.FCIdx[layer_index - 1]]
            fc = self.classifier[self.FCIdx[layer_index]]
            fc.Waux = torch.zeros(
                (fc.out_features, fcPrev.in_features), requires_grad=True
            ).to(device=dev)
            fc.Waux.retain_grad()

    def forwardForGradMax(self, x):
        level = 0
        for m in self.backbone:
            if isinstance(m, torch.nn.Conv2d):
                m.x = x.clone()
                if level > 0:
                    convPrev = self.backbone[self.convIdx[level - 1]]
                    assert isinstance(convPrev, torch.nn.Conv2d)
                    xlm1 = convPrev.x
                    kHeightAux, kWidthAux = m.Waux.size(2), m.Waux.size(3)

                    stride_aux = getattr(m, "Waux_stride", (1, 1))
                    Waux_conv_xlm1 = F.conv2d(
                        xlm1,
                        weight=m.Waux,
                        bias=None,
                        stride=stride_aux,
                        padding=(kHeightAux // 2, kWidthAux // 2),
                    )

                    y = m(x)

                    x = y + Waux_conv_xlm1

                else:
                    x = m(x)
                level += 1
            else:
                x = m(x)

        if self.withAvgPool:
            x = F.avg_pool2d(x, kernel_size=x.size(3)).view(x.size(0), x.size(1))
        else:
            x = torch.flatten(x, 1)

        level = 0
        for m in self.classifier:
            if isinstance(m, torch.nn.Linear):
                m.x = x.clone()
                if level == 0:
                    if len(self.convIdx) != 0:
                        convPrev = self.backbone[self.convIdx[-1]]
                        assert isinstance(convPrev, torch.nn.Conv2d)
                        xlm1 = convPrev.x

                        kHeightAux, kWidthAux = m.Waux.size(2), m.Waux.size(3)

                        Waux_conv_xlm1 = F.conv2d(
                            xlm1, weight=m.Waux, bias=None, stride=1
                        )

                        x = m(x) + torch.flatten(Waux_conv_xlm1, 1)
                    else:
                        x = m(x)
                else:
                    fcPrev = self.classifier[self.FCIdx[level - 1]]
                    assert isinstance(fcPrev, torch.nn.Linear)
                    xlm1 = fcPrev.x
                    Waux_xlm1 = F.linear(xlm1, weight=m.Waux, bias=None)
                    y = m(x)
                    x = y + Waux_xlm1
                level += 1
        return x

    def growGradMax(self, nbToGrow: list = None):
        for conv_index in range(len(self.convIdx) - 1):
            conv = self.backbone[self.convIdx[conv_index]]
            if conv_index == 0:
                conv.inputsToAdd = None
            convNext = self.backbone[self.convIdx[conv_index + 1]]

            if nbToGrow is not None:
                conv.nbToGrow = nbToGrow[conv_index]
            else:
                conv.nbToGrow = int(conv.out_channels * self.growRatio)

            if hasattr(self, "nbChannelsOutConvMax"):
                if (
                    conv.out_channels + conv.nbToGrow
                    > self.nbChannelsOutConvMax[conv_index]
                ):
                    conv.nbToGrow = (
                        self.nbChannelsOutConvMax[conv_index] - conv.out_channels
                    )

            W = convNext.weight
            nbToGrowLimit = W.size(0) * W.size(2) * W.size(3)
            if conv.nbToGrow > nbToGrowLimit:
                warnings.warn(
                    f"GradMax limited the requested VGG growth from "
                    f"{conv.nbToGrow} to {nbToGrowLimit} channels.",
                    RuntimeWarning,
                )
                conv.nbToGrow = nbToGrowLimit

            if conv.nbToGrow > 0:
                W = conv.weight
                Cin = W.size(1)
                kernelHeight1, kernelWidth1 = W.size(2), W.size(3)
                W = convNext.weight
                Cout = W.size(0)
                kernelHeight2, kernelWidth2 = W.size(2), W.size(3)

                A = (
                    convNext.Waux.grad.unfold(dimension=2, size=kernelHeight2, step=1)
                    .unfold(3, kernelWidth2, 1)
                    .permute(0, 4, 5, 1, 2, 3)
                    .reshape(
                        Cout * kernelHeight2 * kernelWidth2,
                        Cin * kernelHeight1 * kernelWidth1,
                    )
                )

                if A.isnan().sum().item() != 0:
                    warnings.warn(
                        f"GradMax stopped because convolution {conv_index} produced a NaN "
                        "auxiliary gradient.",
                        RuntimeWarning,
                    )
                    self.isDead = True
                    return

                sv = torch.linalg.svd(A)

                U = sv[0]
                convNext.inputsToAdd = (
                    U[:, 0 : conv.nbToGrow]
                    .reshape(Cout, kernelHeight2, kernelWidth2, conv.nbToGrow)
                    .permute(0, 3, 1, 2)
                )
            else:
                convNext.inputsToAdd = None

        if len(self.FCIdx) != 0:
            fcNext = self.classifier[self.FCIdx[0]]
            if len(self.convIdx) != 0:
                conv = self.backbone[self.convIdx[-1]]
                if nbToGrow is not None:
                    conv.nbToGrow = nbToGrow[len(self.convIdx) - 1]
                else:
                    conv.nbToGrow = int(conv.out_channels * self.growRatio)
                if hasattr(self, "nbChannelsOutConvMax"):
                    last_conv_index = len(self.convIdx) - 1
                    if (
                        conv.out_channels + conv.nbToGrow
                        > self.nbChannelsOutConvMax[last_conv_index]
                    ):
                        conv.nbToGrow = (
                            self.nbChannelsOutConvMax[last_conv_index]
                            - conv.out_channels
                        )

                W = fcNext.weight
                nbToGrowLimit = W.size(0)
                if conv.nbToGrow > nbToGrowLimit:
                    warnings.warn(
                        f"GradMax limited the requested VGG growth from "
                        f"{conv.nbToGrow} to {nbToGrowLimit} channels.",
                        RuntimeWarning,
                    )
                    conv.nbToGrow = nbToGrowLimit

                if conv.nbToGrow > 0:
                    W = conv.weight
                    Cin = W.size(1)

                    Cout = fcNext.out_features
                    height = self.sizeBeforeFlatten[0]
                    width = self.sizeBeforeFlatten[1]

                    A = (
                        fcNext.Waux.grad.unfold(dimension=2, size=height, step=1)
                        .unfold(3, width, 1)
                        .permute(0, 4, 5, 1, 2, 3)
                        .reshape(Cout * height * width, -1)
                    )

                    if A.isnan().sum().item() != 0:
                        warnings.warn(
                            "GradMax stopped because the convolution-to-classifier "
                            "auxiliary gradient contains NaN values.",
                            RuntimeWarning,
                        )
                        self.isDead = True
                        return

                    sv = torch.linalg.svd(A)
                    U = sv[0]
                    fcNext.inputsToAdd = (
                        U[:, 0 : conv.nbToGrow]
                        .reshape(Cout, height, width, conv.nbToGrow)
                        .permute(0, 3, 1, 2)
                        .reshape(Cout, conv.nbToGrow * height * width)
                    )
                else:
                    fcNext.inputsToAdd = None
            else:
                fcNext.inputsToAdd = None
        else:
            if len(self.convIdx) != 0:
                conv = self.backbone[self.convIdx[-1]]
                conv.nbToGrow = 0

        for fc_index in range(len(self.FCIdx) - 1):
            fc = self.classifier[self.FCIdx[fc_index]]
            fcNext = self.classifier[self.FCIdx[fc_index + 1]]

            if nbToGrow is not None and fc_index + len(self.convIdx) < len(nbToGrow):
                fc.nbToGrow = nbToGrow[fc_index + len(self.convIdx)]
            else:
                fc.nbToGrow = int(fc.out_features * self.growRatio)
            if hasattr(self, "nbFeaturesOutFCMax"):
                if fc.out_features + fc.nbToGrow > self.nbFeaturesOutFCMax[fc_index]:
                    fc.nbToGrow = self.nbFeaturesOutFCMax[fc_index] - fc.out_features

            W = fcNext.weight
            nbToGrowLimit = W.size(0)
            if fc.nbToGrow > nbToGrowLimit:
                warnings.warn(
                    f"GradMax limited the requested VGG classifier growth from "
                    f"{fc.nbToGrow} to {nbToGrowLimit} features.",
                    RuntimeWarning,
                )
                fc.nbToGrow = nbToGrowLimit

            if fc.nbToGrow > 0:
                if fcNext.Waux.grad.isnan().sum().item() != 0:
                    warnings.warn(
                        f"GradMax stopped because classifier layer {fc_index} produced a "
                        "NaN auxiliary gradient.",
                        RuntimeWarning,
                    )
                    self.isDead = True
                    return

                sv = torch.linalg.svd(fcNext.Waux.grad)

                U = sv[0]
                fcNext.inputsToAdd = U[:, 0 : fc.nbToGrow]
            else:
                fcNext.inputsToAdd = None

        if len(self.FCIdx) > 0:
            fcLast = self.classifier[self.FCIdx[-1]]
            fcLast.nbToGrow = 0

        for conv_index in range(len(self.convIdx)):
            conv = self.backbone[self.convIdx[conv_index]]

            if conv.inputsToAdd is not None:
                to_add = conv.inputsToAdd

                to_add = grow.scale_new_input_weights(
                    conv,
                    to_add.to(conv.weight.dtype),
                    scale=getattr(self, "gradmax_init_scale", 1.0),
                    scale_method=getattr(self, "gradmax_scale_method", "mean_norm"),
                )

                W = conv.weight
                newWeight = torch.concat((W, to_add), dim=1)
                conv.weight = torch.nn.Parameter(newWeight)
                conv.in_channels = conv.weight.size(1)

            if conv.nbToGrow > 0:
                grow.add_zero_outputs(conv, conv.nbToGrow)
                if self.withBatchNorm:
                    bn = None
                    conv_pos = self.convIdx[conv_index]
                    for j in range(conv_pos + 1, len(self.backbone)):
                        if isinstance(self.backbone[j], torch.nn.BatchNorm2d):
                            bn = self.backbone[j]
                            break
                        if isinstance(self.backbone[j], torch.nn.Conv2d):
                            break
                    assert bn is not None, (
                        "No BatchNorm2d layer follows this convolution"
                    )
                    grow.add_batchnorm_channels(bn, conv.nbToGrow)

        for fc_index in range(len(self.FCIdx)):
            fc = self.classifier[self.FCIdx[fc_index]]

            if fc.inputsToAdd is not None:
                to_add = fc.inputsToAdd

                to_add = grow.scale_new_input_weights(
                    fc,
                    to_add.to(fc.weight.dtype),
                    scale=getattr(self, "gradmax_init_scale", 1.0),
                    scale_method=getattr(self, "gradmax_scale_method", "mean_norm"),
                )

                W = fc.weight
                newWeight = torch.concat((W, to_add), dim=1)
                fc.weight = torch.nn.Parameter(newWeight)
                fc.in_features = fc.weight.size(1)

            if fc.nbToGrow > 0:
                grow.add_zero_outputs(fc, fc.nbToGrow)


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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ReLU1Fn.apply(x)


def _replace_relu(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, ReLU1())
        else:
            _replace_relu(child)


def build_model(
    num_classes: int,
    device: torch.device,
    width_multiplier: float,
    seed: int,
) -> SimpleChainImageClassificationNetwork:
    torch.manual_seed(int(seed))
    blocklist = [[1], [2], [4, 4], [8, 8], [8, 8]]
    base_width = max(1, int(64 * width_multiplier))
    backbone_config = [[base_width * factor for factor in block] for block in blocklist]

    model = SimpleChainImageClassificationNetwork(
        input_size=(32, 32),
        input_channels=3,
        device=device,
    )
    model.build(
        input_size=(32, 32),
        input_channels=3,
        backbone_config=backbone_config,
        classifier_config=[],
        with_batchnorm=False,
        with_maxpool=False,
        with_bias=False,
        with_relu=True,
        vgg_style=True,
        final_conv_out_channels=num_classes,
        final_conv_stride=2,
        final_conv_add_bn_relu=False,
        bn_after_relu=True,
    )
    model.gradmax_scale_method = "mean_norm"
    model.gradmax_init_scale = 0.5
    _replace_relu(model.backbone)
    return model.to(device)


def preprocess(x: torch.Tensor) -> torch.Tensor:
    return x


def _conv_layers(
    model: SimpleChainImageClassificationNetwork,
) -> list[tuple[int, nn.Conv2d]]:
    return [
        (index, layer)
        for index, layer in enumerate(model.backbone)
        if isinstance(layer, nn.Conv2d)
    ]


def growth_plan(
    seed_model: SimpleChainImageClassificationNetwork,
    big_model: SimpleChainImageClassificationNetwork,
    grow_steps: int,
) -> list[list[int]]:
    seed_convs = _conv_layers(seed_model)
    big_convs = _conv_layers(big_model)
    differences = [
        int(big.out_channels) - int(seed.out_channels)
        for (_, seed), (_, big) in zip(seed_convs, big_convs)
    ]
    plan = [[0 for _ in differences] for _ in range(grow_steps)]
    for layer_index, difference in enumerate(differences):
        base, remainder = divmod(int(difference), int(grow_steps))
        for grow_index in range(grow_steps):
            plan[grow_index][layer_index] = base + (grow_index < remainder)
    return plan


@torch.no_grad()
def _expand_empirical(
    model: SimpleChainImageClassificationNetwork,
    layer_index: int,
    amount: int,
) -> None:
    """Initialize added Mode E blocks from the inherited weight variance."""
    layer = model.backbone[layer_index]
    if not isinstance(layer, nn.Conv2d):
        raise TypeError(f"backbone[{layer_index}] is not Conv2d")

    old_weight = layer.weight.detach()
    old_out = int(layer.out_channels)
    new_out = old_out + int(amount)
    expanded = nn.Conv2d(
        layer.in_channels,
        new_out,
        layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
        bias=layer.bias is not None,
        device=old_weight.device,
        dtype=old_weight.dtype,
    )
    expanded.weight[:old_out].copy_(old_weight)
    std = float(old_weight.float().std().item())
    expanded.weight[old_out:].normal_(0.0, std)
    if layer.bias is not None:
        expanded.bias.zero_()
        expanded.bias[:old_out].copy_(layer.bias.detach())
    model.backbone[layer_index] = expanded
    model._expand_associated_batchnorm(layer_index, old_out, new_out)

    next_index = model._find_next_conv_index(layer_index)
    if next_index is None:
        if len(model.classifier) != 0:
            raise NotImplementedError(
                "Empirical classifier growth is not used by this VGG configuration."
            )
        return

    next_layer = model.backbone[next_index]
    old_next_weight = next_layer.weight.detach()
    old_in = int(next_layer.in_channels)
    expanded_next = nn.Conv2d(
        old_in + int(amount),
        next_layer.out_channels,
        next_layer.kernel_size,
        stride=next_layer.stride,
        padding=next_layer.padding,
        dilation=next_layer.dilation,
        groups=next_layer.groups,
        bias=next_layer.bias is not None,
        device=old_next_weight.device,
        dtype=old_next_weight.dtype,
    )
    expanded_next.weight[:, :old_in].copy_(old_next_weight)
    next_std = float(old_next_weight.float().std().item())
    expanded_next.weight[:, old_in:].normal_(0.0, next_std)
    if next_layer.bias is not None:
        expanded_next.bias.copy_(next_layer.bias.detach())
    model.backbone[next_index] = expanded_next


def grow_model(
    model: SimpleChainImageClassificationNetwork,
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
        model.growGradMax(nbToGrow=additions)
        model._refresh_gradmax_indices()
        return

    previous_addition = 0
    conv_indices = [index for index, _ in _conv_layers(model)]
    for position, layer_index in enumerate(conv_indices):
        amount = additions[position] if position < len(additions) else 0
        if amount <= 0:
            previous_addition = 0
            continue

        if mode == "mode_a":
            model._expand_conv_layer_for_mode(
                layer_index, amount, "mode_a", previous_addition
            )
        elif mode == "mode_b":
            model._expand_conv_layer_for_mode(
                layer_index, amount, "mode_b", previous_addition
            )
        elif mode == "mode_c":
            model._expand_conv_layer_for_mode(
                layer_index, amount, "mode_c", previous_addition
            )
        elif mode == "mode_d":
            model._expand_conv_layer_for_mode(
                layer_index, amount, "mode_d", previous_addition
            )
        elif mode == "mode_e":
            _expand_empirical(model, layer_index, amount)
        else:
            raise ValueError(f"Unknown VGG initialization mode: {mode}")
        previous_addition = amount

    model._refresh_gradmax_indices()
