import torch, math
import torch.nn as nn
import torch.nn.functional as F

import grow


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.ReLU, init_type='kaiming', with_bias: bool = True):
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

        self.growRatio = getattr(self, 'growRatio', 0.25)
        self.gradMaxMode = getattr(self, 'gradMaxMode', 1)
        self.gradmax_scale_method = getattr(self, 'gradmax_scale_method', 'mean_norm')
        self.gradmax_init_scale = getattr(self, 'gradmax_init_scale', 1.0)
        self.isDead = getattr(self, 'isDead', False)

        self._gradmax_debug = getattr(self, '_gradmax_debug', False)
        self._gradmax_dbg_id = getattr(self, '_gradmax_dbg_id', None)

        if hasattr(self, '_refresh_gradmax_indices'):
            self._refresh_gradmax_indices()

    def _initialize_weights(self, init_type):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self._init_layer_weights(layer, init_type)
        self._init_layer_weights(self.output_layer, init_type)

    def _init_layer_weights(self, layer, init_type):
        if init_type.lower() == 'kaiming':
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        elif init_type.lower() == 'xavier':
            nn.init.xavier_normal_(layer.weight)
        else:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
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
            raise TypeError(f"第 {layer_index} 层不是 nn.Linear")

        in_features = old_layer.in_features
        old_out_features = old_layer.out_features
        new_out_features = old_out_features + num_new_neurons

        old_weight = old_layer.weight.data

        new_weight = torch.empty(
            new_out_features, in_features,
            device=old_weight.device, dtype=old_weight.dtype
        )

        new_weight[:old_out_features, :] = old_weight

        fan_in = in_features
        std = math.sqrt(2.0 / fan_in) if fan_in > 0 else 0.0
        new_weight[old_out_features:, :].normal_(0, std)
        print(f"mode 1 layer {layer_index}: std = {std}")

        expanded_layer = nn.Linear(in_features, new_out_features, bias=self.with_bias)
        expanded_layer.weight = nn.Parameter(new_weight)

        if self.with_bias:
            old_bias = old_layer.bias.data
            new_bias = torch.empty(
                new_out_features,
                device=old_weight.device, dtype=old_weight.dtype
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
                next_out_features, new_in_features,
                device=next_old_weight.device, dtype=next_old_weight.dtype
            )

            new_weight_next[:, :next_in_features] = next_old_weight
            new_weight_next[:, next_in_features:] = 0.0

            expanded_next_layer = nn.Linear(new_in_features, next_out_features, bias=self.with_bias)
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
            new_weight_out = torch.empty(out_out_features, new_out_in_features, 
                                         device=out_old_weight.device, dtype=out_old_weight.dtype)

            new_weight_out[:, :out_in_features] = out_old_weight
            new_weight_out[:, out_in_features:] = 0.0

            expanded_output_layer = nn.Linear(new_out_in_features, out_out_features, bias=self.with_bias)
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
            new_out_features, in_features,
            device=old_weight.device, dtype=old_weight.dtype
        )
        new_weight[:old_out_features, :] = old_weight

        if num_new_neurons > 0:

            fan_in = last_old_in_features
            std = math.sqrt(2.0 / fan_in) if fan_in > 0 else 0.0
            new_weight[old_out_features:, :last_old_in_features].normal_(0, std)

            new_weight[old_out_features:, last_old_in_features:] = 0.0
            print(f"mode 2 layer {layer_index}: std = {std}")

        expanded_layer = nn.Linear(in_features, new_out_features, bias=self.with_bias)
        expanded_layer.weight = nn.Parameter(new_weight)

        if self.with_bias:
            old_bias = old_layer.bias.data
            new_bias = torch.empty(
                new_out_features,
                device=old_weight.device, dtype=old_weight.dtype
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
                next_out_features, new_in_features,
                device=next_old_weight.device, dtype=next_old_weight.dtype
            )
            new_weight_next[:, :next_in_features] = next_old_weight
            new_weight_next[:, next_in_features:] = 0.0

            expanded_next_layer = nn.Linear(new_in_features, next_out_features, bias=self.with_bias)
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
            new_weight_out = torch.empty(out_out_features, new_out_in_features, 
                                         device=out_old_weight.device, dtype=out_old_weight.dtype)
            new_weight_out[:, :out_in_features] = out_old_weight
            new_weight_out[:, out_in_features:] = 0.0

            expanded_output_layer = nn.Linear(new_out_in_features, out_out_features, bias=self.with_bias)
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
            new_out_features, in_features,
            device=old_weight.device, dtype=old_weight.dtype
        )


        new_weight[:old_out_features, :] = old_weight

        fan_in = in_features
        std = math.sqrt(2.0 / fan_in) if fan_in > 0 else 0.0
        new_weight[old_out_features:, :].normal_(0, std)
        print(f"mode 3 layer {layer_index}: std = {std}")

        expanded_layer = nn.Linear(in_features, new_out_features, bias=self.with_bias)
        expanded_layer.weight = nn.Parameter(new_weight)

        if self.with_bias:
            old_bias = old_layer.bias.data
            new_bias = torch.empty(
                new_out_features,
                device=old_weight.device, dtype=old_weight.dtype
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
                next_out_features, new_in_features,
                device=next_old_weight.device, dtype=next_old_weight.dtype
            )

            new_weight_next[:, :next_in_features] = next_old_weight

            fan_in2 = new_in_features
            std2 = math.sqrt(2.0 / fan_in2) if fan_in2 > 0 else 0.0
            new_weight_next[:, next_in_features:].normal_(0, std2)
            print(f"mode 3 layer {layer_index+1}: std_0 = {std2}")

            expanded_next_layer = nn.Linear(new_in_features, next_out_features, bias=self.with_bias)
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
            new_weight_out = torch.empty(out_out_features, new_out_in_features,
                                         device=out_old_weight.device, dtype=out_old_weight.dtype)

            new_weight_out[:, :out_in_features] = out_old_weight

            fan_in3 = new_out_in_features
            std3 = math.sqrt(2.0 / fan_in3) if fan_in3 > 0 else 0.0
            new_weight_out[:, out_in_features:].normal_(0, std3)

            expanded_output_layer = nn.Linear(new_out_in_features, out_out_features, bias=self.with_bias)
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
            new_out_features, in_features,
            device=old_weight.device, dtype=old_weight.dtype
        )


        new_weight[:old_out_features, :] = old_weight

        std = old_weight.std().item() if old_weight.numel() > 1 else 0.0
        new_weight[old_out_features:, :].normal_(0, std)
        print(f"mode 4 layer {layer_index}: std = {std}")

        expanded_layer = nn.Linear(in_features, new_out_features, bias=self.with_bias)
        expanded_layer.weight = nn.Parameter(new_weight)

        if self.with_bias:
            old_bias = old_layer.bias.data
            new_bias = torch.empty(
                new_out_features,
                device=old_weight.device, dtype=old_weight.dtype
            )
            new_bias[:old_out_features] = old_bias
            new_bias[old_out_features:] = 0.0
            expanded_layer.bias = nn.Parameter(new_bias)

        self.layers[layer_index] = expanded_layer


        if layer_index < len(self.layers) - 1:
            next_layer = self.layers[layer_index + 1]
            if not isinstance(next_layer, nn.Linear):
                raise TypeError(f"第 {layer_index+1} 层不是 nn.Linear")

            next_old_weight = next_layer.weight.data
            next_in_features = next_layer.in_features
            next_out_features = next_layer.out_features

            new_in_features = next_in_features + num_new_neurons
            new_weight_next = torch.empty(
                next_out_features, new_in_features,
                device=next_old_weight.device, dtype=next_old_weight.dtype
            )

            new_weight_next[:, :next_in_features] = next_old_weight

            std2 = next_old_weight.std().item() if next_old_weight.numel() > 1 else 0.0
            new_weight_next[:, next_in_features:].normal_(0, std2)
            print(f"mode 4 layer {layer_index+1}: std_0 = {std2}")

            expanded_next_layer = nn.Linear(new_in_features, next_out_features, bias=self.with_bias)
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
            new_weight_out = torch.empty(out_out_features, new_out_in_features,
                                         device=out_old_weight.device, dtype=out_old_weight.dtype)

            new_weight_out[:, :out_in_features] = out_old_weight

            std3 = out_old_weight.std().item() if out_old_weight.numel() > 1 else 0.0
            new_weight_out[:, out_in_features:].normal_(0, std3)

            expanded_output_layer = nn.Linear(new_out_in_features, out_out_features, bias=self.with_bias)
            expanded_output_layer.weight = nn.Parameter(new_weight_out)
            if self.with_bias:
                expanded_output_layer.bias = nn.Parameter(out_old_bias)
            self.output_layer = expanded_output_layer
            

    def _get_all_linear_layers(self):
        return list(self.layers) + [self.output_layer]

    def _refresh_gradmax_indices(self):
        self.FCIdx = list(range(len(self._get_all_linear_layers())))

    def initForGradMax(self):
        dev = next(self.parameters()).device

        self._refresh_gradmax_indices()

        fc_layers = self._get_all_linear_layers()
        for l in range(1, len(fc_layers)):
            fcPrev = fc_layers[l - 1]
            fc = fc_layers[l]
            fc.Waux = torch.zeros(
                (fc.out_features, fcPrev.in_features),
                requires_grad=True,
                device=dev,
                dtype=fc.weight.dtype,
            )
            fc.Waux.retain_grad()

            if getattr(self, "_gradmax_debug", False):
                tag = f"#{self._gradmax_dbg_id}" if getattr(self, "_gradmax_dbg_id", None) is not None else ""
                print(
                    f"[GradMaxDBG init]{tag} layer={l} W={tuple(fc.weight.shape)} "
                    f"Waux={tuple(fc.Waux.shape)} prev_in={fcPrev.in_features}"
                )

    def forwardForGradMax(self, x):

        fc_layers = self._get_all_linear_layers()
        if len(fc_layers) >= 2:
            for l in range(1, len(fc_layers)):
                if not hasattr(fc_layers[l], "Waux"):
                    raise RuntimeError("调用 forwardForGradMax() 前请先调用 initForGradMax() 初始化 Waux")

        # hidden layers
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

        # output layer
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

    def _addFiltersZero(self, fc: nn.Linear, nb: int):

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
        dev = next(self.parameters()).device
        _ = dev  

        self._refresh_gradmax_indices()

        fc_layers = self._get_all_linear_layers()
        if len(fc_layers) <= 1:
            return 

        dbg = getattr(self, "_gradmax_debug", False)
        dbg_id = getattr(self, "_gradmax_dbg_id", None)

        for l in range(0, len(fc_layers) - 1):
            fc = fc_layers[l]
            fcNext = fc_layers[l + 1]

            if l == 0:
                fc.inputsToAdd = None

            if nbToGrow is not None and l < len(nbToGrow):
                fc.nbToGrow = int(nbToGrow[l])
            else:
                fc.nbToGrow = int(fc.out_features * self.growRatio)
                
            if hasattr(self, "nbFeaturesOutFCMax"):
                max_list = getattr(self, "nbFeaturesOutFCMax")
                if isinstance(max_list, (list, tuple)):
                    if l < len(max_list):
                        max_allowed = int(max_list[l])
                    else:
                        max_allowed = int(max_list[-1])
                else:
                    max_allowed = int(max_list)

                if fc.out_features + fc.nbToGrow > max_allowed:
                    fc.nbToGrow = max_allowed - fc.out_features

            if dbg:
                tag = f"#{dbg_id}" if dbg_id is not None else ""
                print(
                    f"[GradMaxDBG prep]{tag} fc_l={l} "
                    f"w={tuple(fc.weight.shape)} -> next_w={tuple(fcNext.weight.shape)} nbToGrow={int(fc.nbToGrow)}"
                )

            if fc.nbToGrow > 0:

                if torch.isnan(fcNext.Waux.grad).any().item() != 0:
                    print("DEAD NET. growGradMax, mode=", self.gradMaxMode, "FC", l)
                    self.isDead = True
                    return

                sv = torch.linalg.svd(fcNext.Waux.grad)
                U = sv[0]

                k = int(fc.nbToGrow)
                if k <= 0:
                    fcNext.inputsToAdd = None
                else:
                    if k <= U.size(1):
                        fcNext.inputsToAdd = U[:, :k]
                    else:
                        rep = (k + U.size(1) - 1) // U.size(1)
                        fcNext.inputsToAdd = U.repeat(1, rep)[:, :k]
            else:
                fcNext.inputsToAdd = None

        fc_layers[-1].nbToGrow = 0

        for l in range(0, len(fc_layers)):
            fc = fc_layers[l]

            if getattr(fc, "inputsToAdd", None) is not None:
                to_add = fc.inputsToAdd

                to_add = grow.scaleNewInputWeights(
                    fc,
                    to_add.to(fc.weight.dtype),
                    scale=getattr(self, "gradmax_init_scale", 1.0),
                    scale_method=getattr(self, "gradmax_scale_method", "mean_norm"),
                )

                W = fc.weight
                newWeight = torch.concat((W, to_add), dim=1)
                fc.weight = torch.nn.Parameter(newWeight)
                fc.in_features = fc.weight.size(1)

            if getattr(fc, "nbToGrow", 0) > 0:
                self._addFiltersZero(fc, int(fc.nbToGrow))

        self._refresh_gradmax_indices()