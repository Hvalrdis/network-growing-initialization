import copy
import math
import torch
import torch.nn.functional as F
from torch import Tensor
import grow


# A simple-chain network with a backbone, a flattener module and classifier part
# The backbone is made up of conv layers with bias, nonlinear activations, maxpool and batchnorm
# The classifier part is made up of dense layers and nonlinear activations
class SimpleChainImageClassificationNetwork(torch.nn.Module):
    def __init__(self, input_size, input_channels, device=torch.device('cpu')):
        super().__init__()
        self.backbone = torch.nn.Sequential()
        self.classifier = torch.nn.Sequential()
        self.input_channels = input_channels
        self.input_size = input_size
        self.device = device
        '''
        self.gradMaxMode = 2
        self.growRatio = 0.5
        self.recordGradientVariance = False
        '''
        
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
        # === 新增：对齐官方 VGG + GradMax 的构建选项 ===
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
                        # Conv -> Act -> bn
                        if act is not None:
                            backboneLayers.append(act)
                        if bn is not None:
                            backboneLayers.append(bn)
                    else:
                        # Conv -> bn -> Act（old）
                        if bn is not None:
                            backboneLayers.append(bn)
                        if act is not None:
                            backboneLayers.append(act)

                    if with_maxpool and (iLayer == len(seq) - 1):
                        backboneLayers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))

                    in_channels = out_channels

            # logits conv
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
                    torch.nn.init.kaiming_normal_(logits_conv.weight, nonlinearity="relu")
                backboneLayers.append(logits_conv)
                in_channels = final_conv_out_channels

                if final_conv_add_bn_relu:
                    bn = torch.nn.BatchNorm2d(final_conv_out_channels) if with_batchnorm else None
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
        if not hasattr(self, 'growRatio'):
            self.growRatio = 0.0
        if not hasattr(self, 'gradMaxMode'):
            self.gradMaxMode = 2
        self.isDead = False
        self._refresh_gradmax_indices()

        # Enregistrer l'état initial
        self.initial_state_dict = copy.deepcopy(self.state_dict())


    # Renvoyer des valeurs d'activation intermédiaires
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

    def expand_conv_layer(self, layer_index, nb_increase):
        layer = self.backbone[layer_index]
        if not isinstance(layer, torch.nn.Conv2d):
            raise ValueError(f"Layer {layer_index} is not a convolutional layer.")
        else:
            with torch.no_grad():
                old_weight = layer.weight.data
                old_bias = layer.bias.data if layer.bias is not None else None
                old_out_channels = layer.out_channels
                new_out_channels = old_out_channels + nb_increase
                
                new_layer = torch.nn.Conv2d(layer.in_channels, new_out_channels, layer.kernel_size, stride=layer.stride,
                                            padding=layer.padding, dilation=layer.dilation, groups=layer.groups,
                                            bias=(old_bias is not None), device=self.device)

                # Initialiser les poids et les biais
                new_layer.weight.data[:old_out_channels, :, :, :] = old_weight
            
                if old_out_channels < new_out_channels:
                    fan_in = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
                    std = math.sqrt(2.0 / fan_in)
                    new_layer.weight.data[old_out_channels:, :, :, :].normal_(0, std)
                    print(f'mode 1 layer {layer_index}: std = {std}')

                if old_bias is not None:
                    new_bias = torch.zeros(new_out_channels, device=self.device)
                    new_bias[:old_out_channels] = old_bias
                    new_layer.bias.data = new_bias

                self.backbone[layer_index] = new_layer.to(self.device)

                # Ajuster la couche BatchNorm associée
                if layer_index + 1 < len(self.backbone) and isinstance(self.backbone[layer_index + 1], torch.nn.BatchNorm2d):
                    old_batchnorm = self.backbone[layer_index + 1]
                    new_batchnorm = torch.nn.BatchNorm2d(new_out_channels, device=self.device)

                    new_batchnorm.weight.data[:old_out_channels] = old_batchnorm.weight.data
                    new_batchnorm.bias.data[:old_out_channels] = old_batchnorm.bias.data
                    new_batchnorm.running_mean[:old_out_channels] = old_batchnorm.running_mean
                    new_batchnorm.running_var[:old_out_channels] = old_batchnorm.running_var

                    new_batchnorm.weight.data[old_out_channels:] = 1
                    new_batchnorm.bias.data[old_out_channels:] = 0
                    new_batchnorm.running_mean[old_out_channels:] = 0
                    new_batchnorm.running_var[old_out_channels:] = 1

                    self.backbone[layer_index + 1] = new_batchnorm.to(self.device)

                # Trouver la couche de convolution suivante
                next_conv_index = layer_index + 1
                while next_conv_index < len(self.backbone) and not isinstance(self.backbone[next_conv_index], torch.nn.Conv2d):
                    next_conv_index += 1

                if next_conv_index < len(self.backbone):
                    next_layer = self.backbone[next_conv_index]

                    old_weight_next_layer = next_layer.weight.data
                    old_bias_next_layer = next_layer.bias.data if next_layer.bias is not None else None
                    new_in_channels_next_layer = next_layer.in_channels + nb_increase

                    new_next_layer = torch.nn.Conv2d(new_in_channels_next_layer, next_layer.out_channels, next_layer.kernel_size,
                                                    stride=next_layer.stride, padding=next_layer.padding,
                                                    dilation=next_layer.dilation, groups=next_layer.groups,
                                                    bias=(old_bias_next_layer is not None), device=self.device)
                    
                    new_next_layer.weight.data[:, :next_layer.in_channels, :, :] = old_weight_next_layer
                    new_next_layer.weight.data[:, next_layer.in_channels:, :, :] = 0
                    
                    if old_bias_next_layer is not None:
                        new_next_layer.bias.data = old_bias_next_layer.clone()

                    self.backbone[next_conv_index] = new_next_layer.to(self.device)

            if self.is_last_conv_layer(layer_index):
                output_size = self.update_output_size()
                self.adjust_classifier(output_size, mode=1)

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
        if isinstance(first_layer, torch.nn.Linear):
            old_weight = first_layer.weight.data
            old_bias = first_layer.bias.data if first_layer.bias is not None else None
            
            # Convertir les poids en tenseur 4D [N,C,H,W]
            weight_4d = old_weight.reshape(
                old_weight.size(0),  # nb filtres (channels)
                old_weight.size(1) // (self.sizeBeforeFlatten[0] * self.sizeBeforeFlatten[1]), # in_channel = in_features//featuremap
                self.sizeBeforeFlatten[0], # height of featuremap
                self.sizeBeforeFlatten[1] # width of featuremap
            )
            
            added_features = new_in_features - weight_4d.size(1) * self.sizeBeforeFlatten[0] * self.sizeBeforeFlatten[1]
            init_shape = (
                weight_4d.size(0), 
                added_features // (self.sizeBeforeFlatten[0] * self.sizeBeforeFlatten[1]), 
                self.sizeBeforeFlatten[0], 
                self.sizeBeforeFlatten[1]
            )

            fan_in = new_in_features 
            
            if mode in (1, 2, 5):
                init_weight = torch.zeros(init_shape, device=self.device)

            elif mode == 3:
                std = math.sqrt(2.0 / fan_in)
                init_weight = torch.empty(init_shape, device=self.device).normal_(0, std)

            elif mode == 4:
                std = old_weight.std().item()
                init_weight = torch.empty(init_shape, device=self.device).normal_(0, std)

            else:
                raise ValueError(f"Unknown mode for classifier expansion: {mode}")

                
            new_weight_4d = torch.cat((weight_4d, init_weight), dim=1)
            
            # Reconvertir le tenseur de poids 4D en 2D [N,M] (nb filtres, in_features)
            new_weight = new_weight_4d.reshape(old_weight.size(0), new_in_features)
            
            # Créer un nouveau premier couche et laisser les restants inchangés
            new_first_layer = torch.nn.Linear(new_in_features, first_layer.out_features, bias=(old_bias is not None))
            new_first_layer.weight.data = new_weight
            if old_bias is not None:
                new_first_layer.bias.data = old_bias
            
            # Mettre à jour le classificateur
            layers = list(self.classifier.children())[1:]  # Conserver tous les couches sauf la premier
            new_classifier = torch.nn.Sequential(new_first_layer, *layers)
            self.classifier = new_classifier.to(self.device)
        else:
            raise TypeError("The first layer of the classifier is not a linear layer.")

    def expand_conv_layer_mode2(self, layer_index, nb_increase, last_nb_increase):
        layer = self.backbone[layer_index]
        if not isinstance(layer, torch.nn.Conv2d):
            raise ValueError(f"Layer {layer_index} is not a convolutional layer.")
        else:
            with torch.no_grad():
                old_weight = layer.weight.data
                old_bias = layer.bias.data if layer.bias is not None else None
                old_out_channels = layer.out_channels
                new_out_channels = old_out_channels + nb_increase
                last_old_in_channels = layer.in_channels - last_nb_increase
                
                new_layer = torch.nn.Conv2d(layer.in_channels, new_out_channels, layer.kernel_size, stride=layer.stride,
                                            padding=layer.padding, dilation=layer.dilation, groups=layer.groups,
                                            bias=(old_bias is not None), device=self.device)

                # Initialiser les poids et les biais
                new_layer.weight.data[:old_out_channels, :, :, :] = old_weight
                if old_out_channels < new_out_channels:
                    fan_in = last_old_in_channels * layer.kernel_size[0] * layer.kernel_size[1]
                    std = math.sqrt(2.0 / fan_in)
                    new_layer.weight.data[old_out_channels:, :last_old_in_channels, :, :].normal_(0, std)
                    new_layer.weight.data[old_out_channels:, last_old_in_channels:, :, :] = 0
                    print(f'mode 2 layer {layer_index}: std = {std}')

                if old_bias is not None:
                    new_bias = torch.zeros(new_out_channels, device=self.device)
                    new_bias[:old_out_channels] = old_bias
                    new_layer.bias.data = new_bias

                self.backbone[layer_index] = new_layer.to(self.device)

                # Ajuster la couche BatchNorm associée
                if layer_index + 1 < len(self.backbone) and isinstance(self.backbone[layer_index + 1], torch.nn.BatchNorm2d):
                    old_batchnorm = self.backbone[layer_index + 1]
                    new_batchnorm = torch.nn.BatchNorm2d(new_out_channels, device=self.device)

                    new_batchnorm.weight.data[:old_out_channels] = old_batchnorm.weight.data
                    new_batchnorm.bias.data[:old_out_channels] = old_batchnorm.bias.data
                    new_batchnorm.running_mean[:old_out_channels] = old_batchnorm.running_mean
                    new_batchnorm.running_var[:old_out_channels] = old_batchnorm.running_var

                    new_batchnorm.weight.data[old_out_channels:] = 1
                    new_batchnorm.bias.data[old_out_channels:] = 0
                    new_batchnorm.running_mean[old_out_channels:] = 0
                    new_batchnorm.running_var[old_out_channels:] = 1

                    self.backbone[layer_index + 1] = new_batchnorm.to(self.device)

                # Trouver la couche de convolution suivante
                next_conv_index = layer_index + 1
                while next_conv_index < len(self.backbone) and not isinstance(self.backbone[next_conv_index], torch.nn.Conv2d):
                    next_conv_index += 1

                if next_conv_index < len(self.backbone):
                    next_layer = self.backbone[next_conv_index]

                    old_weight_next_layer = next_layer.weight.data
                    old_bias_next_layer = next_layer.bias.data if next_layer.bias is not None else None

                    new_in_channels_next_layer = next_layer.in_channels + nb_increase

                    new_next_layer = torch.nn.Conv2d(new_in_channels_next_layer, next_layer.out_channels, next_layer.kernel_size,
                                                    stride=next_layer.stride, padding=next_layer.padding,
                                                    dilation=next_layer.dilation, groups=next_layer.groups,
                                                    bias=(old_bias_next_layer is not None), device=self.device)
                    
                    new_next_layer.weight.data[:, :next_layer.in_channels, :, :] = old_weight_next_layer
                    new_next_layer.weight.data[:, next_layer.in_channels:, :, :] = 0
                    
                    if old_bias_next_layer is not None:
                        new_next_layer.bias.data = old_bias_next_layer.clone()

                    self.backbone[next_conv_index] = new_next_layer.to(self.device)

            if self.is_last_conv_layer(layer_index):
                output_size = self.update_output_size()
                self.adjust_classifier(output_size, mode=2)

    def expand_conv_layer_mode3(self, layer_index, nb_increase):
        layer = self.backbone[layer_index]
        if not isinstance(layer, torch.nn.Conv2d):
            raise ValueError(f"Layer {layer_index} is not a convolutional layer.")
        else:
            with torch.no_grad():
                old_weight = layer.weight.data
                old_bias = layer.bias.data if layer.bias is not None else None
                old_out_channels = layer.out_channels
                new_out_channels = old_out_channels + nb_increase
                
                new_layer = torch.nn.Conv2d(layer.in_channels, new_out_channels, layer.kernel_size, stride=layer.stride,
                                      padding=layer.padding, dilation=layer.dilation, groups=layer.groups,
                                      bias=(old_bias is not None), device=self.device)

                new_layer.weight.data[:old_out_channels, :, :, :] = old_weight

                fan_in = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
                std = math.sqrt(2.0 / fan_in)
                new_layer.weight.data[old_out_channels:, :, :, :].normal_(0, std)
                print(f'mode 3 layer {layer_index}: std = {std}')

                if old_bias is not None:
                    new_bias = torch.zeros(new_out_channels, device=self.device)
                    new_bias[:old_out_channels] = old_bias
                    new_layer.bias.data = new_bias

                self.backbone[layer_index] = new_layer.to(self.device)

                # Ajuster la couche BatchNorm associée
                if layer_index + 1 < len(self.backbone) and isinstance(self.backbone[layer_index + 1], torch.nn.BatchNorm2d):
                    old_batchnorm = self.backbone[layer_index + 1]
                    new_batchnorm = torch.nn.BatchNorm2d(new_out_channels, device=self.device)

                    new_batchnorm.weight.data[:old_out_channels] = old_batchnorm.weight.data
                    new_batchnorm.bias.data[:old_out_channels] = old_batchnorm.bias.data
                    new_batchnorm.running_mean[:old_out_channels] = old_batchnorm.running_mean
                    new_batchnorm.running_var[:old_out_channels] = old_batchnorm.running_var

                    new_batchnorm.weight.data[old_out_channels:] = 1
                    new_batchnorm.bias.data[old_out_channels:] = 0
                    new_batchnorm.running_mean[old_out_channels:] = 0
                    new_batchnorm.running_var[old_out_channels:] = 1

                    self.backbone[layer_index + 1] = new_batchnorm.to(self.device)

                # Trouver la couche de convolution suivante
                next_conv_index = layer_index + 1
                while next_conv_index < len(self.backbone) and not isinstance(self.backbone[next_conv_index], torch.nn.Conv2d):
                    next_conv_index += 1

                if next_conv_index < len(self.backbone):

                    next_layer = self.backbone[next_conv_index]

                    old_weight_next_layer = next_layer.weight.data
                    old_bias_next_layer = next_layer.bias.data if next_layer.bias is not None else None
                    new_in_channels_next_layer = next_layer.in_channels + nb_increase

                    new_next_layer = torch.nn.Conv2d(new_in_channels_next_layer, next_layer.out_channels, next_layer.kernel_size,
                                               stride=next_layer.stride, padding=next_layer.padding,
                                               dilation=next_layer.dilation, groups=next_layer.groups,
                                               bias=(old_bias_next_layer is not None), device=self.device)

                    new_next_layer.weight.data[:, :next_layer.in_channels, :, :] = old_weight_next_layer

                    fan_in2 = (next_layer.in_channels + nb_increase) * layer.kernel_size[0] * layer.kernel_size[1]
                    std2 = math.sqrt(2.0 / fan_in2)
                    new_next_layer.weight.data[:, next_layer.in_channels:, :, :].normal_(0, std2)
                    print(f'mode 3 layer {next_conv_index}: std_0 = {std2}')
                    
                    if old_bias_next_layer is not None:
                        new_next_layer.bias.data = old_bias_next_layer.clone()

                    self.backbone[next_conv_index] = new_next_layer.to(self.device)

            if self.is_last_conv_layer(layer_index):
                output_size = self.update_output_size()
                self.adjust_classifier(output_size, mode=3)

    def expand_conv_layer_mode4(self, layer_index, nb_increase):
        layer = self.backbone[layer_index]
        if not isinstance(layer, torch.nn.Conv2d):
            raise ValueError(f"Layer {layer_index} is not a convolutional layer.")
        else:
            with torch.no_grad():
                old_weight = layer.weight.data
                old_bias = layer.bias.data if layer.bias is not None else None
                old_out_channels = layer.out_channels
                new_out_channels = old_out_channels + nb_increase
                
                new_layer = torch.nn.Conv2d(layer.in_channels, new_out_channels, layer.kernel_size, stride=layer.stride,
                                      padding=layer.padding, dilation=layer.dilation, groups=layer.groups,
                                      bias=(old_bias is not None), device=self.device)

                new_layer.weight.data[:old_out_channels, :, :, :] = old_weight

                std = old_weight.std().item()
                new_layer.weight.data[old_out_channels:, :, :, :].normal_(0, std)
                print(f'mode 4 layer {layer_index}: std = {std}')

                if old_bias is not None:
                    new_bias = torch.zeros(new_out_channels, device=self.device)
                    new_bias[:old_out_channels] = old_bias
                    new_layer.bias.data = new_bias

                self.backbone[layer_index] = new_layer.to(self.device)

                # Ajuster la couche BatchNorm associée
                if layer_index + 1 < len(self.backbone) and isinstance(self.backbone[layer_index + 1], torch.nn.BatchNorm2d):
                    old_batchnorm = self.backbone[layer_index + 1]
                    new_batchnorm = torch.nn.BatchNorm2d(new_out_channels, device=self.device)

                    new_batchnorm.weight.data[:old_out_channels] = old_batchnorm.weight.data
                    new_batchnorm.bias.data[:old_out_channels] = old_batchnorm.bias.data
                    new_batchnorm.running_mean[:old_out_channels] = old_batchnorm.running_mean
                    new_batchnorm.running_var[:old_out_channels] = old_batchnorm.running_var

                    new_batchnorm.weight.data[old_out_channels:] = 1
                    new_batchnorm.bias.data[old_out_channels:] = 0
                    new_batchnorm.running_mean[old_out_channels:] = 0
                    new_batchnorm.running_var[old_out_channels:] = 1

                    self.backbone[layer_index + 1] = new_batchnorm.to(self.device)

                # Trouver la couche de convolution suivante
                next_conv_index = layer_index + 1
                while next_conv_index < len(self.backbone) and not isinstance(self.backbone[next_conv_index], torch.nn.Conv2d):
                    next_conv_index += 1

                if next_conv_index < len(self.backbone):
                    next_layer = self.backbone[next_conv_index]

                    old_weight_next_layer = next_layer.weight.data
                    old_bias_next_layer = next_layer.bias.data if next_layer.bias is not None else None
                    new_in_channels_next_layer = next_layer.in_channels + nb_increase

                    new_next_layer = torch.nn.Conv2d(new_in_channels_next_layer, next_layer.out_channels, next_layer.kernel_size,
                                               stride=next_layer.stride, padding=next_layer.padding,
                                               dilation=next_layer.dilation, groups=next_layer.groups,
                                               bias=(old_bias_next_layer is not None), device=self.device)

                    new_next_layer.weight.data[:, :next_layer.in_channels, :, :] = old_weight_next_layer

                    std2 = old_weight_next_layer.std().item()
                    new_next_layer.weight.data[:, next_layer.in_channels:, :, :].normal_(0, std2)
                    print(f'mode 4 layer {next_conv_index}: std_0 = {std2}')
                    
                    if old_bias_next_layer is not None:
                        new_next_layer.bias.data = old_bias_next_layer.clone()

                    self.backbone[next_conv_index] = new_next_layer.to(self.device)

            if self.is_last_conv_layer(layer_index):
                output_size = self.update_output_size()
                self.adjust_classifier(output_size, mode=4)


    def _get_conv_indices(self):
        return [i for i, layer in enumerate(self.backbone) if isinstance(layer, torch.nn.Conv2d)]

    def _get_fc_indices(self):
        return [i for i, layer in enumerate(self.classifier) if isinstance(layer, torch.nn.Linear)]

    def _refresh_gradmax_indices(self):
        self.convIdx = self._get_conv_indices()
        self.FCIdx = self._get_fc_indices()

    
    def initForGradMax(self):
        # assert self.gradMaxMode>=1 and self.gradMaxMode<=4
        dev = self.device

        for l in range(1, len(self.convIdx)):
            convPrev = self.backbone[self.convIdx[l-1]]
            conv = self.backbone[self.convIdx[l]]
            assert isinstance(convPrev, torch.nn.Conv2d) and isinstance(conv, torch.nn.Conv2d)
            W = conv.weight
            kernelHeight2, kernelWidth2 = W.size(2), W.size(3)
            W = convPrev.weight
            kernelHeight1, kernelWidth1 = W.size(2), W.size(3)
            kernelHeight, kernelWidth = kernelHeight1+kernelHeight2-1, kernelWidth1+kernelWidth2-1
            conv.Waux = torch.zeros((conv.out_channels, convPrev.in_channels, kernelHeight, kernelWidth), requires_grad=True).to(device=dev)
            conv.Waux.retain_grad()
            
            # combined stride
            s1 = convPrev.stride if isinstance(convPrev.stride, tuple) else (convPrev.stride, convPrev.stride)
            s2 = conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)
            conv.Waux_stride = tuple(
                (a + b) if (a > 1 and b > 1) else (a + b - 1)
                for a, b in zip(s1, s2)
            )
            
            # ===== debug =====
            if getattr(self, "_gradmax_debug", False):
                print(
                    "[GradMaxDBG init] "
                    f"level={l} prev_stride={s1} cur_stride={s2} Waux_stride={conv.Waux_stride} "
                    f"Waux={tuple(conv.Waux.shape)}"
                )
        
        if len(self.convIdx)!=0 and len(self.FCIdx)!=0: #classifier 为空时就会跳过 FC 的 Waux 分支
            fc = self.classifier[self.FCIdx[0]]
            convPrev = self.backbone[self.convIdx[-1]]
            height = self.sizeBeforeFlatten[0]
            width = self.sizeBeforeFlatten[1]

            fc.Waux = torch.zeros((fc.out_features, convPrev.in_channels, height*2, width*2), requires_grad=True).to(device=dev)
            fc.Waux.retain_grad()

        for l in range(1,len(self.FCIdx)):
            fcPrev = self.classifier[self.FCIdx[l-1]]
            fc = self.classifier[self.FCIdx[l]]
            fc.Waux = torch.zeros((fc.out_features, fcPrev.in_features), requires_grad=True).to(device=dev)
            fc.Waux.retain_grad()

    # It is assumed that loss.backward() is called after each batch is passed through forwardForGradMax
    # So that dL/da_{l+1} @ x_{l-1}^T is accumulated into Waux.grad 
    def forwardForGradMax(self, x):
        # assert self.gradMaxMode>=1 and self.gradMaxMode<=4

        level = 0
        for m in self.backbone:
            if isinstance(m, torch.nn.Conv2d):
                m.x = x.clone()
                if level>0:
                    convPrev = self.backbone[self.convIdx[level-1]]
                    assert isinstance(convPrev,torch.nn.Conv2d)
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
                    
                    if getattr(self, "_gradmax_debug", False):
                        s_prev = convPrev.stride if isinstance(convPrev.stride, tuple) else (convPrev.stride, convPrev.stride)
                        s_cur = m.stride if isinstance(m.stride, tuple) else (m.stride, m.stride)
                        print(
                            "[GradMaxDBG fwd] "
                            f"level={level} prev_stride={s_prev} cur_stride={s_cur} aux_stride={stride_aux} | "
                            f"x={tuple(x.shape)} y={tuple(y.shape)} xlm1={tuple(xlm1.shape)} aux_out={tuple(Waux_conv_xlm1.shape)}"
                        )

                    x = y + Waux_conv_xlm1

                else:
                    x = m(x)
                level+=1
            else:
                x = m(x)
        
        # print('x.shape=', x.shape)
        if self.withAvgPool:
            x = F.avg_pool2d(x,kernel_size=x.size(3)).view(x.size(0),x.size(1))
        else:
            x = torch.flatten(x,1)
        
        level = 0
        for m in self.classifier:
            if isinstance(m, torch.nn.Linear):
                m.x = x.clone()
                if level==0:
                    if len(self.convIdx)!=0:
                        convPrev = self.backbone[self.convIdx[-1]]
                        assert isinstance(convPrev,torch.nn.Conv2d)
                        xlm1 = convPrev.x

                        height = self.sizeBeforeFlatten[0]
                        width = self.sizeBeforeFlatten[1]

                        kHeight1, kWidth1 = convPrev.weight.size(2), convPrev.weight.size(3)
                        kHeightAux, kWidthAux = m.Waux.size(2), m.Waux.size(3)
                        # print('xlm1.shape=', xlm1.shape)
                        Waux_conv_xlm1 = F.conv2d(xlm1, weight=m.Waux, bias=None, stride=1) # , padding=(kHeight1//2, kWidth1//2)) # [:,:,::2,::2]
                        # print(Waux_conv_xlm1.shape)
                        x = m(x) + torch.flatten(Waux_conv_xlm1, 1)
                    else:
                        x = m(x)
                else:
                    fcPrev = self.classifier[self.FCIdx[level-1]]
                    assert isinstance(fcPrev,torch.nn.Linear)
                    xlm1 = fcPrev.x
                    Waux_xlm1 = F.linear(xlm1, weight=m.Waux, bias=None)
                    y = m(x)
                    x = y+Waux_xlm1
                level+=1
        return x

    # Compute dL/da_{l+1} @ x_{l-1}^T for each layer l+1
    def growGradMax(self, nbToGrow:list=None):
        dev = self.device
        dbg = getattr(self, "_gradmax_debug", False)
        dbg_id = getattr(self, "_gradmax_dbg_id", None)

        gain = self.gainDefault

        # Set number of output channels to add, to all conv layers except the last one
        # Set inputs to add, to the next conv layers 
        for l in range(0, len(self.convIdx)-1):
            conv = self.backbone[self.convIdx[l]]
            if l==0:
                conv.inputsToAdd = None
            convNext = self.backbone[self.convIdx[l+1]]
            
            if not (nbToGrow is None):
                conv.nbToGrow = nbToGrow[l]
            else:
                conv.nbToGrow = int(conv.out_channels*self.growRatio)

            if hasattr(self, 'nbChannelsOutConvMax'):
                if conv.out_channels + conv.nbToGrow > self.nbChannelsOutConvMax[l]:
                    conv.nbToGrow = self.nbChannelsOutConvMax[l] - conv.out_channels

            if dbg:
                tag = f"#{dbg_id}" if dbg_id is not None else ""
                print(
                    f"[GradMaxDBG prep]{tag} conv_l={l} "
                    f"conv@{self.convIdx[l]} w={tuple(conv.weight.shape)} s={conv.stride} "
                    f"-> next@{self.convIdx[l+1]} w_next={tuple(convNext.weight.shape)} s_next={convNext.stride} "
                    f"nbToGrow={int(conv.nbToGrow)}"
                )
            W = convNext.weight
            nbToGrowLimit = W.size(0)*W.size(2)*W.size(3)
            if conv.nbToGrow>nbToGrowLimit:
                # convNext.out_channels
                print('WARNING: growGradMax. Nb new outputs=', conv.nbToGrow, 'was limited to', nbToGrowLimit)
                conv.nbToGrow = nbToGrowLimit

            # A = convNext.Waux.grad
            if conv.nbToGrow>0:
                W = conv.weight
                Cin = W.size(1)
                kernelHeight1, kernelWidth1 = W.size(2), W.size(3)
                W = convNext.weight
                Cout = W.size(0)
                kernelHeight2, kernelWidth2 = W.size(2), W.size(3)

                A = convNext.Waux.grad.unfold(dimension=2, size=kernelHeight2, step=1).unfold(3,kernelWidth2,1) \
                    .permute(0,4,5,1,2,3).reshape(Cout*kernelHeight2*kernelWidth2, Cin*kernelHeight1*kernelWidth1)

                if A.isnan().sum().item()!=0:
                    print('DEAD NET. growGradMax, mode=',self.gradMaxMode,'conv',l)
                    self.printNbChannelsOut()
                    self.isDead = True
                    return
                    # print(conv.x)
                    # print(convNext.x)
                    # print(convNext.Waux.grad)
                    # exit()
                sv = torch.linalg.svd(A)

                U = sv[0]
                convNext.inputsToAdd = U[:,0:conv.nbToGrow].reshape(Cout,kernelHeight2,kernelWidth2,conv.nbToGrow).permute(0,3,1,2)
            else:
                convNext.inputsToAdd = None

        # Set number of output channels to add, to the last conv layer
        # Set inputs to add, to the first FC layer
        if len(self.FCIdx) != 0:
            fcNext = self.classifier[self.FCIdx[0]]
            if len(self.convIdx)!=0:
                conv = self.backbone[self.convIdx[-1]]
                if not (nbToGrow is None):
                    conv.nbToGrow = nbToGrow[len(self.convIdx)-1]
                else:
                    conv.nbToGrow = int(conv.out_channels*self.growRatio)
                if hasattr(self,'nbChannelsOutConvMax'):
                    # Limit size
                    if conv.out_channels+conv.nbToGrow>self.nbChannelsOutConvMax[l]:
                        conv.nbToGrow = self.nbChannelsOutConvMax[l]-conv.out_channels

                W = fcNext.weight
                nbToGrowLimit = W.size(0)
                if conv.nbToGrow>nbToGrowLimit:
                    print('WARNING: growGradMax. Nb new outputs=', conv.nbToGrow, 'was limited to', nbToGrowLimit)
                    conv.nbToGrow = nbToGrowLimit

                if conv.nbToGrow>0:
                    W = conv.weight
                    Cin = W.size(1)

                    Cout = fcNext.out_features
                    height = self.sizeBeforeFlatten[0]
                    width = self.sizeBeforeFlatten[1]

                    A = fcNext.Waux.grad.unfold(dimension=2, size=height, step=1).unfold(3,width,1) \
                        .permute(0,4,5,1,2,3).reshape(Cout*height*width, -1)

                    if A.isnan().sum().item()!=0:
                        print('DEAD NET. growGradMax, mode=',self.gradMaxMode,'conv->FC')
                        self.printNbChannelsOut()
                        self.isDead = True
                        return

                    sv = torch.linalg.svd(A)
                    U = sv[0]
                    fcNext.inputsToAdd = U[:,0:conv.nbToGrow].reshape(Cout,height,width,conv.nbToGrow).permute(0,3,1,2) \
                        .reshape(Cout,conv.nbToGrow*height*width)
                else:
                    fcNext.inputsToAdd = None
            else:
                fcNext.inputsToAdd = None
        else:
            if len(self.convIdx)!=0:
                conv = self.backbone[self.convIdx[-1]]
                conv.nbToGrow = 0

        # Set number of output features to add, to all FC layers except the last one
        # Set inputs to add, to the next FC layers
        for l in range(0, len(self.FCIdx)-1):
            fc = self.classifier[self.FCIdx[l]]
            fcNext = self.classifier[self.FCIdx[l+1]]
        
            # The number of features to add, k, cannot be greater than 
            # the number of output features of the next layer c_{l+1}
            # U,\Sigma,V^T = svd(Waux)
            # U is of size c_{l+1} x c_{l+1}
            if not (nbToGrow is None) and l+len(self.convIdx)<len(nbToGrow):
                fc.nbToGrow = nbToGrow[l+len(self.convIdx)]
            else:
                fc.nbToGrow = int(fc.out_features * self.growRatio)
            if hasattr(self, 'nbFeaturesOutFCMax'):
                # Limit size
                if fc.out_features + fc.nbToGrow > self.nbFeaturesOutFCMax[l]:
                    fc.nbToGrow = self.nbFeaturesOutFCMax[l] - fc.out_features
            
            W = fcNext.weight
            nbToGrowLimit = W.size(0)
            if fc.nbToGrow>nbToGrowLimit:
                # convNext.out_channels
                print('WARNING: growGradMax. Nb new outputs=', fc.nbToGrow, 'was limited to', nbToGrowLimit)
                fc.nbToGrow = nbToGrowLimit

            # A = convNext.Waux.grad
            if fc.nbToGrow>0:
                # W = fc.weight
                # Cin = W.size(1)
                # W = fcNext.weight
                # Cout = W.size(0)
                #A = fcNext.Waux.grad
                if fcNext.Waux.grad.isnan().sum().item()!=0:
                    print('DEAD NET. growGradMax, mode=',self.gradMaxMode,'FC',l)
                    self.printNbChannelsOut()
                    self.isDead = True
                    return

                sv = torch.linalg.svd(fcNext.Waux.grad)
                # print(sv) # conv.Waux.grad.shape, (conv.Waux.grad**2).sum().item())

                # Take the top k left-singular vectors
                U = sv[0]
                fcNext.inputsToAdd = U[:,0:fc.nbToGrow]
            else:
                fcNext.inputsToAdd = None

        if len(self.FCIdx) > 0:
            fcLast = self.classifier[self.FCIdx[-1]]
            fcLast.nbToGrow = 0

        # Grow conv layers (inputs and outputs)
        for l in range(0, len(self.convIdx)):
            conv = self.backbone[self.convIdx[l]]

            if conv.inputsToAdd!=None:
                to_add = conv.inputsToAdd

                to_add = grow.scaleNewInputWeights(
                    conv,
                    to_add.to(conv.weight.dtype),
                    scale=getattr(self, "gradmax_init_scale", 1.0),
                    scale_method=getattr(self, "gradmax_scale_method", "mean_norm"),
                )

                W = conv.weight
                newWeight = torch.concat((W, to_add), dim=1)
                conv.weight = torch.nn.Parameter(newWeight)
                conv.in_channels = conv.weight.size(1)

            if conv.nbToGrow>0:
                grow.addFiltersZero(conv, conv.nbToGrow)
                if self.withBatchNorm:
                    # 兼容 Conv->BN->ReLU 以及 Conv->ReLU->BN,向后找最近的 BatchNorm2d
                    bn = None
                    conv_pos = self.convIdx[l]
                    for j in range(conv_pos + 1, len(self.backbone)):
                        if isinstance(self.backbone[j], torch.nn.BatchNorm2d):
                            bn = self.backbone[j]
                            break
                        if isinstance(self.backbone[j], torch.nn.Conv2d):
                            break
                    assert bn is not None, "withBatchNorm=True 但在该 Conv 后未找到 BatchNorm2d"
                    grow.addChannelsBatchNorm(bn, conv.nbToGrow)

        # Grow FC layers (inputs and outputs)
        for l in range(0, len(self.FCIdx)):
            fc = self.classifier[self.FCIdx[l]]

            if l==len(self.FCIdx)-1:
                gain = 1

            if fc.inputsToAdd!=None:
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

            if fc.nbToGrow>0:
                grow.addFiltersZero(fc, fc.nbToGrow)