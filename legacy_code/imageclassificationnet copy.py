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
        """
        对齐官方 VGG（CIFAR）建议：
          - with_maxpool=False
          - vgg_style=True（第1个 block stride=1；从第2个 block 起：每个 block 的第1个 conv stride=2）
          - bn_after_relu=True（Conv -> ReLU -> BN）
          - final_conv_out_channels=num_classes, final_conv_stride=2, final_conv_add_bn_relu=False
        """
        assert input_channels == 1 or input_channels == 3

        # 统一 input_size
        if isinstance(input_size, (tuple, list)):
            assert len(input_size) == 2
            H, W = int(input_size[0]), int(input_size[1])
        else:
            H = W = int(input_size)

        backboneLayers = []
        in_channels = int(input_channels)

        if len(backbone_config) != 0:
            # backbone_config: list[list[int]]，每个子 list 是一个 block（若干个 conv 的输出通道数）
            for seq_id, seq in enumerate(backbone_config):
                for iLayer, out_channels in enumerate(seq):
                    out_channels = int(out_channels)

                    # === stride 规则 ===
                    if with_maxpool:
                        # 走 MaxPool 下采样时，卷积层 stride 全部为 1
                        st = 1
                    else:
                        if vgg_style:
                            # 官方 VGG：第1个 block 全 stride=1；
                            # 从第2个 block 起，每个 block 的第1个 conv stride=2，其余 stride=1
                            st = 2 if (seq_id > 0 and iLayer == 0) else 1
                        else:
                            # 旧逻辑：每个 block 的最后一个 conv stride=2（保留兼容）
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

                    # === Conv / Norm / Activation 的顺序 ===
                    bn = torch.nn.BatchNorm2d(out_channels) if with_batchnorm else None
                    act = torch.nn.ReLU(inplace=True) if with_relu else None

                    if bn_after_relu:
                        # Conv -> Act -> Norm（官方 VGG）
                        if act is not None:
                            backboneLayers.append(act)
                        if bn is not None:
                            backboneLayers.append(bn)
                    else:
                        # Conv -> Norm -> Act（旧顺序，保留兼容）
                        if bn is not None:
                            backboneLayers.append(bn)
                        if act is not None:
                            backboneLayers.append(act)

                    # MaxPool 放在每个 block 末尾
                    if with_maxpool and (iLayer == len(seq) - 1):
                        backboneLayers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))

                    in_channels = out_channels

            # === 追加 logits conv（官方 VGG：stride=2，然后 Flatten；通常不接 BN/ReLU）===
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
                    # 默认 False（对齐官方）；若你手动开 True 才会走这里
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

            # === 用 dummy forward 精确计算 sizeBeforeFlatten / in_features ===
            with torch.no_grad():
                dummy = torch.zeros((1, int(input_channels), H, W), device=self.device)
                y = self.backbone(dummy)
            self.sizeBeforeFlatten = (int(y.size(2)), int(y.size(3)))
            in_features = int(y.size(1) * y.size(2) * y.size(3))

        else:
            # 兼容“没有 backbone 的纯线性输入”
            assert isinstance(input_size, int)
            self.backbone = torch.nn.Sequential().to(device=self.device)
            self.sizeBeforeFlatten = (1, 1)
            in_features = int(in_channels * int(input_size))

        # === classifier（允许为空；官方 VGG 就是空）===
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

        # === GradMax(导师版本) 需要的属性 ===
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

        # 允许无 FC classifier：官方 VGG 的做法是 final conv 输出 num_classes 后直接 flatten 作为 logits
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

                # GradMax（官方语义）：当前层新增输出通道的 incoming 置 0，保证函数不变
                if old_out_channels < new_out_channels:
                    new_layer.weight.data[old_out_channels:, :, :, :].zero_()


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
                    old_bias_next_layer = next_layer.bias.data if layer.bias is not None else None
                    new_in_channels_next_layer = next_layer.in_channels + nb_increase

                    new_next_layer = torch.nn.Conv2d(new_in_channels_next_layer, next_layer.out_channels, next_layer.kernel_size,
                                                    stride=next_layer.stride, padding=next_layer.padding,
                                                    dilation=next_layer.dilation, groups=next_layer.groups,
                                                    bias=(old_bias_next_layer is not None), device=self.device)
                    
                    new_next_layer.weight.data[:, :next_layer.in_channels, :, :] = old_weight_next_layer
                    new_next_layer.weight.data[:, next_layer.in_channels:, :, :] = 0

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
            
            # 注意：Python 里 `if mode == 1 or 2` 永远为真（因为 `or 2` 总是 True）。
            # 这里我们要判断 mode 是否属于某几个取值。
            if mode in (1, 2, 5):
                # Mode 1/2/5: initialisation à 0 (préserve la fonction au moment du grow)
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
                    old_bias_next_layer = next_layer.bias.data if layer.bias is not None else None
                    new_in_channels_next_layer = next_layer.in_channels + nb_increase

                    new_next_layer = torch.nn.Conv2d(new_in_channels_next_layer, next_layer.out_channels, next_layer.kernel_size,
                                                    stride=next_layer.stride, padding=next_layer.padding,
                                                    dilation=next_layer.dilation, groups=next_layer.groups,
                                                    bias=(old_bias_next_layer is not None), device=self.device)
                    
                    new_next_layer.weight.data[:, :next_layer.in_channels, :, :] = old_weight_next_layer
                    new_next_layer.weight.data[:, next_layer.in_channels:, :, :] = 0

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

                    self.backbone[next_conv_index] = new_next_layer.to(self.device)

            if self.is_last_conv_layer(layer_index):
                output_size = self.update_output_size()
                self.adjust_classifier(output_size, mode=4)

    # =====================
    # Mode 5: GradMax (conv->conv) - 初始化下一层新增输入通道权重（SVD）
    # =====================
    def _get_conv_indices(self):
        return [i for i, layer in enumerate(self.backbone) if isinstance(layer, torch.nn.Conv2d)]

    def _get_fc_indices(self):
        return [i for i, layer in enumerate(self.classifier) if isinstance(layer, torch.nn.Linear)]

    def _refresh_gradmax_indices(self):
        self.convIdx = self._get_conv_indices()
        self.FCIdx = self._get_fc_indices()

        self.grow_layer_tuples = [
            (self.convIdx[i], self.convIdx[i + 1]) for i in range(len(self.convIdx) - 1)
        ]
        

    def compute_gradmax_conv_inputs_to_add(self, nb_to_grow_per_conv, scale_method="he", scale=1.0):
        """从 Waux.grad 构造矩阵 A，并用 SVD 取 top-k 左奇异向量来初始化下一层新增输入通道权重。

        返回:
            dict: key 为“下一层 conv 在 backbone 里的 index”，value 为张量形状 (Cout_next, k, kH2, kW2)
        """
        conv_indices = getattr(self, "_gradmax_conv_indices", self._get_conv_indices())
        inputs_to_add = {}

        for l in range(0, len(conv_indices) - 1):
            k = int(nb_to_grow_per_conv[l]) if l < len(nb_to_grow_per_conv) else 0
            if k <= 0:
                continue

            conv_prev = self.backbone[conv_indices[l]]
            conv_next = self.backbone[conv_indices[l + 1]]

            if (not hasattr(conv_next, "Waux")) or (conv_next.Waux.grad is None):
                raise RuntimeError("GradMax: Waux.grad 为空。请先 forward_for_gradmax(...) + loss.backward().")

            Cin = conv_prev.in_channels
            kH1, kW1 = conv_prev.weight.size(2), conv_prev.weight.size(3)

            Cout = conv_next.out_channels
            kH2, kW2 = conv_next.weight.size(2), conv_next.weight.size(3)

            # A: (Cout*kH2*kW2, Cin*kH1*kW1)
            A = (
                conv_next.Waux.grad
                .unfold(2, kH2, 1)
                .unfold(3, kW2, 1)
                .permute(0, 4, 5, 1, 2, 3)
                .reshape(Cout * kH2 * kW2, Cin * kH1 * kW1)
            )

            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
            k_eff = min(k, U.size(1))
            if k_eff <= 0:
                continue

            # (Cout*kH2*kW2, k_eff) -> (Cout, k_eff, kH2, kW2)
            w = (
                U[:, :k_eff]
                .reshape(Cout, kH2, kW2, k_eff)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

            # 设定幅度（只改“大小”，不改“方向”）
            if scale_method == "he":
                # Grow 后 conv_next 的 in_channels 会增加 k
                fan_in = (conv_next.in_channels + k_eff) * kH2 * kW2
                std = math.sqrt(2.0 / fan_in)
                target_norm = math.sqrt(Cout * kH2 * kW2) * std
                w = w * (scale * target_norm)
            elif scale_method == "none":
                w = w * scale
            else:
                raise ValueError(f"Unknown scale_method: {scale_method}")

            inputs_to_add[conv_indices[l + 1]] = w

        return inputs_to_add

    def expand_conv_layer_mode5(self, layer_index, nb_increase, gradmax_cols=None):
        """Mode5：按 Mode1 的顺序 grow（先扩当前层 out，再扩下一层 in），
        但把“下一层新增输入通道权重块”用 GradMax(SVD) 初始化。

        gradmax_cols: Tensor or None
            形状应为 (Cout_next, nb_increase, kH, kW)。若 None，则退化为 Mode1（新增列为 0）。
        """
        layer = self.backbone[layer_index]
        if not isinstance(layer, torch.nn.Conv2d):
            raise ValueError(f"Layer {layer_index} is not a convolutional layer.")

        with torch.no_grad():
            old_weight = layer.weight.data
            old_bias = layer.bias.data if layer.bias is not None else None
            old_out_channels = layer.out_channels
            new_out_channels = old_out_channels + nb_increase

            new_layer = torch.nn.Conv2d(
                layer.in_channels,
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
                fan_in = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
                std = math.sqrt(2.0 / fan_in)
                new_layer.weight.data[old_out_channels:, :, :, :].normal_(0, std)
                print(f"mode 5 layer {layer_index}: std = {std}")

            if old_bias is not None:
                new_bias = torch.zeros(new_out_channels, device=self.device)
                new_bias[:old_out_channels] = old_bias
                new_layer.bias.data = new_bias

            self.backbone[layer_index] = new_layer.to(self.device)

            # BatchNorm 同步扩维（同 Mode1）
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

            # 找下一层 conv，扩它的 in_channels（新增列）
            next_conv_index = layer_index + 1
            while next_conv_index < len(self.backbone) and not isinstance(self.backbone[next_conv_index], torch.nn.Conv2d):
                next_conv_index += 1

            if next_conv_index < len(self.backbone):
                next_layer = self.backbone[next_conv_index]
                old_weight_next_layer = next_layer.weight.data
                old_bias_next_layer = next_layer.bias.data if next_layer.bias is not None else None
                old_in_channels_next_layer = next_layer.in_channels
                new_in_channels_next_layer = old_in_channels_next_layer + nb_increase

                new_next_layer = torch.nn.Conv2d(
                    new_in_channels_next_layer,
                    next_layer.out_channels,
                    next_layer.kernel_size,
                    stride=next_layer.stride,
                    padding=next_layer.padding,
                    dilation=next_layer.dilation,
                    groups=next_layer.groups,
                    bias=(old_bias_next_layer is not None),
                    device=self.device,
                )

                # 复制旧列
                new_next_layer.weight.data[:, :old_in_channels_next_layer, :, :] = old_weight_next_layer

                # 新增列：默认 0；若给了 GradMax，就用它覆盖
                if gradmax_cols is None:
                    new_next_layer.weight.data[:, old_in_channels_next_layer:, :, :] = 0
                else:
                    if tuple(gradmax_cols.shape) != (
                        next_layer.out_channels,
                        nb_increase,
                        next_layer.kernel_size[0],
                        next_layer.kernel_size[1],
                    ):
                        raise ValueError(
                            f"GradMax cols shape mismatch: expected "
                            f"{(next_layer.out_channels, nb_increase, next_layer.kernel_size[0], next_layer.kernel_size[1])}, "
                            f"got {tuple(gradmax_cols.shape)}"
                        )
                    new_next_layer.weight.data[:, old_in_channels_next_layer:, :, :] = gradmax_cols.to(
                        device=self.device, dtype=new_next_layer.weight.dtype
                    )

                if old_bias_next_layer is not None:
                    new_next_layer.bias.data = old_bias_next_layer

                self.backbone[next_conv_index] = new_next_layer.to(self.device)

        if self.is_last_conv_layer(layer_index):
            output_size = self.update_output_size()
            self.adjust_classifier(output_size, mode=5)
            
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
                    Waux_conv_xlm1 = F.conv2d(xlm1, weight=m.Waux, bias=None, stride=1, padding=(kHeightAux//2, kWidthAux//2))
                    y = m(x) # + Waux_conv_xlm1
                    if Waux_conv_xlm1.size(2)!=y.size(2) or Waux_conv_xlm1.size(3)!=y.size(3):
                        x = y+Waux_conv_xlm1[:,:,::2,::2]
                    else:
                        x = y+Waux_conv_xlm1
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
                # Limit size（这里是 last conv，对应 convIdx 的最后一个）
                idx_last = len(self.convIdx) - 1
                if conv.out_channels + conv.nbToGrow > self.nbChannelsOutConvMax[idx_last]:
                    conv.nbToGrow = self.nbChannelsOutConvMax[idx_last] - conv.out_channels
                    if conv.nbToGrow < 0:
                        conv.nbToGrow = 0

            
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
            # ✅ 没有 FC（你现在的 VGG logits conv + flatten 就是这个分支）
            # 最后一层 logits conv 不参与 “conv->FC” 的 GradMax，且强制不 grow（避免改变 logits 维度）
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
                    # 兼容 Conv->BN->ReLU 以及 Conv->ReLU->BN：向后找最近的 BatchNorm2d
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