import grow
import torch

class SimpleChainImageClassificationNetwork(torch.nn.Module):

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
        
        if len(self.convIdx)!=0:
            fc = self.classifier[0]
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
            if hasattr(self,'nbChannelsOutConvMax'):
                # Limit size
                if conv.out_channels+conv.nbToGrow>self.nbChannelsOutConvMax[l]:
                    conv.nbToGrow = self.nbChannelsOutConvMax[l]-conv.out_channels
            
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
                # convNext.out_channels
                print('WARNING: growGradMax. Nb new outputs=', conv.nbToGrow, 'was limited to', nbToGrowLimit)
                conv.nbToGrow = nbToGrowLimit
            # print('Last conv layer :', conv.nbToGrow)

            if conv.nbToGrow>0:
                W = conv.weight
                Cin = W.size(1)
                # kernelHeight1, kernelWidth1 = W.size(2), W.size(3)

                Cout = fcNext.out_features

                height = self.sizeBeforeFlatten[0]
                width = self.sizeBeforeFlatten[1]
                # if self.gradMaxMode==1:
                #     A = fcNext.Waux.reshape(Cout, conv.in_channels, height, width).permute(0,2,3,1) \
                #         .reshape(Cout*height*width, conv.in_channels)
                # else:
                A = fcNext.Waux.grad.unfold(dimension=2, size=height, step=1).unfold(3,width,1) \
                    .permute(0,4,5,1,2,3).reshape(Cout*height*width, -1) # Cin*kernelHeight1*kernelWidth1)
            
                # Cout, Cin, kernelHeight, kernelWidth = A.size(0), A.size(1), A.size(2), A.size(3)
                if A.isnan().sum().item()!=0:
                    print('DEAD NET. growGradMax, mode=',self.gradMaxMode,'conv->FC')
                    self.printNbChannelsOut()
                    self.isDead = True
                    return

                sv = torch.linalg.svd(A)
                
                U = sv[0]
                fcNext.inputsToAdd = U[:,0:conv.nbToGrow].reshape(Cout,height,width,conv.nbToGrow).permute(0,3,1,2) \
                    .reshape(Cout,conv.nbToGrow*height*width)
                # m.inputsToAdd /= torch.sqrt((m.inputsToAdd**2).sum())
            else:
                fcNext.inputsToAdd = None
        else:
            fcNext.inputsToAdd = None

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
                fc.nbToGrow = int(fc.out_channels*self.growRatio)
            if hasattr(self,'nbFeaturesOutFCMax'):
                # Limit size
                if fc.out_features+conv.nbToGrow>self.nbFeaturesOutFCMax[l]:
                    fc.nbToGrow = self.nbFeaturesOutFCMax[l]-fc.out_features
            
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

        fcLast = self.classifier[self.FCIdx[-1]]
        fcLast.nbToGrow = 0

        # Grow conv layers (inputs and outputs)
        for l in range(0, len(self.convIdx)):
            conv = self.backbone[self.convIdx[l]]

            if conv.inputsToAdd!=None:
                W = conv.weight
                newWeight = torch.concat((W,conv.inputsToAdd), dim=1)
                conv.weight = torch.nn.Parameter(newWeight)
                conv.in_channels = conv.weight.size(1)

            if conv.nbToGrow>0:
                grow.addFiltersZero(conv, conv.nbToGrow)
                if self.withBatchNorm:
                    bn = self.backbone[self.convIdx[l]+1]
                    assert isinstance(bn,torch.nn.BatchNorm2d)
                    grow.addChannelsBatchNorm(bn, conv.nbToGrow)

        # Grow FC layers (inputs and outputs)
        for l in range(0, len(self.FCIdx)):
            fc = self.classifier[self.FCIdx[l]]

            if l==len(self.FCIdx)-1:
                gain = 1

            if fc.inputsToAdd!=None:
                W = fc.weight
                newWeight = torch.concat((W,fc.inputsToAdd), dim=1)
                fc.weight = torch.nn.Parameter(newWeight)
                fc.in_features = fc.weight.size(1)

            if fc.nbToGrow>0:
                grow.addFiltersZero(fc, fc.nbToGrow)