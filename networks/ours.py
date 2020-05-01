# full assembly of the sub-parts to form the complete net
from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, cfg):

        self._cfg = cfg

        n_channels = cfg.MODEL.IN_CHANNELS
        n_classes = cfg.MODEL.OUT_CHANNELS
        conv_block = double_conv

        super(UNet, self).__init__()

        first_chan = cfg.MODEL.TOPOLOGY[0]
        self.inc = inconv(n_channels, first_chan, conv_block)
        self.outc = outconv(first_chan, n_classes)
        self.multiscale_context_enabled = False
        self.multiscale_context_type = False

        up_block = up

        # Variable scale
        down_topo = cfg.MODEL.TOPOLOGY
        down_dict = OrderedDict()
        n_layers = len(down_topo)
        up_topo = [first_chan] # topography upwards
        up_dict = OrderedDict()

        # Downward layers
        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers-1
            in_dim = down_topo[idx]
            out_dim = down_topo[idx+1] if is_not_last_layer else down_topo[idx] # last layer

            layer = down(in_dim, out_dim, conv_block)

            print(f'down{idx+1}: in {in_dim}, out {out_dim}')
            down_dict[f'down{idx+1}'] = layer
            up_topo.append(out_dim)
        self.down_seq = nn.ModuleDict(down_dict)
        bottleneck_dim = out_dim

        # context layer
        if self.multiscale_context_enabled:
            self.multiscale_context = MultiScaleContextForUNet(cfg, bottleneck_dim)

        # Upward layers
        for idx in reversed(range(n_layers)):
            is_not_last_layer = idx != 0
            x1_idx = idx
            x2_idx = idx - 1 if is_not_last_layer else idx
            in_dim = up_topo[x1_idx] * 2
            out_dim = up_topo[x2_idx]

            layer = up_block(in_dim, out_dim, conv_block, bilinear=cfg.MODEL.SIMPLE_INTERPOLATION)

            print(f'up{idx+1}: in {in_dim}, out {out_dim}')
            up_dict[f'up{idx+1}'] = layer

        self.up_seq = nn.ModuleDict(up_dict)

    def forward(self, x):
        x1 = self.inc(x)

        inputs = [x1]
        # Downward U:
        for layer in self.down_seq.values():
            out = layer(inputs[-1])
            inputs.append(out)

        #Multiscale context
        if self.multiscale_context_enabled:
            bottleneck_features = inputs.pop()
            context = self.multiscale_context(bottleneck_features)
            inputs.append(context)

        # Upward U:
        inputs.reverse()
        x1 = inputs.pop(0)
        for idx, layer in enumerate(self.up_seq.values()):
            x2 = inputs[idx]
            x1 = layer(x1, x2)  # x1 for next up layer

        out = self.outc(x1)

        return out

class MultiScaleContextForUNet(nn.Module):
    def __init__(self, cfg, bottlneck_dim):
        super().__init__()
        self._cfg = cfg
        self.multiscale_context_type = cfg.MODEL.MULTISCALE_CONTEXT.TYPE
        self.context = self.build_multiscale_context(bottlneck_dim)

    def build_multiscale_context(self, bottleneck_dim):
        context_layers = []
        for i, layer_dilation in enumerate(self._cfg.MODEL.MULTISCALE_CONTEXT.DILATION_TOPOLOGY):
            layer = ContextLayer(bottleneck_dim, layer_dilation)
            context_layers.append(layer)
        if self.multiscale_context_type == 'Simple':
            context = nn.Sequential(*context_layers)
        if self.multiscale_context_type == 'PyramidSum':
            context =  nn.ModuleList(context_layers)
        if self.multiscale_context_type == 'ParallelSum':
            context =  nn.ModuleList(context_layers)
        return context

    def forward(self, x):
        if self.multiscale_context_type == 'Simple':
            context = self.context(x)
        elif self.multiscale_context_type == 'PyramidSum':
            context = x
            for layer in self.context:
                context = layer(context)
        elif self.multiscale_context_type == 'ParallelSum':
            context = x
            for layer in self.context:
                context += layer(x)

        return context


# sub-parts of the U-Net model
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class triple_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(triple_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class attention_block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.W2 = nn.Sequential(
            nn.Conv2d(in_ch // 2, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.MaxPool2d(2)
        )

        self.W1 = nn.Conv2d(in_ch // 2, out_ch, kernel_size=1, stride=1, padding=0, bias=True)

        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2, ):  # X1 is upstream, X2 is skip

        xl_size_orig = x2.size()
        xl_ = self.W2(x2)
        x1 = self.W1(x1)

        psi = self.psi(xl_ + x1)

        upsampled_psi = F.interpolate(psi, size=xl_size_orig[2:], mode='bilinear', align_corners=False)

        # scale features with attention
        attention = upsampled_psi.expand_as(x2)

        return attention * x2


class ContextLayer(nn.Module):
    def __init__(self, channels, dilation, include_activation=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(channels),
        )
        if include_activation:
            self.conv.add_module('relu', nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(inconv, self).__init__()
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(down, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block, bilinear=True, ):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.detach().size()[2] - x1.detach().size()[2]
        diffX = x2.detach().size()[3] - x1.detach().size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class attention_up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block, bilinear=True, ):
        super().__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.attention = attention_block(in_ch, out_ch)
        self.conv = conv_block(in_ch, out_ch)
        print('in', in_ch, 'out', out_ch)

    def forward(self, x1, x2):
        x2 = self.attention(x1, x2)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x