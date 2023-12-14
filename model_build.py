import torch
from torch import nn
from cls.mmcls.models.backbones import Temporal_block


class Spatial_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False):
        super(Spatial_conv2d, self).__init__()
        self.transpose = transpose
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=padding, output_padding=stride // 2)
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        y = self.act(self.norm(y))
        return y

class Spatial_Convlayer(nn.Module):
    def __init__(self, C_in, C_out, stride, kernel_size=3, padding=1, transpose=False):
        super(Spatial_Convlayer, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = Spatial_conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                padding=padding, transpose=transpose)

    def forward(self, x):
        y = self.conv(x)
        return y


def stride_generator(N, reverse=False):  # N=4
    strides = [1, 2, 1, 2] * 10  # [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]

class Encoder(nn.Module):  ##input : C_in, output: C_hid
    def __init__(self, C_in, C_hid, L_layer, kernel_size):
        super(Encoder, self).__init__()
        strides = stride_generator(L_layer)
        self.enc = nn.Sequential(
            Spatial_Convlayer(C_in, C_hid, stride=strides[0], kernel_size=kernel_size[0], padding=(kernel_size[0]-1)//2),
            *[Spatial_Convlayer(C_hid, C_hid, stride=strides[i], kernel_size=kernel_size[i], padding=(kernel_size[i]-1)//2) for i in range(1, L_layer)]
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # B*4, 3, 128, 128
        out = []
        out.append(self.enc[0](x))
        for i in range(1, len(self.enc)):
            out.append(self.enc[i](out[i-1]))
        return out


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, T_in, T_out, L_layer, kernel_size):
        super(Decoder, self).__init__()
        strides = stride_generator(L_layer)
        self.T_in = T_in
        self.C_hid = C_hid
        self.T_out = T_out
        self.C_out = C_out

        self.dec = nn.Sequential(
            *[Spatial_Convlayer(C_hid, C_hid, stride=strides[i], kernel_size=kernel_size[i], padding=(kernel_size[i] - 1) // 2,
                     transpose=True) for i in range(L_layer - 1, 0, -1)],
            Spatial_Convlayer(C_hid, C_hid, stride=strides[0], kernel_size=kernel_size[0], padding=(kernel_size[0] - 1) // 2,
                   transpose=True)
        )
        self.relu = nn.ReLU(inplace=True)
        self.readout = nn.Conv2d(C_hid, C_out, 1)
        self.channel_ex = nn.Conv2d(self.T_in * self.C_hid, self.T_out * self.C_out, 1)

    def forward(self, x):
        data = x[-1]
        for i in range(0, len(self.dec) - 1):
            out = self.dec[i](data)
            data = out + x[-(i + 2)]
        Y = self.dec[-1](data)
        BT, _, H, W = Y.shape
        Y = self.readout(Y)
        return Y


class FATranslator(nn.Module):
    def __init__(self, channel_in, L_layer, reduction=8, layer_config=(1, 8, 2, 8)):
        super(FATranslator, self).__init__()
        self.net = Temporal_block.Temporal_block( layer_config=layer_config,
                                                  in_channels=channel_in,
                                                  stem_channels=channel_in,
                                                  base_channels=channel_in // reduction,
                                                  L_layer=L_layer,
                                                  expansion=reduction).to("cuda")

        self.channel_in = channel_in  # T * hid_S

    def forward(self, x):
        for i in range(len(x)):
            BT, _, H, W = x[i].shape  # [B*T, hid_s, h, w]
            x[i] = x[i].reshape(-1, self.channel_in, H, W)  # [B, T*hid_s, h,w]
        x = self.net(x)
        for i in range(len(x)):
            _, _, H, W = x[i].shape
            x[i] = x[i].reshape(BT, -1, H, W)
        return x


class SDFNet(nn.Module):
    def __init__(self, shape_in, shape_out, hid_channel=64, L_layer=4, encoder_kernel_size=[3, 5, 7, 5], layer_config=(1, 8, 2, 8), reduction=8):
        super(SDFNet, self).__init__()
        T, C, H, W = shape_in
        self.T_out, self.C_out, _, _ = shape_out
        self.enc = Encoder(C, hid_channel, L_layer, encoder_kernel_size)
        self.hid1 = FATranslator(T * hid_channel, L_layer, reduction, layer_config)
        self.dec = Decoder(hid_channel, self.C_out, T, self.T_out, L_layer, encoder_kernel_size)

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)
        data = self.enc(x)  # data is a list    [B*T, hid_s, h, w]
        data = self.hid1(data)  # [B*T, hid_s, H, W]
        Y = self.dec(data)
        Y = Y.reshape(B, self.T_out, self.C_out, H, W)
        return Y