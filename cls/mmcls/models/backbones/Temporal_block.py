"""
Code modified based on the original code by Li, D. et al. from their GitHub repository (https://https://github.com/d-li14/involution).
Special thanks to Li, D. et al. for their contributions!

note: Three different temporal blocks are located at line 72-91.
date: August 16, 2023.

e-mail: any questions, please contact with me: ganlq@std.uestc.edu.cn
"""


import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm

from .base_backbone import BaseBackbone
from ..utils.involution_cuda import involution



class Bottleneck(nn.Module):
    def __init__(self,
                 feature_count,
                 in_channels,
                 out_channels,
                 num_block,
                 expansion=4,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        self.num_block = num_block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.feature_count = feature_count

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        """
        #  *****************************************************************
        #  1  1st encoder ------------   S-block    ------------ decoder 1st
        #  2  2nd encoder ------------   M-block    ------------ decoder 2nd
        #  3  3rd encoder ------------   M-block    ------------ decoder 3rd
        #  4  4th encoder ------------   D-block    ------------ decoder 4th
        #  *****************************************************************
        """
        if self.feature_count == 0:
            self.conv2 = involution(self.mid_channels, 1, self.conv2_stride)
            # self.conv2 = nn.Conv2d(self.mid_channels,self.mid_channels,kernel_size=1,padding=0, stride=1)
        elif self.feature_count == 1:
            self.conv2 = involution(self.mid_channels, 15, self.conv2_stride)
            # self.conv2 = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=15, padding=(15-1)//2, stride=1)
        elif self.feature_count == 2:
            # self.conv2 = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=15, padding=(15-1)//2, stride=1)
            self.conv2 = involution(self.mid_channels, 15, self.conv2_stride)
        else:
            # self.conv2 = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=21, padding=(21-1)//2, stride=1)
            self.conv2 = involution(self.mid_channels, 21, self.conv2_stride)

        self.dropout = nn.Dropout(p=0.5)
        # *******************************************************************
        # *******************************************************************
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            # out = self.dropout(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)
            if self.num_block >=6:
                out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

class Layer(nn.Sequential):
    def __init__(self, feature_count, block, num_blocks, in_channels, out_channels, expansion, stride, **kwargs):
        layers = []
        layers.append(
            block(
                feature_count=feature_count,
                in_channels=in_channels,
                out_channels=out_channels,
                num_block=num_blocks,
                expansion=expansion,
                stride=stride,
                **kwargs))
        in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(
                block(
                    feature_count=feature_count,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_block=num_blocks,
                    expansion=expansion,
                    stride=1,
                    **kwargs))
        super(Layer, self).__init__(*layers)

class Temporal_block(BaseBackbone):
    def __init__(self,
                 layer_config,
                 in_channels,
                 stem_channels,
                 base_channels,
                 L_layer,
                 expansion,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 ):
        super(Temporal_block, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.frozen_stages = frozen_stages

        _in_channels = stem_channels
        _out_channels = base_channels * expansion #base_channels=64

        self._make_MAX_POOL(in_channels, stem_channels)
        self.layers = []
        for i, num_blocks in enumerate(layer_config):
            layer = self._make_layer(
                feature_count = i,
                block = Bottleneck,
                num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                expansion=expansion,
                stride=1,
                dilation=1)
            #_in_channels = _out_channels

            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, layer)
            self.layers.append(layer_name)
        self._freeze_stages()

    def _make_layer(self, **kwargs):
        return Layer(**kwargs)

    def _make_MAX_POOL(self, in_channels, stem_channels):
        self.stem = nn.Sequential(
            ConvModule(
                in_channels,
                stem_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=True),
            involution(stem_channels // 2, 3, 1),
            nn.BatchNorm2d(stem_channels // 2),
            nn.ReLU(inplace=True),
            ConvModule(
                stem_channels // 2,
                stem_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        super(Temporal_block, self).init_weights(pretrained)
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)

    def forward(self, x):
        x[-1] = self.stem(x[-1])
        x[-1] = self.maxpool(x[-1])
        outs = []
        for i, layer_name in enumerate(self.layers):
            res_layer = getattr(self, layer_name)
            outs.append(res_layer(x[i]))
        return outs

    def train(self, mode=True):
        super(Temporal_block, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()