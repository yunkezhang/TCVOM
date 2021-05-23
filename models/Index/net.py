import os
import sys
import math
from time import time
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hlaspp import ASPP
from .hlindex import HolisticIndexBlock, DepthwiseO2OIndexBlock, DepthwiseM2OIndexBlock
from .hldecoder import *
from .hlconv import *

def pred(inp, oup, conv_operator, k, batch_norm):
    # the last 1x1 convolutional layer is very important
    hlConv2d = hlconv[conv_operator]
    return nn.Sequential(
        hlConv2d(inp, oup, k, 1, batch_norm),
        nn.Conv2d(oup, oup, k, 1, padding=k//2, bias=False)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, dilation, expand_ratio, batch_norm):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        BatchNorm2d = batch_norm

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.kernel_size = 3
        self.dilation = dilation

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )

    def fixed_padding(self, inputs, kernel_size, dilation):
        kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
        return padded_inputs

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.kernel_size == (3, 3):
                m.dilation = (dilate, dilate)
                m.padding = (dilate, dilate)

    def forward(self, x):
        x_pad = self.fixed_padding(x, self.kernel_size, dilation=self.dilation)
        if self.use_res_connect:
            return x + self.conv(x_pad)
        else:
            return self.conv(x_pad)

class IndexMattingEncoder(nn.Module):
    def __init__(
        self, 
        output_stride=32, 
        width_mult=1., 
        apply_aspp=True,
        freeze_bn=False,
        use_nonlinear=True,
        use_context=True,
        ):
        super(IndexMattingEncoder, self).__init__()
        self.width_mult = width_mult
        self.output_stride = output_stride

        BatchNorm2d = nn.BatchNorm2d

        block = InvertedResidual
        aspp = ASPP

        index_block = DepthwiseM2OIndexBlock

        initial_channel = 32
        current_stride = 1
        rate = 1
        inverted_residual_setting = [
            # expand_ratio, input_chn, output_chn, num_blocks, stride, dilation
            [1, initial_channel, 16, 1, 1, 1],
            [6, 16, 24, 2, 2, 1],
            [6, 24, 32, 3, 2, 1],
            [6, 32, 64, 4, 2, 1],
            [6, 64, 96, 3, 1, 1],
            [6, 96, 160, 3, 2, 1],
            [6, 160, 320, 1, 1, 1],
        ]

        ### encoder ###
        # building the first layer
        # assert input_size % output_stride == 0
        initial_channel = int(initial_channel * width_mult)
        self.layer0 = conv_bn(4, initial_channel, 3, 2, BatchNorm2d)
        self.layer0.apply(partial(self._stride, stride=1)) # set stride = 1
        current_stride *= 2
        # building bottleneck layers
        for i, setting in enumerate(inverted_residual_setting):
            s = setting[4]
            inverted_residual_setting[i][4] = 1 # change stride
            if current_stride == output_stride:
                rate *= s
                inverted_residual_setting[i][5] = rate
            else:
                current_stride *= s
        self.layer1 = self._build_layer(block, inverted_residual_setting[0], BatchNorm2d)
        self.layer2 = self._build_layer(block, inverted_residual_setting[1], BatchNorm2d, downsample=True)
        self.layer3 = self._build_layer(block, inverted_residual_setting[2], BatchNorm2d, downsample=True)
        self.layer4 = self._build_layer(block, inverted_residual_setting[3], BatchNorm2d, downsample=True)
        self.layer5 = self._build_layer(block, inverted_residual_setting[4], BatchNorm2d)
        self.layer6 = self._build_layer(block, inverted_residual_setting[5], BatchNorm2d, downsample=True)
        self.layer7 = self._build_layer(block, inverted_residual_setting[6], BatchNorm2d)

        # freeze encoder batch norm layers
        if freeze_bn:
            self.freeze_bn()

        # define index blocks
        if output_stride == 32:
            self.index0 = index_block(32, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
            self.index2 = index_block(24, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
            self.index3 = index_block(32, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
            self.index4 = index_block(64, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
            self.index6 = index_block(160, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
        elif output_stride == 16:
            self.index0 = index_block(32, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
            self.index2 = index_block(24, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
            self.index3 = index_block(32, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
            self.index4 = index_block(64, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
        else:
            raise NotImplementedError
        
        ### context aggregation ###
        if apply_aspp:
            self.dconv_pp = aspp(320, 160, output_stride=output_stride, batch_norm=BatchNorm2d)
        else:
            self.dconv_pp = conv_bn(320, 160, 1, 1, BatchNorm2d)

        self._initialize_weights()

    def _build_layer(self, block, layer_setting, batch_norm, downsample=False):
        t, p, c, n, s, d = layer_setting
        input_channel = int(p * self.width_mult)
        output_channel = int(c * self.width_mult)

        layers = []
        for i in range(n):
            if i == 0:
                d0 = d
                if downsample:
                    d0 = d // 2 if d > 1 else 1
                layers.append(block(input_channel, output_channel, s, d0, expand_ratio=t, batch_norm=batch_norm))
            else:
                layers.append(block(input_channel, output_channel, 1, d, expand_ratio=t, batch_norm=batch_norm))
            input_channel = output_channel

        return nn.Sequential(*layers)

    def _stride(self, m, stride):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.kernel_size == (3, 3):
                m.stride = stride
                return

    def forward(self, x):
        # encode
        l0 = self.layer0(x)                                 # 4x320x320
        idx0_en, idx0_de = self.index0(l0)
        l0 = idx0_en * l0
        l0p = 4 * F.avg_pool2d(l0, (2, 2), stride=2)        # 32x160x160

        l1 = self.layer1(l0p)                               # 16x160x160
        l2 = self.layer2(l1)                                # 24x160x160
        idx2_en, idx2_de = self.index2(l2)
        l2 = idx2_en * l2
        l2p = 4 * F.avg_pool2d(l2, (2, 2), stride=2)        # 24x80x80
        
        l3 = self.layer3(l2p)                               # 32x80x80       
        idx3_en, idx3_de = self.index3(l3)  
        l3 = idx3_en * l3
        l3p = 4 * F.avg_pool2d(l3, (2, 2), stride=2)        # 32x40x40

        l4 = self.layer4(l3p)                               # 64x40x40
        idx4_en, idx4_de = self.index4(l4)
        l4 = idx4_en * l4
        l4p = 4 * F.avg_pool2d(l4, (2, 2), stride=2)        # 64x20x20

        l5 = self.layer5(l4p)                               # 96x20x20
        l6 = self.layer6(l5)                                # 160x20x20
        if self.output_stride == 32:
            idx6_en, idx6_de = self.index6(l6)
            l6 = idx6_en * l6
            l6p = 4 * F.avg_pool2d(l6, (2, 2), stride=2)    # 160x10x10
        elif self.output_stride == 16:
            l6p, idx6_de = l6, None

        l7 = self.layer7(l6p)                               # 320x10x10

        # pyramid pooling
        l = self.dconv_pp(l7)                               # 160x10x10

        return [l, l6, idx6_de, l5, l4, idx4_de, l3, idx3_de, l2, idx2_de, l1, l0, idx0_de]

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class IndexMattingDecoder(nn.Module):
    def __init__(self,
        conv_operator='std_conv',
        decoder_kernel_size=5,
        ):
        super(IndexMattingDecoder, self).__init__()
        decoder_block = IndexedUpsamlping
        BatchNorm2d = nn.BatchNorm2d

        self.decoder_layer6 = decoder_block(160*2, 96, conv_operator=conv_operator, kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.decoder_layer5 = decoder_block(96*2, 64, conv_operator=conv_operator, kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.decoder_layer4 = decoder_block(64*2, 32, conv_operator=conv_operator, kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.decoder_layer3 = decoder_block(32*2, 24, conv_operator=conv_operator, kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.decoder_layer2 = decoder_block(24*2, 16, conv_operator=conv_operator, kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.decoder_layer1 = decoder_block(16*2, 32, conv_operator=conv_operator, kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.decoder_layer0 = decoder_block(32*2, 32, conv_operator=conv_operator, kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)

        self.pred = pred(32, 1, conv_operator, k=decoder_kernel_size, batch_norm=BatchNorm2d)

    def forward(self, inputs):
        l, l6, idx6_de, l5, l4, idx4_de, l3, idx3_de, l2, idx2_de, l1, l0, idx0_de = inputs
        # decode                                # out
        l = self.decoder_layer6(l, l6, idx6_de) # OS=16
        l = self.decoder_layer5(l, l5)
        l = self.decoder_layer4(l, l4, idx4_de) # OS=8
        l = self.decoder_layer3(l, l3, idx3_de) # OS=4
        l = self.decoder_layer2(l, l2, idx2_de) # OS=2
        l = self.decoder_layer1(l, l1)
        l = self.decoder_layer0(l, l0, idx0_de) # OS=1

        l = self.pred(l)
        return l

class IndexMatting(nn.Module):
    def __init__(self):
        super(IndexMatting, self).__init__()
        self.encoder = IndexMattingEncoder()
        self.decoder = IndexMattingDecoder()

    def forward(self, x, **kwargs):
        l = self.encoder(x)
        l = self.decoder(l)
        return l