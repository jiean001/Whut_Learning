#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2019-01-13
#
# Author: jiean001
#########################################################
import torch
import torch.nn as nn
try:
    from ..networks.base_networks import conv_norm_relu_block
    from ..networks.base_networks import convTranspose_norm_relu_block
    from ..networks.base_networks import ResnetBlock
    from ..networks.base_networks import get_norm_layer
    from ..models.model_factory import register_generator
except:
    from networks.base_networks import conv_norm_relu_block
    from networks.base_networks import convTranspose_norm_relu_block
    from networks.base_networks import ResnetBlock
    from networks.base_networks import get_norm_layer
    from models.model_factory import register_generator


class Extract_Style_Feature(nn.Module):
    def __init__(self, input_nc, ngf, norm_layer, gpu_ids):
        super(Extract_Style_Feature, self).__init__()
        self.gpu_ids = gpu_ids
        self.extract_s_f = nn.Sequential(
            conv_norm_relu_block(norm_layer=norm_layer, input_nc=input_nc, ngf=ngf, kernel_size=7, padding=3, stride=1,
                                 relu='relu'),
            conv_norm_relu_block(norm_layer=norm_layer, input_nc=ngf * 1, ngf=ngf * 3, kernel_size=3, padding=1,
                                 stride=2,
                                 relu='relu'),
            conv_norm_relu_block(norm_layer=norm_layer, input_nc=ngf * 3, ngf=ngf * 9, kernel_size=3, padding=1,
                                 stride=2,
                                 relu='relu'),
            ResnetBlock(dim=ngf * 9, padding_type='zero', norm_layer=norm_layer, use_dropout=False, norm_type='batch'),
            ResnetBlock(dim=ngf * 9, padding_type='zero', norm_layer=norm_layer, use_dropout=False, norm_type='batch'),
            ResnetBlock(dim=ngf * 9, padding_type='zero', norm_layer=norm_layer, use_dropout=False, norm_type='batch'),
            ResnetBlock(dim=ngf * 9, padding_type='zero', norm_layer=norm_layer, use_dropout=False, norm_type='batch'),
            ResnetBlock(dim=ngf * 9, padding_type='zero', norm_layer=norm_layer, use_dropout=False, norm_type='batch'),
            ResnetBlock(dim=ngf * 9, padding_type='zero', norm_layer=norm_layer, use_dropout=False, norm_type='batch')

        )

    def forward(self, input_style_binary):
        if self.gpu_ids and isinstance(input_style_binary.data, torch.cuda.FloatTensor):
            out = nn.parallel.data_parallel(self.extract_s_f, input_style_binary, self.gpu_ids)
        else:
            out = self.S_layer_1(input_style_binary)
        return out


class Extract_Standard_Content_Feature(nn.Module):
    def __init__(self, input_nc, ngf, norm_layer, gpu_ids):
        super(Extract_Standard_Content_Feature, self).__init__()
        self.gpu_ids = gpu_ids
        self.C_layer_1 = conv_norm_relu_block(norm_layer=norm_layer, input_nc=input_nc, ngf=ngf, kernel_size=7, padding=3, stride=1,
                                 relu='relu')
        self.C_layer_2 = conv_norm_relu_block(norm_layer=norm_layer, input_nc=ngf * 1, ngf=ngf * 3, kernel_size=3, padding=1,
                                 stride=2,
                                 relu='relu')
        self.C_layer_3 = nn.Sequential(
            conv_norm_relu_block(norm_layer=norm_layer, input_nc=ngf * 3, ngf=ngf * 9, kernel_size=3, padding=1,
                                 stride=2,
                                 relu='relu'),
            ResnetBlock(dim=ngf * 9, padding_type='zero', norm_layer=norm_layer, use_dropout=False, norm_type='batch'),
            ResnetBlock(dim=ngf * 9, padding_type='zero', norm_layer=norm_layer, use_dropout=False, norm_type='batch'),
            ResnetBlock(dim=ngf * 9, padding_type='zero', norm_layer=norm_layer, use_dropout=False, norm_type='batch'),
            ResnetBlock(dim=ngf * 9, padding_type='zero', norm_layer=norm_layer, use_dropout=False, norm_type='batch'),
            ResnetBlock(dim=ngf * 9, padding_type='zero', norm_layer=norm_layer, use_dropout=False, norm_type='batch'),
            ResnetBlock(dim=ngf * 9, padding_type='zero', norm_layer=norm_layer, use_dropout=False, norm_type='batch')
        )

    def forward(self, input_standard_binary):
        if self.gpu_ids and isinstance(input_standard_binary.data, torch.cuda.FloatTensor):
            layer1 = nn.parallel.data_parallel(self.C_layer_1, input_standard_binary, self.gpu_ids)
            layer2 = nn.parallel.data_parallel(self.C_layer_2, layer1, self.gpu_ids)
            out = nn.parallel.data_parallel(self.C_layer_3, layer2, self.gpu_ids)
        else:
            layer1 = self.C_layer_1(input_standard_binary)
            layer2 = self.C_layer_2(layer1)
            out = self.C_layer_3(layer2)
        return layer1, layer2, out


class Split_Feature_Content(nn.Module):
    def __init__(self, output_nc, ngf, norm_layer, gpu_ids):
        super(Split_Feature_Content, self).__init__()
        self.gpu_ids = gpu_ids
        self.reverse_layer1 = nn.Sequential(
            ResnetBlock(dim=ngf * 18, padding_type='zero', norm_layer=norm_layer, use_dropout=False, norm_type='batch'),
            ResnetBlock(dim=ngf * 18, padding_type='zero', norm_layer=norm_layer, use_dropout=False, norm_type='batch'),
            ResnetBlock(dim=ngf * 18, padding_type='zero', norm_layer=norm_layer, use_dropout=False, norm_type='batch'),
            ResnetBlock(dim=ngf * 18, padding_type='zero', norm_layer=norm_layer, use_dropout=False, norm_type='batch'),
            ResnetBlock(dim=ngf * 18, padding_type='zero', norm_layer=norm_layer, use_dropout=False, norm_type='batch'),
            ResnetBlock(dim=ngf * 18, padding_type='zero', norm_layer=norm_layer, use_dropout=False, norm_type='batch'),
            convTranspose_norm_relu_block(norm_layer=norm_layer, input_nc=ngf * 18, ngf=ngf * 3, kernel_size=3,
                                          padding=1, stride=2, output_padding=1)
        )
        self.reverse_layer2 = nn.Sequential(
            convTranspose_norm_relu_block(norm_layer=norm_layer, input_nc=ngf * 6, ngf=ngf * 1, kernel_size=3,
                                          padding=1, stride=2, output_padding=1)
        )
        self.reverse_layer3 = nn.Sequential(
            convTranspose_norm_relu_block(norm_layer=norm_layer, input_nc=ngf * 2, ngf=output_nc, kernel_size=7,
                                          padding=3),
            nn.Tanh()
        )

    # the size of input is [B, C, H, W]
    def forward(self, b_style_feature, b_content_feature, b_content_feature_l1, b_content_feature_l2):
        if self.gpu_ids and isinstance(b_style_feature.data, torch.cuda.FloatTensor):
            input = torch.cat((b_style_feature, b_content_feature), dim=1)
            layer1 = nn.parallel.data_parallel(self.reverse_layer1, input, self.gpu_ids)
            input = torch.cat((layer1, b_content_feature_l2), dim=1)
            layer2 = nn.parallel.data_parallel(self.reverse_layer2, input, self.gpu_ids)
            input = torch.cat((layer2, b_content_feature_l1), dim=1)
            out = nn.parallel.data_parallel(self.reverse_layer3, input, self.gpu_ids)
        else:
            input = torch.cat((b_style_feature, b_content_feature), dim=1)
            layer1 = self.reverse_layer1(input)
            input = torch.cat((layer1, b_content_feature_l2), dim=1)
            layer2 = self.reverse_layer2(input)
            input = torch.cat((layer2, b_content_feature_l1), dim=1)
            out = self.reverse_layer3(input)
        return out


class Content_Generator(nn.Module):
    def __init__(self, input_nc, style_num, output_nc, ngf, norm_layer, gpu_ids):
        super(Content_Generator, self).__init__()
        self.Enc_Style_B = Extract_Style_Feature(input_nc=input_nc*style_num, ngf=ngf, norm_layer=norm_layer, gpu_ids=gpu_ids)
        self.Enc_Content_Std = Extract_Standard_Content_Feature(input_nc=input_nc, ngf=ngf, norm_layer=norm_layer, gpu_ids=gpu_ids)
        self.Dec_SC_IMG = Split_Feature_Content(output_nc=output_nc, ngf=ngf, norm_layer=norm_layer, gpu_ids=gpu_ids)

    def forward(self, input_style_binary, input_standard_binary):
        b_style_feature = self.Enc_Style_B.forward(input_style_binary)
        b_content_feature_l1, b_content_feature_l2, b_content_feature = self.Enc_Content_Std(input_standard_binary)
        out = self.Dec_SC_IMG(b_style_feature, b_content_feature, b_content_feature_l1, b_content_feature_l2)
        return out


@register_generator('content_generator')
def load_content_generator(**kwargs):
    input_nc = kwargs['input_nc']
    style_num = kwargs['style_num']
    output_nc = kwargs['output_nc']
    ngf = kwargs['ngf']
    norm_layer = kwargs['norm_layer']
    gpu_ids = kwargs['gpu_ids']

    return Content_Generator(input_nc, style_num, output_nc, ngf, get_norm_layer(norm_layer), gpu_ids)