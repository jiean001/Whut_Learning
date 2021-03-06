#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-12-26
#
# Author: jiean001
#########################################################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

try:
    from ..options.default_settings import Tensor
except:
    from options.default_settings import Tensor


class Gaussian_Noise_Layer(nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        super(Gaussian_Noise_Layer, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        input = input.float()
        noise = Tensor(np.random.normal(self.mean, self.std, size=input.size()))
        return input+noise


def mlp_lrelu_wn(input_dim, output_dim, activation=nn.LeakyReLU, activation_parameter=None,
                          is_wn=False):
    if is_wn:
        out = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(input_dim, output_dim))
        )
    else:
        out = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

    if activation:
        if activation_parameter is not None:
            out.add_module(
                'activate',
                activation(activation_parameter)
            )
        else:
            out.add_module(
                'activate',
                activation()
            )

    return out


class Gaussian_MLP_LRelu_WN(nn.Module):
    def __init__(self, input_dim, num_classes, output_dim, activation=nn.LeakyReLU, activation_parameter=0.2,
                          is_wn=False, wn_dim=1, noise_mean=0.0, noise_std=0.3):
        super(Gaussian_MLP_LRelu_WN, self).__init__()
        self.gaussian_noise_layer = Gaussian_Noise_Layer(noise_mean, noise_std)
        self.mlp_lrelu_wn = mlp_lrelu_wn(input_dim=input_dim+num_classes, num_classes=num_classes,
                                         output_dim=output_dim, activation=activation, activation_parameter=activation_parameter,
                                         is_wn=is_wn, wn_dim=wn_dim)

    # label is the one-hot
    def forward(self, sample):
        x = sample[0]
        label = sample[1]
        gaussian_noise_layer = self.gaussian_noise_layer(x)
        gaussian_x_and_label = concat(gaussian_noise_layer, label, dim=1)
        return self.mlp_lrelu_wn(gaussian_x_and_label)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def concat(x, y, dim=1):
    return torch.cat([x, y], dim)


def dropout(x, rate, is_training, inplace=False):
    return F.dropout(input=x, p=rate, training=is_training, inplace=inplace)


def conv_bn_relu_maxpool_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

def conv_bn_lrelu_maxpool_block(in_channels, out_channels, leaky=0.2, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(leaky, True),
        nn.MaxPool2d(2)
    )


def conv_bn_relu_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


def conv_norm_relu_block(norm_layer, input_nc, ngf, kernel_size, padding, stride=1, relu='relu'):
    model = [nn.Conv2d(input_nc, ngf, kernel_size=kernel_size, padding=padding, stride=stride)]
    if norm_layer:
        model += [norm_layer(ngf)]

    if relu == 'relu':
        model += [nn.ReLU(True)]
    elif relu == 'Lrelu':
        model += [nn.LeakyReLU(0.2, True)]
    return nn.Sequential(*model)


def conv_bn_lrelu_block(in_channels, out_channels, leaky=0.2, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(leaky, True)
    )


def convTranspose_norm_relu_block(norm_layer, input_nc, ngf, kernel_size, padding, stride=1, output_padding=0):
    model = [nn.ConvTranspose2d(input_nc, ngf,
                                kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
             norm_layer(int(ngf)),
             nn.ReLU(True)]
    return nn.Sequential(*model)


def linear_sigmoid(in_dim):
    return nn.Sequential(
        nn.Linear(in_dim, 1),
        nn.Sigmoid()
    )


def conv_bn_lrelu_block():
    pass


def gaussian_mlp_lrelu_wn(input_dim, num_classes, output_dim, activation=nn.LeakyReLU, activation_parameter=0.2,
                          is_wn=False, wn_dim=1, noise_mean=0.0, noise_std=0.3):
    return nn.Sequential(
        Gaussian_MLP_LRelu_WN(input_dim=input_dim, num_classes=num_classes, output_dim=output_dim,
                              activation=activation, activation_parameter=activation_parameter, is_wn=is_wn, wn_dim=wn_dim,
                              noise_mean=noise_mean, noise_std=noise_std)
    )


def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'layer':
        norm_layer = functools.partial(nn.LayerNorm, affine=False)
    else:
        norm_layer = None
        print('normalization layer [%s] is not found' %(norm_type))
    return norm_layer


class Base_MLP_Network(nn.Module):
    def __init__(self, input_dim, num_classes, out_dim_list, activation, is_wn,
                 activation_parameter, wn_dim, noise_mean, noise_input_std, noise_hidden_std, way):
        super(Base_MLP_Network, self).__init__()
        self.mlpnet = nn.Sequential(
            Flatten()
        )

        self.mlpnet.add_module(
            'dis_layer0',
            gaussian_mlp_lrelu_wn(input_dim=input_dim, num_classes=num_classes,
                                  output_dim=out_dim_list[0], activation=activation,
                                  is_wn=is_wn, activation_parameter=activation_parameter, wn_dim=wn_dim,
                                  noise_mean=noise_mean,
                                  noise_std=noise_input_std)
        )

        for i in range(1, len(out_dim_list) - 1):
            self.mlpnet.add_module(
                'dis_layer{:s}'.format(str(i)),
                gaussian_mlp_lrelu_wn(input_dim=out_dim_list[i - 1], num_classes=num_classes,
                                      output_dim=out_dim_list[i], activation=activation,
                                      is_wn=is_wn, activation_parameter=activation_parameter, wn_dim=wn_dim,
                                      noise_mean=noise_mean,
                                      noise_std=noise_hidden_std)
            )

        self.mlpnet.add_module(
            'dis_layer{:s}'.format(str(len(out_dim_list))),
            gaussian_mlp_lrelu_wn(input_dim=out_dim_list[len(out_dim_list) - 2], num_classes=num_classes,
                                  output_dim=out_dim_list[len(out_dim_list) - 1], activation=nn.Sigmoid,
                                  is_wn=False, activation_parameter=None, wn_dim=wn_dim,
                                  noise_mean=noise_mean,
                                  noise_std=noise_hidden_std)
        )
        self.num_layer = len(out_dim_list)
        self.way = way

    def forward(self, x, y_one_hot, is_initial=True):
        if not is_initial:
            from torch.autograd import Variable
            try:
                from ..utils.data_utils import cal_one_hot
            except:
                from utils.data_utils import cal_one_hot

            batch_size = x.size(0)
            assert batch_size == y_one_hot.size(0)
            num = x.size(1)
            assert num == y_one_hot.size(1)

            x = Variable(x)
            y = Variable(y_one_hot)

            y_one_hot = self.cal_one_hot(y, batch_size, num, self.way)
            x = x.view(batch_size * num, *x.size()[2:])

        out = self.mlpnet[0].forward(x)
        for i in range(self.num_layer):
            sample = (out, y_one_hot)
            out = self.mlpnet[i + 1].forward(sample)
        return out


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, norm_type='batch'):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, norm_type)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, norm_type):
        conv_block = []
        p = 0
        # TODO: support padding types
        assert (padding_type == 'zero')
        p = 1

        # TODO: InstanceNorm

        conv_block += conv_norm_relu_block(norm_layer, dim, dim, 3, p)[:]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        else:
            conv_block += [nn.Dropout(0.0)]

        if norm_type == 'batch' or norm_type == 'instance':
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                           norm_layer(dim)]
        else:
            assert ("norm not defined")

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
