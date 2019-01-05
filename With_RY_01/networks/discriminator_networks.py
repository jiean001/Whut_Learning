#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-12-25
#
# Author: jiean001
#########################################################
import torch.nn as nn
import torch

from torch.autograd import Variable
try:
    from ..models.model_factory import register_discriminator
    from ..utils.data_utils import cal_one_hot
except:
    from models.model_factory import register_discriminator
    from utils.data_utils import cal_one_hot
from .base_networks import Base_MLP_Network
from .base_networks import conv_bn_relu_block
from .base_networks import conv_bn_relu_maxpool_block
from .base_networks import Flatten
from .base_networks import linear_sigmoid


class MLP_Network(nn.Module):
    def __init__(self, mlpnet, num_layer, way):
        super(MLP_Network, self).__init__()
        self.mlpnet = mlpnet
        self.num_layer = num_layer
        self.way = way

    # x: batch, num, channel, hight, width
    # y: batch, num
    def forward(self, x, y):
        batch_size = x.size(0)
        assert batch_size == y.size(0)

        num = x.size(1)
        assert num == y.size(1)

        x = Variable(x)
        y = Variable(y)

        y_one_hot = cal_one_hot(y, batch_size, num, self.way)
        x = x.view(batch_size*num, *x.size()[2:])

        out = self.mlpnet(x, y_one_hot)
        return out

    def get_graph(self):
        return self


@register_discriminator('mlp_discriminator')
def load_mlp_discriminator(**kwargs):
    input_dim = kwargs['input_dim']
    # label_dim = kwargs['label_dim']
    num_classes = kwargs['num_classes']
    out_dim_list = kwargs['out_dim_list']
    activation = kwargs['activation']
    is_wn = kwargs['is_wn']
    activation_parameter = kwargs['activation_parameter']
    wn_dim = kwargs['wn_dim']
    noise_mean = kwargs['noise_mean']
    noise_hidden_std = kwargs['noise_hidden_std']
    noise_input_std = kwargs['noise_input_std']
    way = kwargs['way']

    mlpnet = Base_MLP_Network(input_dim, num_classes, out_dim_list, activation, is_wn,
                 activation_parameter, wn_dim, noise_mean, noise_input_std, noise_hidden_std, way)

    return MLP_Network(mlpnet, num_layer=len(out_dim_list), way=way)


class Related_Network(nn.Module):
    def __init__(self, f, g, way, shot, f_shape):
        super(Related_Network, self).__init__()
        self.f = f
        self.g = g
        self.way = way
        self.shot = shot
        self.f_shape = f_shape

    # the input is the labeled support set
    def calculate_prototypical(self, xs):
        # the size of xs is: (B, way*shot, C, W, H)
        self.batch_size = xs.size(0)
        xs = Variable(xs).float()
        xs = xs.view(self.batch_size * self.way * self.shot, *xs.size()[2:])
        # the size of z_xs is: (B*way*shot, z_dim]
        z_xs = self.f.forward(xs)
        z_dim = z_xs.size(-1)
        # the size of proto is: (B, way, z_dim])
        self.proto = z_xs.view(self.batch_size, self.way, self.shot, z_dim).mean(2)
        self.z_dim = self.proto.size(-1)

    # def get_crspd_proto_based_label(self, y):
    #     assert self.batch_size == y.size(0)
    #     # y is the one-hot
    #     num_query_or_unlabeled = y.size(1)
    #     # the size of y_one_hot is (B, N, way, 1)
    #     y_one_hot = cal_one_hot(y, self.batch_size, num_query_or_unlabeled, self.way).view(self.batch_size, num_query_or_unlabeled, self.way, 1)
    #     for i in range(self.batch_size):
    #         for j in range(num_query_or_unlabeled):
    #             if i == 0 and j == 0:
    #                 # the size of self.proto[i] is (way, z_dim)
    #                 # the size of y_one_hot[i][j] is (way, 1)
    #                 # the size of crspd_proto is (1, z_dim)
    #                 crspd_protp = self.proto[i].mul(y_one_hot[i][j]).sum(0).view(1, self.z_dim)
    #             else:
    #                 tmp_crspd_protp = self.proto[i].mul(y_one_hot[i][j]).sum(0).view(1, self.z_dim)
    #                 crspd_protp = torch.cat([crspd_protp, tmp_crspd_protp], 0)
    #     # the size of the crspd_protp is: (BxN, C, H, W)
    #     return crspd_protp.view(crspd_protp.size(0), *self.f_shape[:])

    def get_crspd_proto_based_label(self, y):
        assert self.batch_size == y.size(0)
        # y is the one-hot
        num_query_or_unlabeled = y.size(1)
        # the size of y_one_hot is (B*N, 1, way)
        y_one_hot = cal_one_hot(y, self.batch_size, num_query_or_unlabeled, self.way).view(self.batch_size * num_query_or_unlabeled, 1, self.way)
        # the size of proto is (B*N, way, Z)
        proto = self.proto.unsqueeze(1)
        proto = proto.expand(self.batch_size, num_query_or_unlabeled, self.way, self.z_dim)
        proto = proto.contiguous().view((-1, self.way, self.z_dim))

        crspd_protp = torch.bmm(y_one_hot, proto)
        return crspd_protp.view(crspd_protp.size(0), *self.f_shape[:])

    def get_g_input(self, x, y):
        assert self.batch_size == x.size(0)
        x = x.view(self.batch_size * x.size(1), *x.size()[2:])
        z_x = self.f.forward(x)
        z_x = z_x.view(z_x.size(0), *self.f_shape[:])
        z_x_corresponding_proto = self.get_crspd_proto_based_label(y)

        assert z_x.size() == z_x_corresponding_proto.size()
        # the size of g_input is: (B*N, 2*f_out_c, f_out_h, f_out_w)
        return torch.cat([z_x, z_x_corresponding_proto], 1)

    def get_f_output(self, x):
        return self.f.forward(x)

    def forward(self, x, y, input_type='query'):
        if input_type == 'query':
            y = Variable(y)
        else:
            pass
            # y = y.detach()
        x = Variable(x).float()
        g_input = self.get_g_input(x, y)
        return self.g.forward(g_input)

    def evaluate_forward(self, x, y):
        x = Variable(x).float()
        y = Variable(y)
        g_input = self.get_g_input(x, y)
        return self.g.forward(g_input)

    # # the input is:
    # # the unlabeled set and its corresponding predicted label
    # # or
    # # the query set and its corresponding label
    def forward_bak(self, x, y):
        # if requires_grad:
        x = Variable(x).float()
        y = Variable(y)
        # else:
        #     self.f.eval()
        #     self.g.eval()
        #     x = Variable(x, requires_grad=False).float()
        #     y = Variable(y, requires_grad=False)
        g_input = self.get_g_input(x, y)
        return self.g.forward(g_input)
    #
    #



@register_discriminator('related_discriminator')
def load_related_discriminator(**kwargs):
    input_dim = kwargs['input_dim']        # default is [1, 28, 28]
    out_dim_list = kwargs['out_dim_list']  # default is:[32, 64, 64, 64, 64, 64]
    way = kwargs['way']
    shot = kwargs['shot']
    f_shape = (out_dim_list[3], input_dim[1] // 2 // 2, input_dim[2] // 2 // 2)

    f = nn.Sequential(
        conv_bn_relu_maxpool_block(input_dim[0], out_dim_list[0]),         # (32, 14, 14)
        conv_bn_relu_maxpool_block(out_dim_list[0], out_dim_list[1]),      # (64, 7, 7)
        conv_bn_relu_block(out_dim_list[1], out_dim_list[2]),              # (64, 7, 7)
        conv_bn_relu_block(out_dim_list[2], out_dim_list[3]),              # (64, 7, 7)
        Flatten()
    )

    g = nn.Sequential(
        conv_bn_relu_maxpool_block(2*out_dim_list[3], out_dim_list[4]),           # (128, 3, 3)
        conv_bn_relu_maxpool_block(out_dim_list[4], out_dim_list[5]),             # (64, 1, 1)
        Flatten(),
        linear_sigmoid(out_dim_list[5])
    )

    return Related_Network(f, g, way, shot, f_shape)
