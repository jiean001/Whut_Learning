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
    from .base_networks import Base_MLP_Network
    from ..utils.data_utils import  cal_one_hot
except:
    from models.model_factory import register_discriminator
    from utils.data_utils import cal_one_hot
from .base_networks import Base_MLP_Network


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
