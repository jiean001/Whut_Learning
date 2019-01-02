#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-12-25
#
# Author: jiean001
#########################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
try:
    from ..models.model_factory import register_classifier
    from ..utils.metric import batch_euclidean_dist
    from ..utils.metric import euclidean_dist
    from ..options.default_settings import Tensor
except:
    from models.model_factory import register_classifier
    from utils.metric import batch_euclidean_dist
    from utils.metric import euclidean_dist
    from options.default_settings import Tensor
from .base_networks import conv_bn_relu_maxpool_block
from .base_networks import Flatten


class Prototypical_Net(nn.Module):
    def __init__(self, encoder, is_cuda, way, test_way, shot, unlabeled, query):
        super(Prototypical_Net, self).__init__()
        self.encoder = encoder
        self.is_cuda = is_cuda
        self.way = way
        self.test_way = test_way
        self.shot = shot
        self.unlabeled = unlabeled
        self.query = query
    def cal_proto_and_zu(self, z, b, z_dim, n_xu_or_xq, way):
        proto = z[:b*way*self.shot].view(b, way, self.shot, z_dim).mean(2)
        zu_or_zq = z[b*way*self.shot:].view(b, n_xu_or_xq, z_dim)
        return proto, zu_or_zq

    def predict_label_only(self, xu):
        self.encoder.eval()
        assert self.batch_size == xu.size(0)
        n_unlabeled = self.size(1)
        batch_input = xu.view(self.batch_size * n_unlabeled, *xu.size()[2:])
        z_unlabeled = self.encoder(batch_input).view(self.batch_size, n_unlabeled, -1)
        dists = batch_euclidean_dist(z_unlabeled, self.z_proto)
        log_p_y = F.log_softmax(-dists, dim=2).view(self.batch_size * self.way, n_unlabeled // self.way, -1)
        _, y_hat = log_p_y.max(2)
        y_hat = y_hat.view(self.batch_size, n_unlabeled)
        return y_hat

    def predict_label(self, xu):
        self.encoder.eval()
        pass

    def forward_base(self, xs, xu_or_xq, yq=None, op_type='train'):
        self.batch_size = xs.size(0)
        assert xu_or_xq.size(0) == self.batch_size

        if op_type == 'train':
            way = self.way
        elif op_type == 'val' or op_type == 'test':
            way = self.test_way

        if self.is_cuda:
            xs = xs.cuda()
            xu_or_xq = xu_or_xq.cuda()

        n_xu_or_xq = xu_or_xq.size(1)

        xs = xs.view(self.batch_size * way * self.shot, *xs.size()[2:])
        xu_or_xq = xu_or_xq.view(self.batch_size * n_xu_or_xq, *xu_or_xq.size()[2:])
        input_batch = torch.cat([xs, xu_or_xq], 0)
        z = self.encoder.forward(input_batch.float())
        z_dim = z.size(-1)

        self.z_proto, zu_or_zq = self.cal_proto_and_zu(z, self.batch_size, z_dim, n_xu_or_xq, way)
        # the shape of dists is:    [Batch, Way*Query, Way(Distance)]
        # the shape of zu_or_zq is: [Batch, Way*Query, Z_DIM]
        # the shape of z_proto is:  [Batch, Way, Z_DIM]
        dists = batch_euclidean_dist(zu_or_zq, self.z_proto)
        # the shape of dists is:    [Batch, Way*Query, Way(probability)]
        log_p_dists = F.log_softmax(-dists, dim=2)
        # the shape of y_hat is:    [Batch, Way*Query]
        _, y_hat = log_p_dists.max(2)

        if yq is not None:
            assert self.batch_size == yq.size(0)
            target_inds = yq.view(yq.size(0)*yq.size(1), 1, 1).long()
            if self.is_cuda:
                target_inds = target_inds.cuda()
            log_p_y = log_p_dists.view(log_p_dists.size(0)*log_p_dists.size(1), 1, way)
            loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
            acc_val = torch.eq(y_hat, target_inds.view(self.batch_size, -1)).float().mean()

            return y_hat, loss_val, {
                'loss': loss_val.item(),
                'acc': acc_val.item()
            }
        return y_hat, log_p_dists

    def forward(self, sample, run_type=1):
        # xs: (B, way*shot, C, W, H)
        xs = Variable(sample['xs'])  # support labeled

        if run_type == 0:   # unlabeled
            # xc: (B, N, C, H, W)
            xu = Variable(sample['xu'])  # support unlabeled
            return self.forward_base(xs, xu, op_type='train')
        elif run_type == 1:  # query
            # xq: (B, N, C, H, W)
            # yq: (B, N, 1)
            xq = Variable(sample['xq'])  # query
            yq = Variable(sample['yq'], requires_grad=False)  # query label
            return self.forward_base(xs, xq, yq, 'train')
        else:
            raise ValueError("Unknown type {:s}".format(type))

    def forward_test(self, sample, run_type=1):
        self.encoder.eval()
        # xs: (B, way*shot, C, W, H)
        xs = Variable(sample['xs'])  # support labeled

        if run_type == 0:   # unlabeled
            # xc: (B, N, C, H, W)
            xu = Variable(sample['xu'])  # support unlabeled
            return self.forward_base(xs, xu, op_type='test')
        elif run_type == 1:  # query
            # xq: (B, N, C, H, W)
            # yq: (B, N, 1)
            xq = Variable(sample['xq'], requires_grad=False)  # query
            yq = Variable(sample['yq'], requires_grad=False)  # query label
            return self.forward_base(xs, xq, yq, 'test')
        else:
            raise ValueError("Unknown type {:s}".format(type))

    def get_graph(self):
        return self.encoder


@register_classifier('prototype_classifier')
def load_prototype_classifier(**kwargs):
    is_cuda = kwargs['cuda']
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']
    way = kwargs['way']
    test_way = kwargs['test_way']
    shot = kwargs['shot']
    unlabeled = kwargs['unlabeled']
    query = kwargs['query']

    encoder = nn.Sequential(
        conv_bn_relu_maxpool_block(x_dim[0], hid_dim),  # (64, 14, 14)
        conv_bn_relu_maxpool_block(hid_dim, hid_dim),   # (64, 7, 7)
        conv_bn_relu_maxpool_block(hid_dim, hid_dim),   # (64, 3, 3)
        conv_bn_relu_maxpool_block(hid_dim, z_dim),     # (64, 1, 1)
        Flatten()
    )

    return Prototypical_Net(encoder, is_cuda, way, test_way, shot, unlabeled, query)
