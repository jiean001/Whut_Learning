# !/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-12-31
#
# Author: jiean001
#
# data基本操作
#########################################################

import torch


def cal_one_hot(y, batch_size, num, way):
    y = y.view(batch_size * num, 1).cpu()
    one_hot = torch.zeros(batch_size * num, way).scatter_(1, y, 1)
    return one_hot.cuda()
