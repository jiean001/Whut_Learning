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
from PIL import Image


# 计算one-hot编码
def cal_one_hot(y, batch_size, num, way, is_cuda=True):
    y = y.view(batch_size * num, 1).cpu()
    one_hot = torch.zeros(batch_size * num, way).scatter_(1, y, 1)
    if is_cuda:
        one_hot = one_hot.cuda()
    return one_hot


# 图片的默认打开方式
# open_type = 'L' or 'RGB'
def default_img_loader(path, width=64, height=64, open_type='RGB'):
    if width and height:
        return Image.open(path).convert(open_type).resize((width, height))
    return Image.open(path).convert(open_type)


# 打开一张图片
def get_one_img(img_path, loader=default_img_loader, open_type='L', transform=None, fineSize=64, is_cuda=False):
    img = loader(img_path, fineSize, fineSize, open_type=open_type)
    if transform is not None:
        img = transform(img)
    if is_cuda:
        return img
    return img


# 判断是否为图片
def is_image(input_path):
    return input_path.endswith('.png') or input_path.endswith('.jpg')
