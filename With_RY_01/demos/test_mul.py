from utils.data_utils import cal_one_hot
from options.default_settings import Tensor

import numpy as np
import torch


B = 3
way = 5
z_dim = 4
n = 2

proto = Tensor(np.random.random(size=(B, way, z_dim)))
y = Tensor(np.random.randint(0, way, size=(B, n))).long()


def get_crspd_proto_based_label(y):
    # y is the one-hot
    # the size of y_one_hot is (B, N, W)
    y_one_hot = cal_one_hot(y, B, n, way).view(B, n, way, 1).cpu()
    crspd_proto = proto[0].mul(y_one_hot[0][0]).sum(0).view(1, z_dim)
    for j in range(1, n):
        tmp_proto = proto[0].mul(y_one_hot[0][j]).sum(0).view(1, z_dim)
        crspd_proto = torch.cat([crspd_proto, tmp_proto], 0)
    for i in range(1, B):
        for j in range(0, n):
            tmp_proto = proto[i].mul(y_one_hot[i][j]).sum(0).view(1, z_dim)
            crspd_proto = torch.cat([crspd_proto, tmp_proto], 0)
    print(crspd_proto)

    print('proto:', proto, proto.size())
    print('y', y, y.size())

get_crspd_proto_based_label(y)