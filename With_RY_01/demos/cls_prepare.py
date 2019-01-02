import torch
import numpy as np

b = 12
way = 5
shot = 3
n = 10
c = 1
h = 28
w = 28

Tensor = torch.FloatTensor

xs = Tensor(np.random.random((b, way*shot, c, h, w)))
xq = Tensor(np.random.random((b, n, c, h, w)))

n_xs = xs.size(1)
n_xq = xq.size(1)
x_one_barch = torch.cat([xs, xq], 1)

x_batch = x_one_barch.view(b*(n_xs+n_xq), *x_one_barch.size()[2:])

print('xs size:', xs.size(), 'xq size:', xq.size(), 'x_one size:', x_one_barch.size(), 'x_batch size:', x_batch.size())