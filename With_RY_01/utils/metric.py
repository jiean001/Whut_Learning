import torch


def batch_euclidean_dist_old(x, y):
    b = x.size(0)
    assert b == y.size(0)

    n = x.size(1)
    m = y.size(1)
    d = x.size(2)
    assert d == y.size(2)

    _x = x[0].unsqueeze(1).expand(n, m, d)
    _y = y[0].unsqueeze(0).expand(n, m, d)
    for i in range(1, b):
        try:
            _x = torch.stack([_x, x[i].unsqueeze(1).expand(n, m, d)], 0)
            _y = torch.stack([_y, y[i].unsqueeze(0).expand(n, m, d)], 0)
        except:
            _x = torch.cat([_x, x[i].unsqueeze(1).expand(n, m, d).unsqueeze(0)], 0)
            _y = torch.cat([_y, y[i].unsqueeze(0).expand(n, m, d).unsqueeze(0)], 0)
    return torch.pow(_x - _y, 2).sum(3)


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)


def batch_euclidean_dist(x, y):
    b = x.size(0)
    assert b == y.size(0)

    n = x.size(1)
    m = y.size(1)
    d = x.size(2)
    assert d == y.size(2)

    _x = x.unsqueeze(2).expand(b, n, m, d)
    _y = y.unsqueeze(1).expand(b, n, m, d)
    return torch.pow(_x - _y, 2).sum(3)


def test():
    import numpy as np
    Tensor = torch.FloatTensor
    batch_size = 8
    z_dim = 4
    way = 5
    num = 3

    x = Tensor(np.random.random(size=(batch_size, num, z_dim)))
    y = Tensor(np.random.random(size=(batch_size, way, z_dim)))
    tmp = batch_euclidean_dist(x, y)
    tmp2 = batch_euclidean_dist_old(x, y)
    for i in range(batch_size):
        tmp3 = euclidean_dist(x[i], y[i])
        print(tmp[i] == tmp3)
        # print(tmp3.size())
        # print('---', tmp[i], '\n===', tmp2[i], '\n\n\n\n')
# test()
