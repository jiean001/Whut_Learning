#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-07-09
#
# Author: jiean001
#
# 自定义DataLoader
#########################################################
try:
    from .data_factory import register_dataloader
    from .base_data_loader import BaseDataLoader
    from .custom_dataset import Custom_Dataset
except:
    from data_factory import register_dataloader
    from base_data_loader import BaseDataLoader
    from custom_dataset import Custom_Dataset
import torch
import os


class Customed_DataLoader(BaseDataLoader):
    def __init__(self, dataset_root, dataset_name, split,  num_way, num_shot, num_unlabel,
                num_query, label_ratio, aug_90, seed, batch_size, nThreads, episodes):
        super(Customed_DataLoader, self).__init__()

        dataset_folder = os.path.join(dataset_root, dataset_name)
        dataset = Custom_Dataset(folder=dataset_folder, split=split,
                                 num_way=num_way, num_shot=num_shot, num_unlabel=num_unlabel,
                                 num_query=num_query, label_ratio=label_ratio, aug_90=aug_90, seed=seed
                                 )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(nThreads))
        self.dataset = dataset
        self._data = Customed_Data(data_loader, episodes)

    def name(self):
        return 'Customed_DataLoader'

    def load_data(self):
        return self._data

    def __len__(self):
        return 1000000
        # return min(len(self.dataset), self.opt.max_dataset_size)


class Customed_Data(object):
    def __init__(self, data_loader, max_dataset_size):
        self.data_loader = data_loader
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def next(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration
        support_images, support_unlabel_images, query_images, query_labels = next(self.data_loader_iter)
        return {'xs':support_images,
                'xu': support_unlabel_images,
                'xq': query_images,
                'yq': query_labels}

    def __next__(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration
        support_images, support_unlabel_images, query_images, query_labels = next(self.data_loader_iter)
        return {'xs': support_images,
                'xu': support_unlabel_images,
                'xq': query_images,
                'yq': query_labels}


@register_dataloader('omniglot_dataloader')
def load_omniglot_dataloader(**kwargs):
    dataset_root = kwargs['dataset_root']
    dataset_name = kwargs['dataset_name']
    split = kwargs['split']
    label_ratio = kwargs['label_ratio']
    aug_90 = kwargs['aug_90']
    seed = kwargs['seed']
    nThreads = kwargs['nThreads']

    num_way = kwargs['way']
    num_shot = kwargs['shot']
    num_unlabel = kwargs['unlabeled']
    num_query = kwargs['query']
    batch_size = kwargs['batch_size']
    episodes = kwargs['episodes']

    return Customed_DataLoader(dataset_root, dataset_name, split,  num_way, num_shot, num_unlabel,
                num_query, label_ratio, aug_90, seed, batch_size, nThreads, episodes)
