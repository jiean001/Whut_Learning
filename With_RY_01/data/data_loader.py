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
    from ..data.data_factory import register_dataloader
    from ..data.base_data_loader import BaseDataLoader
    from ..data.custom_dataset import Custom_Dataset
    from ..data.custom_dataset import NST_Dataset
    from ..utils.data_utils import default_img_loader
except:
    from data.data_factory import register_dataloader
    from data.base_data_loader import BaseDataLoader
    from data.custom_dataset import Custom_Dataset
    from data.custom_dataset import NST_Dataset
    from utils.data_utils import default_img_loader
import torch
import os
import numpy as np
import torchvision.transforms as transforms


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
            shuffle=True,
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
        return {'xs': support_images,
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


class NST_Customed_DataLoader(BaseDataLoader):
    def __init__(self, dataset_dir, dataset_name, dataset_type, is_rgb, style_character_num, one_style_choice_num,
                 default_img_loader, fineSize, is_cuda, batch_size, nThreads, max_dataset_size=100000):
        super(NST_Customed_DataLoader, self).__init__()
        transform = transforms.Compose([
            transforms.Resize(fineSize, fineSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])

        dataset = NST_Dataset(dataset_dir=dataset_dir, dataset_name=dataset_name, dataset_type=dataset_type,
                              is_rgb=is_rgb, style_character_num=style_character_num,
                              one_style_choice_num=one_style_choice_num, loader=default_img_loader, transform=transform, fineSize=fineSize, is_cuda=is_cuda)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True, num_workers=int(nThreads))
        self.max_dataset_size = max_dataset_size
        self.dataset = dataset
        self._data = NST_Customed_Data(data_loader, self.max_dataset_size)

    def name(self):
        return 'NST_Customed_DataLoader'

    def load_data(self):
        return self._data

    def __len__(self):
        return min(len(self.dataset), self.max_dataset_size)


class NST_Customed_Data(object):
    def __init__(self, data_loader, max_dataset_size):
        super(NST_Customed_Data, self).__init__()
        self.data_loader = data_loader
        self.max_dataset_size = min(data_loader.__len__(), max_dataset_size)

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def next(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration
        ts_style_img_lst, ts_standard_img, ts_gt_img = next(self.data_loader_iter)
        return {'style_imgs': ts_style_img_lst,
                'std_img': ts_standard_img,
                'gt_img': ts_gt_img}

    def __next__(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration
        ts_style_img_lst, ts_standard_img, ts_gt_img = next(self.data_loader_iter)
        return {'style_imgs': ts_style_img_lst,
                'std_img': ts_standard_img,
                'gt_img': ts_gt_img}


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


@register_dataloader('NST_dataloader')
def load_omniglot_dataloader(**kwargs):
    dataset_dir = kwargs['dataset_dir']
    dataset_name = kwargs['dataset_name']
    dataset_type = kwargs['dataset_type']
    is_rgb = kwargs['is_rgb']
    style_character_num = kwargs['style_character_num']
    one_style_choice_num = kwargs['one_style_choice_num']
    _default_img_loader = kwargs['default_img_loader']
    fineSize = kwargs['fineSize']
    is_cuda = kwargs['is_cuda']
    batch_size = kwargs['batch_size']
    nThreads = kwargs['nThreads']
    max_dataset_size = kwargs['max_dataset_size']
    if _default_img_loader == 'DEFAULT':
        img_loader = default_img_loader

    return NST_Customed_DataLoader(dataset_dir, dataset_name, dataset_type, is_rgb, style_character_num,
                                   one_style_choice_num, img_loader, fineSize, is_cuda, batch_size,
                                   nThreads, max_dataset_size=max_dataset_size)
