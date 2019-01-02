# !/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-12-31
#
# Author: Yi Rong
#
# data基本操作
#########################################################
try:
    from .episode import  Episode
except:
    from episode import Episode
import torch.utils.data as data
import numpy as np
import os
import pickle as pkl


class Custom_Dataset(data.Dataset):
    def __init__(self, folder, split, num_way=5, num_shot=1, num_unlabel=5,
                 num_query=-1, label_ratio=1, aug_90=True, seed=0):
        super(Custom_Dataset, self).__init__()
        self._folder = folder
        self._split = split
        self._num_way = num_way
        self._num_shot = num_shot
        self._num_unlabel = num_unlabel
        self._label_ratio = label_ratio
        self._num_query = num_query
        self._aug_90 = aug_90
        self._seed = seed
        self._rnd = np.random.RandomState(seed)

        self.read_dataset()

    def read_dataset(self):
        split_str = self._split + '_vinyals'
        aug_str = '_aug90' if self._aug_90 else ''
        data_folder = os.path.join(self._folder, split_str + aug_str + '.pkl')
        print(data_folder)
        if os.path.exists(data_folder):
            try:
                with open(data_folder, 'rb') as data_file:
                    data = pkl.load(data_file, encoding='bytes')
                    self._images = data[b'images']
                    self._labels = data[b'labels']
                    self._label_str = data[b'label_str']
            except:
                with open(data_folder, 'rb') as data_file:
                    data = pkl.load(data_file)
                    self._images = data['images']
                    self._labels = data['labels']
                    self._label_str = data['label_str']

        self._num_class = len(self._label_str)
        self.read_label_split()
        self.build_split_set()

    def read_label_split(self):
        self._label_idx = np.array(self.label_split(), dtype=np.int64)

    def label_split(self):
        rnd = np.random.RandomState(self._seed)
        num_class = self._num_class
        num_image = self._labels.shape[0]
        image_ids = np.arange(num_image)

        label_split = []
        self._label_dict = {}

        for class_idx in range(num_class):
            class_images = image_ids[self._labels == class_idx]
            self._label_dict[class_idx] = class_images
            rnd.shuffle(class_images)
            label_split.extend(class_images[:int(len(class_images) * self._label_ratio)])
        return sorted(label_split)

    def build_split_set(self):
        self._label_idx = np.array(self._label_idx)
        self._label_idx_set = set(self._label_idx)

        self._unlabel_idx = list(filter(lambda _idx:
                   _idx not in self._label_idx_set, range(self._labels.shape[0])))
        self._unlabel_idx = np.array(self._unlabel_idx, dtype=np.int64)

        if len(self._unlabel_idx) > 0:
            self._unlabel_idx_set = set(self._unlabel_idx)
        else:
            self._unlabel_idx_set = set()

    def __getitem__(self, index):
        num_class = self._num_class
        class_ids = np.arange(num_class)
        self._rnd.shuffle(class_ids)

        support_image_ids = []
        support_labels = []
        support_unlabel_image_ids = []

        query_image_ids = []
        query_labels = []

        is_training = self._split in ["train"]
        assert is_training or self._split in ["val", "test"]

        for way_idx in range(self._num_way):
            class_idx = class_ids[way_idx]
            class_image_ids = self._label_dict[class_idx]

            label_ids = list(
                filter(lambda _id: _id in self._label_idx_set, class_image_ids))

            unlabel_ids = list(
                filter(lambda _id: _id in self._unlabel_idx_set, class_image_ids))

            self._rnd.shuffle(label_ids)
            self._rnd.shuffle(unlabel_ids)

            support_image_ids.extend(label_ids[:self._num_shot])
            support_labels.extend([way_idx] * self._num_shot)

            if self._num_query == -1:
                if is_training:
                    num_query = len(label_ids) - self._num_shot
                else:
                    num_query = len(label_ids) - self._num_shot - self._num_unlabel
            else:
                num_query = self._num_query
                if is_training:
                    assert num_query <= len(label_ids) - self._num_shot
                else:
                    assert num_query <= len(label_ids) - self._num_shot - self._num_unlabel

            query_image_ids.extend(label_ids[self._num_shot: self._num_shot + num_query])
            query_labels.extend([way_idx] * num_query)

            if is_training:
                support_unlabel_image_ids.extend(unlabel_ids[:self._num_unlabel])
            else:
                support_unlabel_image_ids.extend(
                    label_ids[self._num_shot + num_query: self._num_shot + num_query + self._num_unlabel])

            query_ids_set = set(query_image_ids)
            for _id in support_unlabel_image_ids:
                assert _id not in query_ids_set

        support_images = self._images[support_image_ids] / 255.0
        support_unlabel_images = self._images[support_unlabel_image_ids] / 255.0
        support_labels = np.array(support_labels)
        query_images = self._images[query_image_ids] / 255.0
        query_labels = np.array(query_labels)

        return Episode(support_images,
                       support_labels,
                       query_images,
                       query_labels,
                       support_unlabel_images).get_data_split()

    def __len__(self):
        return 1000000
