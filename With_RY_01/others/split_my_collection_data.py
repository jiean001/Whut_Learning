#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-12-25
#
# Author: jiean001
#
# 将自定义的数据集分割成train,val,test
#
# train:val:test = 6:1:3
#########################################################
import glob
import os
import random
import shutil
import cv2

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

src_root_dir = r'/home/share/dataset/my_collection/scene_text/'
dataset_name_lst = ['format_255', 'format_background']
split_dict = {'train': 0.6, 'val': 0.1, 'test': 0.3}

def split_dataset(src_root_dir=src_root_dir, dataset_name_lst=dataset_name_lst, split_dict=split_dict):
    dataset_dir1 = os.path.join(src_root_dir, dataset_name_lst[0])
    dataset_dir2 = os.path.join(src_root_dir, dataset_name_lst[1])
    imgs_lst = glob.glob('%s/*' %(dataset_dir1))
    random.shuffle(imgs_lst)
    used_ratio = 0
    total_imgs = len(imgs_lst)
    for item in split_dict.items():
        target_dir_1 = os.path.join(dataset_dir1, item[0])
        target_dir_2 = os.path.join(dataset_dir2, item[0])
        # print(target_dir_1, target_dir_2)
        mkdirs(target_dir_1)
        mkdirs(target_dir_2)
        start_index = int(total_imgs*used_ratio)
        used_ratio += item[1]
        end_index = int(total_imgs*used_ratio)
        for i in range(start_index, end_index):
            base_name = os.path.basename(imgs_lst[i])
            srcfile1 = os.path.join(dataset_dir1, base_name)
            srcfile2 = os.path.join(dataset_dir2, base_name)
            dstfile1 = os.path.join(target_dir_1, base_name)
            dstfile2 = os.path.join(target_dir_2, base_name)
            shutil.move(srcfile1, dstfile1)
            shutil.move(srcfile2, dstfile2)
            print(srcfile1, '--->', dstfile1)
            print(srcfile2, '--->', dstfile2)
        # print(start_index, end_index)
    # print(len(imgs_lst), imgs_lst[-1], dataset_dir1)

# resize standard image
dst_sir = os.path.join(src_root_dir, 'standard')
def resize_stand_img(src_dir=r'/home/anna/new_format_255', dst_sir=dst_sir, format_size=(64, 64)):
    mkdirs(dst_sir)
    imgs_dir_lst = glob.glob('%s/*' %(src_dir))
    for img in imgs_dir_lst:
        base_name = os.path.basename(img)
        img = cv2.imread(img)
        img_new = cv2.resize(img, format_size, interpolation=cv2.INTER_CUBIC)
        dst_file = os.path.join(dst_sir, base_name)
        print(dst_file)
        cv2.imwrite(dst_file, img_new)


if __name__ == '__main__':
    # split_dataset()
    resize_stand_img()