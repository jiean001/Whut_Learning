#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-12-25
#
# Author: jiean001
#########################################################
import torch
import torch.nn as nn

# data settings
default_dataloader_name = 'omniglot_dataloader'
default_dataset_root = r'/home/share/dataset/FSL/'
default_dataset_name = 'omniglot'
default_batch_size = 16
default_seed = 0
default_nThreads = 3
# train
default_split = 'train'
default_way = 10
default_shot = 1
default_unlabeled = 5
default_query = 5
default_label_ratio = 0.5
default_aug_90 = True
default_batches_per_epoch = 100
default_train_episodes = default_batches_per_epoch
# test
default_test_split = 'test'
default_test_way = 5
default_test_shot = 1
default_test_query = 5
default_test_episodes = 100

# classifier settings
default_classifier_name = 'prototype_classifier'
default_x_dim = '1,28,28'
default_hid_dim = 64
default_z_dim = 64
default_cls_opt = 'Adam'
default_cls_lr = 0.001
default_cls_lrS = 20
default_cls_lrG = 0.5

# discriminator settings
# mlp_discriminator  related_discriminator related_continuous_discriminator
# related_continuous_only_g_discriminator
default_discriminator_name = 'related_continuous_only_g_discriminator'
# mlp_discriminator
# default_input_dim = 28*28
# related_discriminator
default_input_dim = default_x_dim
default_num_classes = default_way
# mlp_discriminator
# default_out_dim_list = '1000, 500, 250, 250, 250, 1'
# related_discriminator
default_out_dim_list = '64, 64, 64, 64, 64, 64'
# default_activation = 'nn.LeakyReLU'
default_activation = 'nn.ReLU'
default_activation_parameter = 0.0
default_is_wn = True
default_wn_dim = 1
default_noise_mean = 0.0
default_noise_hidden_std = 0.5
default_noise_input_std = 0.3
default_dis_opt = 'Adam'
default_dis_lr = 0.001
default_dis_lrS = 20
default_dis_lrG = 0.5
default_continuous_type = 'cat'

# global env settings
default_is_training = True
default_is_testing = False
default_is_retrain = False
default_cuda = True
default_use_tensorboardX = True
default_epoches = 1000

default_checkpoint_dir = './checkpoint'
default_log_dir = 'log'
default_tfX_comment = 'the tensorboardX commit'
default_gpu_ids = '0,1'
default_start_epoch = 0

# protypical_mlp_network
# protypical_related_network
# protopical_related_continuous_network
# p_r_c_g
default_model_name = 'p_r_c_g'

# neural style transfer
# data start
default_nst_dataloader_name = 'NST_dataloader'
# /home/share/dataset/MCGAN/SEPARATE/
# /home/share/dataset/my_collection/scene_text/
default_nst_dataset_dir = r'/home/share/dataset/MCGAN/SEPARATE/'
# format_255 Capitals64
default_nst_dataset_name = 'Capitals64'
default_nst_dataset_type = 'train'
default_nst_is_rgb = False
default_nst_style_character_num = 3
default_nst_one_style_choice_num = 5
default_nst_loader = 'DEFAULT'
default_nst_fineSize = 64
default_nst_is_cuda = True
default_nst_batch_size = 32
default_nst_nThreads = 1
default_nst_max_dataset_size = 1000000
# data end
# generator start
default_nst_g_model_name = 'content_generator'
default_nst_g_input_nc = 1
default_nst_g_style_num = default_nst_style_character_num
default_nst_g_output_nc = default_nst_g_input_nc
default_nst_g_ngf = 64
# batch instance layer
default_nst_g_norm_layer = 'batch'
default_nst_g_gpu_ids = default_gpu_ids
default_nst_g_lr = 0.0002
default_nst_g_beta1 = 0.5
# generator end


# variable settings
Tensor = torch.cuda.FloatTensor if default_cuda else torch.FloatTensor
