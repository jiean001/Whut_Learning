#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-12-25
#
# Author: jiean001
#########################################################


import torch.nn as nn
import argparse
import os
from .default_settings import *
try:
    from ..utils.opt_utils import filter_multi_opt
    from ..utils.dir_util import mkdirs
except:
    from utils.opt_utils import filter_multi_opt
    from utils.dir_util import mkdirs


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.parse()

    def initialize(self):
        # -------------- global options begin -------------
        self.parser.add_argument('--global.model_name', type=str, default=default_model_name, metavar='MODELNAME',
                                 help="data loader name (default: {:s})".format(default_dataloader_name))
        self.parser.add_argument('--global.use_tensorboardX', action='store_true', default=default_use_tensorboardX,
                                 help='use tensorboardX to visiable')
        self.parser.add_argument('--global.tfX_comment', type=str, default=default_tfX_comment,
                                 help='use tensorboardX to visiable')
        self.parser.add_argument('--global.checkpoint_dir', type=str, default=default_checkpoint_dir,
                                 help='the checkpoint dir'.format(str(default_checkpoint_dir)))
        self.parser.add_argument('--global.log_dir', type=str, default=default_log_dir,
                                 help='the log dir'.format(str(default_log_dir)))
        self.parser.add_argument('--global.gpu_ids', type=str, default=default_gpu_ids, help='gpu ids: e.g. 0  0,1,2, 0,2')
        self.parser.add_argument('--global.cuda', action='store_true', default=default_cuda,
                                 help="run in CUDA mode (default: True)")
        self.parser.add_argument('--global.is_training', action='store_true', default=default_is_training,
                                 help="is training (default: {:s})".format(str(default_is_training)))
        self.parser.add_argument('--global.is_testing', action='store_true', default=default_is_testing,
                                 help="is training (default: {:s})".format(str(default_is_testing)))
        self.parser.add_argument('--global.is_retrain', action='store_true', default=default_is_retrain,
                                 help="is training (default: {:s})".format(str(default_is_retrain)))
        self.parser.add_argument('--global.epoches', '--epochs', type=int, help='number of epochs to train for', default=default_epoches)
        # -------------- global options end  --------------

        # -------------- data options begin  ------------------------------
        self.parser.add_argument('--data.dataloader_name', type=str, default=default_dataloader_name, metavar='DLNAME',
                                 help="data loader name (default: {:s})".format(default_dataloader_name))
        self.parser.add_argument('--data.dataset_root', type=str, default=default_dataset_root, metavar='DTROOT',
                                 help="dataset root (default: {:s})".format(default_dataset_root))
        self.parser.add_argument('--data.dataset_name', type=str, default=default_dataset_name, metavar='DS',
                            help="data set name (default: {:s})".format(default_dataset_name))
        self.parser.add_argument('--data.seed', type=int, default=default_seed, metavar='BATCH',
                                 help="seed default: {:s}".format(str(default_seed)))
        self.parser.add_argument('--data.nThreads', type=int, default=default_nThreads, metavar='BATCH',
                                 help="number of thread, default is {:s}".format(str(default_nThreads)))
        self.parser.add_argument('--data.aug_90', action='store_true', default=default_aug_90,
                                 help="run in CUDA mode (default: True)")
        # train
        self.parser.add_argument('--data.split', type=str, default=default_split, metavar='SP',
                                 help="split name (default: {:s})".format(default_split))
        self.parser.add_argument('--data.way', type=int, default=default_way, metavar='WAY',
                            help="number of classes per episode (default: 60)")
        self.parser.add_argument('--data.shot', type=int, default=default_shot, metavar='SHOT',
                            help="number of labeled support examples per class (default: 5)")
        self.parser.add_argument('--data.unlabeled', type=int, default=default_unlabeled, metavar='UNLABELED',
                                 help="number of unlabeled support examples per class (default: 5)")
        self.parser.add_argument('--data.query', type=int, default=default_query, metavar='QUERY',
                            help="number of query examples per class (default: 5)")
        self.parser.add_argument('--data.batch_size', type=int, default=default_batch_size, metavar='BATCH',
                                 help="number of query examples per class (default: 5)")
        self.parser.add_argument('--data.train_episodes', type=int, default=default_train_episodes, metavar='NTRAIN',
                            help="number of train episodes per epoch (default: 100)")
        self.parser.add_argument('--data.label_ratio', type=int, default=default_label_ratio, metavar='NTRAIN',
                                 help="number of train episodes per epoch (default: {:s})".format(str(default_label_ratio)))
        # test
        self.parser.add_argument('--data.test_split', type=str, default=default_test_split, metavar='TSP',
                                 help="split name (default: {:s})".format(default_test_split))
        self.parser.add_argument('--data.test_way', type=int, default=default_test_way, metavar='TESTWAY',
                            help="number of classes per episode in test. 0 means same as data.way (default: 5)")
        self.parser.add_argument('--data.test_shot', type=int, default=default_test_shot, metavar='TESTSHOT',
                            help="number of support examples per class in test. 0 means same as data.shot (default: 0)")
        self.parser.add_argument('--data.test_query', type=int, default=default_test_query, metavar='TESTQUERY',
                            help="number of query examples per class in test. 0 means same as data.query (default: 15)")
        self.parser.add_argument('--data.test_episodes', type=int, default=default_test_episodes, metavar='NTEST',
                            help="number of test episodes per epoch (default: 100)")

        self.parser.add_argument('--data.trainval', action='store_true',
                            help="run in train+validation mode (default: False)")
        self.parser.add_argument('--data.sequential', action='store_true',
                            help="use sequential sampler instead of episodic (default: False)")
        # -------------- data options end    ------------------------------

        # -------------- classifier options begin  -----------------------
        self.parser.add_argument('--classifier.model_name', type=str, default=default_classifier_name, metavar='CSRNAME',
                            help="classifier name (default: {:s})".format(default_classifier_name))
        self.parser.add_argument('--classifier.x_dim', type=str, default=default_x_dim, metavar='XDIM',
                            help="dimensionality of input images (default: '1,28,28')")
        self.parser.add_argument('--classifier.hid_dim', type=int, default=default_hid_dim, metavar='HIDDIM',
                            help="dimensionality of hidden layers (default: 64)")
        self.parser.add_argument('--classifier.z_dim', type=int, default=default_z_dim, metavar='ZDIM',
                            help="dimensionality of input images (default: 64)")
        self.parser.add_argument('--classifier.optim', type=str, default=default_cls_opt, metavar='CLSOPTIM',
                                 help="the classifier optimizer: {:s}".format(default_cls_opt))
        self.parser.add_argument('--classifier.lr', '--learning_rate', type=float,
                                 help='learning rate for the model, default={:s}'.format(str(default_cls_lr)), default=default_cls_lr)
        self.parser.add_argument('--classifier.lrS', '--lr_scheduler_step', type=int,
                                 help='StepLR learning rate scheduler step, default={:s}'.format(str(default_cls_lrS)), default=default_cls_lrS)
        self.parser.add_argument('--classifier.lrG', '--lr_scheduler_gamma', type=float,
                                 help='StepLR learning rate scheduler gamma, default={:s}'.format(str(default_cls_lrG)), default=default_cls_lrG)

        # -------------- classifier options end    -----------------------

        # -------------- generator options begin  -------------------------
        # -------------- generator options end    -------------------------

        # -------------- discriminator options begin   ---------------------
        self.parser.add_argument('--discriminator.model_name', type=str, default=default_discriminator_name, metavar='DISNAME',
                                 help="discriminator name (default: {:s})".format(default_discriminator_name))
        self.parser.add_argument('--discriminator.input_dim', type=str, default=default_input_dim,
                                 metavar='DISIDIM',
                                 help="the input dimension of the discriminator (default: {:s})".format(str(default_input_dim)))
        self.parser.add_argument('--discriminator.num_classes', type=int, default=default_num_classes,
                                 metavar='DISNOC',
                                 help="the class number (default: {:s})".format(str(default_num_classes)))
        self.parser.add_argument('--discriminator.out_dim_list', type=str, default=default_out_dim_list,
                                 metavar='DISODIM',
                                 help="the output dimension of the discriminator (default: {:s})".format(str(default_out_dim_list)))
        self.parser.add_argument('--discriminator.activation', type=str, default=default_activation,
                                 metavar='DISACT',
                                 help="the activation function (default: {:s})".format(str(default_activation)))
        self.parser.add_argument('--discriminator.activation_parameter', type=str, default=default_activation_parameter,
                                 metavar='DISACTP',
                                 help="the activation function parameters (default: {:s})".format(str(default_activation_parameter)))
        self.parser.add_argument('--discriminator.is_wn', action='store_true', default=default_is_wn,
                                 help="discriminator is use weight normalization (default: {:s})".format(str(default_is_wn)))
        self.parser.add_argument('--discriminator.wn_dim', type=int, default=default_wn_dim,
                                 metavar='DISWNd',
                                 help="weight normalization dimension (default: {:s})".format(str(default_wn_dim)))
        self.parser.add_argument('--discriminator.noise_mean', type=float, default=default_noise_mean,
                                 metavar='DISNSM',
                                 help="the mean of gaussian noise(default: {:s})".format(str(default_noise_mean)))
        self.parser.add_argument('--discriminator.noise_hidden_std', type=str, default=default_noise_hidden_std,
                                 metavar='DISNSHSTD',
                                 help="the std of hidden layer gaussian noise (default: {:s})".format(str(default_noise_hidden_std)))
        self.parser.add_argument('--discriminator.noise_input_std', type=str, default=default_noise_input_std,
                                 metavar='DISNSISTD',
                                 help="the std of input layer gaussian noise (default: {:s})".format(str(default_noise_input_std)))
        self.parser.add_argument('--discriminator.optim', type=str, default=default_dis_opt, metavar='DISSOPTIM',
                                 help="the classifier optimizer: {:s}".format(default_dis_opt))
        self.parser.add_argument('--discriminator.lr', '--disc_learning_rate', type=float,
                                 help='learning rate for the model, default={:s}'.format(str(default_dis_lr)),
                                 default=default_cls_lr)
        self.parser.add_argument('--discriminator.lrS', '--dis_lr_scheduler_step', type=int,
                                 help='StepLR learning rate scheduler step, default={:f}'.format(default_dis_lrS),
                                 default=default_cls_lrS)
        self.parser.add_argument('--discriminator.lrG', '--dis_lr_scheduler_gamma', type=float,
                                 help='StepLR learning rate scheduler gamma, default={:f}'.format(
                                     default_cls_lrG), default=default_dis_lrG)

        # -------------- discriminator options end    ----------------------

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()
        args = vars(opt)

        tag_lst = ['global', 'data', 'classifier', 'discriminator']
        opt_dict = filter_multi_opt(args, tag_lst=tag_lst)

        self.global_opt = opt_dict['global']
        self.data_opt = opt_dict['data']
        self.classifier_opt = opt_dict['classifier']
        self.discriminator_opt = opt_dict['discriminator']

        # -- deal the details of global settings begin -- #
        # deal single gpu
        if isinstance(self.global_opt['gpu_ids'], str):
            self.global_opt['gpu_ids'] = [int(self.global_opt['gpu_ids'])]

        # create checkpoint and log folder
        self.global_opt['log_dir'] = os.path.join(self.global_opt['checkpoint_dir'], self.global_opt['log_dir'])
        mkdirs(self.global_opt['log_dir'])
        # -- deal the details of global settings end -- #

        # print log #
        if self.global_opt['is_training']:
            file_name = os.path.join(self.global_opt['checkpoint_dir'], 'train_opt.txt')
        else:
            file_name = os.path.join(self.global_opt['checkpoint_dir'], 'test_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        # -- deal the details of classifier settings begin -- #
        # add cuda to classifier
        self.classifier_opt['cuda'] = self.global_opt['cuda']
        self.classifier_opt['way'] = self.data_opt['way']
        self.classifier_opt['shot'] = self.data_opt['shot']
        self.classifier_opt['unlabeled'] = self.data_opt['unlabeled']
        self.classifier_opt['query'] = self.data_opt['query']
        self.classifier_opt['test_way'] = self.data_opt['test_way']
        # -- deal the details of classifier settings end -- #

        # -- deal the details of discriminator settings begin -- #
        if 'nn.LeakyReLU' == self.discriminator_opt['activation']:
            self.discriminator_opt['activation'] = nn.LeakyReLU
        elif 'nn.ReLU' == self.discriminator_opt['activation']:
            self.discriminator_opt['activation'] = nn.ReLU
        self.discriminator_opt['way'] = self.data_opt['way']
        self.discriminator_opt['shot'] = self.data_opt['shot']
        # -- deal the details of discriminator settings begin -- #


