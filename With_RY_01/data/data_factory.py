#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-12-31
#
# Author: jiean001
#
# the factory of dataloader
#########################################################


DATALOADER_REGISTRY = {}


def register_dataloader(dataloader_name):
    def decorator(f):
        DATALOADER_REGISTRY[dataloader_name] = f
        return f
    return decorator


def get_dataloader(data_opt):
    if data_opt['dataloader_name'] in DATALOADER_REGISTRY:
        return DATALOADER_REGISTRY[data_opt['dataloader_name']](**data_opt)
    else:
        raise ValueError("Unknown classifier {:s}".format(data_opt['dataloader_name']))
