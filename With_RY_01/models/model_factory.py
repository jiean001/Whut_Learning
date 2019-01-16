#!/usr/bin/python
# -*- coding: UTF-8 -*-

#########################################################
# Create on 2018-12-25
#
# Author: jiean001
#
# the factory of dataloader, model, classifer, generator
#             and discriminator
#########################################################


MODEL_REGISTRY = {}
CLASSIFIER_REGISTRY = {}
GENERATOR_REGISTRY = {}
DISCRIMINATOR_REGISTRY = {}
DATALOADER_REGISTRY = {}


def register_dataloader(dataloader_name):
    def decorator(f):
        DATALOADER_REGISTRY[dataloader_name] = f
        return f
    return decorator


def get_dataloader(data_opt):
    if data_opt['loader_name'] in DATALOADER_REGISTRY:
        return DATALOADER_REGISTRY[data_opt['loader_name']](**data_opt)
    else:
        raise ValueError("Unknown classifier {:s}".format(data_opt['loader_name']))


def register_classifier(classifier_name):
    def decorator(f):
        CLASSIFIER_REGISTRY[classifier_name] = f
        return f
    return decorator


def get_classifier(classifier_name, classifier_opt):
    if classifier_name in CLASSIFIER_REGISTRY:
        return CLASSIFIER_REGISTRY[classifier_name](**classifier_opt)
    else:
        raise ValueError("Unknown classifier {:s}".format(classifier_name))


def register_discriminator(discriminator_name):
    def decorator(f):
        DISCRIMINATOR_REGISTRY[discriminator_name] = f
        return f
    return decorator


def get_discriminator(discriminator_name, discriminator_opt):
    if discriminator_name in DISCRIMINATOR_REGISTRY:
        return DISCRIMINATOR_REGISTRY[discriminator_name](**discriminator_opt)
    else:
        raise ValueError("Unknown discriminator {:s}".format(discriminator_name))


def register_generator(generator_name):
    def decorator(f):
        GENERATOR_REGISTRY[generator_name] = f
        return f
    return decorator


def get_generator(generator_opt):
    generator_name = generator_opt['model_name']
    if generator_name in GENERATOR_REGISTRY:
        return GENERATOR_REGISTRY[generator_name](**generator_opt)
    else:
        raise ValueError("Unknown generator {:s}".format(generator_name))


def register_model(model_name):
    def decorator(f):
        MODEL_REGISTRY[model_name] = f
        return f
    return decorator


def get_model(model_name, model_opt):
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name](**model_opt)
    else:
        raise ValueError("Unknown model {:s}".format(model_name))
