# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:26:18 2017

@author: VANLOI
"""
from __future__ import print_function, division
import numpy as np
import math


"calculate batch_size on different size of dataset"
def stopping_para_vae(train_size):
    n_batch  = 0
    b_size = 0
    max_stop = 2000
    if (train_size <= 2000):
        n_batch  = 20
        b_size = int(train_size/n_batch)
    else:
        b_size = 100
        n_batch  = int(train_size/b_size)

    patience = round(max_stop/n_batch)
    every_valid = 1
    return patience, every_valid, b_size, n_batch


def stopping_para_shrink(train_size):
    n_batch  = 0
    b_size = 0
    max_stop = 2000.0
    if (train_size <= 2000):
        n_batch  = 20
        b_size = int(train_size/n_batch)
    else:
        b_size = 100
        n_batch  = int(train_size/b_size)

    every_valid = 5
    "update times after validation = every_valid x n_batch"
    patience =  round(max_stop/(every_valid*n_batch))

    return patience, every_valid, b_size, n_batch






