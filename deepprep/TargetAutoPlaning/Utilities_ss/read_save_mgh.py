# -*- coding: utf-8 -*-
'''
@File   : read_save.py
Created on 2024-07-10 10:44:10
@Author : ssli
'''
'''
Modified on 2024-07-10 10:44:25
Modified by ssli
'''

import os

import numpy as np
import pandas as pd
import ants
import nibabel as nib

from glob import glob
import matplotlib.pyplot as plt

def read_mgh(file):
    '''
    file: str, the path to the mgh file

    Return the data in the mgh file
    '''
    img = nib.load(file).get_fdata()
    if len(img.shape) == 4:
        img = img.reshape(-1, img.shape[-1], order='F')
    elif len(img.shape) == 3:
        img = img.reshape(-1, order='F')
    return img

def save_mgh(data, file):
    '''
    data: np.array, the data to be saved
    file: str, the path to save the data

    Save the data as a mgh file
    '''
    if data.dtype == np.float64 or data.dtype == bool or data.dtype == np.int64:
        data = data.astype('float32')
    ## 检查数据中是否有nan
    if np.isnan(data).any():
        print('There are nan in the data')
    img = nib.MGHImage(data, np.eye(4))
    nib.save(img, file)
    return file

def reaed_func_gii(file):
    '''
    file: str, the path to read the data, ends with .func.gii

    Read the data from a func.gii file
    '''
    data_array = nib.load(file).darrays
    if len(data_array) == 1:
        data = data_array[0].data
    else:
        data = [data_array[i].data for i in range(len(data_array))]
        data = np.array(data).transpose()
    return data