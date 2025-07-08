# -*- coding: utf-8 -*-
'''
@File   : read_save_nii.py
Created on 2024-07-18 19:32:31
@Author : ssli
Description: read and save nii file
'''
'''
Modified on 2024-07-18 19:32:36
Modified by ssli
'''

import os

import numpy as np
import pandas as pd
import ants
import nibabel as nib

from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import sys
# sys.path.append('/home/ssli/liss_workspace/Utilities_ss')
from Utilities_ss import read_save_mgh as rsm
from Utilities_ss import convert

def read_nii(file):
    '''
    file: str, the path to the nii file, the nii file should be a 3D or 4D file

    Return the data in the nii file, if the nii file is a 4D file, the data will be reshaped to (N1 \* N2\* N3, N4)
    '''
    img = ants.image_read(file)
    data = img.numpy()
    if len(data.shape) == 4:
        data = data.reshape(data.shape[0]*data.shape[1]*data.shape[2], data.shape[3], order='F')
    elif len(data.shape) == 3:
        data = data
    else:
        raise ValueError('The nii file should be a 3D or 4D file')
    return data

def save_nii(data, file, template='/home/ssli/liss_workspace/Downloads/MNI2mm_template.nii.gz'):
    '''
    data: np.array, the data to be saved
    file: str, the path to save the nii file

    Description: save the data to the nii file
    '''
    ### read the template
    template_img = ants.image_read(template)
    ### create the image
    data = data.astype(np.float32)
    img = ants.from_numpy(data, origin=template_img.origin, spacing=template_img.spacing, direction=template_img.direction)
    ### save the image
    ants.image_write(img, file)