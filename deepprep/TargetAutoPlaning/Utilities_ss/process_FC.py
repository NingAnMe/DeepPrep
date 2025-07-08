# %%
# -*- coding: utf-8 -*-
'''
@File   : process_FC.py
Created on 2024-11-24 16:08:15
@Author : ssli
'''
'''
Modified on 2024-11-24 16:08:36
Modified by ssli
'''

# %%
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
from Utilities_ss import read_save_nii as rsn
from Utilities_ss import convert


def compute_fc(seed, bold):
    '''
    description: Calculate the FC map of multiple seeds and bold data. 
    This function can be replaced by the function `cal_corr`, which is more general.

    seed: np.array, shape (n, T), n seeds each with T time points

    bold: np.array, shape (m, T), m regions each with T time points

    return: np.array, shape (n, m), the FC map where each column corresponds to one seed
    '''
    seed = np.array(seed, dtype=np.float64)
    bold = np.array(bold, dtype=np.float64)

    # Normalize seeds
    seed_norm = (seed - np.mean(seed, axis=1, keepdims=True)) / np.std(seed, axis=1, keepdims=True)

    # Normalize BOLD data
    bold_norm = (bold - np.mean(bold, axis=1, keepdims=True)) / np.std(bold, axis=1, keepdims=True)

    # Compute FC map
    fcmap = np.dot(bold_norm, seed_norm.T) / bold.shape[1]  # Shape: (m, n)
    fcmap = fcmap.T  # Shape: (n, m)

    return fcmap

def cal_corr(s_series, t_series=None):
    """
    Calculate Pearson's correlation coefficient matrix of s_series and t_series.
    Note: There is a nan_to_num function returning results.

    Parameters
    ----------
    s_series : np.ndarry.
        s matrix with [N1, T]
    t_series : np.ndarry., optional
        t matrix with [N2, T], by default None

    Returns
    -------
    np.ndarry.
        Pearson's correlation coefficient matrix of s_series and t_series with [N1, N2]
    """
    if len(s_series.shape) == 1:
        s_series = np.expand_dims(s_series, 0)
    if t_series is None:
        t_series = s_series
    if s_series.shape[1] != t_series.shape[1]:
        raise Exception(f's_series has shape {s_series.shape}, but t_series has shape {t_series.shape}')
   
    s_series = s_series.T - np.mean(s_series.T, 0)
    s_series = s_series * 1 / np.sqrt(sum(np.square(s_series), 0))

    t_series = t_series.T - np.mean(t_series.T, 0)
    t_series = t_series * 1 / np.sqrt(sum(np.square(t_series), 0))
    corr_mat = np.dot(s_series.T, t_series)
    return np.nan_to_num(corr_mat)