
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import nibabel as nib
# from pathlib import Path
# from tqdm import tqdm
# from yfzhu.io.util import _read_surf_data
# from yfzhu.plot.wb_view import wb_view_fc, wb_view_annot
from .pBFS import indi_Gordon17_with_SCAN_parc, write_confidence_to_annot_Gordon17_with_SCAN, indi_APP_18_parc, write_confidence_to_annot_APP_18
# from Figure1.utils import set_environ

import os

import numpy as np
# import pandas as pd
# import ants
import nibabel as nib

from glob import glob
# import matplotlib.pyplot as plt
# import seaborn as sns
# import sys
# sys.path.append('/home/ssli/liss_workspace/Utilities_ss')
from Utilities_ss import read_save_mgh as rsm
# from Utilities_ss import convert
from Utilities_ss import read_save_nii as rsn

def sm_confidence(sm, parc_path, subj, n_iter, n_parc, sess, hemi='lh'):
    
    # set_environ()
    """
    Applies surface-based smoothing to network confidence data.

    Args:
        sm (int): Number of smoothing iterations.
        parc_path (str): Path to the iteration-specific parcellation data.
        subj (str): Subject identifier.
        n_iter (int): Iteration number.
        n_parc (int): Number of parcellations.

    Returns:
        str: Path to the directory where the smoothed data is saved.
    """
    
    sm_dir = os.path.join(parc_path, f'Iter_{n_iter}_sm{sm}')
    if not os.path.exists(sm_dir): os.makedirs(sm_dir)
    if hemi == 'lh':
        for parc in range(n_parc):
            cmd = f'mri_surf2surf --srcsubject fsaverage6 \
            --sval {parc_path}/Iter_{n_iter}/{hemi}.NetworkConfidence_{parc+1}.mgh \
            --trgsubject fsaverage6 --tval {sm_dir}/{hemi}.NetworkConfidence_{parc+1}_fs6.mgh \
            --hemi {hemi} --nsmooth-in {sm} > /dev/null 2>&1'
            os.system(cmd)
    elif hemi == 'rh':
        for parc in range(n_parc):
            cmd = f'mri_surf2surf --srcsubject fsaverage6 \
            --sval {parc_path}/Iter_{n_iter}/{hemi}.NetworkConfidence_{parc+1}.mgh \
            --trgsubject fsaverage6 --tval {sm_dir}/{hemi}.NetworkConfidence_{parc+1}_fs6.mgh \
            --hemi {hemi} --nsmooth-in {sm} > /dev/null 2>&1'
            os.system(cmd)
    else:
        for hemi in ['lh', 'rh']:
            for parc in range(n_parc):
                cmd = f'mri_surf2surf --srcsubject fsaverage6 \
                --sval {parc_path}/Iter_{n_iter}/{hemi}.NetworkConfidence_{parc+1}.mgh \
                --trgsubject fsaverage6 --tval {sm_dir}/{hemi}.NetworkConfidence_{parc+1}_fs6.mgh \
                --hemi {hemi} --nsmooth-in {sm} > /dev/null 2>&1'
                os.system(cmd)
    return sm_dir

def write_confidence_to_annot_Gordon17_with_SCAN(smed_path, hemis = ['lh', 'rh']):
    """
    Writes confidence values to annotation files for Gordon17 with SCAN.

    Args:
        smed_path (str): The path to the directory containing the input files.
        hemi (list): List of hemisphere names.

    Returns:
        None
    """
    ctb=np.array([[0, 0, 0, 255],
                        [255, 255, 0, 0],
                        [120, 18, 134, 0],
                        [230, 148, 34, 0],
                        [255, 0, 0, 0],
                        [74, 155, 60, 0],
                        [0, 118, 14, 0],
                        [205, 62, 78, 0],
                        [255, 152, 213, 0],
                        [196, 58, 250, 0],
                        [220, 180, 140, 0],
                        [42, 204, 164, 0],
                        [12, 48, 255, 0],
                        [200, 248, 164, 0],
                        [122, 135, 50, 0],
                        [119, 140, 176, 0],
                        [0, 0, 130, 0],
                        [70, 130, 180, 0],
                        [128, 0, 76, 0]])

    
    parc = 18
    name = ['Other'] + [f'Net_{i+1}' for i in range(17)] + ['SCAN']
    conf_thr = 0.1
    conf_high_thr = 0.6
    
    for hemi in hemis:
        confidence = np.zeros((parc, 40962))
        for i in range(parc):
            mgh_file = os.path.join(smed_path, f'{hemi}.NetworkConfidence_{i+1}_fs6.mgh')
            parc_smed_conf = nib.load(mgh_file).get_fdata().reshape(-1, order='F')
            confidence[i, :] = parc_smed_conf
        
        conf, membershape = confidence.max(axis=0), confidence.argmax(axis=0)+1
        high_conf_membershape = membershape.copy()
        
        membershape[conf < conf_thr] = -1
        high_conf_membershape = membershape.copy()
        high_conf_membershape[conf < conf_high_thr] = -1
        
        parc_path = os.path.join(smed_path, f'{hemi}.parc_result.annot')
        nib.freesurfer.write_annot(parc_path, membershape, ctb, name)
        nib.freesurfer.read_annot(parc_path)
        
        parc_path = os.path.join(smed_path, f'{hemi}.parc_result_highconf.annot')
        nib.freesurfer.write_annot(parc_path, high_conf_membershape, ctb, name)

def load_data(d_path, sub, run, ext):
    fs6_lh_data_path = d_path / f'{sub}/preprocess/{sub}/surf/lh.{sub}_bld{run}_rest_reorient_skip_faln_mc_g1000000000_bpss_resid{ext}'
    fs6_lh_data = rsm.read_mgh(fs6_lh_data_path)
    fs6_rh_data_path = d_path / f'{sub}/preprocess/{sub}/surf/rh.{sub}_bld{run}_rest_reorient_skip_faln_mc_g1000000000_bpss_resid{ext}'
    fs6_rh_data = rsm.read_mgh(fs6_rh_data_path)
    return fs6_lh_data, fs6_rh_data

def parcellation_18(data_path, out_path, subject, sess, n_iter=10, conf=2.5, weight_group=1):
    res_path = os.path.join(out_path, f'{subject}/{sess}/Parcellation18/iter{n_iter}_c{int(conf*10)}_w{weight_group}')
    if not os.path.exists(res_path): os.makedirs(res_path)

    ## check if have been processed
    if os.path.exists(os.path.join(res_path, 'Iter_10_sm4', 'lh.parc_result.annot')):
        print(f'{subject} {sess} 18 parcellation has been processed.')
        return 1

    if sess == 'temp':
        lh_data_path = glob(f'{data_path}/{subject}/func/*hemi-L_space-fsaverage6_desc-fwhm_bold.nii.gz')
        rh_data_path = glob(f'{data_path}/{subject}/func/*hemi-R_space-fsaverage6_desc-fwhm_bold.nii.gz')
    else:
        ## load data
        lh_data_path = glob(f'{data_path}/{subject}/{sess}/func/*hemi-L_space-fsaverage6_desc-fwhm_bold.nii.gz')
        rh_data_path = glob(f'{data_path}/{subject}/{sess}/func/*hemi-R_space-fsaverage6_desc-fwhm_bold.nii.gz')
    if len(lh_data_path) == 0 or len(rh_data_path) == 0:
        print(f'No fs6 func data for {subject} {sess}. Please download it from the server.')
        return 0
    lh_data = np.hstack([rsn.read_nii(d) for d in sorted(lh_data_path)])
    rh_data = np.hstack([rsn.read_nii(d) for d in sorted(rh_data_path)])
    all_data = np.vstack([lh_data, rh_data])
    print(all_data.shape)
    # do parc
    indi_Gordon17_with_SCAN_parc(all_data, n_iter, conf, res_path, subject, weight_group, sess)
    # smooth confidence
    smed_dir = sm_confidence(4, res_path, subject, n_iter, 18, sess, hemi='all')
    # smed confidence to annot
    write_confidence_to_annot_Gordon17_with_SCAN(smed_dir, hemis=['lh', 'rh'])
    return 1

def parcellation_18_APP(data_path, out_path, subject, sess, n_iter=10, conf=2.5, weight_group=1):
    res_path = os.path.join(out_path, f'{subject}/{sess}/Parcellation18/iter{n_iter}_c{int(conf*10)}_w{weight_group}')
    if not os.path.exists(res_path): os.makedirs(res_path)

    ## check if have been processed
    if os.path.exists(os.path.join(res_path, 'Iter_10_sm4', 'lh.parc_result.annot')):
        print(f'{subject} {sess} 18 parcellation has been processed.')
        return 1

    if sess == 'temp':
        lh_data_path = glob(f'{data_path}/{subject}/func/*hemi-L_space-fsaverage6_desc-fwhm_bold.nii.gz')
        rh_data_path = glob(f'{data_path}/{subject}/func/*hemi-R_space-fsaverage6_desc-fwhm_bold.nii.gz')
    else:
        ## load data
        lh_data_path = glob(f'{data_path}/{subject}/{sess}/func/*hemi-L_space-fsaverage6_desc-fwhm_bold.nii.gz')
        rh_data_path = glob(f'{data_path}/{subject}/{sess}/func/*hemi-R_space-fsaverage6_desc-fwhm_bold.nii.gz')
    if len(lh_data_path) == 0 or len(rh_data_path) == 0:
        print(f'No fs6 func data for {subject} {sess}. Please download it from the server.')
        return 0
    lh_data = np.hstack([rsn.read_nii(d) for d in sorted(lh_data_path)])
    rh_data = np.hstack([rsn.read_nii(d) for d in sorted(rh_data_path)])
    all_data = np.vstack([lh_data, rh_data])
    print(all_data.shape)
    # do parc
    indi_APP_18_parc(all_data, n_iter, conf, res_path, subject, weight_group, sess)
    # smooth confidence
    smed_dir = sm_confidence(4, res_path, subject, n_iter, 18, sess, hemi='all')
    # smed confidence to annot
    write_confidence_to_annot_APP_18(smed_dir)
    return 1