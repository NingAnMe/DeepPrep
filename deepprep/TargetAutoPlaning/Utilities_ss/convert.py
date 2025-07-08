# -*- coding: utf-8 -*-
'''
@File   : convert.py
Created on 2024-07-08 20:33:10
@Author : ssli
Description: this script is used to convert between different file formats.
'''
'''
Modified on 2024-07-08 20:33:26
Modified by ssli
'''

import os
import numpy as np
import nibabel as nib
from pathlib import Path

def set_hemi_info(file_path, hemi):
    '''
    Add the hemisphere information to the file so that the workbench can recognize it.
    '''
    if hemi == 'lh':
        hemi_info = 'CORTEX_LEFT'
    elif hemi == 'rh':
        hemi_info = 'CORTEX_RIGHT'
    else:
        raise Exception(f'hemi error!')
    cmd = f'wb_command -set-structure "{file_path}" {hemi_info}'
    os.system(cmd)
    return file_path

def label_2_annot(nvtx, label, color, label_name, annot_path):
    """
    Convert a label to an annotation file.

    Parameters:
    - nvtx (int): The total number of vertices.
    - label (int): The label to convert.
    - color (tuple): The RGB color of the label.
    - label_name (str): The name of the label.
    - annot_path (str): The path to save the annotation file.

    Returns:
    None
    """
    label_mask = np.zeros(nvtx).astype(int)
    label_mask[label] = 1
    rand_int = int(np.random.rand()*1000000)
    ctb = np.array([[255, 255, 255, 0, 0],
           [int(color[0]), 
            int(color[1]), 
            int(color[2]),
            0, 
            rand_int]], dtype=np.int32)
    nib.freesurfer.write_annot(annot_path, label_mask, ctb, ['other', label_name])
    return annot_path

def func_gii_2_label(func_gii_path, label_gii_path, label_txt_path, verbose=False, discard_others=True):
    """
    Convert a func.gii file to a label.gii file using wb_command.
    More information about command used here can be found at https://humanconnectome.org/software/workbench-command/-metric-label-import

    Parameters:
    func_gii_path (str): The path to the input func.gii file.
    label_gii_path (str): The path to save the output label.gii file.
    label_txt_path (str): The path to the label text file.
    verbose (bool, optional): If True, display the command output. Defaults to False.
    discard_others (bool, optional): If True, discard all other labels. Defaults to True.

    Returns:
    str: Path to the output label.gii file.
    """
    cmd = f'wb_command -metric-label-import {func_gii_path} {label_txt_path} {label_gii_path}'
    if discard_others:
        cmd += ' -discard-others'
    if not verbose:
        cmd += ' > /dev/null'
    os.system(cmd)
    return label_gii_path

def annot_2_label(surf_path, annot_path, label_gii_path, verbose=False):
    """
    Convert an annotation file to a label file using FreeSurfer's mris_convert command.

    Parameters:
    surf_path (str): The path to the surface file.
    annot_path (str): The path to the annotation file.
    label_gii_path (str): The path to save the resulting label file.
    verbose (bool, optional): If True, print the command output. Defaults to False.

    Returns:
    None
    """
    cmd = f'mris_convert --annot {annot_path} {surf_path} {label_gii_path}'
    if not verbose:
        cmd += ' > /dev/null'
    os.system(cmd)
    return label_gii_path

def label_gii_2_border(surf_gii, label_gii_path, boder_path, verbose=False):
    """
    Convert a label.gii file to a border file using wb_command.

    Parameters:
    surf_gii (str): Path to the surface file in GIFTI format.
    label_gii_path (str): Path to the input label.gii file.
    boder_path (str): Path to save the output border file.
    verbose (bool, optional): If True, display the command output. Defaults to False.

    Returns:
    str: Path to the output border file.

    """
    cmd = f'wb_command -label-to-border {surf_gii} {label_gii_path} {boder_path}'
    if not verbose:
        cmd += ' > /dev/null'
    os.system(cmd)
    return str(label_gii_path).replace("label.gii", "border")

def mgh_2_func_gii(mgh_file, output_func_gii_file, verbose=False, hemi=None):
    """
    Convert a mgh file to a func.gii file using mri_convert.

    Parameters:
    mgh_file (str)
    output_func_gii_file (str)
    """
    cmd = f'mri_convert "{mgh_file}" "{output_func_gii_file}"'
    if not verbose:
        cmd += ' > /dev/null'
    os.system(cmd)

    if hemi in ['lh', 'rh']:
        output_func_gii_file = set_hemi_info(output_func_gii_file, hemi)
    elif hemi is not None:
        raise Exception(f'hemi not recognized! hemi should be "lh" or "rh"')

    return output_func_gii_file

def mri_surf2surf(srcsubject, trgsubject, hemi, srcsurf, trgsurf, verbose=False):
    '''
    srcsubject: str, the subject of src
    trgsubject: str, the subject of trg
    hemi: str, the hemisphere
    srcsurf: str, the path of src surf
    trgsurf: str, the path of trg surf

    description: convert the src surf to trg surf
    '''
    cmd = f'mri_surf2surf --srcsubject {srcsubject} --trgsubject {trgsubject} --hemi {hemi} --sval {srcsurf} --tval {trgsurf}'
    if not verbose:
        cmd += ' > /dev/null'
    os.system(cmd)
    return trgsurf

def mri_vol2surf(src_vol, src_subject, hemi, trg_subject, trg_surf, projfrac=0.5, verbose=False):
    '''
    src_vol: str, the path of src volume, ie. the input volume file
    src_subject: str, the subject of src
    hemi: str, the hemisphere
    trg_subject: str, the subject of trg
    trg_surf: str, the path of trg surf
    projfrac: float, the projection fraction;  (0->1)fractional projection along normal 
    '''
    cmd = f'mri_vol2surf --mov {src_vol} --regheader {src_subject} --hemi {hemi} --trgsubject {trg_subject} --o {trg_surf} --projfrac {projfrac}'
    if not verbose:
        cmd += ' > /dev/null'
    os.system(cmd)
    return trg_surf

def mri_annot2annot(srcsubject, trgsubject, hemi, srcannot, trgannot, verbose=False):
    '''
    srcsubject: str, the subject of src
    trgsubject: str, the subject of trg
    hemi: str, the hemisphere
    srcannot: str, the path of src annot
    trgannot: str, the path of trg annot

    description: convert the src surf to trg surf
    '''
    cmd = f'mri_surf2surf --srcsubject {srcsubject} --trgsubject {trgsubject} --hemi {hemi} --sval-annot {srcannot} --tval {trgannot}'
    if not verbose:
        cmd += ' > /dev/null'
    os.system(cmd)
    # print(f'{cmd}')
    return trgannot

def metrics_fsavg_2_fsLR(input_func_gii, hemi, output_file, src_reso='fsaverage6', trg_reso='32k', method='largest'):
    """Trans a fsaverage overlay 2 fsLR resolution.

    Parameters
    ----------
    input_func_gii : 
        func.gii file of overlat
    hemi : str
        lh, rh
    src_reso : str, optional
        fsaverage4, fsaverage5, fsaverage6, fsaverage , by default 'fsaverage6'
    trg_reso : str, optional
        32K, 59K, 164K, by default '32K'
    merhods : str, optional
        use only the value of the vertex with the largest weight, by default 'largest', see 'wb_command -metric-resample'
    """
    nvtx = {'fsaverage': '164k', 'fsaverage6': '41k', 'fsaverage5': '10k', 'fsaverage4': '3k'}
    if hemi == 'lh':
        hemi_LR = 'L'
    elif hemi == 'rh':
        hemi_LR = 'R'
    else:
        raise Exception(f'hemi error!')
    template_path = Path(__file__).parents[0] / 'resource/standard_mesh_atlases'

    fs_sphere = f'{template_path}/resample_fsaverage/{src_reso}_std_sphere.{hemi_LR}.{nvtx[src_reso]}_fsavg_{hemi_LR}.surf.gii'
    fs_area = f'{template_path}/resample_fsaverage/{src_reso}.{hemi_LR}.midthickness_va_avg.{nvtx[src_reso]}_fsavg_{hemi_LR}.shape.gii'
    fsLR_sphere = f'{template_path}/resample_fsaverage/fs_LR-deformed_to-fsaverage.{hemi_LR}.sphere.{trg_reso}_fs_LR.surf.gii'
    fsLR_area = f'{template_path}/resample_fsaverage/fs_LR.{hemi_LR}.midthickness_va_avg.{trg_reso}_fs_LR.shape.gii'

    cmd = f'wb_command -metric-resample {input_func_gii} {fs_sphere} {fsLR_sphere} ADAP_BARY_AREA {output_file} -area-metrics {fs_area} {fsLR_area}'
    if method == 'largest':
        cmd += ' -largest'
    os.system(cmd)
    
    return output_file