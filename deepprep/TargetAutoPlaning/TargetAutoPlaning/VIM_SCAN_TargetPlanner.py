import os
from pathlib import Path
# import zipfile
import warnings
import nibabel as nib
# import pyvista as pv
import numpy as np
from scipy.special import expit
from scipy.spatial import KDTree
from .planner import Planner, surf2mesh, mri_surf2surf, expand_mask, plot_target_on_sagittal_slice, concatenate_images, add_target_roi_mask_T1, mri_annot2annot, plot_target_on_coronal_slice
import ants

import sh, sys
import shutil

def save_nii(data, file, template='./MNI2mm_template.nii.gz'):
    '''
    data: np.array, the data to be saved
    file: str, the path to save the nii file

    Description: save the data to the nii file
    '''
    file = str(file)
    template = str(template)
    ### read the template
    template_img = ants.image_read(template)
    ### create the image
    data = data.astype(np.float32)
    img = ants.from_numpy(data, origin=template_img.origin, spacing=template_img.spacing, direction=template_img.direction)
    ### save the image
    ants.image_write(img, file)

def get_cluster_coordinates_v2(filename):
    """
    选择面积最大的cluster的中心点作为OptimalTarget

    从指定的cluster_stats.txt文件中读取cluster中FC最大点的坐标。

    参数:
    filename (str): cluster_stats.txt文件的路径。

    返回:
    tuple: 包含VoxX, VoxY, 和 VoxZ坐标的元组。
    """
    # 读取文件
    with open(filename, 'r') as file:
        lines = file.readlines()

    # 检查文件是否至少有28行
    if len(lines) < 28:
        warnings.warn(f"Does not have cluster in thalamus")
        return 0, 0, 0, 0

    # 定位到文件的第28行并提取相关信息
    cluster_line = lines[27].strip()

    max_volsize_max = -np.inf
    max_volsize_coords = np.nan, np.nan, np.nan
    FC_value = np.nan

    # 遍历每一行以找到体积最大的cluster的坐标
    if cluster_line:
        for line in lines[27:]:
            parts = line.strip().split()
            volsize_max = float(parts[1].replace(',', ''))

            if volsize_max > max_volsize_max:
                max_volsize_max = volsize_max
                FC_value = float(parts[6].replace(',', ''))
                VoxX = float(parts[3].replace(',', ''))
                VoxY = float(parts[4].replace(',', ''))
                VoxZ = float(parts[5].replace(',', ''))
                max_volsize_coords = VoxX, VoxY, VoxZ

        if FC_value > 0:
            return max_volsize_coords[0],max_volsize_coords[1],max_volsize_coords[2], FC_value
        else:
            print("All FC max values are not positive.")
            return 0, 0, 0, 0

    else:
        print("cluster1 not found in the file.")
        warnings.warn(f"Does not have cluster in thalamus")
        return 0, 0, 0, 0

def compute_centroid(matrix):
    """
    计算3D矩阵的几何中心。

    参数:
    matrix (numpy.ndarray): 一个3D矩阵。

    返回:
    tuple: 包含x, y, z坐标的元组。
    """
    x_indices, y_indices, z_indices = np.where(matrix > 0)

    if len(x_indices) == 0:
        return (np.nan, np.nan, np.nan)

    x_center = np.mean(x_indices)
    y_center = np.mean(y_indices)
    z_center = np.mean(z_indices)

    return (x_center, y_center, z_center)

class VIM_SCAN_TargetPlanner(Planner):
    def __init__(self, verbose=False, workdir=None, logdir=None):
        super().__init__(verbose, workdir, logdir)

        self.lh_pial_mesh = None
        self.rh_pial_mesh = None

        self.lh_candi_mask = None
        self.rh_candi_mask = None
        self.lh_fs6_candi_mask = None
        self.rh_fs6_candi_mask = None

        self.lh_surf_bold = None
        self.rh_surf_bold = None

        self.lh_fs6_heatmap = None
        self.rh_fs6_heatmap = None
        self.lh_heatmap = None
        self.rh_heatmap = None

        self.heatmap = None

        self.lh_sulc = None
        self.rh_sulc = None

        # candidate target constraints
        # ?h_parc213_[native/fs6]_surf.annot parcellation index
        self.func_indexes = (18,)

        self.heatmap_percentile = 95

        self.warning_info = []

        self.VIM_mask_MNI152NLin6Asym_2mm_nn = None

    def __reset(self):
        self.lh_pial_mesh = None
        self.rh_pial_mesh = None

        self.lh_fs6_candi_mask = None
        self.rh_fs6_candi_mask = None
        self.lh_candi_mask = None
        self.rh_candi_mask = None

        self.lh_surf_bold = None
        self.rh_surf_bold = None

        self.lh_fs6_heatmap = None
        self.rh_fs6_heatmap = None
        self.rh_heatmap = None
        self.lh_heatmap = None

        self.lh_sulc = None
        self.rh_sulc = None

        self.lh_dorsal_targets = None
        self.rh_dorsal_targets = None

    # def __get_Indi_213_parcel(self, DeepPrep_postprocess_data_path, subj):
    #     raise NotImplementedError
    #     return

    # def __set_anat_info(self, DeepPrep_postprocess_data_path, subj, sulc_percentile=80):
    #     # lh pial mesh
    #     lh_pial_file = os.path.join(os.environ.get("SUBJECTS_DIR"), subj, 'surf', 'lh.pial')
    #     lh_pial_mesh = surf2mesh(lh_pial_file)
    #     self.lh_pial_mesh = lh_pial_mesh.copy()
    #     # rh pial mesh
    #     rh_pial_file = os.path.join(os.environ.get("SUBJECTS_DIR"), subj, 'surf', 'rh.pial')
    #     rh_pial_mesh = surf2mesh(rh_pial_file)
    #     self.rh_pial_mesh = rh_pial_mesh.copy()

    #     # 使用reconall生成的sulc文件
    #     # lh
    #     lh_sulc_file = os.path.join(os.environ.get('SUBJECTS_DIR'), subj, 'surf', 'lh.sulc')
    #     lh_sulc_data = nib.freesurfer.read_morph_data(lh_sulc_file)
    #     ## reconall生成的sulc文件的沟回是负值，需要转换为正值
    #     lh_sulc_data = -lh_sulc_data
    #     lh_sulc_file_mgh = self.workdir / 'lh_sulc.mgh'
    #     self.save_mgh(lh_sulc_data, lh_sulc_file_mgh)
    #     lh_fs6_sulc_file = self.workdir / 'lh_fs6_sulc.mgh'
    #     mri_surf2surf(subj, lh_sulc_file_mgh, 'fsaverage6', lh_fs6_sulc_file, 'lh')
    #     self.lh_fs6_sulc = nib.load(lh_fs6_sulc_file).get_fdata().flatten()
    #     self.lh_sulc = lh_sulc_data
    #     # rh
    #     rh_sulc_file = os.path.join(os.environ.get('SUBJECTS_DIR'), subj, 'surf', 'rh.sulc')
    #     rh_sulc_data = nib.freesurfer.read_morph_data(rh_sulc_file)
    #     rh_sulc_data = -rh_sulc_data
    #     rh_sulc_file_mgh = self.workdir / 'rh_sulc.mgh'
    #     self.save_mgh(rh_sulc_data, rh_sulc_file_mgh)
    #     rh_fs6_sulc_file = self.workdir / 'rh_fs6_sulc.mgh'
    #     mri_surf2surf(subj, rh_sulc_file_mgh, 'fsaverage6', rh_fs6_sulc_file, 'rh')
    #     self.rh_fs6_sulc = nib.load(rh_fs6_sulc_file).get_fdata().flatten()
    #     self.rh_sulc = rh_sulc_data

    #     # lh fs6 precentral mask
    #     lh_fs6_aparc_file = self.resource_dir / 'lh_fs6_aparc.annot'
    #     lh_fs6_aparc = nib.freesurfer.read_annot(lh_fs6_aparc_file)
    #     lh_fs6_precentral_mask = lh_fs6_aparc[0] == 24

    #     # rh fs6 precentral mask
    #     rh_fs6_aparc_file = self.resource_dir / 'rh_fs6_aparc.annot'
    #     rh_fs6_aparc = nib.freesurfer.read_annot(rh_fs6_aparc_file)
    #     rh_fs6_precentral_mask = rh_fs6_aparc[0] == 24

    #     # compute sulc threshold
    #     lh_fs6_sulc_threshold = np.percentile(self.lh_fs6_sulc[lh_fs6_precentral_mask == 1], sulc_percentile)
    #     rh_fs6_sulc_threshold = np.percentile(self.rh_fs6_sulc[rh_fs6_precentral_mask == 1], sulc_percentile)
    #     self.lh_sulc_threshold = lh_fs6_sulc_threshold
    #     self.rh_sulc_threshold = rh_fs6_sulc_threshold

    def __get_func_parc18(self, DeepPrep_postprocess_data_path, subj):

        # 18 parcellation 18 ROI
        workdir_split = str(self.workdir).split('/')
        parcellation18_dir = '/'.join(workdir_split + ['/Parcellation18'])
        ## lh
        lh_fs6_parc18_file = os.path.join(parcellation18_dir, 'iter10_c25_w1', 'Iter_10_sm4', 'lh.parc_result_highconf.annot')
        lh_fs6_parc18 = nib.freesurfer.read_annot(lh_fs6_parc18_file)
        lh_fs6_parc18_labels = lh_fs6_parc18[0]
        self.lh_fs6_func_labels = lh_fs6_parc18_labels.copy()
        ## rh
        rh_fs6_parc18_file = os.path.join(parcellation18_dir, 'iter10_c25_w1', 'Iter_10_sm4', 'rh.parc_result_highconf.annot')
        rh_fs6_parc18 = nib.freesurfer.read_annot(rh_fs6_parc18_file)
        rh_fs6_parc18_labels = rh_fs6_parc18[0]
        self.rh_fs6_func_labels = rh_fs6_parc18_labels.copy()

        # lh 18 function parcellation labels
        lh_parc18_fs6_surf_file = os.path.join(self.workdir, 'lh_parc18_fs6_surf.annot')
        lh_parc18_native_surf_file = os.path.join(self.workdir, 'lh_parc18_native_surf.annot')
        mri_annot2annot('fsaverage6', lh_parc18_fs6_surf_file, subj, lh_parc18_native_surf_file, 'lh')

        # rh 18 function parcellation labels
        rh_parc18_fs6_surf_file = os.path.join(self.workdir, 'rh_parc18_fs6_surf.annot')
        rh_parc18_native_surf_file = os.path.join(self.workdir, 'rh_parc18_native_surf.annot')
        mri_annot2annot('fsaverage6', rh_parc18_fs6_surf_file, subj, rh_parc18_native_surf_file, 'rh')


    def __compute_lang_fc(self, app_data_path, subj):
        ## 计算想要的靶点的fc map
        lh_fs6_lang_idx = self.labels2mask(self.lh_fs6_func_labels, self.func_indexes).astype(bool)
        rh_fs6_lang_idx = self.labels2mask(self.rh_fs6_func_labels, self.func_indexes).astype(bool)
        lh_fs6_lang_data = self.lh_surf_bold[lh_fs6_lang_idx, :]
        rh_fs6_lang_data = self.rh_surf_bold[rh_fs6_lang_idx, :]

        seed_vector = np.vstack((lh_fs6_lang_data, rh_fs6_lang_data)).mean(axis=0)
        # self.lh_fs6_lang_fc = self.compute_surf_fc(seed_vector, self.lh_surf_bold)
        # self.rh_fs6_lang_fc = self.compute_surf_fc(seed_vector, self.rh_surf_bold)

        # lh_fs6_lang_fc_file = self.workdir / 'lh_fs6_lang_fc.mgh'
        # self.save_mgh(self.lh_fs6_lang_fc, lh_fs6_lang_fc_file)
        # rh_fs6_lang_fc_file = self.workdir / 'rh_fs6_lang_fc.mgh'
        # self.save_mgh(self.rh_fs6_lang_fc, rh_fs6_lang_fc_file)

        ## lh
        VIM_mask__MNI152NLin6Asym_2mm_nn = self.resource_dir / 'lh_thalamus_MNINLin6Asym_2mm.nii.gz'
        VIM_mask = ants.image_read(str(VIM_mask__MNI152NLin6Asym_2mm_nn)).numpy()
        self.VIM_mask_MNI152NLin6Asym_2mm_nn = VIM_mask.copy()

        self.lh_fs6_lang_fc = self.compute_vol_fc_with_mask(seed_vector, self.vol_bold, mask=VIM_mask) ## just calculate the fc map in VIM region
        ## set negative values to zero
        self.lh_fs6_lang_fc[self.lh_fs6_lang_fc < 0] = 0
        ## save the fc map
        save_nii(self.lh_fs6_lang_fc, self.workdir / 'lh_VIM_SCAN_fc.nii.gz', template= self.resource_dir / 'lh_thalamus_MNINLin6Asym_2mm.nii.gz')
        self.lh_fs6_lang_fc_file = self.workdir / 'lh_VIM_SCAN_fc.nii.gz'

        ## rh
        VIM_mask__MNI152NLin6Asym_2mm_nn = self.resource_dir / 'rh_thalamus_MNINLin6Asym_2mm.nii.gz'
        VIM_mask = ants.image_read(str(VIM_mask__MNI152NLin6Asym_2mm_nn)).numpy()
        self.VIM_mask_MNI152NLin6Asym_2mm_nn = VIM_mask.copy()

        self.rh_fs6_lang_fc = self.compute_vol_fc_with_mask(seed_vector, self.vol_bold, mask=VIM_mask) ## just calculate the fc map in VIM region
        ## set negative values to zero
        self.rh_fs6_lang_fc[self.rh_fs6_lang_fc < 0] = 0
        ## save the fc map
        save_nii(self.rh_fs6_lang_fc, self.workdir / 'rh_VIM_SCAN_fc.nii.gz', template= self.resource_dir / 'rh_thalamus_MNINLin6Asym_2mm.nii.gz')
        self.rh_fs6_lang_fc_file = self.workdir / 'rh_VIM_SCAN_fc.nii.gz'

        # # project lh fs6 Lang fc to native space
        # lh_lang_fc_file = self.workdir / 'lh_lang_fc.mgh'
        # mri_surf2surf('fsaverage6', lh_fs6_lang_fc_file, subj, lh_lang_fc_file, 'lh')
        # self.lh_lang_fc = nib.load(lh_lang_fc_file).get_fdata().flatten()
        # if self.verbose:
        #     lh_fs6_pial_mesh = self.lh_fs6_pial_mesh.copy()
        #     lh_fs6_pial_mesh.point_data['label'] = self.lh_fs6_lang_fc.copy()
        #     lh_fs6_pial_mesh.save(self.logdir / 'lh_fs6_lang_fc.vtk')
        #     lh_pial_mesh = self.lh_pial_mesh.copy()
        #     lh_pial_mesh.point_data['label'] = self.lh_lang_fc.copy()
        #     lh_pial_mesh.save(self.logdir / 'lh_lang_fc.vtk')

        # # project rh fs6 Lang fc to native space
        # rh_lang_fc_file = self.workdir / 'rh_lang_fc.mgh'
        # mri_surf2surf('fsaverage6', rh_fs6_lang_fc_file, subj, rh_lang_fc_file, 'rh')
        # self.rh_lang_fc = nib.load(rh_lang_fc_file).get_fdata().flatten()
        # if self.verbose:
        #     rh_fs6_pial_mesh = self.rh_fs6_pial_mesh.copy()
        #     rh_fs6_pial_mesh.point_data['label'] = self.rh_fs6_lang_fc.copy()
        #     rh_fs6_pial_mesh.save(self.logdir / 'rh_fs6_lang_fc.vtk')
        #     rh_pial_mesh = self.rh_pial_mesh.copy()
        #     rh_pial_mesh.point_data['label'] = self.rh_lang_fc.copy()
        #     rh_pial_mesh.save(self.logdir / 'rh_lang_fc.vtk')

    def __compute_heatmap(self, app_data_path, subj, sulc_threshold=5):
        # lh_sulc_weight = expit(3 * (self.lh_fs6_sulc + 1.5))
        # ## 修改为适配reconall生成的sulc
        # lh_sulc_weight = expit(3 * (self.lh_fs6_sulc - self.lh_sulc_threshold))
        # lh_fs6_fusion_fc = self.lh_fs6_lang_fc * lh_sulc_weight

        # lh_fs6_fusion_fc_file = self.workdir / 'lh_fs6_fusion_fc.mgh'
        # self.save_mgh(lh_fs6_fusion_fc, lh_fs6_fusion_fc_file)
        # lh_fusion_fc_file = self.workdir / 'lh_fusion_fc.mgh'
        # mri_surf2surf('fsaverage6', lh_fs6_fusion_fc_file, subj, lh_fusion_fc_file, 'lh')
        # lh_fusion_fc = nib.load(lh_fusion_fc_file).get_fdata().flatten()
        # self.lh_fs6_heatmap = lh_fs6_fusion_fc
        # self.lh_heatmap = lh_fusion_fc

        # rh_sulc_weight = expit(3 * (self.rh_fs6_sulc - self.rh_sulc_threshold))   # 为什么使用这样的权重？根据观察回表面对应的值决定的
        # rh_fs6_fusion_fc = self.rh_fs6_lang_fc * rh_sulc_weight

        # rh_fs6_fusion_fc_file = self.workdir / 'rh_fs6_fusion_fc.mgh'
        # self.save_mgh(rh_fs6_fusion_fc, rh_fs6_fusion_fc_file)
        # rh_fusion_fc_file = self.workdir / 'rh_fusion_fc.mgh'
        # mri_surf2surf('fsaverage6', rh_fs6_fusion_fc_file, subj, rh_fusion_fc_file, 'rh')
        # rh_fusion_fc = nib.load(rh_fusion_fc_file).get_fdata().flatten()
        # self.rh_fs6_heatmap = rh_fs6_fusion_fc
        # self.rh_heatmap = rh_fusion_fc

        # set the VIM region heatmap, that is, the FC map in the VIM region
        self.heatmap = self.lh_fs6_lang_fc.copy()        

    def __set_target(self, index, rank, subj, hemi):
        '''
        Set target based on the index and hemisphere and compute the target fc map.
        '''
        target_fc_file = None
        target_data = None
        fs6_target_data = None

        if hemi == 'lh':
            lh_target_idx = index
            lh_heatmap_value = self.lh_heatmap[lh_target_idx]
            if lh_heatmap_value < 0:
                warnings.warn(
                    f'Expected heatmap value > 0, but lh SCAN target{rank}(vertex index: lh{lh_target_idx}) heatmap value is {lh_heatmap_value} < 0.')
            lh_target = np.zeros((self.lh_sulc.shape))
            lh_target[lh_target_idx] = 1
            lh_target_file = self.workdir / f'left_target{rank}_{lh_target_idx}_lh.mgh'
            self.save_mgh(lh_target, lh_target_file)

            lh_fs6_target_file = self.workdir / f'left_target{rank}_{lh_target_idx}_fs6_lh.mgh'
            mri_surf2surf(subj, lh_target_file, 'fsaverage6', lh_fs6_target_file, 'lh')
            lh_fs6_target = nib.load(lh_fs6_target_file).get_fdata().flatten()
            lh_fs6_target_idx = np.argmax(lh_fs6_target)

            ## 寻找临近的点，以ROI内的平均时间序列作为种子
            lh_fs6_target_mask = np.zeros_like(lh_fs6_target)
            lh_fs6_target_mask[lh_fs6_target_idx] = 1
            lh_fs6_target_mask_expanded = expand_mask(self.lh_fs6_pial_mesh, lh_fs6_target_mask, num_rings=2)
            lh_fs6_target_seed = self.lh_surf_bold[lh_fs6_target_mask_expanded, :]
            lh_fs6_target_seed = lh_fs6_target_seed.mean(axis=0)

            lh_fs6_target_fc = self.compute_surf_fc(lh_fs6_target_seed, self.lh_surf_bold)
            lh_fs6_target_fc_file = self.workdir / f'left_target{rank}_{lh_target_idx}_fc_fs6_lh.mgh'
            self.save_mgh(lh_fs6_target_fc, lh_fs6_target_fc_file)
            lh_target_fc_file = self.workdir / f'left_target{rank}_{lh_target_idx}_fc_lh.mgh'
            mri_surf2surf('fsaverage6', lh_fs6_target_fc_file, subj, lh_target_fc_file, 'lh')
            target_fc_file = lh_target_fc_file
            target_data = lh_target
            fs6_target_data = lh_fs6_target

            ## compute the target fc map of right hemisphere
            rh_fs6_target_fc = self.compute_surf_fc(lh_fs6_target_seed, self.rh_surf_bold)
            rh_fs6_target_fc_file = self.workdir / f'left_target{rank}_{lh_target_idx}_fc_fs6_rh.mgh'
            self.save_mgh(rh_fs6_target_fc, rh_fs6_target_fc_file)
            rh_target_fc_file = self.workdir / f'left_target{rank}_{lh_target_idx}_fc_rh.mgh'
            mri_surf2surf('fsaverage6', rh_fs6_target_fc_file, subj, rh_target_fc_file, 'rh')

        elif hemi == 'rh':
            rh_target_idx = index
            rh_heatmap_value = self.rh_heatmap[rh_target_idx]
            if rh_heatmap_value < 0:
                warnings.warn(
                    f'Expected heatmap value > 0, but rh SCAN target{rank}(vertex index: rh{rh_target_idx}) heatmap value is {rh_heatmap_value} < 0.')
            rh_target = np.zeros((self.rh_sulc.shape))
            rh_target[rh_target_idx] = 1
            rh_target_file = self.workdir / f'right_target{rank}_{rh_target_idx}_rh.mgh'
            self.save_mgh(rh_target, rh_target_file)

            rh_fs6_target_file = self.workdir / f'right_target{rank}_{rh_target_idx}_fs6_rh.mgh'
            mri_surf2surf(subj, rh_target_file, 'fsaverage6', rh_fs6_target_file, 'rh')
            rh_fs6_target = nib.load(rh_fs6_target_file).get_fdata().flatten()
            rh_fs6_target_idx = np.argmax(rh_fs6_target)

            ## 寻找临近的点，以ROI内的平均时间序列作为种子
            rh_fs6_target_mask = np.zeros_like(rh_fs6_target)
            rh_fs6_target_mask[rh_fs6_target_idx] = 1
            rh_fs6_target_mask_expanded = expand_mask(self.rh_fs6_pial_mesh, rh_fs6_target_mask, num_rings=2)
            rh_fs6_target_seed = self.rh_surf_bold[rh_fs6_target_mask_expanded, :]
            rh_fs6_target_seed = rh_fs6_target_seed.mean(axis=0)

            rh_fs6_target_fc = self.compute_surf_fc(rh_fs6_target_seed, self.rh_surf_bold)
            rh_fs6_target_fc_file = self.workdir / f'right_target{rank}_{rh_target_idx}_fc_fs6_rh.mgh'
            self.save_mgh(rh_fs6_target_fc, rh_fs6_target_fc_file)
            rh_target_fc_file = self.workdir / f'right_target{rank}_{rh_target_idx}_fc_rh.mgh'
            mri_surf2surf('fsaverage6', rh_fs6_target_fc_file, subj, rh_target_fc_file, 'rh')
            target_fc_file = rh_target_fc_file
            target_data = rh_target
            fs6_target_data = rh_fs6_target

            ## compute the target fc map of left hemisphere
            lh_fs6_target_fc = self.compute_surf_fc(rh_fs6_target_seed, self.lh_surf_bold)
            lh_fs6_target_fc_file = self.workdir / f'right_target{rank}_{rh_target_idx}_fc_fs6_lh.mgh'
            self.save_mgh(lh_fs6_target_fc, lh_fs6_target_fc_file)
            lh_target_fc_file = self.workdir / f'right_target{rank}_{rh_target_idx}_fc_lh.mgh'
            mri_surf2surf('fsaverage6', lh_fs6_target_fc_file, subj, lh_target_fc_file, 'lh')

        else:
            error_msg = f'Invalid hemisphere: {hemi}'
            raise ValueError(error_msg)
        
        return target_fc_file, target_data, fs6_target_data

    def __target_search(self, subj):

        ## lh
        cluster_mask_file = self.workdir / 'lh_VIM_MNI152NLin6Asym_2mm_cluster_mask.nii.gz'
        cluster_label_file = self.workdir / 'lh_VIM_MNI152NLin6Asym_2mm_cluster_label.nii.gz'
        cluster_stats_file = self.workdir / 'lh_VIM_MNI152NLin6Asym_2mm_cluster_stats.txt'
        
        # thmax = np.max(self.lh_fs6_lang_fc) 
        # thmin = thmax * 0.5

        thmax = 0.6 # 可以根据需要调整，不同数据集可能需要不同的阈值
        thmin = 0.3 # 可以根据需要调整，不同数据集可能需要不同的阈值
        # minsize = 50 # 可以根据需要调整，不同数据集可能需要不同的阈值

        thmin = 0.0001 ## 排除掉 0 的情况

        if thmax > 0:
            # sh.mri_volcluster(*shargs, _out=sys.stdout)
            # cmd = f"mri_volcluster --in {self.lh_fs6_lang_fc_file} --sum {cluster_stats_file} --out {cluster_mask_file} --ocn {cluster_label_file} --thmin {thmin} --thmax {thmax} --minsize {minsize} > /dev/null"
            cmd = f"mri_volcluster --in {self.lh_fs6_lang_fc_file} --sum {cluster_stats_file} --out {cluster_mask_file} --ocn {cluster_label_file} --thmin {thmin} > /dev/null"
            os.system(cmd)

            Optimal_target_and_maxFC = get_cluster_coordinates_v2(cluster_stats_file)
            ## 手动计算最大cluster的几何中心
            label_data = ants.image_read(str(cluster_label_file)).numpy()
            label_data[label_data != 1] = 0
            coord = compute_centroid(label_data)
            coord = [int(coord[0] + 0.5), int(coord[1] + 0.5), int(coord[2] + 0.5)]
            Optimal_target_and_maxFC = [coord[0], coord[1], coord[2], Optimal_target_and_maxFC[-1]]
        else:
            Optimal_target_and_maxFC = [0, 0, 0, 0]
        # ## 手动计算每个cluster的几何中心
        # centroid_list = []
        # fc_list = []
        # label_data = ants.image_read(cluster_label_file).numpy()
        # for label in np.unique(label_data):
        #     label_data_tmp = label_data.copy()
        #     label_data_tmp[label_data != label] = 0
        #     if label == 0:
        #         continue
        #     else:
        #         coord = compute_centroid(label_data_tmp)
        #         fc = self.fs6_lang_fc[int(coord[0]+0.5), int(coord[1]+0.5), int(coord[2]+0.5)]
        #         # if fc > 0:
        #         if fc:
        #             print(f"lable:{label}, coorf:{coord}, fc:{fc}")
        #             centroid_list.append(coord)
        #             fc_list.append(fc)

        Optimal_target = [int(Optimal_target_and_maxFC[0]), int(Optimal_target_and_maxFC[1]), int(Optimal_target_and_maxFC[2])]
        FC_max = Optimal_target_and_maxFC[-1]

        temp_data = np.zeros_like(self.lh_fs6_lang_fc)
        temp_data[Optimal_target[0], Optimal_target[1], Optimal_target[2]] = 1
        save_nii(temp_data, self.workdir / 'lh_optimal_target.nii.gz', template=self.resource_dir / 'VIM_MNI152NLin6Asym_2mm_nn.nii.gz')

        # mri_vol2vol('MNI152NLin6Asym', self.workdir / 'lh_optimal_target.nii.gz', subj, self.workdir / 'lh_optimal_target_native.nii.gz')

        # self.lh_targets = [(Optimal_target, FC_max)]
        self.lh_targets = [{'index': Optimal_target, 'score': FC_max}]

        ## rh
        cluster_mask_file = self.workdir / 'rh_VIM_MNI152NLin6Asym_2mm_cluster_mask.nii.gz'
        cluster_label_file = self.workdir / 'rh_VIM_MNI152NLin6Asym_2mm_cluster_label.nii.gz'
        cluster_stats_file = self.workdir / 'rh_VIM_MNI152NLin6Asym_2mm_cluster_stats.txt'

        # thmax = np.max(self.rh_fs6_lang_fc)
        # thmin = thmax * 0.5

        thmax = 0.6 # 可以根据需要调整，不同数据集可能需要不同的阈值
        thmin = 0.3 # 可以根据需要调整，不同数据集可能需要不同的阈值
        # minsize = 50 # 可以根据需要调整，不同数据集可能需要不同的阈值

        thmin = 0.0001 ## 排除掉 0 的情况

        if thmax > 0:
            # cmd = f"mri_volcluster --in {self.rh_fs6_lang_fc_file} --sum {cluster_stats_file} --out {cluster_mask_file} --ocn {cluster_label_file} --thmin {thmin} --thmax {thmax} --minsize {minsize} > /dev/null"
            cmd = f"mri_volcluster --in {self.rh_fs6_lang_fc_file} --sum {cluster_stats_file} --out {cluster_mask_file} --ocn {cluster_label_file} --thmin {thmin} > /dev/null"
            os.system(cmd)

            Optimal_target_and_maxFC = get_cluster_coordinates_v2(cluster_stats_file)
            ## 手动计算最大cluster的几何中心
            label_data = ants.image_read(str(cluster_label_file)).numpy()
            label_data[label_data != 1] = 0
            coord = compute_centroid(label_data)
            coord = [int(coord[0] + 0.5), int(coord[1] + 0.5), int(coord[2] + 0.5)]
            Optimal_target_and_maxFC = [coord[0], coord[1], coord[2], Optimal_target_and_maxFC[-1]]
        else:
            Optimal_target_and_maxFC = [0, 0, 0, 0]
        

        # ## 手动计算每个cluster的几何中心
        # centroid_list = []
        # fc_list = []
        # label_data = ants.image_read(cluster_label_file).numpy()
        # for label in np.unique(label_data):
        #     label_data_tmp = label_data.copy()
        #     label_data_tmp[label_data != label] = 0
        #     if label == 0:
        #         continue
        #     else:
        #         coord = compute_centroid(label_data_tmp)
        #         fc = self.fs6_lang_fc[int(coord[0]+0.5), int(coord[1]+0.5), int(coord[2]+0.5)]
        #         # if fc > 0:
        #         if fc:
        #             print(f"lable:{label}, coorf:{coord}, fc:{fc}")
        #             centroid_list.append(coord)
        #             fc_list.append(fc)

        Optimal_target = [int(Optimal_target_and_maxFC[0]), int(Optimal_target_and_maxFC[1]), int(Optimal_target_and_maxFC[2])]
        FC_max = Optimal_target_and_maxFC[-1]

        temp_data = np.zeros_like(self.lh_fs6_lang_fc)
        temp_data[Optimal_target[0], Optimal_target[1], Optimal_target[2]] = 1
        save_nii(temp_data, self.workdir / 'rh_optimal_target.nii.gz', template=self.resource_dir / 'VIM_MNI152NLin6Asym_2mm_nn.nii.gz')

        # mri_vol2vol('MNI152NLin6Asym', self.workdir / 'rh_optimal_target.nii.gz', subj, self.workdir / 'rh_optimal_target_native.nii.gz')

        # self.rh_targets = [(Optimal_target, FC_max)]
        self.rh_targets = [{'index': Optimal_target, 'score': FC_max}]


    def __get_target_info(self, subject):
        '''
        get the target information from the planner
        '''
        ## indi information
        lh_indi_target = self.lh_targets[0]
        rh_indi_target = self.rh_targets[0]
        lh_indi_target_index = lh_indi_target['index']
        rh_indi_target_index = rh_indi_target['index']
        lh_indi_target_score = lh_indi_target['score']
        rh_indi_target_score = rh_indi_target['score']

        lh_indi_target_voxel_coord_MNI_space = np.array(lh_indi_target_index, dtype=float)
        rh_indi_target_voxel_coord_MNI_space = np.array(rh_indi_target_index, dtype=float)

        ## read in the MNINlin6Asym volume template
        MNI152NLin6Asym_2mm_nn = self.resource_dir / 'lh_thalamus_MNINLin6Asym_2mm.nii.gz'
        MNI152NLin6Asym_2mm_nn_img = nib.load(MNI152NLin6Asym_2mm_nn)
        lh_MNI_target_volRAS = np.dot(MNI152NLin6Asym_2mm_nn_img.affine, np.append(lh_indi_target_voxel_coord_MNI_space, 1))
        rh_MNI_target_volRAS = np.dot(MNI152NLin6Asym_2mm_nn_img.affine, np.append(rh_indi_target_voxel_coord_MNI_space, 1))

        ## get T1 nii.gz file
        T1_path = os.path.join(os.environ['SUBJECTS_DIR'], subject, 'mri/T1.mgz')
        ### convert to nii.gz
        T1_nii_path = os.path.join(self.workdir, 'T1.nii.gz')
        cmd = f'mri_convert {T1_path} {T1_nii_path} > /dev/null'
        os.system(cmd)
        brain_path = os.path.join(os.environ['SUBJECTS_DIR'], subject, 'mri/brain.mgz')
        brain_nii_path = os.path.join(self.workdir, 'brain.nii.gz')
        cmd = f'mri_convert {brain_path} {brain_nii_path} > /dev/null'
        os.system(cmd)
        
        ## transform the MNI voxel coordinates to Indi RAS coordinates
        lh_indi_target_voxel_coord_T1w_space = [0, 0, 0] ## to be filled
        rh_indi_target_voxel_coord_T1w_space = [0, 0, 0] ## to be filled
        ### create a target file in MNI space
        template_file = self.resource_dir / 'tpl-MNI152NLin6Asym_res-02_desc-brain_T1w.nii.gz'
        template = ants.image_read(str(template_file))
        #### lh
        target_mask = np.zeros((91, 109, 91), dtype=np.float32)
        target_mask[int(lh_indi_target_voxel_coord_MNI_space[0]), int(lh_indi_target_voxel_coord_MNI_space[1]), int(lh_indi_target_voxel_coord_MNI_space[2])] = 1
        target_mask_image = ants.from_numpy(target_mask, origin=template.origin, spacing=template.spacing, direction=template.direction)
        ants.image_write(target_mask_image, os.path.join(self.workdir, 'lh_target_MNI.nii.gz'))
        #### rh
        target_mask = np.zeros((91, 109, 91), dtype=np.float32)
        target_mask[int(rh_indi_target_voxel_coord_MNI_space[0]), int(rh_indi_target_voxel_coord_MNI_space[1]), int(rh_indi_target_voxel_coord_MNI_space[2])] = 1
        target_mask_image = ants.from_numpy(target_mask, origin=template.origin, spacing=template.spacing, direction=template.direction)
        ants.image_write(target_mask_image, os.path.join(self.workdir, 'rh_target_MNI.nii.gz'))

        script_dir = '/'.join(str(self.resource_dir).split('/')[:-1])
        ### convert the MNI target mask to T1w space
        cmd = f"python {script_dir}/mri_synthmorph_joint.py \
            -t {os.path.join(self.workdir, 'MNI_to_T1_warp.nii.gz')} \
            -o {os.path.join(self.workdir, 'lh_target_T1.nii.gz')} \
            {template_file} \
            {brain_nii_path} \
            -mp /opt/model/SynthMorph/models \
            -a {os.path.join(self.workdir, 'lh_target_MNI.nii.gz')} \
            -ao {os.path.join(self.workdir, 'lh_target_T1.nii.gz')} \
            -a2 {os.path.join(self.workdir, 'rh_target_MNI.nii.gz')} \
            -a2o {os.path.join(self.workdir, 'rh_target_T1.nii.gz')}"
        os.system(cmd)
            
        # python mri_synthmorph_joint.py -t /mnt/HardDisk3/AutoPoint_test/trans.nii.gz -o /mnt/HardDisk3/AutoPoint_test/MNINlin6Asym_indi.nii.gz /mnt/HardDisk3/AutoPoint_test/tpl-MNI152NLin6Asym_res-02_desc-brain_T1w.nii.gz /mnt/HardDisk3/AutoPoint_test/data_from_Wei/Preprocess_DeepPrep_0630/Recon/sub-Stroke014/mri/brain.mgz -mp /opt/model/SynthMorph/models -a /mnt/HardDisk3/AutoPoint_test/lh_thalamus_MNINLin6Asym_2mm.nii.gz -ao /mnt/HardDisk3/AutoPoint_test/lh_thalamus_indi.nii.gz

        ## read in the T1w space target mask
        lh_target_T1_nii = ants.image_read(os.path.join(self.workdir, 'lh_target_T1.nii.gz')).numpy()
        rh_target_T1_nii = ants.image_read(os.path.join(self.workdir, 'rh_target_T1.nii.gz')).numpy()
        ## get the voxel coordinates in T1w space, the maximum value is the target voxel
        lh_indi_target_voxel_coord_T1w_space = np.array(np.unravel_index(np.argmax(lh_target_T1_nii), lh_target_T1_nii.shape), dtype=float)
        rh_indi_target_voxel_coord_T1w_space = np.array(np.unravel_index(np.argmax(rh_target_T1_nii), rh_target_T1_nii.shape), dtype=float)

        ## copy the target file to the subject's directory
        lh_target_T1_nii_path = os.path.join(self.workdir, 'lh_target_T1.nii.gz')
        rh_target_T1_nii_path = os.path.join(self.workdir, 'rh_target_T1.nii.gz')
        shutil.copyfile(lh_target_T1_nii_path, os.path.join(self.workdir, f'../{subject}_T1target_lh.nii.gz'))
        shutil.copyfile(rh_target_T1_nii_path, os.path.join(self.workdir, f'../{subject}_T1target_rh.nii.gz'))

        ## read in the Indi Volume template
        Indi_Volume = os.path.join(os.environ.get("SUBJECTS_DIR"), subject, 'mri', 'T1.mgz')
        Indi_Volume_img = nib.load(Indi_Volume)
        Indi_Volume_affine = Indi_Volume_img.affine
        lh_indi_target_volRAS = np.dot(np.linalg.inv(Indi_Volume_affine), np.append(lh_indi_target_voxel_coord_MNI_space, 1))
        rh_indi_target_volRAS = np.dot(np.linalg.inv(Indi_Volume_affine), np.append(rh_indi_target_voxel_coord_MNI_space, 1))

        ## add the target voxel coordinates in T1w space

        add_target_roi_mask_T1(T1_nii_path, lh_indi_target_voxel_coord_T1w_space, os.path.join(self.workdir, 'lh_T1target.nii.gz'))
        add_target_roi_mask_T1(T1_nii_path, rh_indi_target_voxel_coord_T1w_space, os.path.join(self.workdir, 'rh_T1target.nii.gz'))

        ## create a dictionary to store the target information
        target_info_dict = {
            'name': subject,
            'lh_Indi_Target_Indix': None,
            'rh_Indi_Target_Indix': None,
            'lh_Indi_Target_Score': float(lh_indi_target_score),
            'rh_Indi_Target_Score': float(rh_indi_target_score),
            'lh_Indi_Target_surfRAS': None,
            'rh_Indi_Target_surfRAS': None,
            'lh_Indi_Target_volRAS': f'{lh_indi_target_volRAS[0]:.2f},{lh_indi_target_volRAS[1]:.2f},{lh_indi_target_volRAS[2]:.2f}',
            'rh_Indi_Target_volRAS': f'{rh_indi_target_volRAS[0]:.2f},{rh_indi_target_volRAS[1]:.2f},{rh_indi_target_volRAS[2]:.2f}',
            'lh_Indi_Target_Voxel_Coord_T1w_Space': f'{round(lh_indi_target_voxel_coord_T1w_space[0])},{round(lh_indi_target_voxel_coord_T1w_space[1])},{round(lh_indi_target_voxel_coord_T1w_space[2])}',
            'rh_Indi_Target_Voxel_Coord_T1w_Space': f'{round(rh_indi_target_voxel_coord_T1w_space[0])},{round(rh_indi_target_voxel_coord_T1w_space[1])},{round(rh_indi_target_voxel_coord_T1w_space[2])}',
            'lh_Indi_Target_Voxel_Coord_MNI_Space': f'{lh_MNI_target_volRAS[0]:.2f},{lh_MNI_target_volRAS[1]:.2f},{lh_MNI_target_volRAS[2]:.2f}',
            'rh_Indi_Target_Voxel_Coord_MNI_Space': f'{rh_MNI_target_volRAS[0]:.2f},{rh_MNI_target_volRAS[1]:.2f},{rh_MNI_target_volRAS[2]:.2f}'
        }

        self.target_info_dict = target_info_dict

        if lh_indi_target_score == 0:
            self.warning_info = f'Warning: cannot find left hemisphere target for {subject}.'
        if rh_indi_target_score == 0:
            self.warning_info = f'Warning: cannot find right hemisphere target for {subject}.'

        return  target_info_dict


    def __plot_results(self, subj):

        ## 创建保存截图的目录
        figures_dir = os.path.join(self.workdir, 'Figures')
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)

        ### 获取SUBJECTS_DIR下的个体表面文件
        lh_pial_file = os.path.join(os.path.join(os.environ.get("SUBJECTS_DIR"), subj, 'surf', 'lh.pial'))
        rh_pial_file = os.path.join(os.path.join(os.environ.get("SUBJECTS_DIR"), subj, 'surf', 'rh.pial'))
        ### 准备截图文件
        lh_pial_file_workdir = os.path.join(self.workdir, 'lh.pial')
        rh_pial_file_workdir = os.path.join(self.workdir, 'rh.pial')
        shutil.copyfile(lh_pial_file, lh_pial_file_workdir)
        shutil.copyfile(rh_pial_file, rh_pial_file_workdir)
        ### 转换为surf.gii文件
        cmd = f'mris_convert {lh_pial_file_workdir} {self.workdir}/lh.pial.surf.gii > /dev/null'
        os.system(cmd)
        cmd = f'mris_convert {rh_pial_file_workdir} {self.workdir}/rh.pial.surf.gii > /dev/null'
        os.system(cmd)

        ## plot parcellation 18
        ### 准备 18 parcellation labels
        lh_fs6_parc18_file = os.path.join(self.workdir, 'lh_parc18_native_surf.annot')
        rh_fs6_parc18_file = os.path.join(self.workdir, 'rh_parc18_native_surf.annot')
        ### 转换为label.gii文件
        cmd = f'mris_convert --annot {lh_fs6_parc18_file} {self.workdir}/lh.pial {self.workdir}/lh_parc18_native_surf.label.gii > /dev/null'
        os.system(cmd)
        cmd = f'mris_convert --annot {rh_fs6_parc18_file} {self.workdir}/rh.pial {self.workdir}/rh_parc18_native_surf.label.gii > /dev/null'
        # print(cmd)
        os.system(cmd)
        ### 截图
        wb_view_scene_file = os.path.join(self.resource_dir, 'wb_view_scene', 'parcellation18.scene')
        ### 复制到工作目录
        shutil.copyfile(wb_view_scene_file, os.path.join(self.workdir, 'parcellation18.scene'))
        ### 执行wb_view命令
        cmd = f'wb_command -show-scene {self.workdir}/parcellation18.scene 1 {figures_dir}/check_parcellation18.png 1200 800 > /dev/null 2>&1'
        os.system(cmd)

        ## 修改工作目录
        os.chdir(self.workdir)

        ## plot lh target
        ### coronal volume
        plot_target_on_coronal_slice(
            os.path.join(os.environ['SUBJECTS_DIR'], subj, 'mri', 'T1.mgz'),
            self.target_info_dict['lh_Indi_Target_Voxel_Coord_T1w_Space'],
            os.path.join(figures_dir, 'lh_target_coronal_slice.png')
        )

        ## plot rh target
        ### coronal volume
        plot_target_on_coronal_slice(
            os.path.join(os.environ['SUBJECTS_DIR'], subj, 'mri', 'T1.mgz'),
            self.target_info_dict['rh_Indi_Target_Voxel_Coord_T1w_Space'],
            os.path.join(figures_dir, 'rh_target_coronal_slice.png')
        )
        
        concatenate_images(
            os.path.join(figures_dir, 'lh_target_coronal_slice.png'),
            os.path.join(figures_dir, 'rh_target_coronal_slice.png'),
            os.path.join(figures_dir, 'target_show_both.png'),
            direction='horizontal'
        )

        workdir_split = str(self.workdir).split('/')
        subj_name = workdir_split[-2]
        sess_name = workdir_split[-1]

        figures_dict = {
            "Parcellation18_Figure": f"{subj_name}/{sess_name}/Figures/check_parcellation18.png",
            "Parcellation213_Figure": None,
            "Target_Figure": f"{subj_name}/{sess_name}/Figures/target_show_both.png",
        }
        self.figures_dict = figures_dict

        return None



    def plan(self, postprocess_data_path, subj, use_group_target=False, sulc_percentile=80):
        self.__reset()
        if isinstance(postprocess_data_path, str):
            postprocess_data_path = Path(postprocess_data_path)
        # if os.path.exists(os.path.join(self.workdir, 'rh_target0_fc.mgh')):
        #     print(f'{subj} has been planned')
        #     return 11
        # self.__set_anat_info(postprocess_data_path, subj, sulc_percentile)
        surf_bolds_path = os.path.join(postprocess_data_path, 'func')
        self.lh_surf_bold = self.load_surf_bolds_DeepPrep(surf_bolds_path, 'lh')
        self.rh_surf_bold = self.load_surf_bolds_DeepPrep(surf_bolds_path, 'rh')
        self.vol_bold = self.load_vol_bolds_DeepPrep(surf_bolds_path)
        if not use_group_target:
            # self.__get_Indi_213_parcel(postprocess_data_path, subj)
            self.__get_func_parc18(postprocess_data_path, subj)
            self.__compute_lang_fc(postprocess_data_path, subj)
            self.__compute_heatmap(postprocess_data_path, subj)
            self.__target_search(subj)
            # self.__check_targets(subj)
        else:
            self.lh_heatmap = np.ones_like(self.lh_sulc)
            self.rh_heatmap = np.ones_like(self.rh_sulc)
        self.__get_target_info(subj)
        self.__plot_results(subj)