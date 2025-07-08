import os
from pathlib import Path
import zipfile
import warnings
import nibabel as nib
import pyvista as pv
import numpy as np
from scipy.spatial import KDTree
from scipy.special import softmax
from scipy.special import expit
from .planner import Planner, surf2mesh, mri_surf2surf, mri_annot2annot, plot_target_on_sagittal_slice, concatenate_images, add_target_roi_mask_T1

import shutil

class MotorTargetPlanner(Planner):
    def __init__(self, verbose=False, workdir=None, logdir=None):
        super().__init__(verbose, workdir, logdir)

        self.lh_pial_mesh = None
        self.lh_anat_labels = None
        self.rh_pial_mesh = None
        self.rh_anat_labels = None

        self.lh_fs6_ventro_candi_mask = None
        self.lh_dorsal_candi_mask = None
        self.lh_fs6_dorsal_candi_mask = None
        self.rh_fs6_ventro_candi_mask = None
        self.rh_dorsal_candi_mask = None
        self.rh_fs6_dorsal_candi_mask = None

        self.lh_surf_bold = None
        self.rh_surf_bold = None

        self.lh_fs6_heatmap = None
        self.lh_heatmap = None
        self.rh_fs6_heatmap = None
        self.rh_heatmap = None

        self.lh_sulc = None
        self.rh_sulc = None

        # candidate target constraints
        # # ventro
        # # ?h_parc213_[native/fs6]_surf.annot parcellation index
        # self.func_ventro_indexes = (45,)
        # # ?h.aparc.annot parcellation index
        # self.anat_ventro_indexes = (18, 20)

        # dorsal
        # ?h_parc18_[native/fs6]_surf.annot parcellation index
        self.func_dorsal_indexes = (18,)
        # ?h.aparc.annot parcellation index
        self.anat_dorsal_indexes = (24,)

        self.dorsal_heatmap_percentile = 90

        self.target_info_dict = None
        self.figures_dict = None

    def __reset(self):
        self.lh_pial_mesh = None
        self.lh_anat_labels = None
        self.rh_anat_labels = None
        self.rh_pial_mesh = None

        self.lh_fs6_ventro_candi_mask = None
        self.rh_fs6_ventro_candi_mask = None
        self.lh_dorsal_candi_mask = None
        self.lh_fs6_dorsal_candi_mask = None
        self.rh_dorsal_candi_mask = None
        self.rh_fs6_dorsal_candi_mask = None
        self.lh_ba6_lable = None
        self.rh_ba6_lable = None

        self.lh_surf_bold = None
        self.rh_surf_bold = None

        self.lh_heatmap = None
        self.lh_fs6_heatmap = None
        self.rh_heatmap = None
        self.rh_fs6_heatmap = None

        self.lh_sulc = None
        self.rh_sulc = None

        self.lh_dorsal_targets = None
        self.rh_dorsal_targets = None

    def __compute_candi_mask(self, app_data_path, subj):
        # warning: 当前靶点候选区域mask计算代码，没有靶点候选区域为空的异常，工程应用中应处理、报告
        # lh pial mesh
        lh_pial_file = os.path.join(os.environ.get("SUBJECTS_DIR"), subj, 'surf', 'lh.pial')
        lh_pial_mesh = surf2mesh(lh_pial_file)
        self.lh_pial_mesh = lh_pial_mesh.copy()
        # rh pial mesh
        rh_pial_file = os.path.join(os.environ.get("SUBJECTS_DIR"), subj, 'surf', 'rh.pial')
        rh_pial_mesh = surf2mesh(rh_pial_file)
        self.rh_pial_mesh = rh_pial_mesh.copy()

        # lh anatomy labels
        labels, ctab, names = nib.freesurfer.read_annot(
            os.path.join(os.environ.get("SUBJECTS_DIR"), subj, 'label', 'lh.aparc.annot'))
        self.lh_anat_labels = labels.copy()
        if self.verbose:
            lh_pial_mesh.point_data['label'] = labels.copy()
            lh_pial_mesh.save(os.path.join(self.logdir, 'lh_anat_aparc.vtk'))
        # rh anatomy labels
        labels, ctab, names = nib.freesurfer.read_annot(
            os.path.join(os.environ.get("SUBJECTS_DIR"), subj, 'label', 'rh.aparc.annot'))
        self.rh_anat_labels = labels.copy()
        if self.verbose:
            rh_pial_mesh.point_data['label'] = labels.copy()
            rh_pial_mesh.save(os.path.join(self.logdir, 'rh_anat_aparc.vtk'))

        # lh medial constraint
        lh_fs6_medial_mask_file = self.resource_dir / 'lh_fs6_medial_mask_surf.mgh'
        lh_fs6_medial_mask = nib.load(lh_fs6_medial_mask_file).get_fdata().flatten()
        # rh medial constraint
        rh_fs6_medial_mask_file = self.resource_dir / 'rh_fs6_medial_mask_surf.mgh'
        rh_fs6_medial_mask = nib.load(rh_fs6_medial_mask_file).get_fdata().flatten()

        # lh 18 function parcellation labels
        lh_parc18_fs6_surf_file = os.path.join(self.workdir, 'lh_parc18_fs6_surf.annot')
        lh_parc18_native_surf_file = os.path.join(self.workdir, 'lh_parc18_native_surf.annot')
        mri_annot2annot('fsaverage6', lh_parc18_fs6_surf_file, subj, lh_parc18_native_surf_file, 'lh')
        labels, ctab, names = nib.freesurfer.read_annot(
            os.path.join(self.workdir, 'lh_parc18_native_surf.annot'))
        self.lh_func_labels = labels.copy()

        # rh 18 function parcellation labels
        rh_parc18_fs6_surf_file = os.path.join(self.workdir, 'rh_parc18_fs6_surf.annot')
        rh_parc18_native_surf_file = os.path.join(self.workdir, 'rh_parc18_native_surf.annot')
        mri_annot2annot('fsaverage6', rh_parc18_fs6_surf_file, subj, rh_parc18_native_surf_file, 'rh')
        labels, ctab, names = nib.freesurfer.read_annot(
            os.path.join(self.workdir, 'rh_parc18_native_surf.annot'))
        self.rh_func_labels = labels.copy()

        # compute_sulc_depth
        # lh
        self.lh_sulc = self.compute_sulc_depth(self.lh_pial_mesh)
        lh_alpha_shape_sulc_file = self.workdir / 'lh_alpha_shape_sulc.mgh'
        self.save_mgh(self.lh_sulc, lh_alpha_shape_sulc_file)
        lh_fs6_alpha_shape_sulc_file = self.workdir / 'lh_fs6_alpha_shape_sulc.mgh'
        mri_surf2surf(subj, lh_alpha_shape_sulc_file, 'fsaverage6', lh_fs6_alpha_shape_sulc_file, 'lh')
        self.lh_fs6_sulc = nib.load(lh_fs6_alpha_shape_sulc_file).get_fdata().flatten()
        # rh
        self.rh_sulc = self.compute_sulc_depth(self.rh_pial_mesh)
        rh_alpha_shape_sulc_file = self.workdir / 'rh_alpha_shape_sulc.mgh'
        self.save_mgh(self.rh_sulc, rh_alpha_shape_sulc_file)
        rh_fs6_alpha_shape_sulc_file = self.workdir / 'rh_fs6_alpha_shape_sulc.mgh'
        mri_surf2surf(subj, rh_alpha_shape_sulc_file, 'fsaverage6', rh_fs6_alpha_shape_sulc_file, 'rh')
        self.rh_fs6_sulc = nib.load(rh_fs6_alpha_shape_sulc_file).get_fdata().flatten()

        # # ventro
        # # lh ventro anatomy candidate region mask
        # lh_ventro_anat_candi_mask = self.labels2mask(self.lh_anat_labels, self.anat_ventro_indexes)
        # if self.verbose:
        #     lh_pial_mesh.point_data['label'] = lh_ventro_anat_candi_mask.copy()
        #     lh_pial_mesh.save(os.path.join(self.logdir, 'lh_ventro_anat_candi_mask.vtk'))

        # # rh ventro anatomy candidate region mask
        # rh_ventro_anat_candi_mask = self.labels2mask(self.rh_anat_labels, self.anat_ventro_indexes)
        # if self.verbose:
        #     rh_pial_mesh.point_data['label'] = rh_ventro_anat_candi_mask.copy()
        #     rh_pial_mesh.save(os.path.join(self.logdir, 'rh_ventro_anat_candi_mask.vtk'))

        # # lh ventro function candidate region mask
        # lh_ventro_func_candi_mask = self.labels2mask(self.lh_func_labels, self.func_ventro_indexes)
        # if self.verbose:
        #     lh_pial_mesh.point_data['label'] = lh_ventro_func_candi_mask.copy()
        #     lh_pial_mesh.save(os.path.join(self.logdir, 'lh_ventro_func_candi_mask.vtk'))

        # # rh ventro function candidate region mask
        # rh_ventro_func_candi_mask = self.labels2mask(self.rh_func_labels, self.func_ventro_indexes)
        # if self.verbose:
        #     rh_pial_mesh.point_data['label'] = rh_ventro_func_candi_mask.copy()
        #     rh_pial_mesh.save(os.path.join(self.logdir, 'rh_ventro_func_candi_mask.vtk'))

        # # lh ventro candidate region mask
        # lh_ventro_candi_mask = (lh_ventro_anat_candi_mask == 1) & (lh_ventro_func_candi_mask == 1)
        # lh_ventro_candi_mask_file = self.workdir / 'lh_ventro_candi_mask.mgh'
        # self.save_mgh(lh_ventro_candi_mask, lh_ventro_candi_mask_file)
        # lh_fs6_ventro_candi_mask_file = self.workdir / 'lh_fs6_ventro_candi_mask.mgh'
        # mri_surf2surf(subj, lh_ventro_candi_mask_file, 'fsaverage6', lh_fs6_ventro_candi_mask_file, 'lh')
        # lh_fs6_ventro_candi_mask = nib.load(lh_fs6_ventro_candi_mask_file).get_fdata().flatten()
        # if not np.any(lh_fs6_ventro_candi_mask):
        #     warnings.warn(
        #         'warning: The lh individual IFG region was empty, using atlas IFG region')
        #     lh_fs6_ifg_mask_file = self.resource_dir / 'lh_fs6_ifg_mask_surf.mgh'
        #     lh_fs6_ventro_candi_mask = nib.load(lh_fs6_ifg_mask_file).get_fdata().flatten()
        #     self.save_mgh(lh_fs6_ventro_candi_mask, lh_fs6_ventro_candi_mask_file)
        #     mri_surf2surf('fsaverage6', lh_fs6_ventro_candi_mask_file, subj, lh_ventro_candi_mask_file, 'lh')
        #     lh_ventro_candi_mask = nib.load(lh_ventro_candi_mask_file).get_fdata().flatten()
        # self.lh_fs6_ventro_candi_mask = lh_fs6_ventro_candi_mask.astype(np.float32)
        # if self.verbose:
        #     lh_pial_mesh.point_data['label'] = lh_ventro_candi_mask.copy()
        #     lh_pial_mesh.save(os.path.join(self.logdir, 'lh_ventro_candi_mask.vtk'))

        # # rh ventro candidate region mask
        # rh_ventro_candi_mask = (rh_ventro_anat_candi_mask == 1) & (rh_ventro_func_candi_mask == 1)
        # rh_ventro_candi_mask_file = self.workdir / 'rh_ventro_candi_mask.mgh'
        # self.save_mgh(rh_ventro_candi_mask, rh_ventro_candi_mask_file)
        # rh_fs6_ventro_candi_mask_file = self.workdir / 'rh_fs6_ventro_candi_mask.mgh'
        # mri_surf2surf(subj, rh_ventro_candi_mask_file, 'fsaverage6', rh_fs6_ventro_candi_mask_file, 'rh')
        # rh_fs6_ventro_candi_mask = nib.load(rh_fs6_ventro_candi_mask_file).get_fdata().flatten()
        # if not np.any(rh_fs6_ventro_candi_mask):
        #     warnings.warn(
        #         'warning: The rh individual IFG region was empty, using atlas IFG region')
        #     rh_fs6_ifg_mask_file = self.resource_dir / 'rh_fs6_ifg_mask_surf.mgh'
        #     rh_fs6_ventro_candi_mask = nib.load(rh_fs6_ifg_mask_file).get_fdata().flatten()
        #     self.save_mgh(rh_fs6_ventro_candi_mask, rh_fs6_ventro_candi_mask_file)
        #     mri_surf2surf('fsaverage6', rh_fs6_ventro_candi_mask_file, subj, rh_ventro_candi_mask_file, 'rh')
        #     rh_ventro_candi_mask = nib.load(rh_ventro_candi_mask_file).get_fdata().flatten()
        # self.rh_fs6_ventro_candi_mask = rh_fs6_ventro_candi_mask.astype(np.float32)
        # if self.verbose:
        #     rh_pial_mesh.point_data['label'] = rh_ventro_candi_mask.copy()
        #     rh_pial_mesh.save(os.path.join(self.logdir, 'rh_ventro_candi_mask.vtk'))

        # dorsal
        # lh dorsal anatomy candidate region mask
        lh_dorsal_anat_candi_mask = np.zeros(len(self.lh_anat_labels))
        anat_candi_idx = np.zeros(len(self.lh_anat_labels), dtype=bool)
        for anat_idx in self.anat_dorsal_indexes:
            anat_candi_idx = anat_candi_idx | (self.lh_anat_labels == anat_idx)
        lh_dorsal_anat_candi_mask[anat_candi_idx] = 1

        # rh dorsal anatomy candidate region mask
        rh_dorsal_anat_candi_mask = np.zeros(len(self.rh_anat_labels))
        anat_candi_idx = np.zeros(len(self.rh_anat_labels), dtype=bool)
        for anat_idx in self.anat_dorsal_indexes:
            anat_candi_idx = anat_candi_idx | (self.rh_anat_labels == anat_idx)
        rh_dorsal_anat_candi_mask[anat_candi_idx] = 1

        # lh dorsal function candidate region mask
        lh_dorsal_func_candi_mask = np.zeros(len(self.lh_anat_labels))
        func_candi_idx = np.zeros(len(self.lh_anat_labels), dtype=bool)
        for func_idx in self.func_dorsal_indexes:
            func_candi_idx = func_candi_idx | (self.lh_func_labels == func_idx)
        lh_dorsal_func_candi_mask[func_candi_idx] = 1
        if self.verbose:
            lh_pial_mesh.point_data['label'] = lh_dorsal_func_candi_mask.copy()
            lh_pial_mesh.save(os.path.join(self.logdir, 'lh_dorsal_func_candi_mask.vtk'))

        # rh dorsal function candidate region mask
        rh_dorsal_func_candi_mask = np.zeros(len(self.rh_anat_labels))
        func_candi_idx = np.zeros(len(self.rh_anat_labels), dtype=bool)
        for func_idx in self.func_dorsal_indexes:
            func_candi_idx = func_candi_idx | (self.rh_func_labels == func_idx)
        rh_dorsal_func_candi_mask[func_candi_idx] = 1
        if self.verbose:
            rh_pial_mesh.point_data['label'] = rh_dorsal_func_candi_mask.copy()
            rh_pial_mesh.save(os.path.join(self.logdir, 'rh_dorsal_func_candi_mask.vtk'))

        # # fs6 SMA
        # lh_fs6_sma_mask_file = self.resource_dir / 'lh_fs6_sma_mask_surf.mgh'
        # lh_fs6_sma_mask = nib.load(lh_fs6_sma_mask_file).get_fdata().flatten()
        # rh_fs6_sma_mask_file = self.resource_dir / 'rh_fs6_sma_mask_surf.mgh'
        # rh_fs6_sma_mask = nib.load(rh_fs6_sma_mask_file).get_fdata().flatten()

        # lh dorsal candidate region mask
        lh_dorsal_candi_mask = (lh_dorsal_anat_candi_mask == 1) & (lh_dorsal_func_candi_mask == 1) # & (lh_ba6_mask == 1)
        lh_dorsal_candi_mask_file = self.workdir / 'lh_dorsal_candi_mask.mgh'
        self.save_mgh(lh_dorsal_candi_mask, lh_dorsal_candi_mask_file)
        lh_fs6_dorsal_candi_mask_file = self.workdir / 'lh_fs6_dorsal_candi_mask.mgh'
        mri_surf2surf(subj, lh_dorsal_candi_mask_file, 'fsaverage6', lh_fs6_dorsal_candi_mask_file, 'lh')
        lh_fs6_dorsal_candi_mask = nib.load(lh_fs6_dorsal_candi_mask_file).get_fdata().flatten()
        lh_fs6_dorsal_candi_mask = (lh_fs6_dorsal_candi_mask > 0) & (lh_fs6_medial_mask == 0) # & (lh_fs6_sma_mask == 1)
        # if not np.any(lh_fs6_dorsal_candi_mask):
        #     warnings.warn(
        #         'warning: The lh individual SMA region was empty, using atlas SMA region')
        #     lh_fs6_dorsal_candi_mask = lh_fs6_sma_mask
        #     self.save_mgh(lh_fs6_dorsal_candi_mask, lh_fs6_dorsal_candi_mask_file)
        self.lh_fs6_dorsal_candi_mask = lh_fs6_dorsal_candi_mask.astype(np.float32)
        self.save_mgh(self.lh_fs6_dorsal_candi_mask, lh_fs6_dorsal_candi_mask_file)
        mri_surf2surf('fsaverage6', lh_fs6_dorsal_candi_mask_file, subj, lh_dorsal_candi_mask_file, 'lh')
        self.lh_dorsal_candi_mask = nib.load(lh_dorsal_candi_mask_file).get_fdata().flatten()
        if self.verbose:
            lh_fs6_pial_mesh = self.lh_fs6_pial_mesh.copy()
            lh_fs6_pial_mesh.point_data['label'] = self.lh_fs6_dorsal_candi_mask.copy()
            lh_fs6_pial_mesh.save(os.path.join(self.logdir, 'lh_fs6_dorsal_candi_mask.vtk'))
            lh_pial_mesh.point_data['label'] = self.lh_dorsal_candi_mask.copy()
            lh_pial_mesh.save(os.path.join(self.logdir, 'lh_dorsal_candi_mask.vtk'))

        # rh dorsal candidate region mask
        rh_dorsal_candi_mask = (rh_dorsal_anat_candi_mask == 1) & (rh_dorsal_func_candi_mask == 1) # & (rh_ba6_mask == 1)
        rh_dorsal_candi_mask_file = self.workdir / 'rh_dorsal_candi_mask.mgh'
        self.save_mgh(rh_dorsal_candi_mask, rh_dorsal_candi_mask_file)
        rh_fs6_dorsal_candi_mask_file = self.workdir / 'rh_fs6_dorsal_candi_mask.mgh'
        mri_surf2surf(subj, rh_dorsal_candi_mask_file, 'fsaverage6', rh_fs6_dorsal_candi_mask_file, 'rh')
        rh_fs6_dorsal_candi_mask = nib.load(rh_fs6_dorsal_candi_mask_file).get_fdata().flatten()
        rh_fs6_dorsal_candi_mask = (rh_fs6_dorsal_candi_mask > 0) & (rh_fs6_medial_mask == 0) # & (rh_fs6_sma_mask == 1)
        # if not np.any(rh_fs6_dorsal_candi_mask):
        #     warnings.warn(
        #         'warning: The rh individual SMA region was empty, using atlas SMA region')
        #     rh_fs6_dorsal_candi_mask = rh_fs6_sma_mask
        #     self.save_mgh(rh_fs6_dorsal_candi_mask, rh_fs6_dorsal_candi_mask_file)
        self.rh_fs6_dorsal_candi_mask = rh_fs6_dorsal_candi_mask.astype(np.float32)
        self.save_mgh(self.rh_fs6_dorsal_candi_mask, rh_fs6_dorsal_candi_mask_file)
        mri_surf2surf('fsaverage6', rh_fs6_dorsal_candi_mask_file, subj, rh_dorsal_candi_mask_file, 'rh')
        self.rh_dorsal_candi_mask = nib.load(rh_dorsal_candi_mask_file).get_fdata().flatten()
        if self.verbose:
            rh_fs6_pial_mesh = self.rh_fs6_pial_mesh.copy()
            rh_fs6_pial_mesh.point_data['label'] = self.rh_fs6_dorsal_candi_mask.copy()
            rh_fs6_pial_mesh.save(os.path.join(self.logdir, 'rh_fs6_dorsal_candi_mask.vtk'))
            rh_pial_mesh.point_data['label'] = self.rh_dorsal_candi_mask.copy()
            rh_pial_mesh.save(os.path.join(self.logdir, 'rh_dorsal_candi_mask.vtk'))

    # def __compute_ifg_fc(self, app_data_path, subj):
    #     # lh
    #     lh_fs6_ifg_seed_idx = self.lh_fs6_ventro_candi_mask == 1
    #     seed_vector = np.mean(self.lh_surf_bold[lh_fs6_ifg_seed_idx], axis=0)
    #     self.lh_lh_fs6_ifg_fc = self.compute_surf_fc(seed_vector, self.lh_surf_bold)
    #     self.lh_rh_fs6_ifg_fc = self.compute_surf_fc(seed_vector, self.rh_surf_bold)
    #     # rh
    #     rh_fs6_ifg_seed_idx = self.rh_fs6_ventro_candi_mask == 1
    #     seed_vector = np.mean(self.rh_surf_bold[rh_fs6_ifg_seed_idx], axis=0)
    #     self.rh_rh_fs6_ifg_fc = self.compute_surf_fc(seed_vector, self.rh_surf_bold)
    #     self.rh_lh_fs6_ifg_fc = self.compute_surf_fc(seed_vector, self.lh_surf_bold)

    #     self.lh_fs6_ifg_fc = (self.lh_lh_fs6_ifg_fc + self.rh_lh_fs6_ifg_fc) / 2
    #     lh_fs6_ifg_fc_file = self.workdir / 'lh_fs6_ifg_fc.mgh'
    #     self.save_mgh(self.lh_fs6_ifg_fc, lh_fs6_ifg_fc_file)

    #     # project lh fs6 IFG fc to native space
    #     lh_ifg_fc_file = self.workdir / 'lh_ifg_fc.mgh'
    #     mri_surf2surf('fsaverage6', lh_fs6_ifg_fc_file, subj, lh_ifg_fc_file, 'lh')
    #     self.lh_ifg_fc = nib.load(lh_ifg_fc_file).get_fdata().flatten()
    #     if self.verbose:
    #         lh_fs6_pial_mesh = self.lh_fs6_pial_mesh.copy()
    #         lh_fs6_pial_mesh.point_data['label'] = self.lh_fs6_ifg_fc.copy()
    #         lh_fs6_pial_mesh.save(os.path.join(self.logdir, 'lh_fs6_ifg_fc.vtk'))
    #         lh_pial_mesh = self.lh_pial_mesh.copy()
    #         lh_pial_mesh.point_data['label'] = self.lh_ifg_fc.copy()
    #         lh_pial_mesh.save(os.path.join(self.logdir, 'lh_ifg_fc.vtk'))

    #     self.rh_fs6_ifg_fc = (self.rh_rh_fs6_ifg_fc + self.lh_rh_fs6_ifg_fc) / 2
    #     rh_fs6_ifg_fc_file = self.workdir / 'rh_fs6_ifg_fc.mgh'
    #     self.save_mgh(self.rh_fs6_ifg_fc, rh_fs6_ifg_fc_file)

    #     # project rh fs6 IFG fc to native space
    #     rh_ifg_fc_file = self.workdir / 'rh_ifg_fc.mgh'
    #     mri_surf2surf('fsaverage6', rh_fs6_ifg_fc_file, subj, rh_ifg_fc_file, 'rh')
    #     self.rh_ifg_fc = nib.load(rh_ifg_fc_file).get_fdata().flatten()
    #     if self.verbose:
    #         rh_fs6_pial_mesh = self.rh_fs6_pial_mesh.copy()
    #         rh_fs6_pial_mesh.point_data['label'] = self.rh_fs6_ifg_fc.copy()
    #         rh_fs6_pial_mesh.save(os.path.join(self.logdir, 'rh_fs6_ifg_fc.vtk'))
    #         rh_pial_mesh = self.rh_pial_mesh.copy()
    #         rh_pial_mesh.point_data['label'] = self.rh_ifg_fc.copy()
    #         rh_pial_mesh.save(os.path.join(self.logdir, 'rh_ifg_fc.vtk'))

    def __compute_heatmap(self, app_data_path, subj):
        # lh 18 function parcellation labels
        # labels, ctab, names = nib.freesurfer.read_annot(
        #     os.path.join(app_data_path, 'Parcellation-18', 'lh_parc18_native_surf.annot'))
        
        workdir_split = str(self.workdir).split('/')
        parcellation18_dir = '/'.join(workdir_split + ['Parcellation18'])
        ## lh
        lh_fs6_parc18_file = os.path.join(parcellation18_dir, 'iter10_c25_w1', 'Iter_10_sm4', 'lh.parc_result_highconf.annot')
        # lh_native_parc18_file = os.path.join(self.workdir, 'lh.func_result_highconf.annot')
        # mri_annot2annot('fsaverage6', lh_fs6_parc18_file, subj, lh_native_parc18_file, 'lh')
        labels, ctab, names = nib.freesurfer.read_annot(lh_fs6_parc18_file)

        lh_fs6_func18_labels = labels.astype('float32')
        # lh_func18_labels_file = self.workdir / 'lh_func18_labels.mgh'
        # self.save_mgh(lh_func18_labels, lh_func18_labels_file)
        # lh_fs6_func18_labels_file = self.workdir / 'lh_fs6_func18_labels.mgh'
        # mri_surf2surf(subj, lh_func18_labels_file, 'fsaverage6', lh_fs6_func18_labels_file, 'lh')
        # lh_fs6_func18_labels = nib.load(lh_fs6_func18_labels_file).get_fdata().flatten()

        ## lh hand Confidence
        lh_fs6_net18_confidence_file = os.path.join(parcellation18_dir, 'iter10_c25_w1', 'Iter_10_sm4', 'lh.NetworkConfidence_18_fs6.mgh')
        lh_fs6_net18_confidence = nib.load(lh_fs6_net18_confidence_file).get_fdata().flatten()

        # rh 18 function parcellation labels
        # labels, ctab, names = nib.freesurfer.read_annot(
        #     os.path.join(app_data_path, 'Parcellation-18', 'rh_parc18_native_surf.annot'))

        ## rh
        rh_fs6_parc18_file = os.path.join(parcellation18_dir, 'iter10_c25_w1', 'Iter_10_sm4', 'rh.parc_result_highconf.annot')
        # rh_native_parc18_file = os.path.join(self.workdir, 'rh.func_result_highconf.annot')
        # mri_annot2annot('fsaverage6', rh_fs6_parc18_file, subj, rh_native_parc18_file, 'rh')
        labels, ctab, names = nib.freesurfer.read_annot(rh_fs6_parc18_file)

        rh_fs6_func18_labels = labels.astype('float32')
        # rh_func18_labels_file = self.workdir / 'rh_func18_labels.mgh'
        # self.save_mgh(rh_func18_labels, rh_func18_labels_file)
        # rh_fs6_func18_labels_file = self.workdir / 'rh_fs6_func18_labels.mgh'
        # mri_surf2surf(subj, rh_func18_labels_file, 'fsaverage6', rh_fs6_func18_labels_file, 'rh')
        # rh_fs6_func18_labels = nib.load(rh_fs6_func18_labels_file).get_fdata().flatten()

        ## rh hand Confidence
        rh_fs6_net18_confidence_file = os.path.join(parcellation18_dir, 'iter10_c25_w1', 'Iter_10_sm4', 'rh.NetworkConfidence_18_fs6.mgh')
        rh_fs6_net18_confidence = nib.load(rh_fs6_net18_confidence_file).get_fdata().flatten()

        # lh Hand Network
        lh_fs6_hand_idx = lh_fs6_func18_labels == 18
        lh_fs6_hand_seed = np.mean(self.lh_surf_bold[lh_fs6_hand_idx, :], axis=0)
        lh_lh_fs6_hand_fc = self.compute_surf_fc(lh_fs6_hand_seed, self.lh_surf_bold)
        lh_rh_fs6_hand_fc = self.compute_surf_fc(lh_fs6_hand_seed, self.rh_surf_bold)

        # rh Hand Network
        rh_fs6_hand_idx = rh_fs6_func18_labels == 18
        rh_fs6_hand_seed = np.mean(self.rh_surf_bold[rh_fs6_hand_idx, :], axis=0)
        rh_lh_fs6_hand_fc = self.compute_surf_fc(rh_fs6_hand_seed, self.lh_surf_bold)
        rh_rh_fs6_hand_fc = self.compute_surf_fc(rh_fs6_hand_seed, self.rh_surf_bold)

        lh_sulc_weight = expit(3 * (self.lh_fs6_sulc + 1.5))
        lh_fs6_hand_fc = (lh_lh_fs6_hand_fc + rh_lh_fs6_hand_fc) / 2
        # lh_fs6_ifg_fc = (self.lh_lh_fs6_ifg_fc + self.rh_lh_fs6_ifg_fc) / 2
        # lh_fs6_ifg_fc = self.lh_lh_fs6_ifg_fc
        # lh_fs6_fusion_fc = ((lh_fs6_hand_fc + lh_fs6_ifg_fc) / 2) * lh_sulc_weight
        lh_fs6_fusion_fc = lh_fs6_hand_fc * lh_sulc_weight
        lh_fs6_fusion_fc = lh_fs6_net18_confidence * lh_sulc_weight

        lh_fs6_fusion_fc_file = self.workdir / 'lh_fs6_fusion_fc.mgh'
        self.save_mgh(lh_fs6_fusion_fc, lh_fs6_fusion_fc_file)
        lh_fusion_fc_file = self.workdir / 'lh_fusion_fc.mgh'
        mri_surf2surf('fsaverage6', lh_fs6_fusion_fc_file, subj, lh_fusion_fc_file, 'lh')
        lh_fusion_fc = nib.load(lh_fusion_fc_file).get_fdata().flatten()
        self.lh_fs6_heatmap = lh_fs6_fusion_fc
        self.lh_heatmap = lh_fusion_fc

        rh_sulc_weight = expit(3 * (self.rh_fs6_sulc + 1.5))
        rh_fs6_hand_fc = (lh_rh_fs6_hand_fc + rh_rh_fs6_hand_fc) / 2
        # rh_fs6_ifg_fc = (self.lh_rh_fs6_ifg_fc + self.rh_rh_fs6_ifg_fc) / 2
        # rh_fs6_ifg_fc = self.lh_rh_fs6_ifg_fc
        # rh_fs6_fusion_fc = ((rh_fs6_hand_fc + rh_fs6_ifg_fc) / 2) * rh_sulc_weight
        rh_fs6_fusion_fc = rh_fs6_hand_fc * rh_sulc_weight
        rh_fs6_fusion_fc = rh_fs6_net18_confidence * rh_sulc_weight

        rh_fs6_fusion_fc_file = self.workdir / 'rh_fs6_fusion_fc.mgh'
        self.save_mgh(rh_fs6_fusion_fc, rh_fs6_fusion_fc_file)
        rh_fusion_fc_file = self.workdir / 'rh_fusion_fc.mgh'
        mri_surf2surf('fsaverage6', rh_fs6_fusion_fc_file, subj, rh_fusion_fc_file, 'rh')
        rh_fusion_fc = nib.load(rh_fusion_fc_file).get_fdata().flatten()
        self.rh_fs6_heatmap = rh_fs6_fusion_fc
        self.rh_heatmap = rh_fusion_fc

    def __target_search(self, subj):
        # dorsal
        # lh
        if self.verbose:
            lh_fs6_dorsal_candi_heatmap = np.full((self.lh_fs6_n_vertex), -1.0)
            tmp_candi_idx = self.lh_fs6_dorsal_candi_mask == 1
            lh_fs6_dorsal_candi_heatmap[tmp_candi_idx] = self.lh_fs6_heatmap[tmp_candi_idx]
            lh_fs6_pial_mesh = self.lh_fs6_pial_mesh.copy()
            lh_fs6_pial_mesh.point_data['label'] = lh_fs6_dorsal_candi_heatmap.copy()
            lh_fs6_pial_mesh.save(os.path.join(self.logdir, 'lh_fs6_dorsal_candi_heatmap.vtk'))

        lh_dorsal_candi_heatmap = np.full_like(self.lh_dorsal_candi_mask, -1.0)
        tmp_candi_idx = self.lh_dorsal_candi_mask == 1
        lh_dorsal_candi_heatmap[tmp_candi_idx] = self.lh_heatmap[tmp_candi_idx]
        if self.verbose:
            lh_pial_mesh = self.lh_pial_mesh.copy()
            lh_pial_mesh.point_data['label'] = lh_dorsal_candi_heatmap.copy()
            lh_pial_mesh.save(os.path.join(self.logdir, 'lh_dorsal_candi_heatmap.vtk'))

        lh_dorsal_heatmap_mesh = self.lh_pial_mesh.copy()
        lh_dorsal_heatmap_mesh.point_data['label'] = lh_dorsal_candi_heatmap.copy()
        self.lh_dorsal_targets = self.cluster_target_search(lh_dorsal_heatmap_mesh, self.dorsal_heatmap_percentile)
        for i, target_dict in enumerate(self.lh_dorsal_targets):
            lh_dorsal_target_idx = target_dict['index']
            lh_dorsal_heatmap_value = lh_dorsal_candi_heatmap[lh_dorsal_target_idx]
            if lh_dorsal_heatmap_value < 0:
                warnings.warn(
                    f'Expected heatmap value > 0, but lh SMA target{i}(vertex index: lh{lh_dorsal_target_idx}) heatmap value is {lh_dorsal_heatmap_value} < 0.')
            lh_dorsal_target = np.zeros((self.lh_dorsal_candi_mask.shape))
            lh_dorsal_target[lh_dorsal_target_idx] = 1
            lh_dorsal_target_file = self.workdir / f'lh_dorsal_target{i}.mgh'
            self.save_mgh(lh_dorsal_target, lh_dorsal_target_file)

            lh_fs6_dorsal_target_file = self.workdir / f'lh_fs6_dorsal_target{i}.mgh'
            mri_surf2surf(subj, lh_dorsal_target_file, 'fsaverage6', lh_fs6_dorsal_target_file, 'lh')
            lh_fs6_dorsal_target = nib.load(lh_fs6_dorsal_target_file).get_fdata().flatten()
            lh_fs6_dorsal_target_idx = np.argmax(lh_fs6_dorsal_target)
            lh_fs6_dorsal_target_seed = self.lh_surf_bold[lh_fs6_dorsal_target_idx, :]
            lh_fs6_dorsal_target_fc = self.compute_surf_fc(lh_fs6_dorsal_target_seed, self.lh_surf_bold)
            lh_fs6_dorsal_target_fc_file = self.workdir / f'lh_fs6_dorsal_target{i}_fc.mgh'
            self.save_mgh(lh_fs6_dorsal_target_fc, lh_fs6_dorsal_target_fc_file)
            lh_dorsal_target_fc_file = self.workdir / f'lh_dorsal_target{i}_fc.mgh'
            mri_surf2surf('fsaverage6', lh_fs6_dorsal_target_fc_file, subj, lh_dorsal_target_fc_file, 'lh')
            lh_dorsal_target_fc = nib.load(lh_dorsal_target_fc_file).get_fdata().flatten()

            if self.verbose:
                lh_fs6_pial_mesh = self.lh_fs6_pial_mesh.copy()
                lh_fs6_pial_mesh.point_data['label'] = lh_fs6_dorsal_target.copy()
                lh_fs6_pial_mesh.save(os.path.join(self.logdir, f'lh_fs6_dorsal_target{i}.vtk'))
                lh_pial_mesh = self.lh_pial_mesh.copy()
                lh_pial_mesh.point_data['label'] = lh_dorsal_target.copy()
                lh_pial_mesh.save(os.path.join(self.logdir, f'lh_dorsal_target{i}.vtk'))
                lh_pial_mesh.point_data['label'] = lh_dorsal_target_fc.copy()
                lh_pial_mesh.save(os.path.join(self.logdir, f'lh_dorsal_target{i}_fc.vtk'))

        # rh
        if self.verbose:
            rh_fs6_dorsal_candi_heatmap = np.full((self.rh_fs6_n_vertex), -1.0)
            tmp_candi_idx = self.rh_fs6_dorsal_candi_mask == 1
            rh_fs6_dorsal_candi_heatmap[tmp_candi_idx] = self.rh_fs6_heatmap[tmp_candi_idx]
            rh_fs6_pial_mesh = self.rh_fs6_pial_mesh.copy()
            rh_fs6_pial_mesh.point_data['label'] = rh_fs6_dorsal_candi_heatmap.copy()
            rh_fs6_pial_mesh.save(os.path.join(self.logdir, 'rh_fs6_dorsal_candi_heatmap.vtk'))

        rh_dorsal_candi_heatmap = np.full_like(self.rh_dorsal_candi_mask, -1.0)
        tmp_candi_idx = self.rh_dorsal_candi_mask == 1
        rh_dorsal_candi_heatmap[tmp_candi_idx] = self.rh_heatmap[tmp_candi_idx]
        if self.verbose:
            rh_pial_mesh = self.rh_pial_mesh.copy()
            rh_pial_mesh.point_data['label'] = rh_dorsal_candi_heatmap.copy()
            rh_pial_mesh.save(os.path.join(self.logdir, 'rh_dorsal_candi_heatmap.vtk'))

        rh_dorsal_heatmap_mesh = self.rh_pial_mesh.copy()
        rh_dorsal_heatmap_mesh.point_data['label'] = rh_dorsal_candi_heatmap.copy()
        self.rh_dorsal_targets = self.cluster_target_search(rh_dorsal_heatmap_mesh, self.dorsal_heatmap_percentile)
        for i, target_dict in enumerate(self.rh_dorsal_targets):
            rh_dorsal_target_idx = target_dict['index']
            rh_dorsal_heatmap_value = rh_dorsal_candi_heatmap[rh_dorsal_target_idx]
            if rh_dorsal_heatmap_value < 0:
                warnings.warn(
                    f'Expected heatmap value > 0, but rh SMA target{i}(vertex index: rh{rh_dorsal_target_idx}) heatmap value is {rh_dorsal_heatmap_value} < 0.')
            rh_dorsal_target = np.zeros((self.rh_dorsal_candi_mask.shape))
            rh_dorsal_target[rh_dorsal_target_idx] = 1
            rh_dorsal_target_file = self.workdir / f'rh_dorsal_target{i}.mgh'
            self.save_mgh(rh_dorsal_target, rh_dorsal_target_file)

            rh_fs6_dorsal_target_file = self.workdir / f'rh_fs6_dorsal_target{i}.mgh'
            mri_surf2surf(subj, rh_dorsal_target_file, 'fsaverage6', rh_fs6_dorsal_target_file, 'rh')
            rh_fs6_dorsal_target = nib.load(rh_fs6_dorsal_target_file).get_fdata().flatten()
            rh_fs6_dorsal_target_idx = np.argmax(rh_fs6_dorsal_target)
            rh_fs6_dorsal_target_seed = self.rh_surf_bold[rh_fs6_dorsal_target_idx, :]
            rh_fs6_dorsal_target_fc = self.compute_surf_fc(rh_fs6_dorsal_target_seed, self.rh_surf_bold)
            rh_fs6_dorsal_target_fc_file = self.workdir / f'rh_fs6_dorsal_target{i}_fc.mgh'
            self.save_mgh(rh_fs6_dorsal_target_fc, rh_fs6_dorsal_target_fc_file)
            rh_dorsal_target_fc_file = self.workdir / f'rh_dorsal_target{i}_fc.mgh'
            mri_surf2surf('fsaverage6', rh_fs6_dorsal_target_fc_file, subj, rh_dorsal_target_fc_file, 'rh')
            rh_dorsal_target_fc = nib.load(rh_dorsal_target_fc_file).get_fdata().flatten()

            if self.verbose:
                rh_fs6_pial_mesh = self.rh_fs6_pial_mesh.copy()
                rh_fs6_pial_mesh.point_data['label'] = rh_fs6_dorsal_target.copy()
                rh_fs6_pial_mesh.save(os.path.join(self.logdir, f'rh_fs6_dorsal_target{i}.vtk'))
                rh_pial_mesh = self.rh_pial_mesh.copy()
                rh_pial_mesh.point_data['label'] = rh_dorsal_target.copy()
                rh_pial_mesh.save(os.path.join(self.logdir, f'rh_dorsal_target{i}.vtk'))
                rh_pial_mesh.point_data['label'] = rh_dorsal_target_fc.copy()
                rh_pial_mesh.save(os.path.join(self.logdir, f'rh_dorsal_target{i}_fc.vtk'))

    def __get_target_info(self, subject):
        '''
        get the target information from the planner
        '''
        ## indi information
        lh_indi_target = self.lh_dorsal_targets[0]
        rh_indi_target = self.rh_dorsal_targets[0]
        lh_indi_target_index = lh_indi_target['index']
        rh_indi_target_index = rh_indi_target['index']
        lh_indi_target_score = lh_indi_target['score']
        rh_indi_target_score = rh_indi_target['score']
        ## read in the indi's surface file to get the target surfRAS
        lh_native_pial = os.path.join(os.environ['SUBJECTS_DIR'], subject, 'surf/lh.pial')
        coords, _ = nib.freesurfer.read_geometry(lh_native_pial) 
        lh_target_surfRAS = coords[lh_indi_target_index]
        rh_native_pial = os.path.join(os.environ['SUBJECTS_DIR'], subject, 'surf/rh.pial')
        coords, _ = nib.freesurfer.read_geometry(rh_native_pial)
        rh_target_surfRAS = coords[rh_indi_target_index]
        ## read in the indi's volume 2 surfRAS matrix
        indi_T1 = os.path.join(os.environ['SUBJECTS_DIR'], subject, 'mri/T1.mgz')
        indi_T1_img = nib.load(indi_T1)
        indi_vol2surfRAS_matrix = indi_T1_img.header.get_vox2ras_tkr()
        indi_surfRAS2vox_matrix = np.linalg.inv(indi_vol2surfRAS_matrix)
        ## calculate the target volume coordinates
        lh_indi_target_voxel_coord_T1w_space = np.dot(indi_surfRAS2vox_matrix, np.append(lh_target_surfRAS, 1))[:3]
        rh_indi_target_voxel_coord_T1w_space = np.dot(indi_surfRAS2vox_matrix, np.append(rh_target_surfRAS, 1))[:3]
        ## calculate the volRAS
        lh_indi_target_volRAS = np.dot(indi_T1_img.affine, np.append(lh_indi_target_voxel_coord_T1w_space, 1))
        rh_indi_target_volRAS = np.dot(indi_T1_img.affine, np.append(rh_indi_target_voxel_coord_T1w_space, 1))

        ## add the target voxel coordinates in T1w space
        T1_path = os.path.join(os.environ['SUBJECTS_DIR'], subject, 'mri/T1.mgz')
        ### convert to nii.gz
        T1_nii_path = os.path.join(self.workdir, 'T1.nii.gz')
        cmd = f'mri_convert {T1_path} {T1_nii_path} > /dev/null'
        os.system(cmd)

        add_target_roi_mask_T1(T1_nii_path, lh_indi_target_voxel_coord_T1w_space, os.path.join(self.workdir, f'../{subject}_T1target_lh.nii.gz'))
        add_target_roi_mask_T1(T1_nii_path, rh_indi_target_voxel_coord_T1w_space, os.path.join(self.workdir, f'../{subject}_T1target_rh.nii.gz'))

        ## MNI information
        lh_fs6_target_file = os.path.join(self.workdir, 'lh_fs6_dorsal_target0.mgh')
        rh_fs6_target_file = os.path.join(self.workdir, 'rh_fs6_dorsal_target0.mgh')
        lh_fs6_target = nib.load(lh_fs6_target_file).get_fdata().flatten()
        rh_fs6_target = nib.load(rh_fs6_target_file).get_fdata().flatten()
        lh_fs6_target_index = np.argmax(lh_fs6_target)
        rh_fs6_target_index = np.argmax(rh_fs6_target)
        print(f'lh_fs6_target_index: {lh_fs6_target_index}')
        print(f'rh_fs6_target_index: {rh_fs6_target_index}')
        ## read in the fs6's surface file to get the target surfRAS
        lh_fs6_native_pial = os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage6', 'surf/lh.pial')
        coords, _ = nib.freesurfer.read_geometry(lh_fs6_native_pial)
        lh_fs6_target_surfRAS = coords[lh_fs6_target_index]
        rh_fs6_native_pial = os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage6', 'surf/rh.pial')
        coords, _ = nib.freesurfer.read_geometry(rh_fs6_native_pial)
        rh_fs6_target_surfRAS = coords[rh_fs6_target_index]

        ## create a dictionary to store the target information
        target_info_dict = {
            'name': subject,
            'lh_Indi_Target_Indix': f'lh{int(lh_indi_target_index)}',
            'rh_Indi_Target_Indix': f'rh{int(rh_indi_target_index)}',
            'lh_Indi_Target_Score': float(lh_indi_target_score),
            'rh_Indi_Target_Score': float(rh_indi_target_score),
            'lh_Indi_Target_surfRAS': f'{lh_target_surfRAS[0]:.2f},{lh_target_surfRAS[1]:.2f},{lh_target_surfRAS[2]:.2f}',
            'rh_Indi_Target_surfRAS': f'{rh_target_surfRAS[0]:.2f},{rh_target_surfRAS[1]:.2f},{rh_target_surfRAS[2]:.2f}',
            'lh_Indi_Target_volRAS': f'{lh_indi_target_volRAS[0]:.2f},{lh_indi_target_volRAS[1]:.2f},{lh_indi_target_volRAS[2]:.2f}',
            'rh_Indi_Target_volRAS': f'{rh_indi_target_volRAS[0]:.2f},{rh_indi_target_volRAS[1]:.2f},{rh_indi_target_volRAS[2]:.2f}',
            'lh_Indi_Target_Voxel_Coord_T1w_Space': f'{round(lh_indi_target_voxel_coord_T1w_space[0])},{round(lh_indi_target_voxel_coord_T1w_space[1])},{round(lh_indi_target_voxel_coord_T1w_space[2])}',
            'rh_Indi_Target_Voxel_Coord_T1w_Space': f'{round(rh_indi_target_voxel_coord_T1w_space[0])},{round(rh_indi_target_voxel_coord_T1w_space[1])},{round(rh_indi_target_voxel_coord_T1w_space[2])}',
            'lh_Indi_Target_Voxel_Coord_MNI_Space': f'{lh_fs6_target_surfRAS[0]:.2f},{lh_fs6_target_surfRAS[1]:.2f},{lh_fs6_target_surfRAS[2]:.2f}',
            'rh_Indi_Target_Voxel_Coord_MNI_Space': f'{rh_fs6_target_surfRAS[0]:.2f},{rh_fs6_target_surfRAS[1]:.2f},{rh_fs6_target_surfRAS[2]:.2f}'
        }

        self.target_info_dict = target_info_dict

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
        ### surface
        #### read in the lh pial surface
        coords, faces = nib.freesurfer.read_geometry(lh_pial_file_workdir)
        target_coord = coords[self.lh_dorsal_targets[0]['index'], :]
        #### 创建一个保存 lh target 信息的txt文件
        lh_target_info_file = os.path.join(self.workdir, 'lh_target_coord.txt')
        with open(lh_target_info_file, 'w') as f:
            f.write(f'lh_target\n')
            f.write(f'0 255 0 {target_coord[0]:.2f} {target_coord[1]:.2f} {target_coord[2]:.2f}\n')
        #### 创建一个保存 lh target 的 foci 文件
        lh_target_foci_file = os.path.join(self.workdir, 'lh_target.foci')
        cmd = f'wb_command -foci-create {lh_target_foci_file} -class lh_target {lh_target_info_file} lh.pial.surf.gii'
        os.system(cmd)
        #### 截图
        wb_view_scene_file = os.path.join(self.resource_dir, 'wb_view_scene', 'lh_target_Motor.scene')
        ### 复制到工作目录
        shutil.copyfile(wb_view_scene_file, os.path.join(self.workdir, 'lh_target.scene'))
        ### 执行wb_view命令
        cmd = f'wb_command -show-scene {self.workdir}/lh_target.scene 1 {figures_dir}/lh_check_target.png 1200 800 > /dev/null 2>&1'
        os.system(cmd)

        ### sagittal volume
        plot_target_on_sagittal_slice(
            os.path.join(os.environ['SUBJECTS_DIR'], subj, 'mri', 'T1.mgz'),
            self.target_info_dict['lh_Indi_Target_Voxel_Coord_T1w_Space'],
            os.path.join(figures_dir, 'lh_target_sagittal_slice.png')
        )

        ## 合并 lh target 的截图
        concatenate_images(
            os.path.join(figures_dir, 'lh_check_target.png'),
            os.path.join(figures_dir, 'lh_target_sagittal_slice.png'),
            os.path.join(figures_dir, 'target_show_lh.png'),
            direction='vertical'
        )

        ## plot rh target
        ### surface
        #### read in the rh pial surface
        coords, faces = nib.freesurfer.read_geometry(rh_pial_file_workdir)
        target_coord = coords[self.rh_dorsal_targets[0]['index'], :]
        #### 创建一个保存 rh target 信息的txt文件
        rh_target_info_file = os.path.join(self.workdir, 'rh_target_coord.txt')
        with open(rh_target_info_file, 'w') as f:
            f.write(f'rh_target\n')
            f.write(f'0 255 0 {target_coord[0]:.2f} {target_coord[1]:.2f} {target_coord[2]:.2f}\n')
        #### 创建一个保存 rh target 的 foci 文件
        rh_target_foci_file = os.path.join(self.workdir, 'rh_target.foci')
        cmd = f'wb_command -foci-create {rh_target_foci_file} -class rh_target {rh_target_info_file} rh.pial.surf.gii'
        os.system(cmd)
        #### 截图
        wb_view_scene_file = os.path.join(self.resource_dir, 'wb_view_scene', 'rh_target_Motor.scene')
        ### 复制到工作目录
        shutil.copyfile(wb_view_scene_file, os.path.join(self.workdir, 'rh_target.scene'))
        ### 执行wb_view命令
        cmd = f'wb_command -show-scene {self.workdir}/rh_target.scene 1 {figures_dir}/rh_check_target.png 1200 800 > /dev/null 2>&1'
        os.system(cmd)

        ### sagittal volume
        plot_target_on_sagittal_slice(
            os.path.join(os.environ['SUBJECTS_DIR'], subj, 'mri', 'T1.mgz'),
            self.target_info_dict['rh_Indi_Target_Voxel_Coord_T1w_Space'],
            os.path.join(figures_dir, 'rh_target_sagittal_slice.png')
        )

        ## 合并 rh target 的截图
        concatenate_images(
            os.path.join(figures_dir, 'rh_check_target.png'),
            os.path.join(figures_dir, 'rh_target_sagittal_slice.png'),
            os.path.join(figures_dir, 'target_show_rh.png'),
            direction='vertical'
        )
        
        concatenate_images(
            os.path.join(figures_dir, 'target_show_lh.png'),
            os.path.join(figures_dir, 'target_show_rh.png'),
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

    def plan(self, postprocess_data_path, subj):
        self.__reset()
        if isinstance(postprocess_data_path, str):
            postprocess_data_path = Path(postprocess_data_path)
        surf_bolds_path = os.path.join(postprocess_data_path, 'func')
        self.lh_surf_bold = self.load_surf_bolds_DeepPrep(surf_bolds_path, 'lh')
        self.rh_surf_bold = self.load_surf_bolds_DeepPrep(surf_bolds_path, 'rh')
        self.__compute_candi_mask(postprocess_data_path, subj)
        # self.__compute_ifg_fc(postprocess_data_path, subj)
        self.__compute_heatmap(postprocess_data_path, subj)
        self.__target_search(subj)
        self.__get_target_info(subj)
        self.__plot_results(subj)
