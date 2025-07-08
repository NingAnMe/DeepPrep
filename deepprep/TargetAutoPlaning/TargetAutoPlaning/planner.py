import os
import glob
# import shutil
from pathlib import Path
import tempfile
import nibabel as nib
import numpy as np
import pyvista as pv
from scipy import stats
import datetime
import sh
import open3d as o3d
from scipy.spatial import KDTree
from scipy.special import softmax, expit
import pandas as pd
import ants

import pyvista as pv
import numpy as np
from collections import deque

import matplotlib.pyplot as plt

from PIL import Image

def expand_mask(mesh, original_mask, num_rings=3):
    # 构建邻接列表
    edges = mesh.extract_all_edges(use_all_points=True)
    lines = edges.lines.reshape(-1, 3)[:, 1:3]
    adjacency = [[] for _ in range(mesh.n_points)]
    for u, v in lines:
        adjacency[u].append(v)
        adjacency[v].append(u)
    # 去重
    for i in range(mesh.n_points):
        adjacency[i] = list(set(adjacency[i]))
    
    # 初始化距离数组和队列
    distance = np.full(mesh.n_points, -1, dtype=int)
    initial_points = np.where(original_mask)[0]
    queue = deque(initial_points)
    for point_id in initial_points:
        distance[point_id] = 0
    
    # 执行BFS扩展
    while queue:
        u = queue.popleft()
        current_dist = distance[u]
        if current_dist >= num_rings:
            continue
        for v in adjacency[u]:
            if distance[v] == -1:
                distance[v] = current_dist + 1
                queue.append(v)
    
    return distance != -1

def mri_surf2surf(srcsubject, sval, trgsubject, tval, hemi, *args):
    sh.mri_surf2surf(
        '--srcsubject', srcsubject,
        '--sval', sval,
        '--trgsubject', trgsubject,
        '--tval', tval,
        '--hemi', hemi, *args)
    
def mri_annot2annot(srcsubject, sval, trgsubject, tval, hemi, *args):
    sh.mri_surf2surf(
        '--srcsubject', srcsubject,
        '--sval-annot', sval,
        '--trgsubject', trgsubject,
        '--tval', tval,
        '--hemi', hemi, *args)

def mri_vol2vol(srcsubject, sval, trgsubject, tval, *args):
    sh.mri_vol2vol(
        '--mov', sval,
        '--regheader', srcsubject,
        '--trgsubject', trgsubject,
        '--o', tval, *args)

def smooth_surf_data(subject, sval, tval, hemi, nsmooth_out=3):
    sh.mri_surf2surf(
        '--srcsubject', subject,
        '--sval', sval,
        '--trgsubject', subject,
        '--tval', tval,
        '--hemi', hemi,
        '--nsmooth-out', nsmooth_out)

def surf2mesh(surf_file):
    coords, faces = nib.freesurfer.read_geometry(surf_file)
    pv_vertices = coords
    dims = np.full(shape=(faces.shape[0], 1), fill_value=3)
    pv_faces = np.hstack((dims, faces))
    mesh = pv.PolyData(pv_vertices, pv_faces)
    return mesh

def plot_target_on_sagittal_slice(t1_nii_path, target_voxel, output_path):
    """
    在 T1 图像的矢状面绘制靶点位置（白色方块）
    
    参数:
        t1_nii_path (str): T1 图像路径 (.nii 或 .nii.gz)
        target_voxel (tuple): 靶点坐标 (x, y, z)，为体素索引
        output_path (str): 输出图片路径
    """
    # 读取 T1 图像
    t1_img = ants.image_read(t1_nii_path)
    t1_data = t1_img.numpy()

    if type(target_voxel) is str:
        # 如果传入的是字符串，则尝试将其转换为元组
        target_voxel = tuple(map(int, target_voxel.strip('()').split(',')))

    x, y, z = target_voxel
    sagittal_slice = t1_data[x, :, :]

    # 创建绘图
    plt.figure(figsize=(6, 6))
    plt.imshow(sagittal_slice, cmap="gray", origin="upper")
    plt.scatter(z, y, s=50, c='white', marker='s')
    plt.axis('off')

    # 创建输出目录（如果需要）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存图像
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

# def plot_target_on_coronal_slice(t1_nii_path, target_voxel, output_path):
#     """
#     在 T1 图像的冠状面绘制靶点位置（白色方块），并旋转图像使其竖直显示（逆时针 90 度）

#     参数:
#         t1_nii_path (str): T1 图像路径 (.nii 或 .nii.gz)
#         target_voxel (tuple): 靶点坐标 (x, y, z)，为体素索引
#         output_path (str): 输出图片路径
#     """
#     import ants
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import os

#     # 读取 T1 图像
#     t1_img = ants.image_read(t1_nii_path)
#     t1_data = t1_img.numpy()

#     if type(target_voxel) is str:
#         target_voxel = tuple(map(int, target_voxel.strip('()').split(',')))

#     x, y, z = target_voxel
#     coronal_slice = t1_data[:, y, :]  # 固定 y，绘制 x-z 面

#     # 旋转切片图像（逆时针 90 度）
#     coronal_slice_rot = np.rot90(coronal_slice)

#     # 旋转后的坐标位置也需要调整：
#     # 原先是 (z, x)，旋转后变为 (x, depth - z - 1)
#     h, w = coronal_slice.shape
#     new_x = z
#     new_y = w - x - 1

#     # 创建绘图
#     plt.figure(figsize=(6, 6))
#     plt.imshow(coronal_slice_rot, cmap="gray", origin="upper")
#     plt.scatter(new_x, new_y, s=50, c='white', marker='s')
#     plt.axis('off')

#     # 创建输出目录（如果需要）
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     # 保存图像
#     plt.savefig(output_path, bbox_inches='tight', dpi=300)
#     plt.close()


def plot_target_on_coronal_slice(t1_nii_path, target_voxel, output_path):
    """
    在 T1 图像的冠状面绘制靶点位置（白色方块），图像垂直显示（顺时针旋转90度）

    参数:
        t1_nii_path (str): T1 图像路径 (.nii 或 .nii.gz)
        target_voxel (tuple): 靶点坐标 (x, y, z)，为体素索引
        output_path (str): 输出图片路径
    """
    import ants
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    # 读取 T1 图像
    t1_img = ants.image_read(t1_nii_path)
    t1_data = t1_img.numpy()

    if type(target_voxel) is str:
        target_voxel = tuple(map(int, target_voxel.strip('()').split(',')))

    x, y, z = target_voxel
    
    # 获取冠状面切片并进行转置和翻转
    coronal_slice = t1_data[:, y, :]  # 在 y 平面切片，保留 x-z 维度
    
    # 关键修改：顺时针旋转90度 (转置 + 上下翻转)
    rotated_slice = coronal_slice.T[::-1, ::-1]  # 转置后上下翻转

    # 创建绘图
    plt.figure(figsize=(6, 6))
    
    # 显示旋转后的图像
    plt.imshow(rotated_slice, cmap="gray", origin="upper")
    
    # 调整靶点坐标：由于图像旋转，坐标映射关系变化
    # 原坐标(x,z) -> 新坐标(z, x_max - x - 1)
    x_max = coronal_slice.shape[0] - 1  # 获取x轴最大索引
    z_max = coronal_slice.shape[1] - 1  # 获取z轴最大索引
    plt.scatter(x_max - x, z_max - z, s=50, c='white', marker='s')  # 直接使用新坐标系中的位置
    
    plt.axis('off')

    # 创建输出目录（如果需要）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存图像
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def add_target_roi_mask_T1(t1_nii_path, target_voxel, output_mask_path, radius=3):
    img = ants.image_read(t1_nii_path)
    img_np = img.numpy()

    x = target_voxel[0]
    y = target_voxel[1]
    z = target_voxel[2]
    ## round to nearest voxel
    x = int(x + 0.5)
    y = int(y + 0.5)
    z = int(z + 0.5)
    img_np[int(x - radius + 0.5):int(x + radius + 0.5), int(y - radius + 0.5):int(y + radius + 0.5),
    int(z - radius + 0.5):int(z + radius + 0.5)] = 200

    img_masked = ants.from_numpy(img_np, img.origin, img.spacing, img.direction)
    ants.image_write(img_masked, output_mask_path)

def concatenate_images(image1, image2, output_path, direction='horizontal', background=(255, 255, 255)):
    """
    拼接两张图片（支持横向/竖向），小图居中显示
    
    参数:
        image1 (Image): 第一张图片(PIL Image对象)
        image2 (Image): 第二张图片(PIL Image对象)
        direction (str): 拼接方向 ('horizontal' 或 'vertical')
        background (tuple): 背景色RGB值 (默认黑色)
    
    返回:
        Image: 拼接后的新图片
    """
    img1 = Image.open(image1).convert('RGB')
    img2 = Image.open(image2).convert('RGB')
    
    # 获取图片尺寸
    w1, h1 = img1.size
    w2, h2 = img2.size
    
    if direction == 'horizontal':
        # 横向拼接：总宽度为两图之和，高度取最大值
        new_width = w1 + w2
        new_height = max(h1, h2)
        
        # 创建新画布
        new_image = Image.new('RGB', (new_width, new_height), background)
        
        # 计算垂直居中偏移量并粘贴图片
        y_offset1 = (new_height - h1) // 2
        y_offset2 = (new_height - h2) // 2
        new_image.paste(img1, (0, y_offset1))
        new_image.paste(img2, (w1, y_offset2))
        
    elif direction == 'vertical':
        # 竖向拼接：总高度为两图之和，宽度取最大值
        new_width = max(w1, w2)
        new_height = h1 + h2
        
        # 创建新画布
        new_image = Image.new('RGB', (new_width, new_height), background)
        
        # 计算水平居中偏移量并粘贴图片
        x_offset1 = (new_width - w1) // 2
        x_offset2 = (new_width - w2) // 2
        new_image.paste(img1, (x_offset1, 0))
        new_image.paste(img2, (x_offset2, h1))
    
    # 保存图像
    # concat_img = Image.fromarray(new_image.astype(np.uint8))
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_image.save(output_path)

    return new_image


class Planner(object):
    def __init__(self, verbose=False, workdir=None, logdir=None):
        self.verbose = verbose
        # check, clear workdir
        if workdir is None:
            self.workdir = Path(tempfile.TemporaryDirectory().name)
        else:
            if isinstance(workdir, Path):
                self.workdir = workdir
            else:
                self.workdir = Path(workdir)
        if not self.workdir.exists():
            os.makedirs(self.workdir, exist_ok=True)
        else:
            # shutil.rmtree(self.workdir) ## Commented out to avoid deleting the workdir
            os.makedirs(self.workdir, exist_ok=True)

        self.resource_dir = Path(os.path.dirname(__file__)) / 'resource'

        if self.verbose: 
            # logs dir
            if logdir is None:
                self.logdir = Path('logs') / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            else:
                if isinstance(logdir, Path):
                    self.logdir = logdir
                else:
                    self.logdir = Path(logdir)
            self.logdir.mkdir(parents=True, exist_ok=True)

        # load fs6 template
        # lh
        lh_fs6_pial_file = os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage6', 'surf', 'lh.pial')
        self.lh_fs6_pial_mesh = surf2mesh(lh_fs6_pial_file)
        self.lh_fs6_n_vertex = self.lh_fs6_pial_mesh.points.shape[0]
        # rh
        rh_fs6_pial_file = os.path.join(os.environ['SUBJECTS_DIR'], 'fsaverage6', 'surf', 'rh.pial')
        self.rh_fs6_pial_mesh = surf2mesh(rh_fs6_pial_file)
        self.rh_fs6_n_vertex = self.rh_fs6_pial_mesh.points.shape[0]

    def compute_surf_fc(self, seed, surf_bold):
        n_vertex = surf_bold.shape[0]
        surf_fc = np.zeros((n_vertex))
        # for i in range(n_vertex):
        #     r, _ = stats.pearsonr(surf_bold[i, :], seed)
        #     surf_fc[i] = r

        seed = np.array(seed, dtype=np.float64) # Shape: (T)
        seed = seed.reshape(1, -1) # Shape: (1,T)
        bold = np.array(surf_bold, dtype=np.float64) # Shape: (m,T)

        # Normalize seeds
        seed_norm = (seed - np.mean(seed, axis=1, keepdims=True)) / np.std(seed, axis=1, keepdims=True)

        # Normalize BOLD data
        bold_norm = (bold - np.mean(bold, axis=1, keepdims=True)) / np.std(bold, axis=1, keepdims=True)

        # Compute FC map
        fcmap = np.dot(bold_norm, seed_norm.T) / bold.shape[1]  # Shape: (m, n)
        fcmap = fcmap.T  # Shape: (n, m)
        
        surf_fc = fcmap[0,:]

        return surf_fc

    def compute_vol_fc(self, seed, vol_bold):
        n_i, n_j, n_k = vol_bold.shape[:3]
        vol_fc = np.zeros(shape=(n_i, n_j, n_k), dtype=np.float32)
        for i in range(n_i):
            for j in range(n_j):
                for k in range(n_k):
                    r, _ = stats.pearsonr(vol_bold[i, j, k, :], seed)
                    vol_fc[i, j, k] = r
        return vol_fc
    
    def compute_vol_fc_with_mask(self, seed, vol_bold, mask=None):
        n_i, n_j, n_k = vol_bold.shape[:3]
        vol_fc = np.zeros(shape=(n_i, n_j, n_k), dtype=np.float32)
        if mask is None:
            for i in range(n_i):
                for j in range(n_j):
                    for k in range(n_k):
                        r, _ = stats.pearsonr(vol_bold[i, j, k, :], seed)
                        vol_fc[i, j, k] = r
        else:
            # mask = np.array(mask, dtype=bool)
            for i in range(n_i):
                for j in range(n_j):
                    for k in range(n_k):
                        if mask[i, j, k] != 0:
                            r, _ = stats.pearsonr(vol_bold[i, j, k, :], seed)
                            vol_fc[i, j, k] = r
                        else:
                            vol_fc[i, j, k] = 0.0
        return vol_fc

    def load_surf_bolds(self, surf_bolds_path, hemi='lh'):
        surf_bold_files = sorted(glob.glob(os.path.join(
            surf_bolds_path, f'{hemi}.*_rest_reorient_skip_faln_mc_g1000000000_bpss_resid_fsaverage6_sm6.nii.gz')))
        if hemi == 'lh':
            fs6_n_vertex = self.lh_fs6_n_vertex
        else:
            fs6_n_vertex = self.rh_fs6_n_vertex
        surf_bold = np.zeros(shape=(fs6_n_vertex, 0))
        for bold_file in surf_bold_files:
            bold = nib.load(bold_file).get_fdata()
            n_frame = bold.shape[3]
            n_vertex = bold.shape[0] * bold.shape[1] * bold.shape[2]
            bold_surf = bold.reshape((n_vertex, n_frame), order='F')
            surf_bold = np.hstack((surf_bold, bold_surf))
        return surf_bold
    
    def load_surf_bolds_DeepPrep(self, surf_bolds_path, hemi='lh'):
        '''
        Load all the surface bold data in the surf_bolds_path from DeepPrep postprocess resultsss
        '''
        if hemi == 'lh':
            hemi = 'L'
        elif hemi == 'rh':
            hemi = 'R'
        else:
            raise ValueError('Invalid hemisphere')
        surf_bold_files = sorted(glob.glob(os.path.join(
            surf_bolds_path, f'*_task-rest*hemi-{hemi}_space-fsaverage6_*-fwhm_bold.nii.gz')))
        if hemi == 'lh':
            fs6_n_vertex = self.lh_fs6_n_vertex
        else:
            fs6_n_vertex = self.rh_fs6_n_vertex
        surf_bold = np.zeros(shape=(fs6_n_vertex, 0))
        for bold_file in surf_bold_files:
            bold = nib.load(bold_file).get_fdata()
            n_frame = bold.shape[3]
            n_vertex = bold.shape[0] * bold.shape[1] * bold.shape[2]
            bold_surf = bold.reshape((n_vertex, n_frame), order='F')
            surf_bold = np.hstack((surf_bold, bold_surf))
        return surf_bold

    def load_vol_bolds(self, vol_bolds_path):
        vol_bold_files = sorted(glob.glob(os.path.join(vol_bolds_path,
                                                       f'*_bld*_rest_reorient_skip_faln_mc_g1000000000_bpss_resid_FS1mm_MNI1mm_MNI2mm_sm6*.nii.gz')))
        vol_bold = np.zeros(shape=(128, 128, 128, 0))
        for bold_file in vol_bold_files:
            bold = ants.image_read(bold_file)
            bold_np = bold.numpy()
            vol_bold = np.concatenate((vol_bold, bold_np), axis=3)
        return vol_bold
    
    def load_vol_bolds_DeepPrep(self, vol_bolds_path):
        vol_bold_files = sorted(glob.glob(os.path.join(vol_bolds_path,
                                                       f'*_run-*_space-MNI152NLin6Asym_res-2_desc-fwhm_bold.nii.gz')))
        if len(vol_bold_files) == 0:
            vol_bold_files = sorted(glob.glob(os.path.join(vol_bolds_path,
                                                           f'*_run-*_space-MNI152NLin6Asym_res-02_desc-fwhm_bold.nii.gz')))
        vol_bold = np.zeros(shape=(91, 109, 91, 0))
        for bold_file in vol_bold_files:
            bold = ants.image_read(bold_file)
            bold_np = bold.numpy()
            vol_bold = np.concatenate((vol_bold, bold_np), axis=3)
        return vol_bold

    def save_mgh(self, data, file):
        if data.dtype == np.float64 or data.dtype == bool:
            data = data.astype('float32')
        img = nib.MGHImage(data, np.eye(4))
        nib.save(img, file)

    def labels2mask(self, labels, indexes):
        mask = np.zeros(len(labels))
        idx = np.zeros(len(labels), dtype=bool)
        for index in indexes:
            idx = idx | (labels == index)
        mask[idx] = 1
        return mask

    def plan(self, app_data_path, subj):
        raise NotImplementedError

    def __alpha_shape(self, mesh: pv.PolyData, alpha=20):
        points = mesh.points
        v3dv = o3d.utility.Vector3dVector(points)
        pcd = o3d.geometry.PointCloud(v3dv)
        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        vertices = np.asarray(mesh_o3d.vertices)
        triangles = np.asarray(mesh_o3d.triangles)
        dims = np.full(shape=(triangles.shape[0], 1), fill_value=3)
        faces_pv = np.hstack((dims, triangles))
        mesh_pv = pv.PolyData(vertices, faces_pv)
        return mesh_pv

    def compute_sulc_depth(self, mesh: pv.PolyData):
        # smooth
        mesh_smooth = mesh.smooth(n_iter=100, relaxation_factor=0.04)

        # alpha shape
        mesh_alpha_shape = self.__alpha_shape(mesh_smooth)
        mesh_alpha_shape.compute_normals(inplace=True)

        mesh_impl_dist = mesh.compute_implicit_distance(mesh_alpha_shape)
        sulc_depth = mesh_impl_dist['implicit_distance']
        if sulc_depth.mean() > 0:
            sulc_depth = -sulc_depth
        return sulc_depth

    def cluster_target_search(self, heatmap_mesh: pv.PolyData, percentile):
        '''
        该代码实现了一个三维热力图分析算法，主要功能：

            动态阈值分割：基于用户指定的百分位数确定热点区域

            连通区域分析：提取最大连续热点区域

            质心定位：通过softmax加权计算目标中心

            区域排除：防止重复检测相邻区域

            多维度评分：结合区域面积和热力强度进行综合评价

            结果排序：返回排序后的目标点索引及置信度得分
        '''
        # hyper-parameter
        dist_threshold = 5  # 5mm
        ref_candi_region_area = 32
        ref_heatmap = 0.6
        max_num_of_target = 3
        # compute threshold value
        heatmap = heatmap_mesh.point_data['label']
        candi_idx = heatmap > -1.0
        heatmap_candi = heatmap[candi_idx]
        threshold_value = np.percentile(heatmap_candi, percentile)

        targets = list()
        # compute candi region
        candi_mask = heatmap >= threshold_value
        extracted_mesh = heatmap_mesh.extract_points(candi_mask)
        candi_region_mesh = extracted_mesh.extract_largest()
        candi_region_mesh = candi_region_mesh.connectivity().extract_surface()
        total_candi_area = candi_region_mesh.area
        while candi_region_mesh.n_points > 0:
            target_dict = dict()
            # compute target
            candi_heatmap = candi_region_mesh.point_data['label']
            candi_weight = softmax(candi_heatmap)[np.newaxis]
            candi_target = np.dot(candi_weight, candi_region_mesh.points).flatten()
            candi_tree = KDTree(heatmap_mesh.points[candi_idx])
            candi_index = np.where(candi_idx)[0]
            _, idx = candi_tree.query(candi_target)
            target_idx = candi_index[idx]
            target_dict['index'] = target_idx
            target_dict['mean_heatmap'] = candi_heatmap[candi_heatmap > -1.0].mean().item()
            target_dict['n_points'] = candi_region_mesh.n_points
            target_dict['region_area'] = candi_region_mesh.area
            targets.append(target_dict)

            if not np.any(candi_mask) or len(targets) == max_num_of_target:
                break
            target_region_idx = candi_tree.query_ball_point(candi_target, r=dist_threshold)
            target_region_idx = candi_index[target_region_idx]
            heatmap[target_region_idx] = -1.0
            candi_idx = heatmap > -1.0
            candi_mask = heatmap >= threshold_value
            extracted_mesh = heatmap_mesh.extract_points(candi_mask)
            candi_region_mesh = extracted_mesh.extract_largest()
            candi_region_mesh = candi_region_mesh.connectivity().extract_surface()

        df_targert = pd.DataFrame(targets)
        area_scale = min(total_candi_area / ref_candi_region_area, 1.0)
        area_scale = df_targert['region_area'] / total_candi_area * area_scale
        heatmap_scale = expit(5 * ((df_targert['mean_heatmap'] - 0.2) / ref_heatmap))
        df_targert['score'] = area_scale * heatmap_scale
        df_targert = df_targert.sort_values('score', ascending=False)
        targets = df_targert[['index', 'score']].to_dict('records')

        return targets
