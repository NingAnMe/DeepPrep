import os
import sys

import numpy as np
import nibabel as nib
import pickle as pickle
import sh
# from path import Path
from pathlib import Path
from collections import OrderedDict
import shutil

# import scipy.io as sio  # 用于加载 MATLAB LUT 文件

STEP_NAME = "parc213"
RESOLUTION = 213


def DATA_DIR():
    return Path(os.environ['DATA_DIR']).resolve()
    
def INDI_TEMPL_DIR():
    return Path(os.environ['TEMPL_DIR']).resolve()

def SUBJECTS_DIR():
    return Path(os.environ['SUBJECTS_DIR']).resolve()

def OUTPUT_DIR():
    return Path(os.environ['OUTPUT_DIR']).resolve()

def parc213_step(subject, ses='ses-01'):

    surf_path, recon_dir = check_inputs(subject, ses)

    viz_path, parc_path, res_path = prepare_env(
        subject, recon_dir, RESOLUTION
    )

    # Find networks/confidences.
    lhdata, rhdata = norm_surface_data(surf_path)
    print('lhdata shape:', lhdata.shape)
    for maskname in cluster_numbers().keys():
        for (hemi, data) in zip(['lh', 'rh'], [lhdata, rhdata]):
            fs6_parcellation(
                subject, parc_path, maskname, hemi, RESOLUTION, data
            )

    convert_subs_data(
        [subject],
        OUTPUT_DIR() / 'Parc213',
        cluster_numbers().keys(),
        [16, 21, 17, 30, 24],
        [16, 21, 17, 30, 24]) ## default to 213

def check_inputs(subject, ses='ses-01'):
    recon_dir = SUBJECTS_DIR() / subject
    if not recon_dir.exists():
        raise Exception("No reconall directory: " + recon_dir)

    if ses == 'temp':
        preprocess_dir = DATA_DIR() / subject / 'func'
    else:
        preprocess_dir = DATA_DIR() / subject / ses / 'func'
    fs6_residuals_zip_path = preprocess_dir / ('%s_fs6_residuals.zip' % subject)
    if fs6_residuals_zip_path.exists():
        sh.unzip(
            '-o',
            fs6_residuals_zip_path,
            '-d', preprocess_dir,
            _out=sys.stdout
        )

    surf_path = preprocess_dir

    return surf_path, recon_dir


def prepare_env(subject, recon_dir, resolution):
    parc_str = 'Parc%d' % resolution
    viz_path = OUTPUT_DIR() / 'Viz' / parc_str
    viz_path.mkdir(parents=True, exist_ok=True)

    parc_path = OUTPUT_DIR() / parc_str
    parc_path.mkdir(parents=True, exist_ok=True)

    res_path = OUTPUT_DIR() / 'Residuals'
    res_path.mkdir(parents=True, exist_ok=True)

    recon_path = SUBJECTS_DIR()
    # os.environ['SUBJECTS_DIR'] = recon_path

    fs_path = Path(os.environ['FREESURFER_HOME'])
    for fs_avg in ['fsaverage', 'fsaverage4', 'fsaverage5', 'fsaverage6']:
        fs_avg_link = recon_path / fs_avg
        if not fs_avg_link.exists():

            fs_avg_path = fs_path / 'subjects' / fs_avg
            if not fs_avg_path.exists():
                raise Exception("No such directory: " + fs_avg_path)
            # fs_avg_path.symlink(fs_avg_link)  # This is for path.py
            # fs_avg_link.symlink_to(fs_avg_path) # This is for pathlib

            if fs_avg_link.is_symlink():
                # 删除现有的符号链接（无论是否有效）
                fs_avg_link.unlink()

            # 现在安全地创建新链接 ---
            fs_avg_link.symlink_to(fs_avg_path)
            # print(f"Created symlink: {fs_avg_link} -> {fs_avg_path}")

    return viz_path, parc_path, res_path


def norm_surface_data(surf_path, targ_suffix='desc-fwhm_bold.nii.gz'):
    '''
    Read then return normalized surface data.

    surfdir     - Path. Path to surface data.
    targ_suffix - str. Target "endswith" suffix.

    Returns two numpy arrays as a tuple: lhdata, rhdata.
    '''

    lh_paths = _assemble_lh_surf_paths(surf_path, targ_suffix)
    lhdata = []
    rhdata = []
    for lhpath in lh_paths:
        lh_data_raw = _extract_vol(lhpath)
        lhdata.append(_norm_data(lh_data_raw))

        rhpath = surf_path / lhpath.name.replace('hemi-L', 'hemi-R')
        rh_data_raw = _extract_vol(rhpath)
        rhdata.append(_norm_data(rh_data_raw))

    return np.hstack(lhdata), np.hstack(rhdata)


def _assemble_lh_surf_paths(surf_path, targ_suffix):
    # Assemble alphabetically order of surface files.
    # lh_paths = []
    # for fpath in surf_path.iterdir():
    #     # if not fpath.name.startswith('lh'):
    #     #     continue
    #     if not 'hemi-L' in fpath.name:
    #         continue

    #     if not fpath.name.endswith(targ_suffix):
    #         continue

    #     lh_paths.append(fpath)
    lh_paths = []
    for fpath in surf_path.iterdir():  # 保持相同的遍历方式
        # if not fpath.name.startswith('lh'):
        #     continue
        if 'hemi-L' not in fpath.name:  # 直接使用 Path 对象的 name 属性
            continue
        if not fpath.name.endswith(targ_suffix):  # 后缀检查保持相同逻辑
            continue
        lh_paths.append(fpath)

    lh_paths = sorted(lh_paths)

    return lh_paths


def _extract_vol(path, flatten=False):
    '''
    Extract a volume from a NIFTI file saved at `path`.

    flatten - Flag for if the volume should be flattened (True) or converted to
              a column vector (False, default).

    Returns numpy array.
    '''

    img = nib.load(path)
    vol = img.get_fdata(dtype=np.float32)
    if flatten:
        return vol.flatten()

    nvoxels = np.prod(vol.shape[:-1])
    return np.reshape(vol, (nvoxels, vol.shape[-1]), order='F')


def _norm_data(data):
    '''
    Normalize data across rows and return it.
    '''

    n = data.shape[0]
    sig = data.std(axis=1, ddof=1).reshape((n, 1))
    sig[sig == 0] = 1.0
    mu = data.mean(axis=1).reshape((n, 1))

    return (data - mu) / sig


def cluster_numbers(resolution=213):
    '''
    Return an OrderedDict depicting the number of clusters in each lobe
    used for the parcellation.

    resolution - int. Choice of {213, 114, 92}.
    '''
    if resolution == 213:
        return OrderedDict([
            ('Visual', 16),
            ('Motor', 21),
            ('Frontal', 17),
            ('Temporal', 30),
            ('Parietal', 24)
        ])

    if resolution == 152:
        return OrderedDict([
            ('Visual', 12),
            ('Motor', 12),
            ('Frontal', 17),
            ('Temporal', 20),
            ('Parietal', 15)
        ])

    if resolution == 92:
        return OrderedDict([
            ('Visual', 6),
            ('Motor', 6),
            ('Frontal', 13),
            ('Temporal', 10),
            ('Parietal', 11)
        ])
    raise ValueError('Resolution %d not supported' % resolution)


def fs6_parcellation(
        subject,
        out_path,
        maskname,
        hemi,
        resolution,
        data,
        cleanup=True,
        pickle_path=None):
    '''
    subject    - str.
    maskname   - str. From {"Frontal", "Motor", "Parietal", "Temporal",
                 "Visual", "WB"}
    hemi       - str. From {"lh", "rh"}.
    resolution - int. From {213, 144, 92}
    data       - Numpy array.
    cleanup    - Flag for if un-needed files should be deleted.
    pickle_path- If provided as a Path object, pickle files containing
                 in situ variable values are produced.
    '''

    cluster_nos = cluster_numbers(resolution)
    n_cluster = cluster_nos[maskname]
    out_path.mkdir(parents=True, exist_ok=True)

    prior_variability = _prior_variability(hemi)

    area_hemi_path = out_path / ('%s_%s' % (maskname, hemi))
    par_path = area_hemi_path / subject
    par_path.mkdir(parents=True, exist_ok=True)

    mask_ind, mask_data = _mask_ind(hemi, maskname)

    atlas_path = INDI_TEMPL_DIR() / 'Atlas'
    mask_hemi_atlas_path = atlas_path / ('%s_%s' % (maskname, hemi))
    fs6_by_fs3_path = mask_hemi_atlas_path / 'fs6_by_fs3'

    _fs6_parcellation(
        data,
        mask_data,
        mask_ind,
        fs6_by_fs3_path,
        par_path,
        subject,
        prior_variability,
        n_cluster,
        hemi,
        pickle_path)
    if cleanup:
        for i in range(10):
            iter_path = par_path / ('iter_%d' % i)
            # iter_path.rmtree_p()
            if iter_path.exists():
                shutil.rmtree(iter_path)  # 递归删除目录


def _prior_variability(hemi):
    inter_var_path = INDI_TEMPL_DIR() / 'InterVar'
    pv_path = inter_var_path / ('%s.InterVariability_MSC_fs6.mgh' % hemi)
    pv_img = nib.load(pv_path)
    return pv_img.get_fdata().flatten()


def _mask_ind(hemi, maskname):
    mask_path = INDI_TEMPL_DIR() / 'Labels' / ('%s.%s.mgh' % (hemi, maskname))
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata().flatten()
    return mask_data == 1, mask_data


def _fs6_parcellation(
        data,
        mask,
        mask_ind,
        fs6_by_fs3_path,
        par_path,
        subject,
        prior_variability,
        n_cluster,
        hemi,
        pickle_path):
    # First set some parameters.
    n_iter = 10
    confidence_threshold_ = 1.5

    snr, data_mask, var_inv = _priors(data, prior_variability)
    if pickle_path is not None:
        pickle_fpath = pickle_path / (
            '%s_%d_mask_var_snr.p' % (hemi, n_cluster))
        with pickle_fpath.open('w') as f:
            pickle.dump((snr, data_mask, var_inv, data), f)

    # Create the iter_0 seed directory.
    # iter_0_path = par_path / 'iter_0'
    # iter_0_path.rmtree_p()
    iter_0_path = par_path / 'iter_0'
    if iter_0_path.exists():  # 避免删除不存在的路径时抛出异常
        shutil.rmtree(iter_0_path)  # 递归删除目录
    # cluster_path = fs6_by_fs3_path / ('Cluster%d' % n_cluster)
    # cluster_path.copytree(iter_0_path)
    
    cluster_path = fs6_by_fs3_path / f'Cluster{n_cluster}'
    # 替代 copytree 方法
    shutil.copytree(cluster_path, iter_0_path)  # 注意参数顺序反转
    grp_seed_data = None
    for cnt in range(1, n_iter + 1):
        iter_path = par_path / ('iter_%d' % cnt)
        iter_path.mkdir(parents=True, exist_ok=True)
        last_iter_path = par_path / ('iter_%d' % (cnt - 1))

        if cnt == 1:
            confidence_threshold = 0.8
        else:
            confidence_threshold = confidence_threshold_

        grp_net, seed_data, grp_seed_data = _grp_net_seed_data(
            n_cluster,
            last_iter_path,
            cnt,
            hemi,
            confidence_threshold,
            data_mask,
            snr,
            data,
            grp_seed_data
        )
        if pickle_path is not None:
            pickle_fpath = pickle_path / (
                '%s_%d_%d_seeddata.p' % (hemi, n_cluster, cnt)
            )
            with pickle_fpath.open('w') as f:
                pickle.dump((grp_net, seed_data, grp_seed_data, data), f)

        c_value = _vert_corr(data, mask_ind, seed_data)

        if pickle_path is not None:
            pickle_fpath = pickle_path / (
                '%s_%d_%d_vert_corr.p' % (hemi, n_cluster, cnt)
            )
            with pickle_fpath.open('w') as f:
                pickle.dump(c_value, f)

        c_value = _weight_vert_corr(
            grp_net, data, c_value, var_inv, mask_ind, cnt)

        if pickle_path is not None:
            pickle_fpath = pickle_path / (
                '%s_%d_%d_vert_corr_weighted.p' % (hemi, n_cluster, cnt)
            )
            with pickle_fpath.open('w') as f:
                pickle.dump(c_value, f)

        parc_membership, parc_confidence = _parc_membership(c_value, n_cluster)

        if pickle_path is not None:
            pickle_fpath = pickle_path / (
                '%s_%d_%d_parc_memb.p' % (hemi, n_cluster, cnt)
            )
            with pickle_fpath.open('w') as f:
                pickle.dump((parc_membership, parc_confidence), f)

        for n in range(1, n_cluster + 1):
            network_save, confid_save = _assign_network(
                parc_membership, parc_confidence, n, mask, mask_ind
            )

            network_save = network_save.astype(np.float32)
            confid_save = confid_save.astype(np.float32)
            network_fname = 'network_%d_%s.mgh' % (n, hemi)
            network_path = iter_path / network_fname
            network_img = nib.MGHImage(network_save, np.eye(4))
            nib.save(network_img, network_path)

            confid_fname = 'network_confidence_%d_%s.mgh' % (n, hemi)
            confid_path = iter_path / confid_fname
            confid_img = nib.MGHImage(network_save * confid_save, np.eye(4))
            nib.save(confid_img, confid_path)

def convert_subs_data(subs, OutPath, masknames, 
                     bset_clusternum_lh, bset_clusternum_rh, confs=1.5):
    """
    MATLAB 代码转换版 Python 函数
    参数说明保持与原始 MATLAB 代码一致：
    - subs: 受试者ID列表
    - OutPath: 输出根路径
    - masknames: 掩模名称列表
    - bset_clusternum_lh/rh: 左右半球聚类数配置
    - confs: 配置参数标识
    """
    # 创建输出目录结构 (Python 3.5+ 路径操作)
    wb_lh_root = Path(OutPath) / 'WB_lh'
    wb_rh_root = Path(OutPath) / 'WB_rh'
    wb_lh_root.mkdir(parents=True, exist_ok=True)
    wb_rh_root.mkdir(parents=True, exist_ok=True)
    masknames = list(masknames) 

    for sub in subs:  # 遍历受试者
        # start_time = time.time()
        print(f'Combining Data: {sub} ...')

        # # 获取 FreeSurfer 路径 (模拟 MATLAB 的 dir 行为)
        # sub_data_path = next((p for p in (Path(ReconPath)/sub).iterdir() if p.is_dir()), None)
        # fspath = sub_data_path.parent if sub_data_path else None

        # 初始化空矩阵 (使用 float 类型保持与 MATLAB 兼容)
        lhparc = np.zeros(40962, dtype=np.float32)
        rhparc = np.zeros(40962, dtype=np.float32)
        
        # 创建输出目录
        (wb_lh_root / sub ).mkdir(parents=True, exist_ok=True)
        (wb_rh_root / sub ).mkdir(parents=True, exist_ok=True)

        count = 0  # 计数器初始化
        for n in range(5):  # 原始 MATLAB 代码中 n 从 1 到 5
            maskname = masknames[n]  # 注意 Python 索引从 0 开始
            nCluster_lh = bset_clusternum_lh[n]
            nCluster_rh = bset_clusternum_rh[n]

            for i in range(1, nCluster_lh + 1):  # i 从 1 到 nCluster_lh
                count += 1
                # 构建 MGH 文件路径
                lh_mgh_path = (
                    Path(OutPath) / f"{maskname}_lh" 
                    / sub / "iter_10" / f"network_{i}_lh.mgh"
                )
                # 加载 MGH 文件 (使用 nibabel 代替 load_mgh)
                tmp_lh = nib.load(str(lh_mgh_path)).get_fdata().flatten()
                lhparc += count * tmp_lh

                # 右半球处理同理
                rh_mgh_path = (
                    Path(OutPath) / f"{maskname}_rh" 
                    / sub / "iter_10" / f"network_{i}_rh.mgh"
                )
                tmp_rh = nib.load(str(rh_mgh_path)).get_fdata().flatten()
                rhparc += count * tmp_rh

        # 保存结果文件
        output_lh_path = wb_lh_root / sub / f"lh.{sub}_IndiCluster{count}_fs6.mgh"
        nib.save(nib.MGHImage(lhparc.reshape(-1,1,1), np.eye(4)), str(output_lh_path))

        output_rh_path = wb_rh_root / sub / f"rh.{sub}_IndiCluster{count}_fs6.mgh"
        nib.save(nib.MGHImage(rhparc.reshape(-1,1,1), np.eye(4)), str(output_rh_path))

        # print(f"Time elapsed: {time.time()-start_time:.2f} seconds")

def _priors(data, prior_variability):
    '''
    Rturn the SNR vetor, data mask, and inverse variance parameters
    used, as a tuple of numpy arrays, respectively.
    '''
    snr = np.ones(40962)

    data_mask = data[:, 0] != 0

    # Scale prior variability.
    min_pv = prior_variability.min()
    max_pv = prior_variability.max()
    pv_span = max_pv - min_pv
    var = 0.4 + 0.6 * (prior_variability - min_pv) / pv_span
    var_inv = 1.0 / var

    return snr, data_mask, var_inv


def _grp_net_seed_data(
        n_cluster,
        last_iter_path,
        cnt,
        hemi,
        confidence_threshold,
        data_mask,
        snr,
        data,
        grp_seed_data):
    grp_net = []
    seed_data = []
    for i_cluster in range(1, n_cluster + 1):
        if cnt == 1:
            fname = 'NetworkConfidence_%d_%s.mgh' % (i_cluster, hemi)
        else:
            fname = 'network_confidence_%d_%s.mgh' % (i_cluster, hemi)
        vol_path = last_iter_path / fname
        vol_img = nib.load(vol_path)
        vol = vol_img.get_fdata().flatten()
        idx = np.logical_and(vol >= confidence_threshold, data_mask)

        if not np.any(idx):
            maxx = vol.max()
            if maxx > 0:
                idx = vol == maxx

        grp_net.append(idx)
        seed_data.append(np.matmul(snr[idx], data[idx, :]))

    # Weight in the group seed in each iteration, should throw in
    # individual variability map as weight in the future.
    grp_net = np.array(grp_net)
    seed_data = np.array(seed_data)
    if cnt == 1:
        grp_seed_data = seed_data.copy()
    else:
        seed_data += grp_seed_data / (cnt - 1.0)

    return grp_net, seed_data, grp_seed_data


def _vert_corr(data, mask_ind, seed_data):
    # Compute vertex to seed correlation for all vertices.
    mask_data = data[mask_ind, :]
    c_value = _fast_corr(mask_data.transpose(), seed_data.transpose())
    c_value[np.isnan(c_value)] = 0.0
    return c_value


def _fast_corr(s_series_, t_series_):
    '''
    s_series_ - T x N1 np array.
    t_series_ - T x N2 np array.

    Returns N1 x N2 np array correlation matrix.
    '''

    s_series = __norm_series(s_series_)
    t_series = __norm_series(t_series_)

    return np.matmul(s_series.transpose(), t_series)


def __norm_series(series_):
    '''
    series_ - T x N np array.

    Returns normalized (across the first axis) version of series_ of the same
    size.
    '''
    series = series_ - series_.mean(axis=0)

    # series = series_ - np.tile(
    #     series_.mean(axis=0), (series_.shape[0], 1)
    # )
    std = np.sqrt(np.sum(series ** 2.0, axis=0))
    i_pos = std > 0
    series[:, i_pos] = series[:, i_pos] * (1.0 / std[i_pos])

    # series[:, i_pos] *= np.tile(
    #     1./std[i_pos], (series_.shape[0], 1)
    # )

    return series


def _weight_vert_corr(grp_net, data, c_value, var_inv, mask_ind, cnt):
    # Further weight in the group map * inv(Variability) by adding
    # correlation coeffition of 0 ~ 0.5 according to inv(Variaiblity).
    for i, idxfull in enumerate(grp_net):
        tmp = 0.0 * data[:, 0]
        tmp[idxfull] = 1.0
        tmpmask = tmp[mask_ind]
        idx = tmpmask == 1
        c_value[idx, i] += (var_inv[idxfull] - 1.0) / 3.0 / float(cnt)
    return c_value


def _parc_membership(c_value, n_cluster):
    # Determine the network membership of each vertex.
    cluster_data = c_value[:, :n_cluster]
    parc_membership = []
    parc_confidence = []
    for row in cluster_data:
        idx = np.argsort(row)

        if row[idx[-2]] == 0.0:
            parc_confidence.append(row[idx[-1]])
            if row[idx[-1]] == 0.0:
                parc_membership.append(1)
            else:
                parc_membership.append(idx[-1] + 1)
        else:
            parc_confidence.append(row[idx[-1]] / row[idx[-2]])
            parc_membership.append(idx[-1] + 1)

    parc_membership = np.array(parc_membership)
    parc_confidence = np.array(parc_confidence)
    return parc_membership, parc_confidence

def _assign_network(parc_membership, parc_confidence, n, mask, mask_ind):
    network = 0 * parc_membership
    confid = network.copy().astype(np.float64)
    i_member = parc_membership == n
    network[i_member] = 1
    confid[i_member] = parc_confidence[i_member]

    network_save = 0 * mask
    confid_save = network_save.copy()
    network_save[mask_ind] = network
    confid_save[mask_ind] = confid
    return network_save, confid_save

def parc213_DeepPrep(subject, data_path, reconall_dir, output_dir, ses='ses-01'):
    # Set the path to the FreeSurfer directory.
    os.environ['SUBJECTS_DIR'] = reconall_dir

    # Set the path to the output directory.
    os.environ['OUTPUT_DIR'] = output_dir

    # Set the path to the data directory.
    os.environ['DATA_DIR'] = data_path

    # Set the path to the individual template directory.
    current_script_path = os.path.abspath(__file__)
    os.environ['TEMPL_DIR'] = os.path.join(os.path.dirname(current_script_path), 'resource')

    ## check if have been processed
    if os.path.exists(os.path.join(output_dir, 'Parc213/WB_lh', subject)):
        print(f'{subject} 213-parcellation has been processed.')
        return
        
    # Perform the parcellation.
    parc213_step(subject, ses)


# if __name__ == '__main__':
#     subject = "sub-015"
#     os.environ['DATA_DIR'] = '/mnt/HardDisk/lissworkspace/data/PD_POINT/temp/derivatives_postprocess/BOLD'
#     ## get the path of this file
#     current_script_path = os.path.abspath(__file__)
#     os.environ['TEMPL_DIR'] = os.path.join(os.path.dirname(current_script_path), 'resource')
#     os.environ['SUBJECTS_DIR'] = '/mnt/HardDisk/lissworkspace/data/PD_POINT/temp/derivatives/Recon'
#     os.environ['OUTPUT_DIR'] = '/mnt/HardDisk/lissworkspace/data/PD_POINT/temp/FuJian/Parcellation213'
#     parc213_step(subject)
