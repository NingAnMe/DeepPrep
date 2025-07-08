import os
import numpy as np
import nibabel as nib
# from Figure1.utils import set_environ

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# CUR_DIR = '/home/zhangwei/zhangwei_workspace/data/300ROIs'


def sm_confidence(sm, parc_path, subj, n_iter, n_parc, sess):
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
    
    sm_dir = os.path.join(parc_path, subj, sess, f'Iter_{n_iter}_sm{sm}')
    if not os.path.exists(sm_dir): os.makedirs(sm_dir)
    for hemi in ['lh', 'rh']:
        for parc in range(n_parc):
            cmd = f'mri_surf2surf --srcsubject fsaverage6 \
            --sval {parc_path}/{subj}/{sess}/Iter_{n_iter}/{hemi}.NetworkConfidence_{parc+1}.mgh \
            --trgsubject fsaverage6 --tval {sm_dir}/{hemi}.NetworkConfidence_{parc+1}_fs6.mgh \
            --hemi {hemi} --nsmooth-in {sm} > /dev/null 2>&1'
            os.system(cmd)
    return sm_dir

def write_confidence_to_annot_Gordon17_with_SCAN(smed_path):
    """
    Writes confidence values to annotation files for Gordon17 with SCAN.

    Args:
        smed_path (str): The path to the directory containing the input files.

    Returns:
        None
    """
    ctb = np.array([[255, 255, 255, 0],
                    [255,   0,   0, 0],
                    [  0,   0, 153, 0],
                    [255, 255,   0, 0],
                    [255, 178, 102, 0],
                    [  0, 204,   0, 0],
                    [255, 153, 255, 0],
                    [  0, 153, 153, 0],
                    [  1,   1,   1, 0],
                    [ 76,   0, 153, 0],
                    [ 51, 255, 255, 0],
                    [255, 127,   0, 0],
                    [153,  51, 255, 0],
                    [  0,  51, 102, 0],
                    [ 51, 255,  51, 0],
                    [  0,   0, 255, 0],
                    [255, 255, 204, 0],
                    [  0, 102,   0, 0],
                    [128,   0,  76, 0]])
    
    parc = 18
    name = ['Other'] + [f'Net_{i+1}' for i in range(17)] + ['SCAN']
    conf_thr = 0.1
    conf_high_thr = 0.6
    
    for hemi in ['lh', 'rh']:
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

def write_confidence_to_annot_APP_18(smed_path):
    """
    Writes confidence values to annotation files for Gordon17 with SCAN.

    Args:
        smed_path (str): The path to the directory containing the input files.

    Returns:
        None
    """
    ctb = np.array([[  0,   0,   0, 255],
                    [120,  18, 134,   0],
                    [255,   0,   0,   0],
                    [ 70, 130, 180,   0],
                    [ 42, 204, 164,   0],
                    [ 74, 155,  60,   0],
                    [  0, 118,  14,   0],
                    [196,  58, 250,   0],
                    [255, 152, 213,   0],
                    [200, 248, 164,   0],
                    [122, 135,  50,   0],
                    [119, 140, 176,   0],
                    [230, 148,  34,   0],
                    [135,  50,  74,   0],
                    [ 12,  48, 255,   0],
                    [  0,   0, 130,   0],
                    [255, 255,   0,   0],
                    [205,  62,  78,   0],
                    [220, 180, 140,   0]])
    
    parc = 18
    name = ['Other'] + [f'Net_{i+1}' for i in range(17)] + ['SCAN']
    conf_thr = 0.1
    conf_high_thr = 0.6
    
    for hemi in ['lh', 'rh']:
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


def indi_Gordon17_with_SCAN_parc(wb_data, num_iter, conf_thr, output_path, subj_id, weight_group, sess, prior_variability=None, prior_SNR=None):
    # wb_data: 81924 x T
    if prior_variability == None:
        prior_variability = np.ones(81924)
    if prior_SNR == None:
        prior_SNR = np.ones(81924)
    
    parc_number = 18
    atlas_name = 'Gordon17_with_SCAN'
    # atlas_path = os.path.join(CUR_DIR, f'/home/ssli/liss_workspace/Downloads/{atlas_name}')
    
    # 获取当前文件所在的文件夹
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    atlas_path = os.path.join(CUR_DIR, f'resource/{atlas_name}')
    
    # prepare step.
    # Think about the priors, Could include variability and SNR, compuate inv of these.
    if max(prior_variability) - min(prior_variability) > 0:
        # normalize the range to 0.4 ~1. Therefore the inv will be between 1~2.5.
        prior_variability = 0.4 + 0.6*(prior_variability - min(prior_variability))/(max(prior_variability) - min(prior_variability))
    var_inv_wegiht = 1/prior_variability
    
    if max(prior_SNR) - min(prior_SNR) > 0:
        # normalize the range to 0.4 ~1. Therefore the inv will be between 1~2.5.
        prior_SNR = 0.4 + 0.6*(prior_SNR - min(prior_SNR))/(max(prior_SNR) - min(prior_SNR))
    snr_weight = prior_SNR
    
    # load uncared mask if u donnt wanna care about the mid wall. 
    # And the mid wall data will not be computed when calculating the FC.
    lh_mid_wall_mask = nib.load(os.path.join(atlas_path, f'lh.{atlas_name}_fs6_net000_fs6.mgh')).get_fdata().reshape(-1, order='F') > 0 
    rh_mid_wall_mask = nib.load(os.path.join(atlas_path, f'rh.{atlas_name}_fs6_net000_fs6.mgh')).get_fdata().reshape(-1, order='F') > 0
    mid_wall_mask = np.hstack([lh_mid_wall_mask, rh_mid_wall_mask])
    cortex_mask = np.logical_not(mid_wall_mask)
    cortex_label = np.where(cortex_mask)[0]
    
    parc_residual = np.zeros((parc_number, wb_data.shape[1]))
    atlas_mask = np.zeros((81924, parc_number), dtype=bool)
    for iter_i in range(num_iter):
        print(f'Processing Iter_{iter_i+1}...')
        iter_output_path = os.path.join(output_path, f'Iter_{iter_i+1}')
        if not os.path.exists(iter_output_path): os.makedirs(iter_output_path)
        if iter_i == 0:
            for net_j in range(parc_number):
                net_label = f'net{net_j+1:03d}'
                lh_vertex_mask = nib.load(os.path.join(atlas_path, f'lh.{atlas_name}_fs6_{net_label}_fs6.mgh')).get_fdata().reshape(-1, order='F') > 0
                rh_vertex_mask = nib.load(os.path.join(atlas_path, f'rh.{atlas_name}_fs6_{net_label}_fs6.mgh')).get_fdata().reshape(-1, order='F') > 0
                wb_mask = np.hstack([lh_vertex_mask, rh_vertex_mask])
                parc_residual_data = np.dot(var_inv_wegiht[wb_mask], wb_data[wb_mask, :])
                parc_residual[net_j, :] = parc_residual_data
                # save the atlas mask for the inter varaibility
                atlas_mask[:, net_j] = wb_mask
            atlas_parc_residual = parc_residual.copy()
        else:
            for net_j in range(parc_number):
                # get the seed waveforms based on the last parcellation
                lh_last_parc_conf = nib.load(os.path.join(output_path, f'Iter_{iter_i}/lh.NetworkConfidence_{net_j+1}.mgh')).get_fdata().reshape(-1, order='F')
                lh_vertex_mask = lh_last_parc_conf >= conf_thr
                if sum(lh_vertex_mask) == 0:
                    maxx = max(lh_last_parc_conf)
                    lh_vertex_mask = lh_last_parc_conf == maxx
                    print(f'HAAAAHAAAHAAA--Threshold TOOOOOO LARGE for lh.network{net_j+1}')
                
                rh_last_parc_conf = nib.load(os.path.join(output_path, f'Iter_{iter_i}/rh.NetworkConfidence_{net_j+1}.mgh')).get_fdata().reshape(-1, order='F')
                rh_vertex_mask = rh_last_parc_conf >= conf_thr
                if sum(rh_vertex_mask) == 0:
                    maxx = max(rh_last_parc_conf)
                    rh_vertex_mask = rh_last_parc_conf == maxx
                    print(f'HAAAAHAAAHAAA--Threshold TOOOOOO LARGE for rh.network{net_j+1}')
                
                wb_mask = np.hstack([lh_vertex_mask, rh_vertex_mask])
                parc_residual[net_j, :] = np.dot(snr_weight[wb_mask], wb_data[wb_mask, :])
        
        # Weight in the group seed in each iteration, 
        # should throw in individual variability map as weight in the future
        if weight_group:
            if iter_i > 0:
                parc_residual = parc_residual + atlas_parc_residual / iter_i;
        
        # compute vertex to seed correlation for all vertices
        corr_val = np.zeros((81924, parc_number)) # wb fc map to each parc
        for net_j in range(parc_number):
            # get the parc seed waveforms from the atlas or the last iteration
            tmp_residual = np.expand_dims(parc_residual[net_j, :], 0)
            corr_val[cortex_mask, net_j] = np.squeeze(cal_corr(tmp_residual, wb_data[cortex_mask, :]))
        corr_val[np.isnan(corr_val)] = 0
        
        # Further weight in the group map * inv(Variability) by adding
        # correlation coefficient of 0~ 0.5 according to inv(Variability).
        for net_j in range(parc_number):
            net_mask = atlas_mask[:, net_j]
            corr_val[net_mask, net_j] = corr_val[net_mask, net_j] + (((var_inv_wegiht[net_mask] - 1 )/3)/(iter_i + 1))
        
        # Determine the network membership of each vertex
        parc_membership = np.zeros(corr_val.shape[0], dtype=int)
        parc_confidence = np.zeros(corr_val.shape[0], dtype=float)
        
        for v in cortex_label:
            cor_idx = np.argsort(corr_val[v, :])[::-1]
            cor = corr_val[v, cor_idx]
            parc_membership[v] = cor_idx[0]
            
            if cor[0] > 0 and cor[1] <= 0:
                parc_confidence[v] = cor[0] / 0.000001
            else:
                parc_confidence[v] = cor[0] / cor[1]
        parc_membership[mid_wall_mask] = -1
        
        # write data to mgh file
        for net_j in range(parc_number):
            net_label = f'net{net_j+1:03d}'
            network = np.zeros(81924, dtype=int)
            confid = np.zeros(81924, dtype=float)
            network[parc_membership == net_j] = 1
            confid[parc_membership == net_j] = parc_confidence[parc_membership == net_j]
            
            lh_network = network[:40962]
            rh_network = network[40962:]
            nib.save(nib.MGHImage(lh_network.astype(np.float32), np.eye(4)), os.path.join(iter_output_path, f'lh.Network_{net_j+1}.mgh'))
            nib.save(nib.MGHImage(rh_network.astype(np.float32), np.eye(4)), os.path.join(iter_output_path, f'rh.Network_{net_j+1}.mgh'))
            lh_confid = np.multiply(confid[:40962], lh_network)
            rh_confid = np.multiply(confid[40962:], rh_network)
            nib.save(nib.MGHImage(lh_confid.astype(np.float32), np.eye(4)), os.path.join(iter_output_path, f'lh.NetworkConfidence_{net_j+1}.mgh'))
            nib.save(nib.MGHImage(rh_confid.astype(np.float32), np.eye(4)), os.path.join(iter_output_path, f'rh.NetworkConfidence_{net_j+1}.mgh'))

def indi_APP_18_parc(wb_data, num_iter, conf_thr, output_path, subj_id, weight_group, sess, prior_variability=None, prior_SNR=None):
    # wb_data: 81924 x T
    if prior_variability == None:
        prior_variability = np.ones(81924)
    if prior_SNR == None:
        prior_SNR = np.ones(81924)
    
    parc_number = 18
    atlas_name = 'APP_18'
    # atlas_path = os.path.join(CUR_DIR, f'/home/ssli/liss_workspace/Downloads/{atlas_name}')
    
    # 获取当前文件所在的文件夹
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    atlas_path = os.path.join(CUR_DIR, f'resource/{atlas_name}')
    
    # prepare step.
    # Think about the priors, Could include variability and SNR, compuate inv of these.
    if max(prior_variability) - min(prior_variability) > 0:
        # normalize the range to 0.4 ~1. Therefore the inv will be between 1~2.5.
        prior_variability = 0.4 + 0.6*(prior_variability - min(prior_variability))/(max(prior_variability) - min(prior_variability))
    var_inv_wegiht = 1/prior_variability
    
    if max(prior_SNR) - min(prior_SNR) > 0:
        # normalize the range to 0.4 ~1. Therefore the inv will be between 1~2.5.
        prior_SNR = 0.4 + 0.6*(prior_SNR - min(prior_SNR))/(max(prior_SNR) - min(prior_SNR))
    snr_weight = prior_SNR
    
    # load uncared mask if u donnt wanna care about the mid wall. 
    # And the mid wall data will not be computed when calculating the FC.
    lh_mid_wall_mask = nib.load(os.path.join(atlas_path, f'lh.{atlas_name}_fs6_net000_fs6.mgh')).get_fdata().reshape(-1, order='F') > 0 
    rh_mid_wall_mask = nib.load(os.path.join(atlas_path, f'rh.{atlas_name}_fs6_net000_fs6.mgh')).get_fdata().reshape(-1, order='F') > 0
    mid_wall_mask = np.hstack([lh_mid_wall_mask, rh_mid_wall_mask])
    cortex_mask = np.logical_not(mid_wall_mask)
    cortex_label = np.where(cortex_mask)[0]
    
    parc_residual = np.zeros((parc_number, wb_data.shape[1]))
    atlas_mask = np.zeros((81924, parc_number), dtype=bool)
    for iter_i in range(num_iter):
        print(f'Processing Iter_{iter_i+1}...')
        iter_output_path = os.path.join(output_path, f'Iter_{iter_i+1}')
        if not os.path.exists(iter_output_path): os.makedirs(iter_output_path)
        if iter_i == 0:
            for net_j in range(parc_number):
                net_label = f'net{net_j+1:03d}'
                lh_vertex_mask = nib.load(os.path.join(atlas_path, f'lh.{atlas_name}_fs6_{net_label}_fs6.mgh')).get_fdata().reshape(-1, order='F') > 0
                rh_vertex_mask = nib.load(os.path.join(atlas_path, f'rh.{atlas_name}_fs6_{net_label}_fs6.mgh')).get_fdata().reshape(-1, order='F') > 0
                wb_mask = np.hstack([lh_vertex_mask, rh_vertex_mask])
                parc_residual_data = np.dot(var_inv_wegiht[wb_mask], wb_data[wb_mask, :])
                parc_residual[net_j, :] = parc_residual_data
                # save the atlas mask for the inter varaibility
                atlas_mask[:, net_j] = wb_mask
            atlas_parc_residual = parc_residual.copy()
        else:
            for net_j in range(parc_number):
                # get the seed waveforms based on the last parcellation
                lh_last_parc_conf = nib.load(os.path.join(output_path, f'Iter_{iter_i}/lh.NetworkConfidence_{net_j+1}.mgh')).get_fdata().reshape(-1, order='F')
                lh_vertex_mask = lh_last_parc_conf >= conf_thr
                if sum(lh_vertex_mask) == 0:
                    maxx = max(lh_last_parc_conf)
                    lh_vertex_mask = lh_last_parc_conf == maxx
                    print(f'HAAAAHAAAHAAA--Threshold TOOOOOO LARGE for lh.network{net_j+1}')
                
                rh_last_parc_conf = nib.load(os.path.join(output_path, f'Iter_{iter_i}/rh.NetworkConfidence_{net_j+1}.mgh')).get_fdata().reshape(-1, order='F')
                rh_vertex_mask = rh_last_parc_conf >= conf_thr
                if sum(rh_vertex_mask) == 0:
                    maxx = max(rh_last_parc_conf)
                    rh_vertex_mask = rh_last_parc_conf == maxx
                    print(f'HAAAAHAAAHAAA--Threshold TOOOOOO LARGE for rh.network{net_j+1}')
                
                wb_mask = np.hstack([lh_vertex_mask, rh_vertex_mask])
                parc_residual[net_j, :] = np.dot(snr_weight[wb_mask], wb_data[wb_mask, :])
        
        # Weight in the group seed in each iteration, 
        # should throw in individual variability map as weight in the future
        if weight_group:
            if iter_i > 0:
                parc_residual = parc_residual + atlas_parc_residual / iter_i;
        
        # compute vertex to seed correlation for all vertices
        corr_val = np.zeros((81924, parc_number)) # wb fc map to each parc
        for net_j in range(parc_number):
            # get the parc seed waveforms from the atlas or the last iteration
            tmp_residual = np.expand_dims(parc_residual[net_j, :], 0)
            corr_val[cortex_mask, net_j] = np.squeeze(cal_corr(tmp_residual, wb_data[cortex_mask, :]))
        corr_val[np.isnan(corr_val)] = 0
        
        # Further weight in the group map * inv(Variability) by adding
        # correlation coefficient of 0~ 0.5 according to inv(Variability).
        for net_j in range(parc_number):
            net_mask = atlas_mask[:, net_j]
            corr_val[net_mask, net_j] = corr_val[net_mask, net_j] + (((var_inv_wegiht[net_mask] - 1 )/3)/(iter_i + 1))
        
        # Determine the network membership of each vertex
        parc_membership = np.zeros(corr_val.shape[0], dtype=int)
        parc_confidence = np.zeros(corr_val.shape[0], dtype=float)
        
        for v in cortex_label:
            cor_idx = np.argsort(corr_val[v, :])[::-1]
            cor = corr_val[v, cor_idx]
            parc_membership[v] = cor_idx[0]
            
            if cor[0] > 0 and cor[1] <= 0:
                parc_confidence[v] = cor[0] / 0.000001
            else:
                parc_confidence[v] = cor[0] / cor[1]
        parc_membership[mid_wall_mask] = -1
        
        # write data to mgh file
        for net_j in range(parc_number):
            net_label = f'net{net_j+1:03d}'
            network = np.zeros(81924, dtype=int)
            confid = np.zeros(81924, dtype=float)
            network[parc_membership == net_j] = 1
            confid[parc_membership == net_j] = parc_confidence[parc_membership == net_j]
            
            lh_network = network[:40962]
            rh_network = network[40962:]
            nib.save(nib.MGHImage(lh_network.astype(np.float32), np.eye(4)), os.path.join(iter_output_path, f'lh.Network_{net_j+1}.mgh'))
            nib.save(nib.MGHImage(rh_network.astype(np.float32), np.eye(4)), os.path.join(iter_output_path, f'rh.Network_{net_j+1}.mgh'))
            lh_confid = np.multiply(confid[:40962], lh_network)
            rh_confid = np.multiply(confid[40962:], rh_network)
            nib.save(nib.MGHImage(lh_confid.astype(np.float32), np.eye(4)), os.path.join(iter_output_path, f'lh.NetworkConfidence_{net_j+1}.mgh'))
            nib.save(nib.MGHImage(rh_confid.astype(np.float32), np.eye(4)), os.path.join(iter_output_path, f'rh.NetworkConfidence_{net_j+1}.mgh'))
