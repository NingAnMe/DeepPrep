import torch
import pytorch3d.loss

from nibabel.freesurfer import read_geometry

def chamfer_distance(surf1, surf2, device='cpu'):

    points1, _ = read_geometry(surf1)
    points2, _ = read_geometry(surf2)

    # 转为torch tensor
    points1 = torch.tensor(points1.astype('float32'), dtype=torch.float32).unsqueeze(0)
    points2 = torch.tensor(points2.astype('float32'), dtype=torch.float32).unsqueeze(0)

    points1 = points1.to(device)
    points2 = points2.to(device)

    # 计算 Chamfer 距离
    chamfer_loss, _ = pytorch3d.loss.chamfer_distance(
        x=points1,
        y=points2,
        batch_reduction='mean',   # 对整个batch求平均
        point_reduction='mean',   # 对每个点云内部求平均
        norm=2,                   # 使用L2距离
    )

    print(f"Chamfer Distance: {chamfer_loss:.4f}")
    return chamfer_loss.cpu().item()

if __name__ == '__main__':

    surf1 = '/mnt/ngshare/Data_User/lishenshen/BadAnatParcellation_Data/Freesurfer_results/Recon/HNPD040/surf/lh.white'
    surf2 = '/mnt/ngshare/Data_User/lishenshen/BadAnatParcellation_Data/DeepPrep_Preprocess/Recon/sub-040/surf/lh.white'
    chamfer_distance(surf1, surf2, device='cuda')
    surf1 = '/mnt/ngshare/Data_User/lishenshen/BadAnatParcellation_Data/Freesurfer_results/Recon/HNPD040/surf/rh.white'
    surf2 = '/mnt/ngshare/Data_User/lishenshen/BadAnatParcellation_Data/DeepPrep_Preprocess/Recon/sub-040/surf/rh.white'
    chamfer_distance(surf1, surf2, device='cuda')
    surf1 = '/mnt/ngshare/Data_User/lishenshen/BadAnatParcellation_Data/Freesurfer_results/Recon/HNPD060/surf/lh.white'
    surf2 = '/mnt/ngshare/Data_User/lishenshen/BadAnatParcellation_Data/DeepPrep_Preprocess/Recon/sub-060/surf/lh.white'
    chamfer_distance(surf1, surf2, device='cuda')
    surf1 = '/mnt/ngshare/Data_User/lishenshen/BadAnatParcellation_Data/Freesurfer_results/Recon/HNPD060/surf/rh.white'
    surf2 = '/mnt/ngshare/Data_User/lishenshen/BadAnatParcellation_Data/DeepPrep_Preprocess/Recon/sub-060/surf/rh.white'
    chamfer_distance(surf1, surf2, device='cuda')