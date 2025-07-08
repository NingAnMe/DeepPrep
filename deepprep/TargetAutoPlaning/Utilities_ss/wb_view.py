import os
import numpy as np
import nibabel as nib
from pathlib import Path
from PIL import Image


def exec_cmd(cmd, pf=True):
    if pf:
        print(cmd)
    os.system(cmd)


def convert_black_to_transparent(input_image_path, output_image_path):
    image = Image.open(input_image_path)
    
    image_data = image.getdata()
    
    new_image_data = []
    for item in image_data:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            new_image_data.append((0, 0, 0, 0))  # 设置透明色
        else:
            new_image_data.append(item)  
    
    new_image = Image.new("RGBA", image.size)
    new_image.putdata(new_image_data)
    
    new_image.save(output_image_path)


def wb_view_self_gemo(lh_ge, rh_ge, sc_path, lh_overlay, rh_overlay, png_name, sence_file=None, cashe=False):
    '''
    Description: plot self geometry and overlay data on the surface
    
    lh_ge: str, path to the left hemisphere, such as lh.pial or lh.sphere\n
    rh_ge: str, path to the right hemisphere, such as rh.pial or rh.sphere\n
    sc_path: str, path to the output scene directory\n
    lh_overlay: np.ndarray, left hemisphere overlay data, shape should be (n_vertices,)\n
    rh_overlay: np.ndarray, right hemisphere overlay data, shape should be (n_vertices,)\n
    png_name: str, name of the output PNG file\n
    sence_file: str, path to the scene file, if None, use default surf_fc_np_0106.scene\n
    cashe: bool, if True, keep temporary files, default is False
    '''
    pwd = os.getcwd()
    cur_dir = Path(__file__).parents[0]
    cmd = f'mris_convert {lh_ge} {sc_path}/lh.surf.gii > /dev/null 2>&1'
    exec_cmd(cmd, False)
    cmd = f'mris_convert {rh_ge} {sc_path}/rh.surf.gii > /dev/null 2>&1'
    exec_cmd(cmd, False)
    lh_gii = nib.GiftiImage()
    lh_gii.add_gifti_data_array(nib.gifti.GiftiDataArray(data=lh_overlay.astype(np.float32)))
    nib.save(lh_gii, sc_path / 'lh_tmp.func.gii')
    rh_gii = nib.GiftiImage()
    rh_gii.add_gifti_data_array(nib.gifti.GiftiDataArray(data=rh_overlay.astype(np.float32)))
    nib.save(rh_gii, sc_path / 'rh_tmp.func.gii')
    
    if sence_file is None:
        cmd = f'cp -r {cur_dir}/wb_view_resource/sc/surf_fc_np_0106.scene {sc_path}/sc.sence'
        exec_cmd(cmd, False)
    else:
        cmd = f'cp -r {sence_file} {sc_path}/sc.sence'
        exec_cmd(cmd, False)
    
    os.chdir(sc_path)
    black_img = 'tmp.png'
    cmd = f'wb_command -show-scene sc.sence 1 {black_img} 1200 800 > /dev/null 2>&1'
    exec_cmd(cmd, False)
    convert_black_to_transparent(black_img, png_name)
    if not cashe:
        cmd = f'rm -rf tmp.png *h.surf.gii *h_tmp.func.gii sc.sence'
        exec_cmd(cmd, False)
    os.chdir(pwd)


def wb_view_label_gii(sub:str, lh_label_gii:str, rh_label_gii:str, png_name, sc_path, mesh:str='pial', sence_file=None, cashe:bool=False):
    '''
    Description: plot label.gii

    sub: str, fsaverage4 or fsaverage6\n
    lh_label_gii: str, path to the left hemisphere label.gii file\n
    rh_label_gii: str, path to the right hemisphere label.gii file\n
    png_name: str, name of the output PNG file\n
    sc_path: str, path to the output scene directory\n
    mesh: str, mesh type, default is 'pial'\n
    sence_file: str, path to the scene file, if None, use default scene\n
    cashe: bool, if True, keep temporary files, default is False
    '''
    sc_path =Path(sc_path)
    if not Path(sc_path).exists():
        sc_path.mkdir(exist_ok=True, parents=True)
    cur_dir = Path(__file__).parents[0]
    # cur_dir = Path('/home/zhangwei/zhangwei_workspace/code/Parcellation_for_DP/SCAN_parcellation_code/plot')
    cmd = f'cp -r {cur_dir}/wb_view_resource/{sub}/surf/lh.{mesh}.surf.gii {sc_path}/lh.surf.gii'
    exec_cmd(cmd, False)
    cmd = f'cp -r {cur_dir}/wb_view_resource/{sub}/surf/rh.{mesh}.surf.gii {sc_path}/rh.surf.gii'
    exec_cmd(cmd, False)
    if not (sc_path / 'lh_tmp.label.gii').exists():        
        cmd = f'cp -r {lh_label_gii} {sc_path}/lh_tmp.label.gii'
        exec_cmd(cmd, False)
        cmd = f'cp -r {rh_label_gii} {sc_path}/rh_tmp.label.gii'
        exec_cmd(cmd, False)
    
    if sence_file is None:
        cmd = f'cp -r {cur_dir}/wb_view_resource/sc/label_gii.scene {sc_path}/sc.sence'
        exec_cmd(cmd, False)
    else:
        cmd = f'cp -r {sence_file} {sc_path}/sc.sence'
        exec_cmd(cmd, False)
    
    pwd = os.getcwd()
    os.chdir(sc_path)
    
    black_img = 'tmp.png'
    cmd = f'wb_command -show-scene sc.sence 1 {black_img} 1200 800 > /dev/null 2>&1'
    # cmd = f'wb_command -show-scene sc.sence 1 {black_img} 1200 800'
    exec_cmd(cmd, False)

    convert_black_to_transparent(black_img, png_name)
    if not cashe:
        cmd = f'rm -rf tmp.png *h.surf.gii *h_tmp.label.gii sc.sence'
        exec_cmd(cmd, False)
    os.chdir(pwd)

def wb_view_annot(sub:str, lh_annot:str, rh_annot:str, png_name, sc_path, mesh:str='pial', sence_file=None, cashe:bool=False):
    '''
    Description: convert the annot file to label.gii and plot it
    
    sub: str, fsaverage4 or fsaverage6\n
    lh_annot: str, path to the left hemisphere annot file\n
    rh_annot: str, path to the right hemisphere annot file\n
    png_name: str, name of the output PNG file\n
    sc_path: str, path to the output scene directory\n
    mesh: str, mesh type, default is 'pial'\n
    sence_file: str, path to the scene file, if None, use default scene\n
    cashe: bool, if True, keep temporary files, default is False
    '''
    cur_dir = Path(__file__).parents[0]
    # cur_dir = Path('/home/zhangwei/zhangwei_workspace/code/Parcellation_for_DP/SCAN_parcellation_code/plot')
    if not str(lh_annot).endswith('.annot') or not str(rh_annot).endswith('.annot'):
        raise Exception('Annot error.')
    
    sc_path =Path(sc_path)
    if not Path(sc_path).exists():
        sc_path.mkdir(exist_ok=True, parents=True)
    lh_label_gii = sc_path / 'lh_tmp.label.gii'
    rh_label_gii = sc_path / 'rh_tmp.label.gii'
    cmd = f'mris_convert --annot {lh_annot} {cur_dir}/wb_view_resource/{sub}/surf/lh.pial {lh_label_gii} > /dev/null 2>&1'
    exec_cmd(cmd, False)
    cmd = f'mris_convert --annot {rh_annot} {cur_dir}/wb_view_resource/{sub}/surf/rh.pial {rh_label_gii} > /dev/null 2>&1'
    exec_cmd(cmd, False)
    wb_view_label_gii(sub, lh_label_gii, rh_label_gii, png_name, sc_path, mesh, sence_file, cashe)

def wb_view_fc(sub:str, lh_data, rh_data, png_name, sc_path, mesh:str='pial', sence_file=None, cashe:bool=False):
    '''
    Description: plot functional connectivity data on the surface
    
    sub: str, fsaverage4 or fsaverage6\n
    lh_data: np.ndarray, left hemisphere data, shape should be (n_vertices,)\n
    rh_data: np.ndarray, right hemisphere data, shape should be (n_vertices,)\n
    png_name: str, name of the output PNG file\n
    sc_path: str, path to the output scene directory\n
    mesh: str, mesh type, default is 'pial'\n
    sence_file: str, path to the scene file, if None, use default surf_fc_np_0106.scene\n
    cashe: bool, if True, keep temporary files, default is False
    '''
    cur_dir = Path(__file__).parents[0]
    sc_path =Path(sc_path)
    if not Path(sc_path).exists():
        sc_path.mkdir(exist_ok=True, parents=True)
    cmd = f'cp -r {cur_dir}/wb_view_resource/{sub}/surf/lh.{mesh}.surf.gii {sc_path}/lh.surf.gii'
    exec_cmd(cmd, False)
    cmd = f'cp -r {cur_dir}/wb_view_resource/{sub}/surf/rh.{mesh}.surf.gii {sc_path}/rh.surf.gii'
    exec_cmd(cmd, False)
    if sence_file is None:
        cmd = f'cp -r {cur_dir}/wb_view_resource/sc/surf_fc_np_0106.scene {sc_path}/sc.sence'
        exec_cmd(cmd, False)
    else:
        cmd = f'cp -r {sence_file} {sc_path}/sc.sence'
        exec_cmd(cmd, False)
    
    pwd = os.getcwd()
    os.chdir(sc_path)
    
    lh_gii = nib.GiftiImage()
    lh_gii.add_gifti_data_array(nib.gifti.GiftiDataArray(data=lh_data.astype(np.float32)))
    nib.save(lh_gii, sc_path / 'lh_tmp.func.gii')
    rh_gii = nib.GiftiImage()
    rh_gii.add_gifti_data_array(nib.gifti.GiftiDataArray(data=rh_data.astype(np.float32)))
    nib.save(rh_gii, sc_path / 'rh_tmp.func.gii')
    
    black_img = 'tmp.png'
    cmd = f'wb_command -show-scene sc.sence 1 {black_img} 1200 800 > /dev/null 2>&1'
    exec_cmd(cmd, False)
    convert_black_to_transparent(black_img, png_name)
    if not cashe:
        cmd = f'rm -rf tmp.png *h.surf.gii *h_tmp.func.gii sc.sence'
        exec_cmd(cmd, False)
    os.chdir(pwd)