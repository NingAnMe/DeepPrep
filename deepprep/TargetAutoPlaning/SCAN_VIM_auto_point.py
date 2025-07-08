import os
from TargetAutoPlaning import VIM_SCAN_TargetPlanner, parc213_DeepPrep, parcellation_18
from glob import glob
import pandas as pd

import argparse
import os

import json

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process the paths for data, output, and reconall directories.")
    
    # 设定4个参数，并从环境变量中获取默认值
    parser.add_argument('--data_path', type=str, default=os.environ.get('processed_data_dir'),
                        help="Path to the processed data directory, eg: postprocessing-DeepPrep2510/BOLD (default from 'processed_data_dir' environment variable).")
    
    parser.add_argument('--output_path', type=str, default=os.environ.get('auto_SCAN_dir'),
                        help="Path to the output directory (default from 'auto_SCAN_dir' environment variable).")
    
    parser.add_argument('--reconall_dir', type=str, default=os.environ.get('recon_all_dir'),
                        help="Path to the reconall directory, eg: preprocessing-DeepPrep2510/Recon (default from 'recon_all_dir' environment variable).")
    parser.add_argument('--sulc_percentile', type=int, default=80,
                        help="A integer between 0 and 100, bigger value means higher target. Default is 80.")
    parser.add_argument('--subject_list', type=str, default='',
                        help="List of subjects to process, eg: 'subject1,subject2', if not provided, all subjects unber data_path will be processed.")
    parser.add_argument('--FREESURFER_HOME', type=str, default=os.environ.get('FREESURFER_HOME'),
                        help="Path to the FREESURFER_HOME directory, eg: /usr/local/freesurfer (default from 'FREESURFER_HOME' environment variable).")
    # 解析命令行参数
    args = parser.parse_args()
    if args.data_path is None or args.output_path is None or args.reconall_dir is None:
        raise ValueError("Please provide data_path, output_path, and reconall_dir. Use --help for more information.")
    
    return args

def merge_messages(msgs):
    """
    将 msgs 转换为一个字符串:
      - 如果 msgs 是列表, 使用空格拼接所有字符串元素；
      - 如果 msgs 是字符串，直接返回；
      - 如果 msgs 是空列表或 None，返回空字符串；
    """
    # 如果是 None 或空列表，直接返回空字符串
    if not msgs:
        return ""
    
    # 如果是列表，用 ' '.join 拼接
    if isinstance(msgs, list):
        return " ".join(msgs)
    
    # 如果是字符串，直接返回
    if isinstance(msgs, str):
        return msgs
    
    # 如果是其他类型，可根据需求决定如何处理，这里抛异常
    raise TypeError("msgs 必须是列表、字符串或 None。")

def write_json(subect, lh_indi_target, rh_indi_target, lh_group_target, rh_group_target, warning,filename):
    # 将所有 numpy 数据类型转换为原生 Python 数据类型，以便 json 序列化
    # 生成标准JSON结构
    json_structure = {
        "name": subect,
        "lh Indi Target":int(lh_indi_target[0]['index']),
        "rh Indi Target":int(rh_indi_target[0]['index']),
        "lh Group Target":int(lh_group_target),
        "rh Group Target":int(rh_group_target),
        "lh Indi Target Score":float(lh_indi_target[0]['score']),
        "rh Indi Target Score":float(rh_indi_target[0]['score']),
        "Warnings":warning
    }

    # 写入JSON文件
    with open(filename, 'w') as f:
        json.dump(json_structure, f, indent=2)

    # print("JSON文件已生成：target_output.json")

def set_environ(recnall_dir, freesurfer_home):
    # FreeSurfer
    value = None
    if value is None:
        os.environ['FREESURFER_HOME'] = freesurfer_home
        # os.environ['SUBJECTS_DIR'] = '/usr/local/freesurfer/subjects'
        os.environ['SUBJECTS_DIR'] = recnall_dir
        os.environ['PATH'] = f'{freesurfer_home}/bin:' + os.environ['PATH']

def SCAN_target_auto_planing_batch(data_path, output_path, reconall_dir, subject_list=None, sulc_percentile=80, freesurfer_home=None):

    # set_environ(reconall_dir, freesurfer_home)

    if subject_list is None or subject_list == '':
        subject_list = os.listdir(data_path)
    else:
        subject_list = subject_list.split(',')

    print(f"Data Path: {data_path}")
    # subject_list = ['PDHN013']  ## 特殊处理
    ## 设置保存结果的dataframe
    target_records = pd.DataFrame(columns=['subject', 'run', 'lh_SCAN', 'rh_SCAN', 'warning'])
    scores_records = pd.DataFrame(columns=['subject', 'run', 'lh_SCAN', 'rh_SCAN'])
    group_records = pd.DataFrame(columns=['subject', 'run', 'lh_SCAN', 'rh_SCAN'])
    ## 只保留文件夹
    subject_list = [sub for sub in subject_list if os.path.isdir(os.path.join(data_path, sub))]
    print(f"Subject List: {subject_list}")

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    use_group = False
    for subject in subject_list:
        run_list = os.listdir(os.path.join(data_path, subject))
        if run_list[0] == 'func':
            run = 'temp'
            work_dir_run = os.path.join(output_path, subject, run)
            os.makedirs(work_dir_run, exist_ok=True)
            ## 检查是否存在surf文件
            lh_surf_file = os.path.join(data_path, subject, 'func/*hemi-L_space-fsaverage6_desc-fwhm_bold.nii.gz')
            if len(glob(lh_surf_file)) == 0:
                print(f'{subject} does not have bold data.')
                continue
            if not os.path.exists(os.path.join(reconall_dir, subject)):
                print(f'{subject} does not exist in {reconall_dir}')
                continue
            
            if not use_group:
                ## 进行18分区
                # output_path_split = output_path.split('/')
                # output_path_parcellation18 = '/'.join(output_path_split[:-1])
                output_path_parcellation18 = output_path
                return_value = parcellation_18(data_path, output_path_parcellation18, subject, run)
                if return_value == 0:
                    break
                ## 将18分区的结果复制到work_dir_run
                cmd = f'cp {os.path.join(output_path_parcellation18, subject, run, "Parcellation18/iter10_c25_w1", "Iter_10_sm4/lh.parc_result.annot")} {work_dir_run}/lh_parc18_fs6_surf.annot'
                os.system(cmd)
                cmd = f'cp {os.path.join(output_path_parcellation18, subject, run, "Parcellation18/iter10_c25_w1", "Iter_10_sm4/rh.parc_result.annot")} {work_dir_run}/rh_parc18_fs6_surf.annot'
                os.system(cmd)

                # ## 进行213parcel的计算
                # parc213_DeepPrep(subject, data_path, reconall_dir, os.path.join(output_path_parcellation18, 'Parcellation213'), run)
                # ## 将分区结果复制到work_dir_run
                # cmd = f'cp {os.path.join(output_path, "Parcellation213/Parc213/WB_lh", subject, f"lh.{subject}_IndiCluster108_fs6.mgh")} {work_dir_run}/lh_parc213_fs6_surf.mgh'
                # os.system(cmd)
                # cmd = f'cp {os.path.join(output_path, "Parcellation213/Parc213/WB_rh", subject, f"rh.{subject}_IndiCluster108_fs6.mgh")} {work_dir_run}/rh_parc213_fs6_surf.mgh'
                # os.system(cmd)

            ## 进行计算
            planner = VIM_SCAN_TargetPlanner(verbose=False, workdir=work_dir_run)
            DeepPrep_postprocess_data_path = os.path.join(data_path, subject)
            planner.plan(DeepPrep_postprocess_data_path, subject, use_group_target=use_group, sulc_percentile=sulc_percentile)
            if planner.warning_info == []:
                planner.warning_info = 'None'
            print(f'{subject} {run}:')
            if not use_group:
                print(f'    lh SCAN: {planner.lh_targets}')
                print(f'    rh SCAN: {planner.rh_targets}')
                ## 保存结果，只保留第一个靶点
                target_temp = {'subject': subject, 'run': run, 'lh_SCAN': [planner.lh_targets[0]['index']], 'rh_SCAN': [planner.rh_targets[0]['index']], 'warning': merge_messages(planner.warning_info)}
                # print(target_temp)
                target_records = pd.concat([target_records, pd.DataFrame(target_temp, index=[0])], ignore_index=True)
                # score_temp = {'subject': subject, 'run': run, 'lh_SCAN': planner.lh_targets[0]['score'], 'rh_SCAN': planner.rh_targets[0]['score']}
                # scores_records = pd.concat([scores_records, pd.DataFrame(score_temp, index=[0])], ignore_index=True)
            else:
                ## create lh targets and rh targets for json
                planner.lh_targets = [{'index': '0', 'score': '0'}]
                planner.rh_targets = [{'index': '0', 'score': '0'}]
            # group_temp = {'subject': subject, 'run': run, 'lh_SCAN': planner.lh_group_target, 'rh_SCAN': planner.rh_group_target}
            # group_records = pd.concat([group_records, pd.DataFrame(group_temp, index=[0])], ignore_index=True)
            # print(f'    lh group SCAN: {planner.lh_group_target}')
            # print(f'    rh group SCAN: {planner.rh_group_target}')
            # out_json = os.path.join(output_path, subject, 'SCAN_targets_auto.json')
            # write_json(subject, planner.lh_targets, planner.rh_targets, planner.lh_group_target, planner.rh_group_target, planner.warning_info, out_json)

            target_info_dict = planner.target_info_dict
            print(target_info_dict)
            out_json = os.path.join(output_path, subject, 'TARGET_info.json')
            with open(out_json, 'w') as f:
                json.dump(target_info_dict, f, indent=2)

            with open(out_json, 'r') as f:
                    json_data = json.load(f)
            ## 添加图像信息到out_json，添加在后面
            figures_dict = planner.figures_dict
            if figures_dict is not None:   
                json_data.update(figures_dict)
            
            Warning_dict = {
                "Warnings": merge_messages(planner.warning_info)
            }
            json_data.update(Warning_dict)

            ## 保存更新后的json文件
            with open(out_json, 'w') as f:
                json.dump(json_data, f, indent=2)

        else:
            ## 从run_list 中排除anat和figures目录
            run_list = [run for run in run_list if os.path.isdir(os.path.join(data_path, subject, run)) and 'anat' not in run and 'figures' not in run]
            ## 只保留第一个run
            if len(run_list) == 0:
                print(f'{subject} does not have any valid runs.')
                continue
            run_list = [run_list[0]]  # 只处理第一个run
            
            for run in run_list:
                if 'anat' in run or 'figures' in run:
                    continue
                work_dir_run = os.path.join(output_path, subject, run)
                os.makedirs(work_dir_run, exist_ok=True)
                ## 检查是否存在surf文件
                lh_surf_file = os.path.join(data_path, subject, run, 'func/*hemi-L_space-fsaverage6_desc-fwhm_bold.nii.gz')
                if len(glob(lh_surf_file)) == 0:
                    print(f'{subject} {run} does not have bold data.')
                    continue
                if not os.path.exists(os.path.join(reconall_dir, subject)):
                    print(f'{subject} does not exist in {reconall_dir}')
                    continue

                if not use_group:
                    ## 进行18分区
                    # output_path_split = output_path.split('/')
                    # output_path_parcellation18 = '/'.join(output_path_split[:-1])
                    output_path_parcellation18 = output_path
                    return_value = parcellation_18(data_path, output_path_parcellation18, subject, run)
                    if return_value == 0:
                        break
                    ## 将18分区的结果复制到work_dir_run
                    cmd = f'cp {os.path.join(output_path_parcellation18, subject, run, "Parcellation18/iter10_c25_w1", "Iter_10_sm4/lh.parc_result.annot")} {work_dir_run}/lh_parc18_fs6_surf.annot'
                    os.system(cmd)
                    cmd = f'cp {os.path.join(output_path_parcellation18, subject, run, "Parcellation18/iter10_c25_w1", "Iter_10_sm4/rh.parc_result.annot")} {work_dir_run}/rh_parc18_fs6_surf.annot'
                    os.system(cmd)

                    # ## 进行213parcel的计算
                    # parc213_DeepPrep(subject, data_path, reconall_dir, os.path.join(output_path_parcellation18, 'Parcellation213'), run)
                    # ## 将分区结果复制到work_dir_run
                    # cmd = f'cp {os.path.join(output_path_parcellation18, "Parcellation213/Parc213/WB_lh", subject, f"lh.{subject}_IndiCluster108_fs6.mgh")} {work_dir_run}/lh_parc213_fs6_surf.mgh'
                    # os.system(cmd)
                    # cmd = f'cp {os.path.join(output_path_parcellation18, "Parcellation213/Parc213/WB_rh", subject, f"rh.{subject}_IndiCluster108_fs6.mgh")} {work_dir_run}/rh_parc213_fs6_surf.mgh'
                    # os.system(cmd)

                ## 进行计算
                planner = VIM_SCAN_TargetPlanner(verbose=False, workdir=work_dir_run)
                DeepPrep_postprocess_data_path = os.path.join(data_path, subject, run)
                planner.plan(DeepPrep_postprocess_data_path, subject, use_group_target=use_group, sulc_percentile=sulc_percentile)
                if planner.warning_info == []:
                    planner.warning_info = 'None'
                print(f'{subject} {run}:')
                if not use_group:
                    print(f'    lh SCAN: {planner.lh_targets}')
                    print(f'    rh SCAN: {planner.rh_targets}')
                    ## 保存结果，只保留第一个靶点
                    target_temp = {'subject': subject, 'run': run, 'lh_SCAN': [planner.lh_targets[0]['index']], 'rh_SCAN': [planner.rh_targets[0]['index']], 'warning': merge_messages(planner.warning_info)}
                    # print(target_temp)
                    target_records = pd.concat([target_records, pd.DataFrame(target_temp, index=[0])], ignore_index=True)
                    # score_temp = {'subject': subject, 'run': run, 'lh_SCAN': planner.lh_targets[0]['score'], 'rh_SCAN': planner.rh_targets[0]['score']}
                    # scores_records = pd.concat([scores_records, pd.DataFrame(score_temp, index=[0])], ignore_index=True)
                else:
                    ## create lh targets and rh targets for json
                    planner.lh_targets = [{'index': '0', 'score': '0'}]
                    planner.rh_targets = [{'index': '0', 'score': '0'}]
                # group_temp = {'subject': subject, 'run': run, 'lh_SCAN': planner.lh_group_target, 'rh_SCAN': planner.rh_group_target}
                # group_records = pd.concat([group_records, pd.DataFrame(group_temp, index=[0])], ignore_index=True)
                # print(f'    lh group SCAN: {planner.lh_group_target}')
                # print(f'    rh group SCAN: {planner.rh_group_target}')
                # out_json = os.path.join(output_path, subject, 'SCAN_targets_auto.json')
                # write_json(subject, planner.lh_targets, planner.rh_targets, planner.lh_group_target, planner.rh_group_target, planner.warning_info, out_json)
                target_info_dict = planner.target_info_dict
                print(target_info_dict)
                out_json = os.path.join(output_path, subject, 'TARGET_info.json')
                with open(out_json, 'w') as f:
                    json.dump(target_info_dict, f, indent=2)

                with open(out_json, 'r') as f:
                        json_data = json.load(f)
                ## 添加图像信息到out_json，添加在后面
                figures_dict = planner.figures_dict
                if figures_dict is not None:   
                    json_data.update(figures_dict)
                
                Warning_dict = {
                    "Warnings": merge_messages(planner.warning_info)
                }
                json_data.update(Warning_dict)

                ## 保存更新后的json文件
                with open(out_json, 'w') as f:
                    json.dump(json_data, f, indent=2)
        # break # for test
    ## 保存结果
    target_records.to_csv(os.path.join(output_path, 'SCAN_targets_auto.csv'), index=False)
    # scores_records.to_csv(os.path.join(output_path, 'SCAN_scores_auto.csv'), index=False)
    # group_records.to_csv(os.path.join(output_path, 'SCAN_group_auto.csv'), index=False)
            
if __name__ == '__main__':
    args = parse_arguments()
    print(f"Data Path: {args.data_path}")
    print(f"Output Path: {args.output_path}")
    print(f"Reconall Directory: {args.reconall_dir}")
    print(f"FREESURFER_HOME: {args.FREESURFER_HOME}")
    if args.subject_list != '':
        print(f"Subject List: {args.subject_list.split(',')}")

    set_environ(args.reconall_dir, args.FREESURFER_HOME)

    SCAN_target_auto_planing_batch(args.data_path, args.output_path, args.reconall_dir, args.subject_list, args.sulc_percentile, args.FREESURFER_HOME)
