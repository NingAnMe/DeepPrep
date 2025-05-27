#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------
# @Author : Ning An        @Email : Ning An <ninganme0317@gmail.com>

import streamlit as st
import subprocess
import os

st.markdown(f'# 🚀Target Auto Planing')
st.markdown(
    """
DeepPrep is a preprocessing pipeline that can flexibly handle anatomical and functional MRI data end-to-end.
It accommodates various sizes, from a single participant to LARGE-scale datasets, achieving a 10-fold acceleration compared to the state-of-the-art pipeline.

Both the anatomical and functional parts can be run separately. However, preprocessed Recon is a mandatory prerequisite for executing the functional process.

The DeepPrep workflow takes the directory of the dataset to be processed as input, which is required to be in a valid BIDS format.

-----------------
"""
)

device = st.radio("select a device: ", ("auto", "GPU", "CPU"), horizontal=True, help="Specifies the device. The default is auto, which automatically selects a device.")

st.write(f"Preprocess Target ", f"on the '{device}' device.")

deepprep_cmd = ''
docker_cmd = 'docker run -it --rm'
commond_error = False

bids_dir = st.text_input("BIDS Path:", help="refers to the directory of the input dataset, which is required to be in BIDS format.")
if not bids_dir:
    st.error("The BIDS Path must be input!")
    deepprep_cmd += ' {bids_dir}'
    commond_error = True
elif not bids_dir.startswith('/'):
    st.error("The path must be an absolute path starts with '/'.")
    deepprep_cmd += ' {bids_dir}'
    commond_error = True
elif not os.path.exists(bids_dir):
    st.error("The BIDS Path does not exist!")
    deepprep_cmd += ' {bids_dir}'
    commond_error = True
else:
    deepprep_cmd += f' {bids_dir}'

output_dir = st.text_input("Output Path:", help="refers to the directory to save the DeepPrep outputs.")
if not output_dir:
    st.error("The Output Path must be input!")
    deepprep_cmd += ' {output_dir}'
    commond_error = True
elif not output_dir.startswith('/'):
    st.error("The path must be an absolute path starts with '/'.")
    deepprep_cmd += ' {output_dir}'
    commond_error = True
elif output_dir == bids_dir:
    st.error("The Output Path must be different from the BIDS Path!")
    deepprep_cmd += ' {output_dir}'
    commond_error = True

freesurfer_license_file = st.text_input("FreeSurfer license file path", value='/opt/freesurfer/license.txt', help="FreeSurfer license file path. It is highly recommended to replace the license.txt path with your own FreeSurfer license! You can get it for free from https://surfer.nmr.mgh.harvard.edu/registration.html")
if not freesurfer_license_file.startswith('/'):
    st.error("The path must be an absolute path starts with '/'.")
    commond_error = True
elif not os.path.exists(freesurfer_license_file):
    st.error("The FreeSurfer license file Path does not exist!")
    commond_error = True

participant_label = st.text_input("the subject IDs (optional)", placeholder="sub-001,sub-002",
                                  help="Identify the subjects you'd like to process by their IDs, i.e. sub-001,sub-002.")
if participant_label:
    participant_label.replace("'", "")
    participant_label.replace('"', "")
    deepprep_cmd += f" --participant_label '{participant_label}'"

st.write(f'{deepprep_cmd}')

preprocess_dir = os.path.join(output_dir, 'Preprocess')
preprocess_cmd = f"{bids_dir} {output_dir} participant --fs_license_file {freesurfer_license_file} --bold_task_type 'rest' --bold_surface_spaces 'fsaverage6' --bold_volume_space None --bold_skip_frame 2 --bold_bandpass 0.01-0.08 --bold_confounds --device auto --skip_bids_validation --resume"
st.write(preprocess_cmd)

preprocess_bold_dir = os.path.join(preprocess_dir, 'BOLD')
postprocess_dir = os.path.join(output_dir, 'Postprocess')
postprocess_cmd = f"{preprocess_bold_dir} {postprocess_dir} participant --fs_license_file {freesurfer_license_file} --task_id 'rest' --space 'fsaverage6' --confounds_index_file /opt/DeepPrep/deepprep/rest/denoise/12motion_6param_10bCompCor.txt --skip_frame 2 --surface_fwhm 6 --bandpass 0.01-0.08 --skip_bids_validation --resume"
st.write(postprocess_cmd)

postprocess_bold_dir = os.path.join(postprocess_dir, 'BOLD')
target_auto_planing_cmd = f"--data_path /opt/project/PD_AutoTestData/Postprocess_BOLD --output_path /opt/project/PD_AutoTestData/Target --reconall_dir /opt/project/PD_AutoTestData/Recon --FREESURFER_HOME /opt/freesurfer"
st.write(target_auto_planing_cmd)

if device == "GPU":
    deepprep_cmd += f' --device GPU'
    docker_cmd += ' --gpus all'
elif device == "CPU":
    deepprep_cmd += f' --device CPU'
else:
    deepprep_cmd += f' --device auto'

def run_command(cmd):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )

    while True:
        output = process.stdout.readline()
        print(output)
        if output == "" and process.poll() is not None:
            break
        if output:
            yield output + '\n'

    process.wait()

st.write(f'-----------  ------------')
if st.button("Run", disabled=commond_error):
    with st.spinner('Waiting for the process to finish, please do not leave this page...'):
        command = [f"/opt/DeepPrep/deepprep/deepprep.sh {deepprep_cmd}"]
        with st.expander("------------ running log ------------"):
            st.write_stream(run_command(command))
        import time
        time.sleep(2)
    st.success("Done!")
