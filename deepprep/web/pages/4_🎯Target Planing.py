#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------
# @Author : Ning An        @Email : Ning An <ninganme0317@gmail.com>

import streamlit as st
import subprocess
import os

st.markdown(f'# 🎯Target Planing')
st.markdown(
    """
The goal of Target Planing stage is to identify functional circuit-guided neuromodulation targets based on personalized functional parcellation and RSFC. 
The platform supports fully automated planning of both cortical and subcortical targets, adaptable to different clinical requirements and neuromodulation techniques, such as TMS, FUS, and DBS. 
We currently provide four built-in target planning algorithms, including three cortical TMS targets for post-stroke rehabilitation, targeting personalized language-, motor-, and cognitive impairment-related networks, and one subcortical FUS target for tremor relief in PD. 
In addition, the platform is extendable to user-defined target planning algorithms, allowing flexibility for other functional network targets and diseases. 
This  workflow takes the directory of the dataset to be processed as input, which is required to be in a valid BIDS format.

-----------------
"""
)

device = st.radio("select a device: ", ("auto", "GPU", "CPU"), horizontal=True,
                  help="Specifies the device. The default is auto, which automatically selects a device.")

st.write(f"Preprocess Target ", f"on the '{device}' device.")

commond_error = False

bids_dir = st.text_input("BIDS Path:",
                         help="refers to the directory of the input dataset, which is required to be in BIDS format.")
if not bids_dir:
    st.error("The BIDS Path must be input!")
    commond_error = True
elif not bids_dir.startswith('/'):
    st.error("The path must be an absolute path starts with '/'.")
    commond_error = True
elif not os.path.exists(bids_dir):
    st.error("The BIDS Path does not exist!")
    commond_error = True

output_dir = st.text_input("Output Path:", help="refers to the directory to save the DeepPrep outputs.")
if not output_dir:
    st.error("The Output Path must be input!")
    commond_error = True
elif not output_dir.startswith('/'):
    st.error("The path must be an absolute path starts with '/'.")
    commond_error = True
elif output_dir == bids_dir:
    st.error("The Output Path must be different from the BIDS Path!")
    commond_error = True

freesurfer_license_file = st.text_input("FreeSurfer license file path", value='/opt/freesurfer/license.txt',
                                        help="FreeSurfer license file path. It is highly recommended to replace the license.txt path with your own FreeSurfer license! You can get it for free from https://surfer.nmr.mgh.harvard.edu/registration.html")
if not freesurfer_license_file.startswith('/'):
    st.error("The path must be an absolute path starts with '/'.")
    commond_error = True
elif not os.path.exists(freesurfer_license_file):
    st.error("The FreeSurfer license file Path does not exist!")
    commond_error = True

participant_label = st.text_input("the subject IDs (optional)", placeholder="sub-001 sub-002 sub-003",
                                  help="Identify the subjects you'd like to process by their IDs, i.e. sub-001 sub-002 sub-003.")
if participant_label:
    participant_label.replace("'", "")
    participant_label.replace('"', "")
    participant_label_cmd = f" --participant_label '{participant_label}'"
else:
    participant_label_cmd = ""

bold_skip_frame = st.text_input("skip n frames of BOLD data", value="4",
                                help="skip n frames of BOLD fMRI; the default is `2`.")
bold_bandpass = st.text_input("Bandpass filter", value="0.01-0.08", help="the default range is `0.01-0.08`.")
bold_fwhm = st.text_input("fwhm", value="6", help="smooth by fwhm mm; the default is `6`.")
confounds_file = st.text_input("Confounds File Path",
                               value='/opt/DeepPrep/deepprep/rest/denoise/12motion_6param_10bCompCor.txt',
                               help="The path to the text file that contains all the confound names needed for regression.")
if not confounds_file.startswith('/'):
    st.error("The path must be an absolute path that starts with '/'.")
    commond_error = True
elif not os.path.exists(confounds_file):
    st.error("The Confounds File Path does not exist!")
    commond_error = True

target = st.radio("select a target: ",
                  ("Post-stroke cognition", "Post-stroke aphasia", "Post-stroke motor", "PD-tremor", "custom"),
                  horizontal=True, help="")

if target == "Post-stroke aphasia":
    script_name = st.text_input("Target script name",
                                value='/opt/DeepPrep/deepprep/TargetAutoPlaning/Aphasia_auto_point.py', help="",
                                disabled=True)
elif target == "Post-stroke motor":
    script_name = st.text_input("Target script name",
                                value='/opt/DeepPrep/deepprep/TargetAutoPlaning/Motor_auto_point.py', help="",
                                disabled=True)
elif target == "Post-stroke cognition":
    script_name = st.text_input("Target script name",
                                value='/opt/DeepPrep/deepprep/TargetAutoPlaning/Cognition_auto_point.py', help="",
                                disabled=True)
elif target == "PD-tremor":
    script_name = st.text_input("Target script name",
                                value='/opt/DeepPrep/deepprep/TargetAutoPlaning/SCAN_VIM_auto_point.py', help="",
                                disabled=True)
elif target == "custom":
    script_name = st.text_input("Target script name", placeholder='/absolute_path_to_custom_targets_python_script.py',
                                help="-v /path/to/TargetAutoPlaning:/opt/DeepPrep/deepprep/TargetAutoPlaning",
                                disabled=False)

if device == "GPU":
    device_cmd = f' --device GPU'
elif device == "CPU":
    device_cmd = f' --device CPU'
else:
    device_cmd = f' --device auto'

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
        if output == "" and process.poll() is not None:
            break
        if output:
            yield output + '\n'

    stderr = process.stderr.read()
    if stderr:
        yield stderr + '\n'

    process.wait()

def run_command_with_display(cmd, max_lines=50):
    """
    Run command and display latest output in a scrolling manner
    max_lines: Maximum number of lines to display, default 50 lines
    """
    import time
    
    # Create an empty container for displaying output
    output_container = st.empty()
    output_lines = []
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Redirect stderr to stdout
        text=True,
        shell=True,
        bufsize=1,
        universal_newlines=True
    )
    
    while True:
        line = process.stdout.readline()
        if line == "" and process.poll() is not None:
            break
        if line:
            output_lines.append(line.rstrip())
            
            # Keep only the latest max_lines
            if len(output_lines) > max_lines:
                output_lines = output_lines[-max_lines:]
            
            # Update display content
            display_text = '\n'.join(output_lines)
            output_container.code(display_text, language='bash')
            
            # Brief delay to avoid excessive updates
            time.sleep(0.1)
    
    # Final check for stderr
    stderr = process.stderr
    if stderr:
        remaining_stderr = stderr.read()
        if remaining_stderr:
            output_lines.append(f"ERROR: {remaining_stderr}")
            if len(output_lines) > max_lines:
                output_lines = output_lines[-max_lines:]
            display_text = '\n'.join(output_lines)
            output_container.code(display_text, language='bash')
    
    process.wait()
    
    # If process returns non-zero code, display error message
    if process.returncode != 0:
        output_lines.append(f"Process exit code: {process.returncode}")
        display_text = '\n'.join(output_lines)
        output_container.code(display_text, language='bash')
        st.error(f"Command execution failed, exit code: {process.returncode}")
    else:
        st.success("Command executed successfully!")
    
    return process.returncode == 0

# output
preprocess_dir = os.path.join(output_dir, 'Preprocess')
preprocess_cmd = f"{bids_dir} {preprocess_dir} participant {device_cmd} --fs_license_file {freesurfer_license_file} {participant_label_cmd} --bold_task_type 'rest' --bold_surface_spaces 'fsaverage6' --bold_volume_space None --skip_frame {bold_skip_frame} --bandpass {bold_bandpass} --bold_confounds --skip_bids_validation --resume"

# input
preprocess_bold_dir = os.path.join(preprocess_dir, 'BOLD')
# output
postprocess_dir = os.path.join(output_dir, 'Postprocess')
postprocess_cmd = f"{preprocess_bold_dir} {postprocess_dir} participant --fs_license_file {freesurfer_license_file} --task_id 'rest' --space 'fsaverage6' --confounds_index_file {confounds_file} --skip_frame {bold_skip_frame} --surface_fwhm {bold_fwhm} --volume_fwhm {bold_fwhm} --bandpass {bold_bandpass} --skip_bids_validation --resume"

# input
postprocess_bold_dir = os.path.join(postprocess_dir, 'BOLD')
preprocess_recon_dir = os.path.join(preprocess_dir, 'Recon')
# output
target_dir = os.path.join(output_dir, 'Target')
target_cmd = f"--data_path {postprocess_bold_dir} --output_path {target_dir} --reconall_dir {preprocess_recon_dir} --FREESURFER_HOME /opt/freesurfer"

# Added: Multi-select steps and output settings
with st.expander("------------ custom steps ------------"):
    steps_to_run = st.multiselect(
        'Please select steps to run',
        ['preprocess', 'postprocess', 'target'],
        default=['preprocess', 'postprocess', 'target'],
        help='Multiple selection allowed, execute selected steps')

    # Output display settings
    st.subheader("Output Display Settings")
    max_display_lines = 200

    if 'preprocess' in steps_to_run:
        st.write(f"/opt/DeepPrep/deepprep/preprocess.sh {preprocess_cmd}")
    if 'postprocess' in steps_to_run:
        st.write(f"/opt/DeepPrep/deepprep/web/pages/postprocess.sh {postprocess_cmd}")
    if 'target' in steps_to_run:
        st.write(f"/opt/conda/envs/deepprep/bin/python {script_name} {target_cmd}")

if st.button("Run", disabled=commond_error):
    with st.spinner('Waiting for the process to finish, please do not leave this page...'):
        if 'preprocess' in steps_to_run:
            preprocess_command = f"/opt/DeepPrep/deepprep/preprocess.sh {preprocess_cmd}"
            with st.expander("------------ preprocessing log ------------"):
                run_command_with_display(preprocess_command, max_display_lines)
        if 'postprocess' in steps_to_run:
            postprocess_command = f"/opt/DeepPrep/deepprep/web/pages/postprocess.sh {postprocess_cmd}"
            with st.expander("------------ postprocessing log ------------"):
                run_command_with_display(postprocess_command, max_display_lines)
        if 'target' in steps_to_run:
            target_command = f"/opt/conda/envs/deepprep/bin/python {script_name} {target_cmd}"
            target_qc_command = f"/opt/conda/envs/deepprep/bin/python /opt/conda/envs/deepprep/target/target_qc_html.py --input_dir {target_dir} --output_dir {target_dir} --output_name Target_Planing_Report.html"
            with st.expander("------------ target log ------------"):
                run_command_with_display(target_command, max_display_lines)
                run_command_with_display(target_qc_command, max_display_lines)
        import time
        time.sleep(2)
    st.success("Done!")
