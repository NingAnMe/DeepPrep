#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------
# @Author : Ning An        @Email : Ning An <ninganme0317@gmail.com>

import argparse
import subprocess
import os
import sys
from typing import List


def setup_argument_parser():
    """Setup command line argument parser following BIDS-APP specification"""
    parser = argparse.ArgumentParser(
        description='DeepPrep BIDS App - Neuroimaging preprocessing and target planning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/bids /path/to/output participant --participant_label 001 002  --target "Post-stroke motor
  %(prog)s /path/to/bids /path/to/output participant --target "Post-stroke motor"
        """
    )
    
    # BIDS-APP required positional arguments
    parser.add_argument(
        'bids_dir',
        help='The directory with the input dataset formatted according to the BIDS standard.'
    )
    
    parser.add_argument(
        'output_dir',
        help='The directory where the output files should be stored.'
    )
    
    parser.add_argument(
        'analysis_level',
        choices=['participant', 'group'],
        help='Level of the analysis that will be performed. Multiple participant level analyses can be run independently (in parallel) using the same output_dir.'
    )
    
    # BIDS-APP standard optional arguments
    parser.add_argument(
        '--participant_label',
        nargs='+',
        help='The label(s) of the participant(s) that should be analyzed. The label corresponds to sub-<participant_label> from the BIDS spec (so it does not include "sub-"). If this parameter is not provided all subjects should be analyzed.'
    )
    
    parser.add_argument(
        '--skip_bids_validation',
        action='store_true',
        default=False,
        help='Whether or not to perform BIDS dataset validation'
    )
    
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='DeepPrep BIDS App v1.0.0',
        help='Show program version number and exit'
    )
    
    # Core processing parameters
    parser.add_argument(
        '--fs_license_file',
        default='/opt/freesurfer/license.txt',
        help='Path to FreeSurfer license file'
    )
    
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', '0', '1', '2', '3', '4'],
        default='auto',
        help='Device for computation. Can be "auto", "cpu", or GPU device ID (0, 1, 2, etc.)'
    )
    
    parser.add_argument(
        '--cpus',
        type=int,
        help='Number of CPUs/cores available to use'
    )
    
    parser.add_argument(
        '--memory',
        type=int,
        help='Amount of RAM available to use (in GB)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='Resume incomplete workflows'
    )
    
    # Processing stage selection
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['preprocess', 'postprocess', 'target'],
        default=['preprocess', 'postprocess', 'target'],
        help='Processing stages to run (default: all)'
    )
    
    # Core BOLD processing parameters
    parser.add_argument(
        '--bold_skip_frame',
        default='4',
        help='Number of initial BOLD frames to skip (default: 4)'
    )
    
    parser.add_argument(
        '--bandpass',
        default='0.01-0.08',
        help='Bandpass filter range in Hz (default: 0.01-0.08)'
    )
    
    parser.add_argument(
        '--fwhm',
        type=int,
        default=6,
        help='Smoothing kernel FWHM in mm (default: 6)'
    )
    
    parser.add_argument(
        '--confounds_index_file',
        default='/opt/DeepPrep/deepprep/rest/denoise/12motion_6param_10bCompCor.txt',
        help='Path to confounds index file for regression (default: /opt/DeepPrep/deepprep/rest/denoise/12motion_6param_10bCompCor.txt)'
    )
    
    # Target planning parameters
    parser.add_argument(
        '--target',
        choices=['Post-stroke aphasia', 'Post-stroke motor', 'Post-stroke cognition', 'PD-tremor', 'custom'],
        help='Target type for planning (default: Post-stroke aphasia)',
        required=True
    )
    
    parser.add_argument(
        '--custom_target_script',
        help='Path to custom target script (required when target is "custom")'
    )
    
    return parser


def validate_inputs(args) -> bool:
    """Validate input parameters according to BIDS-APP standards"""
    
    # Validate BIDS directory
    if not os.path.exists(args.bids_dir):
        print(f"Error: BIDS directory does not exist: {args.bids_dir}")
        return False
    
    if not os.path.isabs(args.bids_dir):
        print("Error: BIDS directory must be an absolute path")
        return False
    
    # Validate output directory path
    if not os.path.isabs(args.output_dir):
        print("Error: Output directory must be an absolute path")
        return False
    
    # Ensure output directory is different from BIDS directory
    if os.path.abspath(args.output_dir) == os.path.abspath(args.bids_dir):
        print("Error: Output directory must be different from BIDS directory")
        return False
    
    # Validate FreeSurfer license
    if not os.path.exists(args.fs_license_file):
        print(f"Error: FreeSurfer license file does not exist: {args.fs_license_file}")
        return False
    
    # Validate confounds index file
    if not os.path.exists(args.confounds_index_file):
        print(f"Error: Confounds index file does not exist: {args.confounds_index_file}")
        return False
    
    # Validate custom target script if needed
    if args.target == 'custom':
        if not args.custom_target_script:
            print("Error: Custom target script path must be provided when target is 'custom'")
            return False
        if not os.path.exists(args.custom_target_script):
            print(f"Error: Custom target script does not exist: {args.custom_target_script}")
            return False
    
    # Validate participant labels format
    if args.participant_label:
        for label in args.participant_label:
            if label.startswith('sub-'):
                print(f"Error: participant_label should not include 'sub-' prefix: {label}")
                return False
    
    return True


def get_target_script_path(target: str, custom_script: str = None) -> str:
    """Get target analysis script path"""
    script_mapping = {
        'Post-stroke aphasia': '/opt/DeepPrep/deepprep/TargetAutoPlaning/Aphasia_auto_point.py',
        'Post-stroke motor': '/opt/DeepPrep/deepprep/TargetAutoPlaning/Motor_auto_point.py',
        'Post-stroke cognition': '/opt/DeepPrep/deepprep/TargetAutoPlaning/Cognition_auto_point.py',
        'PD-tremor': '/opt/DeepPrep/deepprep/TargetAutoPlaning/SCAN_VIM_auto_point.py',
        'custom': custom_script
    }
    return script_mapping[target]


def build_participant_label_string(participant_labels: List[str]) -> str:
    """Build participant label string compatible with preprocess.sh"""
    if not participant_labels:
        return ""
    
    # Ensure labels don't have 'sub-' prefix
    clean_labels = [label.replace('sub-', '') for label in participant_labels]
    return ' '.join(clean_labels)


def build_device_argument(device: str) -> str:
    """Build device argument compatible with preprocess.sh"""
    if device == 'auto':
        return 'auto'
    elif device == 'cpu':
        return 'cpu'
    else:
        # GPU device ID
        return device


def run_command(cmd: str, step_name: str) -> bool:
    """Execute command with real-time output"""
    print("=" * 60)
    print(f"Executing {step_name}...")
    print(f"Command: {cmd}")
    print("=" * 60)
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Real-time output
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        process.wait()
        
        if process.returncode != 0:
            print(f"Error: {step_name} failed with return code: {process.returncode}")
            return False
        
        print(f"{step_name} completed successfully!")
        print()
        return True
        
    except Exception as e:
        print(f"Error: Exception occurred while executing {step_name}: {e}")
        return False


def build_commands(args):
    """Build processing commands"""
    
    # Prepare directories
    preprocess_dir = os.path.join(args.output_dir, 'derivatives', 'deepprep')
    postprocess_dir = os.path.join(args.output_dir, 'derivatives', 'deepprep-postprocess')  
    target_dir = os.path.join(args.output_dir, 'derivatives', 'deepprep-target')
    
    # Build command arguments
    common_args = []
    
    # Participant label handling
    if args.participant_label:
        participant_str = build_participant_label_string(args.participant_label)
        common_args.append(f"--participant_label '{participant_str}'")
    
    # Device specification
    device_arg = build_device_argument(args.device)
    common_args.append(f"--device {device_arg}")
    
    # FreeSurfer license
    common_args.append(f"--fs_license_file {args.fs_license_file}")
    
    # Processing options
    if args.skip_bids_validation:
        common_args.append("--skip_bids_validation")
    
    if args.resume:
        common_args.append("--resume")
    
    if args.cpus:
        common_args.append(f"--cpus {args.cpus}")
    
    if args.memory:
        common_args.append(f"--memory {args.memory}")
    
    common_args_str = ' '.join(common_args)
    
    commands = {}
    
    # Preprocessing command
    preprocess_args = [
        "--bold_task_type 'rest'",
        "--bold_surface_spaces 'fsaverage6'",
        "--bold_volume_space None",
        f"--bold_skip_frame {args.bold_skip_frame}",
        f"--bandpass {args.bandpass}",
        "--bold_confounds"
    ]
    
    commands['preprocess'] = (
        f"/opt/DeepPrep/deepprep/preprocess.sh {args.bids_dir} {preprocess_dir} {args.analysis_level} "
        f"{common_args_str} {' '.join(preprocess_args)}"
    )
    
    # Postprocessing command  
    postprocess_args = [
        "--task_id 'rest'",
        "--space 'fsaverage6'",
        f"--confounds_index_file {args.confounds_index_file}",
        f"--skip_frame {args.bold_skip_frame}",
        f"--surface_fwhm {args.fwhm}",
        f"--volume_fwhm {args.fwhm}",
        f"--bandpass {args.bandpass}"
    ]
    
    preprocess_bold_dir = os.path.join(preprocess_dir, 'BOLD')
    
    commands['postprocess'] = (
        f"/opt/DeepPrep/deepprep/web/pages/postprocess.sh {preprocess_bold_dir} {postprocess_dir} {args.analysis_level} "
        f"{common_args_str} {' '.join(postprocess_args)}"
    )
    
    # Target analysis command
    if args.analysis_level == 'participant':
        target_script = get_target_script_path(args.target, args.custom_target_script)
        postprocess_bold_dir = os.path.join(postprocess_dir, 'BOLD')
        preprocess_recon_dir = os.path.join(preprocess_dir, 'Recon')
        
        commands['target'] = (
            f"/opt/conda/envs/deepprep/bin/python {target_script} "
            f"--data_path {postprocess_bold_dir} --output_path {target_dir} "
            f"--reconall_dir {preprocess_recon_dir} --FREESURFER_HOME /opt/freesurfer"
        )
    
    return commands


def main():
    """Main function"""
    
    # Handle GUI mode when no arguments provided
    if len(sys.argv) == 1:
        default_cmd = "/opt/DeepPrep/deepprep/preprocess.sh"
        if not run_command(default_cmd, "UNITE GUI"):
            sys.exit(1)
        return
    
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    print("UNITE BIDS App")
    print("=" * 50)
    
    # Validate inputs
    if not validate_inputs(args):
        sys.exit(1)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build processing commands
    commands = build_commands(args)
    
    # Display configuration
    print("\nConfiguration:")
    print(f"  BIDS directory: {args.bids_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Analysis level: {args.analysis_level}")
    print(f"  Device: {args.device}")
    if args.participant_label:
        print(f"  Participants: {args.participant_label}")
    print(f"  Processing stages: {', '.join(args.stages)}")
    if 'target' in args.stages:
        print(f"  Target analysis: {args.target}")
    if 'postprocess' in args.stages:
        print(f"  Confounds index file: {args.confounds_index_file}")
    print()
    
    # Execute processing stages
    success = True
    stage_names = {
        'preprocess': 'Preprocessing',
        'postprocess': 'Postprocessing', 
        'target': 'Target Analysis'
    }
    
    for stage in args.stages:
        if stage in commands:
            if not run_command(commands[stage], stage_names[stage]):
                success = False
                break
        else:
            print(f"Warning: Unknown stage '{stage}'")
    
    if success:
        print("=" * 60)
        print("All processing stages completed successfully!")
        print(f"Output directory: {args.output_dir}")
        print("=" * 60)
    else:
        print("Error: Processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 