import os
import argparse
import bids
import numpy as np
import nibabel as nib


def mp2rage_denoise(inv1_file, inv2_file, uni_file, out_file, reg):
    # loading data
    inv1_data = nib.load(inv1_file).get_fdata(dtype=np.float32)
    inv2_data = nib.load(inv2_file).get_fdata(dtype=np.float32)
    uni_img = nib.load(uni_file)
    uni_data = uni_img.get_fdata(dtype=np.float32)

    # convert uni_data to -0.5 --- 0.5 scale
    uni_convert = np.zeros_like(uni_data)
    integer_format = False
    if np.min(uni_data) >= 0 and np.max(uni_data) >= 0.51:
        uni_convert = (uni_data - np.max(uni_data) / 2) / np.max(uni_data)
        integer_format = True

    # Give the correct polarity to INV1
    inv1_data = np.sign(uni_convert) * inv1_data
    uni_convert_denominator = uni_convert.copy()
    uni_convert_denominator[uni_convert_denominator == 0] = np.inf
    inv1_pos = (-1 * inv2_data + np.sqrt(np.square(inv2_data) - 4 * uni_convert *
                                         (np.square(inv2_data) * uni_convert))) / (-2 * uni_convert_denominator)
    inv1_pos[np.isnan(inv1_pos)] = 0

    inv1_neg = (-1 * inv2_data - np.sqrt(np.square(inv2_data) - 4 * uni_convert *
                                         (np.square(inv2_data) * uni_convert))) / (-2 * uni_convert_denominator)
    inv1_neg[np.isnan(inv1_neg)] = 0

    inv1_pos_index = np.abs(inv1_data - inv1_pos) > np.abs(inv1_data - inv1_neg)
    inv1_data[inv1_pos_index] = inv1_neg[inv1_pos_index]

    inv1_neg_index = np.abs(inv1_data - inv1_pos) <= np.abs(inv1_data - inv1_neg)
    inv1_data[inv1_neg_index] = inv1_pos[inv1_neg_index]

    # lambda calculation
    noise_level = reg * np.mean(inv2_data[-10:, :, -10:])
    t1w = (np.conj(inv1_data) * inv2_data - np.square(noise_level)) / \
          (np.square(inv1_data) + np.square(inv2_data) + 2 * np.square(noise_level))
    # convert the final image to uint (if necessary)
    if integer_format:
        t1w = np.round(4095 * (t1w + 0.5))

    t1w_img = nib.Nifti1Image(t1w, uni_img.affine, uni_img.header)
    nib.save(t1w_img, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: denoise mp2rage data"
    )
    parser.add_argument("--bids-dir", help="directory of BIDS type: /mnt/ngshare2/BIDS/MSC", required=True)
    parser.add_argument('--subject-ids', type=str, nargs='+', default=[], help='specified subject_id, space-separated list')
    parser.add_argument('--save-dir', help='save directory of denoise mp2rage data')
    parser.add_argument('--denoise-factor', type=int, default=5, help='factor to denoise mp2rage')
    args = parser.parse_args()

    if len(args.subject_ids) != 0:
        subject_ids = [subject_id[4:] if subject_id.startswith('sub-') else subject_id for subject_id in args.subject_ids]
    else:
        subject_ids = args.subject_ids
    layout = bids.BIDSLayout(args.bids_dir, derivatives=False)
    subject_dict = {}
    for t1w_file in layout.get(return_type='filename', subject=subject_ids, suffix="T1w", extension=['.nii.gz', '.nii']):
        sub_info = layout.parse_file_entities(t1w_file)
        subject_id = f"sub-{sub_info['subject']}"
        subject_dict.setdefault(subject_id, []).append(t1w_file)

    for subj in subject_dict.keys():
        for t1w_file in subject_dict[subj]:
            inv1_file = t1w_file.replace("T1w", "INV1")
            inv2_file = t1w_file.replace("T1w", "INV2")
            if os.path.exists(inv1_file) and os.path.exists(inv2_file):
                denoise_file = os.path.join(args.save_dir, os.path.basename(t1w_file).replace(".nii.gz", "_denoise.nii.gz"))
                mp2rage_denoise(inv1_file, inv2_file, t1w_file, denoise_file, args.denoise_factor)
