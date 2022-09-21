from nipype.interfaces.base import BaseInterface, \
    BaseInterfaceInputSpec, traits, File, TraitedSpec, Directory, Str
import sys
import sh
import nibabel as nib
import numpy as np


class BoldSkipReorientInputSpec(BaseInterfaceInputSpec):
    preprocess_dir = Directory(exists=True, desc="preprocess dir", mandatory=True)
    subject_id = Str(desc="subject id", mandatory=True)
    bold = File(exists=True, desc="'bold'/run/f'{subj}_bld{run}_rest.nii.gz'", mandatory=True)

    skip_bold = File(exists=False, desc="'bold'/run/f'{subj}_bld{run}_rest_skip.nii.gz'", mandatory=True)
    reorient_skip_bold = File(exists=False, desc="'bold'/run/f'{subj}_bld{run}_rest_reorient_skip.nii.gz'",
                              mandatory=True)


class BoldSkipReorientOutputSpec(TraitedSpec):
    skip_bold = File(exists=True, desc="'bold'/run/f'{subj}_bld{run}_rest_skip.nii.gz'")
    reorient_skip_bold = File(exists=True, desc="'bold'/run/f'{subj}_bld{run}_rest_reorient_skip.nii.gz'")


class BoldSkipReorient(BaseInterface):
    input_spec = BoldSkipReorientInputSpec
    output_spec = BoldSkipReorientOutputSpec

    time = 4.3 / 60  # 运行时间：分钟 / 单run测试时间
    cpu = 2.5  # 最大cpu占用：个
    gpu = 0  # 最大gpu占用：MB

    def dimstr2dimno(self, dimstr):
        if 'x' in dimstr:
            return 0

        if 'y' in dimstr:
            return 1

        if 'z' in dimstr:
            return 2

    def swapdim(self, infile, a, b, c, outfile):
        '''
        infile  - str. Path to file to read and swap dimensions of.
        a       - str. New x dimension.
        b       - str. New y dimension.
        c       - str. New z dimension.
        outfile - str. Path to file to create.

        Returns None.
        '''

        # Read original file.
        img = nib.load(infile)

        # Build orientation matrix.
        ornt = np.zeros((3, 2))
        order_strs = [a, b, c]
        dim_order = list(map(self.dimstr2dimno, order_strs))
        i_dim = np.argsort(dim_order)
        for i, dim in enumerate(i_dim):
            ornt[i, 1] = -1 if '-' in order_strs[dim] else 1

        ornt[:, 0] = i_dim

        # Transform and save.
        newimg = img.as_reoriented(ornt)
        nib.save(newimg, outfile)

    def _run_interface(self, runtime):
        # skip 0 frame
        sh.mri_convert('-i', self.inputs.bold, '-o', self.inputs.skip_bold, _out=sys.stdout)

        # reorient
        self.swapdim(str(self.inputs.skip_bold), 'x', '-y', 'z', str(self.inputs.reorient_skip_bold))

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["skip_bold"] = self.inputs.skip_bold
        outputs["reorient_skip_bold"] = self.inputs.reorient_skip_bold

        return outputs