import os
import numpy as np
import torch
import torch.utils.data as data
import h5py


class TReNDSfMRIDataset(data.Dataset):
    def __init__(self, dir: str):
        super(TReNDSfMRIDataset, self).__init__()
        self.dir = dir
        self.files = os.listdir(dir)

    def __getitem__(self, index):
        path = os.path.join(self.dir, self.files[index])
        # reference: https://www.kaggle.com/mks2192/reading-matlab-mat-files-and-eda
        f = h5py.File(path, 'r')
        data = f['SM_feature']
        array = data[()]
        return array

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    import nilearn as nl
    import nilearn.plotting as nlplt

    ds = TReNDSfMRIDataset('E:/trends-fmri/fMRI_test')
    print(ds[0].shape)

    base_path = 'E:\\trends-fmri'
    mask_filename = os.path.join(base_path, 'fMRI_mask.nii')
    smri_filename = os.path.join(base_path, 'ch2better.nii')
    subject_filename = os.path.join(base_path, 'fMRI_test/10228.mat')
    mask_niimg = nl.image.load_img(mask_filename)

    def load_subject(filename, mask_niimg):
        """
        Load a subject saved in .mat format with
            the version 7.3 flag. Return the subject
            niimg, using a mask niimg as a template
            for nifti headers.

        Args:
            filename    <str>            the .mat filename for the subject data
            mask_niimg  niimg object     the mask niimg object used for nifti headers
        """
        subject_data = None
        with h5py.File(subject_filename, 'r') as f:
            subject_data = f['SM_feature'][()]
        # It's necessary to reorient the axes, since h5py flips axis order
        subject_data = np.moveaxis(subject_data, [0, 1, 2, 3], [3, 2, 1, 0])
        subject_niimg = nl.image.new_img_like(
            mask_niimg, subject_data, affine=mask_niimg.affine, copy_header=True)
        return subject_niimg

    subject_niimg = load_subject(subject_filename, mask_niimg)
    print("Image shape is %s" % (str(subject_niimg.shape)))
    num_components = subject_niimg.shape[-1]
    print("Detected {num_components} spatial maps".format(
        num_components=num_components))
    nlplt.plot_prob_atlas(subject_niimg,
                          bg_img=smri_filename,
                          view_type='filled_contours',
                          draw_cross=False,
                          title='All %d spatial maps' % num_components,
                          threshold='auto')
    nlplt.show()
