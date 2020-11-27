import os
import numpy as np
import torch
import torch.utils.data as data
import h5py
import nilearn as nl


def load_subject(filename: str,
                 mask_niimg):
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
    with h5py.File(filename, 'r') as f:
        subject_data = f['SM_feature'][()]
    # It's necessary to reorient the axes, since h5py flips axis order
    subject_data = np.moveaxis(subject_data, [0, 1, 2, 3], [3, 2, 1, 0])
    subject_niimg = nl.image.new_img_like(
        mask_niimg, subject_data, affine=mask_niimg.affine, copy_header=True)
    return subject_niimg


class TReNDSfMRIDataset(data.Dataset):
    def __init__(self,
                 dir: str,
                 mask_path: str):
        super(TReNDSfMRIDataset, self).__init__()
        self.dir = dir
        self.files = os.listdir(dir)
        self.mask = nl.image.load_img(mask_path)

    def __getitem__(self, index):
        path = os.path.join(self.dir, self.files[index])
        return torch.Tensor(load_subject(path, self.mask).get_data())

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    import nilearn.plotting as nlplt
    import matplotlib.pyplot as plt
    base_path = 'E:\\trends-fmri'
    ds = TReNDSfMRIDataset(os.path.join(base_path, 'fMRI_test'),
                           mask_path=os.path.join(base_path, 'fMRI_mask.nii'))
    print(ds[0].shape)
    smri_filename = os.path.join(base_path, 'ch2better.nii')
    subject_filename = os.path.join(base_path, 'fMRI_test/10228.mat')
    mask_niimg = ds.mask

    subject_niimg = load_subject(subject_filename, mask_niimg)

    grid_size = int(np.ceil(np.sqrt(subject_niimg.shape[0])))
    fig, axes = plt.subplots(grid_size, grid_size,
                             figsize=(grid_size*10, grid_size*10))
    [axi.set_axis_off() for axi in axes.ravel()]
    row = -1
    for i, cur_img in enumerate(nl.image.iter_img(subject_niimg)):
        col = i % grid_size
        if col == 0:
            row += 1
        nlplt.plot_stat_map(cur_img, bg_img=smri_filename, title="IC %d" %
                            i, axes=axes[row, col], threshold=3, colorbar=False)
    plt.show()

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
