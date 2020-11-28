import os
import numpy as np
import nilearn as nl
import nilearn.plotting as nlplt
import matplotlib.pyplot as plt
import nibabel as nib
from dataset.trends_fmri import load_subject, TReNDSfMRIDataset

base_path = 'E:\\trends-fmri'
smri_filename = os.path.join(base_path, 'ch2better.nii')
subject_filename = os.path.join(base_path, 'fMRI_test/10228.mat')

ds = TReNDSfMRIDataset(os.path.join(base_path, 'fMRI_test'),
                       mask_path=os.path.join(base_path, 'fMRI_mask.nii'))

subject_niimg = load_subject(subject_filename, ds.mask)
grid_size = int(np.ceil(np.sqrt(subject_niimg.shape[0])))
fig, axes = plt.subplots(grid_size, grid_size,
                         figsize=(grid_size*10, grid_size*10))
[axi.set_axis_off() for axi in axes.ravel()]
row = -1
for i, cur_img in enumerate(nl.image.iter_img(subject_niimg)):
    col = i % grid_size
    if col == 0:
        row += 1
    nlplt.plot_stat_map(cur_img,
                        bg_img=smri_filename,
                        title="IC %d" % i,
                        axes=axes[row, col],
                        threshold=3, colorbar=False)
plt.show()

img = nl.image.new_img_like(ds.mask,
                            ds[0].numpy(),
                            affine=ds.mask.affine,
                            copy_header=True)
nlplt.plot_prob_atlas(img,
                      bg_img=smri_filename,
                      view_type='filled_contours',
                      draw_cross=False,
                      threshold='auto')
plt.show()
