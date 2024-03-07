import os

import matplotlib.pyplot as plt
import nibabel as nib
import nilearn as nl
import nilearn.plotting as nlplt
import numpy as np

img = nl.image.load_img(
    'E:/openneuro/ds000030-download/sub-10159/anat/sub-10159_T1w.nii.gz')
img = nl.image.load_img(
    'E:/openneuro/ds000113-download/sub-01/ses-forrestgump/func/sub-01_ses-forrestgump_task-forrestgump_acq-dico_run-02_bold.nii.gz'
)
#img = nl.image.load_img('E:/openneuro/ds003151-download/sub-173/ses-hormoneabsent/func/sub-173_ses-hormoneabsent_task-nback_run-2_sbref.nii.gz')
#img = nl.image.load_img('E:/sub-16_ses-mri_task-facerecognition_run-01_bold.nii')
print(img)
