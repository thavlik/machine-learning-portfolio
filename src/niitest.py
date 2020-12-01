import os
import numpy as np
import nilearn as nl
import nilearn.plotting as nlplt
import matplotlib.pyplot as plt
import nibabel as nib

img = nl.image.load_img('E:/sub-16_ses-mri_task-facerecognition_run-01_bold.nii')
print(img)