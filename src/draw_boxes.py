# Source: https://www.kaggle.com/kmader/deeplesion-overview
from glob import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


def read_hu(x): return imread(x).astype(np.float32)-32768


def create_boxes(in_row):
    box_list = []
    for (start_x, start_y, end_x, end_y) in in_row['bbox']:
        box_list += [Rectangle((start_x, start_y),
                               np.abs(end_x-start_x),
                               np.abs(end_y-start_y)
                               )]
    return box_list

root = 'E:\\deeplesion'
base_img_dir = os.path.join(root, 'Images_png')
patient_df = pd.read_csv(os.path.join(root, 'DL_info.csv'))
patient_df = patient_df[:99]
patient_df['bbox'] = patient_df['Bounding_boxes'].map(
    lambda x: np.reshape([float(y) for y in x.split(',')], (-1, 4)))
patient_df['kaggle_path'] = patient_df.apply(lambda c_row: os.path.join(base_img_dir,
                                                                        '{Patient_index:06d}_{Study_index:02d}_{Series_ID:02d}'.format(
                                                                            **c_row),
                                                                        '{Key_slice_index:03d}.png'.format(**c_row)), 1)
_, test_row = next(patient_df.sample(1, random_state=0).iterrows())
fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
c_img = read_hu(test_row['kaggle_path'])
ax1.imshow(c_img, vmin=-1200, vmax=600, cmap='gray')
ax1.add_collection(PatchCollection(
    create_boxes(test_row), alpha=0.25, facecolor='red'))
ax1.set_title('{Patient_age}-{Patient_gender}'.format(**test_row))
plt.show()