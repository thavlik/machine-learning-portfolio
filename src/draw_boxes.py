# Source: https://www.kaggle.com/kmader/deeplesion-overview
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.patches import Rectangle
from skimage.io import imread
from skimage.transform import resize


def read_hu(x):
    return resize(imread(x).astype(np.float32) - 32768, (512, 512))


def create_boxes(in_row):
    box_list = []
    for (start_x, start_y, end_x, end_y) in in_row['bbox']:
        box_list += [
            Rectangle((start_x, start_y), np.abs(end_x - start_x),
                      np.abs(end_y - start_y))
        ]
    return box_list


def create_segmentation(in_img, in_row):
    yy, xx = np.meshgrid(range(in_img.shape[0]),
                         range(in_img.shape[1]),
                         indexing='ij')
    out_seg = np.zeros_like(in_img)
    for (start_x, start_y, end_x, end_y) in in_row['bbox']:
        c_seg = (xx < end_x) & (xx > start_x) & (yy < end_y) & (yy > start_y)
        out_seg += c_seg
    return np.clip(out_seg, 0, 1).astype(np.float32)


root = 'E:\\deeplesion'
base_img_dir = os.path.join(root, 'Images_png')
patient_df = pd.read_csv(os.path.join(root, 'DL_info.csv'))
patient_df = patient_df[:99]
patient_df['bbox'] = patient_df['Bounding_boxes'].map(
    lambda x: np.reshape([float(y) for y in x.split(',')], (-1, 4)))
patient_df['kaggle_path'] = patient_df.apply(
    lambda c_row: os.path.join(
        base_img_dir, '{Patient_index:06d}_{Study_index:02d}_{Series_ID:02d}'.
        format(**c_row), '{Key_slice_index:03d}.png'.format(**c_row)), 1)
"""
_, test_row = next(patient_df.sample(1, random_state=0).iterrows())
fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
c_img = read_hu(test_row['kaggle_path'])
ax1.imshow(c_img, vmin=-1200, vmax=600, cmap='gray')
ax1.add_collection(PatchCollection(
    create_boxes(test_row), alpha=0.25, facecolor='red'))
ax1.set_title('{Patient_age}-{Patient_gender}'.format(**test_row))
"""


def apply_softwindow(x):
    return (255 * plt.cm.gray(0.5 * np.clip((x - 50) / 350, -1, 1) +
                              0.5)[:, :, :3]).astype(np.uint8)


fig, m_axs = plt.subplots(3, 1, figsize=(10, 15))

for ax1, (_, c_row) in zip(m_axs,
                           patient_df.sample(50, random_state=0).iterrows()):
    c_img = read_hu(c_row['kaggle_path'])
    #ax1.imshow(c_img, vmin=-1200, vmax=600, cmap='gray')
    #ax1.add_collection(PatchCollection(
    #    create_boxes(c_row), alpha=0.25, facecolor='red'))
    #ax1.set_title('{Patient_age}-{Patient_gender}'.format(**c_row))
    #ax1.axis('off')
    c_segs = create_segmentation(c_img, c_row).astype(int)
    #ax1.imshow(mark_boundaries(image=apply_softwindow(c_img),
    #                           label_img=c_segs,
    #                           color=(0, 1, 0),
    #                           mode='thick'))
    ax1.imshow(apply_softwindow(c_img))
    ax1.set_title('Segmentation Map')

plt.show()
