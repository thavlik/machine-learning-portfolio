import numpy as np
import torch
from torch import Tensor
from plotly.subplots import make_subplots
from plotly.graph_objects import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.transforms import Resize, ToPILImage, ToTensor
from torchvision.io import write_video
import nilearn as nl
import nilearn.plotting as nlplt
from dataset.trends_fmri import load_subject
from PIL import Image


def plot_title(template: str,
               model: str,
               epoch: int):
    replacements = {
        '${model}': model,
        '${epoch}': epoch,
    }
    interpolated = template
    for k, v in replacements.items():
        interpolated = interpolated.replace(k, str(v))
    return interpolated


def timeseries(orig: Tensor,
               recons: Tensor,
               model_name: str,
               epoch: int,
               out_path: str,
               rows: int,
               cols: int,
               width: int,
               height: int):
    fig = make_subplots(rows=rows, cols=cols)
    raise NotImplementedError
    fig.write_image(out_path,
                    width=width,
                    height=height)


def plot2d(orig: Tensor,
           recons: Tensor,
           model_name: str,
           epoch: int,
           out_path: str,
           rows: int,
           cols: int,
           display: str = 'horizontal',
           scaling: float = 1.0,
           dpi: int = 110,
           title: str = None,
           suptitle: dict = {},
           imshow_args: dict = {}):
    fig = plt.figure(figsize=(cols * scaling, rows * scaling),
                     dpi=dpi)
    grid = ImageGrid(fig,
                     111,  # similar to subplot(111)
                     nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
                     axes_pad=0.1)  # pad between axes in inch.
    i = 0
    n = min(rows * cols, orig.shape[0])
    to_pil = ToPILImage()
    displays = {
        'horizontal': -1,
        'vertical': -2,
    }
    dim = displays[display]
    for _ in range(rows):
        done = False
        for _ in range(cols):
            if i >= n:
                done = True
                break
            img = torch.cat([orig[i], recons[i]], dim=dim)
            img = to_pil(img)
            ax = grid[i]
            ax.axis('off')
            ax.imshow(img, **imshow_args)
            i += 1
        if done:
            break
    if title != None:
        interpolated = plot_title(title,
                                  model_name,
                                  epoch)
        fig.suptitle(interpolated, **suptitle)
    fig.tight_layout()
    fig.savefig(out_path + '.png', bbox_inches='tight')


def plot2d_dcm(*args, **kwargs):
    return plot2d(*args,
                  **kwargs,
                  imshow_args=dict(cmap=plt.cm.bone))


def plot_video(orig: Tensor,
               recons: Tensor,
               model_name: str,
               epoch: int,
               out_path: str,
               rows: int,
               cols: int,
               fps: int,
               thumbnail_width: int = None,
               thumbnail_height: int = None):
    if orig.shape[-2:] != (thumbnail_height, thumbnail_width):
        # Resize each frame
        to_pil = ToPILImage()
        to_tensor = ToTensor()
        resize = Resize((thumbnail_height, thumbnail_width))

        def transform(x):
            return to_tensor(resize(to_pil(x)))

        def resize_batch(batch):
            return torch.cat([
                torch.cat([transform(frame).unsqueeze(dim=0)
                           for frame in video]).unsqueeze(dim=0)
                for video in batch])
        recons = resize_batch(recons)
        orig = resize_batch(orig)

    # Convert [B, T, C, H, W] to [T, C, H, W]

    # Distributing the batch dimension in a grid
    n = min(rows * cols, orig.shape[0])
    i = 0
    video_rows = []
    for _ in range(rows):
        done = False
        # Build each row, one column at a time
        video_cols = []
        for _ in range(cols):
            if i >= n:
                done = True
                break
            # Original on left, recons on right
            video = torch.cat([orig[i], recons[i]], dim=-1)
            video *= 255.0
            video = video.byte()
            video_cols.append(video)
            i += 1
        while len(video_cols) < cols:
            # Append black videos to the empty spaces
            video_cols.append(torch.zeros(video.shape))
        # Concatenate all columns into a row
        video_row = torch.cat(video_cols, dim=-1)
        video_rows.append(video_row)
        if done:
            break

    # Concatenate all rows into a single video
    video_array = torch.cat(video_rows, dim=-2)
    # [T, C, H, W] -> [T, W, H, C] -> [T, H, W, C]
    video_array = torch.transpose(video_array, 1, -1)
    video_array = torch.transpose(video_array, 1, 2)
    # Monochrome to RGB
    video_array = video_array.repeat(1, 1, 1, 3)

    # Export the tensor as a video
    # TODO: improve video quality
    write_video(out_path + '.mp4', video_array, fps)


def fmri_prob_atlas(orig: Tensor,
                    recons: Tensor,
                    out_path: str,
                    rows: int,
                    cols: int,
                    bg_img: str,
                    mask_path: str,
                    display: str = 'vertical',
                    view_type: str = 'filled_contours',
                    draw_cross: bool = False,
                    threshold: str = 'auto',
                    **kwargs):
    mask = nl.image.load_img(mask_path)

    def save_prob_atlas(x, out_path):
        img = nl.image.new_img_like(mask,
                                    x.numpy(),
                                    affine=mask.affine,
                                    copy_header=True)
        nlplt.plot_prob_atlas(img,
                              bg_img=bg_img,
                              view_type=view_type,
                              draw_cross=draw_cross,
                              threshold=threshold)
        plt.savefig(out_path)

    i = 0
    n = min(rows * cols, orig.shape[0])

    try:
        # Save the individual probability atlases to separate png files
        for _ in range(rows):
            done = False
            for _ in range(cols):
                if i >= n:
                    done = True
                    break
                save_prob_atlas(orig[i], out_path + f'_{i}_orig.png')
                save_prob_atlas(recons[i], out_path + f'_{i}_recons.png')
                i += 1
            if done:
                break

        # Arrange all of the png files into a grid
        orig = []
        recons = []
        to_tensor = ToTensor()

        def load_png(path):
            return to_tensor(Image.open(path)).unsqueeze(dim=0)

        i = 0
        for _ in range(rows):
            done = False
            for _ in range(cols):
                if i >= n:
                    done = True
                    break
                orig.append(load_png(out_path + f'_{i}_orig.png'))
                recons.append(load_png(out_path + f'_{i}_orig.png'))
                i += 1
            if done:
                break

        orig = torch.cat(orig, dim=0)
        recons = torch.cat(recons, dim=0)
        plot2d(orig=orig,
               recons=recons,
               out_path=out_path,
               rows=rows,
               cols=cols,
               display=display,
               **kwargs)
    finally:
        try:
            for i in range(n):
                os.remove(out_path + f'_{i}_orig.png')
                os.remove(out_path + f'_{i}_recons.png')
        except:
            pass


def fmri_stat_map_video(orig: Tensor,
                        recons: Tensor,
                        model_name: str,
                        epoch: int,
                        out_path: str,
                        bg_img: str,
                        mask_path: str,
                        rows: int,
                        cols: int):
    mask = nl.image.load_img(mask_path)
    num_frames = orig.shape[1]
    n = min(orig.shape[0], rows * cols)
    for frame in range(num_frames):
        i = 0
        for _ in range(rows):
            done = False
            for _ in range(cols):
                if i >= n:
                    done = True
                    break
                x = orig[i, :, :, :, frame]
                x = nl.image.new_img_like(mask,
                                          x.numpy(),
                                          affine=mask.affine,
                                          copy_header=True)
                nlplt.plot_stat_map(x,
                                    bg_img=bg_img,
                                    threshold=3,
                                    colorbar=False,
                                    output_file=f'{out_path}_orig_{i}_{frame}.tmp.png')
                i += 1
            if done:
                break
    # TODO: arrange images into frames, create video


plot_fn = {
    'timeseries': timeseries,
    'plot2d': plot2d,
    'dcm': plot2d_dcm,
    'fmri_prob_atlas': fmri_prob_atlas,
    'fmri_stat_map_video': fmri_stat_map_video,
    'video': plot_video,
}


def get_plot_fn(name: str):
    if name not in plot_fn:
        raise ValueError(f'Plotting function "{name}" not found, '
                         f'valid options are {plot_fn}')
    return plot_fn[name]


if __name__ == '__main__':
    import os
    from dataset import RSNAIntracranialDataset, TReNDSfMRIDataset

    base_path = 'E:\\trends-fmri'
    ds = TReNDSfMRIDataset(os.path.join(base_path, 'fMRI_test'),
                           mask_path=os.path.join(base_path, 'fMRI_mask.nii'))
    x = torch.cat([ds[i].unsqueeze(0)
                   for i in range(16)], dim=0)
    fmri_stat_map_video(
        orig=x,
        recons=x,
        model_name='test',
        epoch=0,
        out_path='test',
        bg_img='E:/trends-fmri/ch2better.nii',
        mask_path='E:/trends-fmri/fMRI_mask.nii',
        rows=4,
        cols=4,
    )
    fmri_prob_atlas(
        orig=x,
        recons=x,
        model_name='test',
        epoch=0,
        out_path='test',
        bg_img='E:/trends-fmri/ch2better.nii',
        mask_path='E:/trends-fmri/fMRI_mask.nii',
        rows=4,
        cols=4,
        scaling=3.0,
        dpi=330,
        suptitle=dict(y=0.91),
        title='${model}, fMRI Original (top) vs. Reconstruction (bottom), Epoch ${epoch}',
    )

    """
    ds = RSNAIntracranialDataset(dcm_path='E:/rsna-intracranial/stage_2_train',
                                 s3_path='s3://rsna-intracranial/stage_2_train',
                                 download=False)
    fps = 24
    rows = 4
    cols = 4
    num_frames = fps * 10
    batch = []
    offset = 0
    for i in range(rows * cols):
        frames = []
        for _ in range(num_frames):
            frame = ds[offset][0].unsqueeze(0)
            frames.append(frame)
            offset += 1
        video = torch.cat(frames, dim=0).unsqueeze(0)
        batch.append(video)
    batch = torch.cat(batch, dim=0)
    plot_video(batch, batch, 'out.mp4', dict(
        rows=rows,
        cols=cols,
        fps=fps,
        thumbnail_size=512,
    ))
    """
    # plot2d_dcm(batch,
    #           batch,
    #           'plot.png',
    #           dict(rows=4,
    #                cols=4))
