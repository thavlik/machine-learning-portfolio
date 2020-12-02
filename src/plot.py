import os
import numpy as np
import torch
from torch import Tensor
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.graph_objects import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.transforms import Resize, ToPILImage, ToTensor
from torchvision.io import write_video
import nilearn as nl
import nilearn.plotting as nlplt
from dataset.trends_fmri import load_subject
from PIL import Image
import subprocess
from typing import List, Tuple


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


def eeg(orig: Tensor,
        recons: Tensor,
        model_name: str,
        epoch: int,
        out_path: str,
        width: int,
        height: int,
        line_opacity: float = 0.7,
        layout_params: dict = {}):
    batch_size, num_channels, num_samples = orig.shape
    cols = batch_size
    fig = make_subplots(rows=num_channels, cols=cols)
    if layout_params != None:
        if 'title' in layout_params:
            layout_params['title'] = plot_title(template=layout_params['title'],
                                                model=model_name,
                                                epoch=epoch)
        fig.update_layout(**layout_params)
    i = 0
    n = min(cols, batch_size)
    x = np.arange(num_samples) / num_samples
    for col in range(cols):
        if i >= n:
            break
        for channel in range(num_channels):
            yo = orig[i, channel, :]
            yr = recons[i, channel, :]
            fig.add_trace(go.Scatter(
                x=x,
                y=yo,
                mode='lines',
                name=f'Original (Ch. {channel})',
                opacity=line_opacity,
                line=dict(
                    color='red',
                    width=2,
                ),
            ), row=channel+1, col=col+1)
            fig.add_trace(go.Scatter(
                x=x,
                y=yr,
                mode='lines',
                name=f'Recons (Ch. {channel})',
                opacity=line_opacity,
                line=dict(
                    color='blue',
                    width=2,
                ),
            ), row=channel+1, col=col+1)
        i += 1
    fig.write_image(out_path + '.png',
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

    # Save the individual probability atlases to separate png files
    for _ in range(rows):
        done = False
        for _ in range(cols):
            if i >= n:
                done = True
                break
            save_prob_atlas(orig[i], f'{out_path}_{i}_orig.tmp.png')
            save_prob_atlas(recons[i], f'{out_path}_{i}_recons.tmp.png')
            i += 1
        if done:
            break

    # Arrange all of the png files into a grid
    orig = []
    recons = []
    to_tensor = ToTensor()

    def tmpimg_to_tensor(path):
        img = to_tensor(Image.open(path)).unsqueeze(dim=0)
        os.remove(path)
        return img

    i = 0
    for _ in range(rows):
        done = False
        for _ in range(cols):
            if i >= n:
                done = True
                break
            orig.append(tmpimg_to_tensor(f'{out_path}_{i}_orig.tmp.png'))
            recons.append(tmpimg_to_tensor(f'{out_path}_{i}_recons.tmp.png'))
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


def fmri_stat_map_video(orig: Tensor,
                        recons: Tensor,
                        model_name: str,
                        epoch: int,
                        out_path: str,
                        bg_img: str,
                        mask_path: str,
                        rows: int,
                        cols: int,
                        fps: int = 1,
                        format: str = 'gif'):
    mask = nl.image.load_img(mask_path)
    num_frames = orig.shape[1]
    n = min(orig.shape[0], rows * cols)

    def plot_frame(x, out_path):
        x = nl.image.new_img_like(mask,
                                  x.numpy(),
                                  affine=mask.affine,
                                  copy_header=True)
        try:
            nlplt.plot_stat_map(x,
                                bg_img=bg_img,
                                threshold=3,
                                colorbar=False,
                                output_file=out_path)
            img = ToTensor()(Image.open(out_path))
        finally:
            try:
                os.remove(out_path)
            except:
                pass
        return img

    frames = []
    for frame in range(num_frames):
        i = 0
        frame_rows = []
        for _ in range(rows):
            done = False
            frame_cols = []
            for _ in range(cols):
                if i >= n:
                    done = True
                    break
                o = plot_frame(orig[i, :, :, :, frame],
                               f'{out_path}_{i}_{frame}_orig.tmp.png')
                r = plot_frame(recons[i, :, :, :, frame],
                               f'{out_path}_{i}_{frame}_recons.tmp.png')
                f = torch.cat([o, r], dim=-2)
                frame_cols.append(f)
                i += 1
            frame_cols = torch.cat(frame_cols, dim=-1)
            frame_rows.append(frame_cols)
            if done:
                break
        frame_rows = torch.cat(frame_rows, dim=-2)
        ToPILImage()(frame_rows).save(f'{out_path}_{frame}.tmp.png')
        frames.append(frame_rows.unsqueeze(0))

    def path(p):
        if os.name == 'nt':
            return f'$(wslpath {p})'
        return p

    def run(cmd):
        if os.name == 'nt':
            return subprocess.run(['debian.exe', 'run', cmd], capture_output=True)
        proc = subprocess.run(['sh', '-c', cmd], capture_output=True)
        if proc.returncode != 0:
            msg = f'expected exit code 0 from ffmpeg, got exit code {proc.returncode}: {proc.stdout.decode("unicode_escape")}'
            if proc.stderr:
                msg += ' ' + proc.stderr.decode('unicode_escape')
            raise ValueError(msg)

    in_path = path(f'{out_path}_%d.tmp.png')
    webm_path = path(f'{out_path}.webm')
    run(f'ffmpeg -y -framerate {fps} -i {in_path} -c:v libvpx-vp9 -pix_fmt yuva420p -lossless 1 {webm_path}')
    if format == 'gif':
        gif_path = path(f'{out_path}.gif')
        run(f'ffmpeg -y -i {webm_path} {gif_path}')
        os.remove(f'{out_path}.webm')
    elif format != 'webm':
        raise ValueError('unknown format')
    for i in range(num_frames):
        os.remove(f'{out_path}_{i}.tmp.png')


def plot_comparison(result_dict: dict,
                    metric_name: str,
                    out_path: str):
    fig = go.Figure()
    for name, y in result_dict.items():
        x = np.arange(len(y))
        fig.add_trace(go.Scatter(x=x, y=y,
                                 mode='lines',
                                 name=name))
    fig.update_layout(
        title='Comparison',
        xaxis_title="Epoch",
        yaxis_title=metric_name.title(),
        font=dict(
            size=18,
        ),
    )
    dir = os.path.dirname(out_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    fig.write_image(out_path)


plot_fn = {
    'eeg': eeg,
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
                   for i in range(1)], dim=0)
    fmri_stat_map_video(
        orig=x,
        recons=x,
        model_name='test',
        epoch=0,
        out_path='test',
        bg_img='E:/trends-fmri/ch2better.nii',
        mask_path='E:/trends-fmri/fMRI_mask.nii',
        rows=1,
        cols=1,
    )
    fmri_prob_atlas(
        orig=x,
        recons=x,
        model_name='test',
        epoch=0,
        out_path='test',
        bg_img='E:/trends-fmri/ch2better.nii',
        mask_path='E:/trends-fmri/fMRI_mask.nii',
        rows=1,
        cols=1,
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
