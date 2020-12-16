import os
import numpy as np
import torch
from torch import Tensor
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.graph_objects import Figure
import matplotlib.pyplot as plt
from matplotlib import figure
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.transforms import Resize, ToPILImage, ToTensor
from torchvision.io import write_video
import nilearn as nl
import nilearn.plotting as nlplt
from dataset.trends_fmri import load_subject
from PIL import Image
import subprocess
from typing import List, Tuple
from merge_strategy import strategy
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from torchvision.utils import save_image


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
    fig = figure.Figure(figsize=(cols * scaling, rows * scaling),
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
    plt.close(fig)
    plt.close('all')


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
                    num_samples: int,
                    width: int,
                    height: int,
                    out_path: str,
                    scatter_params: dict = {},
                    layout_params: dict = {}):
    p = strategy.merge(dict(line=dict(width=1.0)), scatter_params)
    fig = go.Figure()
    for name, data in result_dict.items():
        x = data[:, 0]
        y = data[:, 1]
        fig.add_trace(go.Scatter(x=x, y=y,
                                 mode='lines',
                                 name=name,
                                 **p))
    params = dict(title=f'{metric_name.title()} Comparison ({num_samples} samples)',
                  width=width,
                  height=height,
                  xaxis_title="Epoch",
                  yaxis_title=metric_name.title(),
                  font=dict(
                      size=18,
                  ))
    params = strategy.merge(params, layout_params)
    fig.update_layout(**params)
    dir = os.path.dirname(out_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    fig.write_image(out_path)


def classifier2d(test_input: Tensor,
                 targets: Tensor,
                 predictions: Tensor,
                 class_names: List[str],
                 baselines: Tensor,
                 out_path: str,
                 background: List[float] = [0.6, 0.6, 0.6],
                 indicator_thickness: int = None,
                 padding: int = None):
    background = torch.Tensor(background)
    # Draw a grid of images, each class gets a column.
    # Next to each image, visually indicate if the model is
    # correct or not. Baseline accuracy should be colored
    # red, and 100% accuracy is bright green.
    if indicator_thickness is None:
        indicator_thickness = int(
            np.clip(test_input[0][0].shape[-1] / 32, 1, 16))
    if padding is None:
        padding = int(np.clip(test_input[0][0].shape[-1] / 16, 1, 32))
    columns = []
    for i, (class_name, examples, preds, targs, baseline) in enumerate(zip(class_names, test_input, predictions, targets, baselines)):
        column = []
        for img, pred, targ in zip(examples, preds, targs):
            class_rel_acc = pred[i] # pred[i] is 1.0 when 100% accurate
            class_rel_acc = class_rel_acc.float().mean()
            class_rel_acc = (class_rel_acc - baseline) / (1.0 - baseline)
            img = add_indicator_to_image(
                img, class_rel_acc, indicator_thickness, after=False)

            rel_acc = torch.round(pred).int() == targ.int()
            rel_acc = rel_acc.float().mean()
            img = add_indicator_to_image(
                img, rel_acc, indicator_thickness, after=True)

            img = pad_image(img, background, padding)
            column.append(img)
        column = torch.cat(column, dim=1)
        columns.append(column)
    img = torch.cat(columns, dim=2)
    if not out_path.endswith('.png'):
        out_path += '.png'
    save_image(img, out_path)


def add_indicator_to_image(img: Tensor,
                           rel_acc: float,
                           thickness: int = 4,
                           after: bool = True):
    max_hue = 0.42
    hue = np.clip(rel_acc * max_hue, 0.0, max_hue)
    color = hsv_to_rgb([hue, 1.0, 1.0])
    height = img.shape[1]
    colorbar = torch.Tensor(color).unsqueeze(
        1).unsqueeze(1).repeat(1, height, thickness)
    img = torch.cat([img, colorbar] if after else [colorbar, img], dim=2)
    return img


def pad_image(img, color, num_pixels):
    color = torch.Tensor(color).unsqueeze(1).unsqueeze(1)
    height = img.shape[1]
    vbar = color.repeat(1, height, num_pixels)
    img = torch.cat([img, vbar], dim=2)
    img = torch.cat([vbar, img], dim=2)
    width = img.shape[2]
    hbar = color.repeat(1, num_pixels, width)
    img = torch.cat([img, hbar], dim=1)
    img = torch.cat([hbar, img], dim=1)
    return img


plot_fn = {
    'eeg': eeg,
    'plot2d': plot2d,
    'classifier2d': classifier2d,
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


def get_random_example_with_label(ds,
                                  label: int,
                                  exclude: List[int],
                                  depth: int = 0,
                                  max_depth: int = 100) -> int:
    n = len(ds)
    start_idx = np.random.randint(0, n)
    for i in range(n - start_idx):
        index = i + start_idx
        y = ds[index][1]
        if bool(y[label]):
            if index in exclude:
                continue
            return index
    if depth >= max_depth:
        raise ValueError(f'cannot find example with label {label}')
    return get_random_example_with_label(ds,
                                         label,
                                         exclude=exclude,
                                         depth=depth+1,
                                         max_depth=max_depth)


if __name__ == '__main__':
    import os
    import pydicom
    from skimage import exposure
    from skimage.transform import resize
    from dataset import RSNAIntracranialDataset, TReNDSfMRIDataset
    from dataset.dicom_util import normalized_dicom_pixels

    ds = RSNAIntracranialDataset(root='E:/rsna-intracranial',
                                 download=False)

    #path = os.path.join(ds.dcm_path, ds.files[10000])
    #img = pydicom.dcmread(path, stop_before_pixels=False)
    #img = normalized_dicom_pixels(img)
    #img = img.squeeze().numpy()
    #plt.imshow(img, cmap=plt.cm.bone)
    # plt.show()
    #img = plt.cm.bone(plt.Normalize()(img))
    # plt.imshow(img)
    # plt.show()

    batch_size = 3
    num_classes = 6
    X = []
    Y = []
    for i in range(num_classes):
        class_indices = []
        for _ in range(batch_size):
            idx = get_random_example_with_label(ds,
                                                i,
                                                exclude=class_indices)
            l = ds[idx][1]
            assert l[i] != 0
            class_indices.append(idx)
        examples = [ds[j] for j in class_indices]

        def process(x):
            x = x.squeeze().numpy()
            x = exposure.equalize_hist(x)
            x = plt.Normalize()(x)
            x = plt.cm.bone(x)
            x = resize(x, (256, 256, 4))
            x = torch.Tensor(x)
            x = torch.transpose(x, 0, -1)
            x = x.squeeze()
            x = x[:3, ...]
            x = torch.transpose(x, 1, 2)
            return x

        x = [process(ex[0]) for ex in examples]
        y = [ex[1] for ex in examples]
        X.append(x)
        Y.append(y)

    # TODO: add class labels
    classifier2d(X, Y, Y.copy(), [
        "Epidural",
        "Intraparenchymal",
        "Intraventricular",
        "Subarachnoid",
        "Subdural",
        "Any",
    ], baselines=[
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
    ], out_path='img.png')

    img = torch.Tensor([0.0, 1.0, 0.0]).unsqueeze(
        1).unsqueeze(1).repeat(1, 16, 16)
    img = add_indicator_to_image(img, 0.9)
    img = pad_image(img, [1.0, 0.0, 0.0], 4)

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
