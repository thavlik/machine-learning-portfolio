import os
import subprocess
import torch
from math import ceil, sqrt
from torch import Tensor
from typing import List, Optional

import matplotlib.pyplot as plt
import nilearn as nl
import nilearn.plotting as nlplt
import numpy as np
import plotly.graph_objects as go
from matplotlib import figure
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image, ImageDraw, ImageFont
from plotly.subplots import make_subplots
from skimage import exposure
from skimage.io import imread
from skimage.segmentation import mark_boundaries
from skimage.transform import resize
from torchvision.io import write_video
from torchvision.transforms import Resize, ToPILImage, ToTensor
from torchvision.utils import save_image

from merge_strategy import deep_merge


def plot_title(template: str, model: str, epoch: int):
    replacements = {
        '${model}': model,
        '${epoch}': epoch,
    }
    interpolated = template
    for k, v in replacements.items():
        interpolated = interpolated.replace(k, str(v))
    return interpolated


def create_segmentation(mask_shape, bbox):
    yy, xx = np.meshgrid(range(mask_shape[0]),
                         range(mask_shape[1]),
                         indexing='ij')
    out_seg = np.zeros((mask_shape[0], mask_shape[1]))
    for box in bbox:
        start_x, start_y, end_x, end_y = box
        c_seg = (xx < end_x) & (xx > start_x) & (yy < end_y) & (yy > start_y)
        out_seg += c_seg
    return np.clip(out_seg, 0, 1).astype(np.float32)


def apply_softwindow(x):
    return (255 * plt.cm.gray(0.5 * np.clip((x - 50) / 350, -1, 1) +
                              0.5)[:, :, :3]).astype(np.uint8)


def localize_lesions(test_input: Tensor,
                     pred_params: Tensor,
                     target_params: Tensor,
                     out_path: str,
                     figsize: Optional[List[float]] = None):
    n = int(ceil(sqrt(test_input.shape[0])))
    rows = n
    cols = n  #test_input.shape[0]
    h, w = test_input.shape[2:]
    if figsize is None:
        figsize = [cols * 5, rows * 5]  # inches
    fig, axs = plt.subplots(rows, cols, figsize=tuple(figsize))
    row = 0
    col = 0
    scaling = torch.Tensor([w, h, w, h])
    for (x, pred_param, targ_param) in zip(test_input, pred_params,
                                           target_params):
        # Convert the bounding box coordinates to pixel coordinates.
        pred_param *= scaling
        targ_param *= scaling
        x = x.numpy().squeeze()
        mask_shape = x.shape
        # Make the CT slice look more like a CT slice.
        x = apply_softwindow(x)
        # Draw the bounding boxes on the image.
        targ_segs = create_segmentation(mask_shape,
                                        [targ_param.numpy()]).astype(int)
        pred_segs = create_segmentation(mask_shape,
                                        [pred_param.numpy()]).astype(int)
        x = mark_boundaries(image=x,
                            label_img=pred_segs,
                            color=(1, 1, 0),
                            mode='thick')
        x = mark_boundaries(image=x,
                            label_img=targ_segs,
                            color=(0, 1, 0),
                            mode='thick')
        axs[row][col].imshow(x)
        col += 1
        if col >= cols:
            col = 0
            row += 1
    out_path += '.png'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')
    return read_image(out_path)


def read_image(path: str):
    img = imread(path)
    img = img[:, :, :3]
    img = np.transpose(img, axes=(2, 0, 1))
    return img / 255.0


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
    if layout_params is not None:
        if 'title' in layout_params:
            layout_params['title'] = plot_title(
                template=layout_params['title'], model=model_name, epoch=epoch)
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
            ),
                          row=channel + 1,
                          col=col + 1)
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
            ),
                          row=channel + 1,
                          col=col + 1)
        i += 1
    if not out_path.endswith('.png'):
        out_path += '.png'
    fig.write_image(out_path, width=width, height=height)
    return read_image(out_path)


def plot2d(orig: Tensor,
           recons: Tensor,
           model_name: str,
           epoch: int,
           out_path: str,
           rows: int,
           cols: int,
           display: str = 'horizontal',
           scaling: float = 1.0,
           dpi: int = 400,
           title: str = None,
           suptitle: dict = {},
           img_filter: str = None,
           imshow_args: dict = {}):
    fig = figure.Figure(figsize=(cols * scaling, rows * scaling), dpi=dpi)
    grid = ImageGrid(
        fig,
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
            o = orig[i]
            r = recons[i]
            if img_filter is not None:
                o = run_img_filter(o, img_filter)
                r = run_img_filter(r, img_filter)
            img = torch.cat([o, r], dim=dim).byte()
            #img = img.numpy()
            #img = np.transpose(img, axes=(2, 0, 1))
            #img = img.astype(np.byte)
            #img = Image.frombytes(img)
            img = to_pil(img)
            ax = grid[i]
            ax.axis('off')
            ax.imshow(img, **imshow_args)
            i += 1
        if done:
            break
    if title is not None:
        interpolated = plot_title(title, model_name, epoch)
        fig.suptitle(interpolated, **suptitle)
    fig.tight_layout()
    if not out_path.endswith('.png'):
        out_path += '.png'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')
    return read_image(out_path)


def plot2d_dcm(*args, **kwargs):
    return plot2d(*args, **kwargs, imshow_args=dict(cmap=plt.cm.bone))


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
                torch.cat(
                    [transform(frame).unsqueeze(dim=0)
                     for frame in video]).unsqueeze(dim=0) for video in batch
            ])

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
            return subprocess.run(['debian.exe', 'run', cmd],
                                  capture_output=True)
        proc = subprocess.run(['sh', '-c', cmd], capture_output=True)
        if proc.returncode != 0:
            msg = f'expected exit code 0 from ffmpeg, got exit code {proc.returncode}: {proc.stdout.decode("unicode_escape")}'
            if proc.stderr:
                msg += ' ' + proc.stderr.decode('unicode_escape')
            raise ValueError(msg)

    in_path = path(f'{out_path}_%d.tmp.png')
    webm_path = path(f'{out_path}.webm')
    run(f'ffmpeg -y -framerate {fps} -i {in_path} -c:v libvpx-vp9 -pix_fmt yuva420p -lossless 1 {webm_path}'
        )
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
    p = deep_merge(dict(line=dict(width=1.0)), scatter_params)
    fig = go.Figure()
    for name, data in result_dict.items():
        x = data[:, 0]
        y = data[:, 1]
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name, **p))
    params = dict(
        title=f'{metric_name.title()} Comparison ({num_samples} samples)',
        width=width,
        height=height,
        xaxis_title="Epoch",
        yaxis_title=metric_name.title(),
        font=dict(size=18, ))
    params = deep_merge(params, layout_params)
    fig.update_layout(**params)
    dir = os.path.dirname(out_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    fig.write_image(out_path)
    return read_image(out_path)


def ct_filter(x):
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


def run_img_filter(x, img_filter: str):
    if img_filter == 'ct':
        return ct_filter(x)
    elif img_filter == 'apply_softwindow':
        x = x.numpy().squeeze()
        x = apply_softwindow(x)
        x = torch.Tensor(x)
        # HxWxC -> CxWxH
        x = torch.transpose(x, 0, -1)
        x = x.squeeze()
        x = x[:3, ...]
        # CxWxH -> CxHxW
        x = torch.transpose(x, 1, 2)
        return x
    raise NotImplementedError


""" A list of colors that are used to color the lines in the classifier1d plot. 
    The list is long enough to ensure that there are enough colors for all possible 
    channels in the EEG dataset, and more. Redundant and similar shades have been
    removed in order to improve the visual appearance of the plot.
"""
colors = [
    'blue',
    'blueviolet',
    'brown',
    'burlywood',
    'cadetblue',
    'chartreuse',
    'chocolate',
    'coral',
    'crimson',
    'darkblue',
    'darkcyan',
    'darkgoldenrod',
    'darkgrey',
    'darkgreen',
    'darkkhaki',
    'darkmagenta',
    'darkolivegreen',
    'darkorange',
    'darkorchid',
    'darkred',
    'darksalmon',
    'darkseagreen',
    'darkslateblue',
    'darkturquoise',
    'darkviolet',
    'deeppink',
    'deepskyblue',
    'dodgerblue',
    'firebrick',
    'forestgreen',
    'fuchsia',
    'gainsboro',
    'gold',
    'goldenrod',
    'green',
    'greenyellow',
    'honeydew',
    'hotpink',
    'indianred',
    'indigo',
    'ivory',
    'khaki',
    'lavender',
    'lavenderblush',
    'lawngreen',
    'lemonchiffon',
    'lightblue',
    'lightcoral',
    'lightcyan',
    'lightgoldenrodyellow',
    'lightgray',
    'lightgreen',
    'lightpink',
    'lightsalmon',
    'lightseagreen',
    'lightskyblue',
    'lightslategrey',
    'lightsteelblue',
    'lightyellow',
    'lime',
    'limegreen',
    'linen',
    'magenta',
    'maroon',
    'mediumaquamarine',
    'mediumblue',
    'mediumorchid',
    'mediumpurple',
    'mediumseagreen',
    'mediumslateblue',
    'mediumspringgreen',
    'mediumturquoise',
    'mediumvioletred',
    'midnightblue',
    'mintcream',
    'mistyrose',
    'moccasin',
    'navy',
    'oldlace',
    'olive',
    'olivedrab',
    'orange',
    'orangered',
    'orchid',
    'palegoldenrod',
    'palegreen',
    'paleturquoise',
    'palevioletred',
    'papayawhip',
    'peachpuff',
    'peru',
    'pink',
    'plum',
    'powderblue',
    'purple',
    'red',
    'rosybrown',
    'royalblue',
    'saddlebrown',
    'salmon',
    'sandybrown',
    'seagreen',
    'seashell',
    'sienna',
    'silver',
    'skyblue',
    'slateblue',
    'slategrey',
    'snow',
    'springgreen',
    'steelblue',
    'tan',
    'teal',
    'thistle',
    'tomato',
    'turquoise',
    'violet',
    'wheat',
    'yellow',
    'yellowgreen',
]


def classifier1d(test_input: Tensor,
                 targets: Tensor,
                 predictions: Tensor,
                 classes: list,
                 out_path: str,
                 width: int = 8192,
                 height: int = 8192,
                 line_opacity: float = 0.3,
                 layout_params: dict = {}):
    batch_size, num_channels, num_samples = test_input.shape
    fig = make_subplots(rows=batch_size, cols=1)
    if layout_params is not None:
        #if 'title' in layout_params:
        #    layout_params['title'] = plot_title(
        #        template=layout_params['title'], model=model_name, epoch=epoch)
        fig.update_layout(**layout_params)
    x = np.array([i / num_samples for i in range(num_samples)])
    i = 0
    for row in range(batch_size):
        for channel in range(num_channels):
            yo = test_input[i, channel, :]
            fig.add_trace(go.Scatter(x=x,
                                     y=yo,
                                     mode='lines',
                                     name=f'Ch. {channel}',
                                     opacity=line_opacity,
                                     line=dict(
                                         color=colors[channel % len(colors)],
                                         width=2,
                                     )),
                          row=row + 1,
                          col=1)
        i += 1
    if not out_path.endswith('.png'):
        out_path += '.png'
    fig.write_image(out_path, width=width, height=height)
    return read_image(out_path)


def classifier1d_multicolumn(test_input: Tensor,
                             targets: Tensor,
                             predictions: Tensor,
                             classes: List[dict],
                             out_path: str,
                             width: int = 2048,
                             height: int = 512,
                             indicator_thickness: Optional[int] = None,
                             line_opacity: float = 0.3,
                             padding: int = 16,
                             layout_params: dict = {}):
    if not out_path.endswith('.png'):
        out_path += '.png'
    cols, batch_size, num_channels, num_samples = test_input.shape
    x = np.array([i / num_samples for i in range(num_samples)])
    output = None
    for col, klass in enumerate(classes):
        col_output = None
        for row in range(batch_size):
            fig = make_subplots(rows=1, cols=1)
            fig.update_layout(**layout_params)
            if row == 0:
                title = klass['name'] + '      ' + str(klass['labels'])
                fig.update_layout(title=dict(text=title, font=dict(size=35)),
                                  margin={'t': 64})
            for channel in range(num_channels):
                y = test_input[col, row, channel, :]
                fig.add_trace(go.Scatter(x=x,
                                         y=y,
                                         mode='lines',
                                         name=f'Ch. {channel}',
                                         opacity=line_opacity,
                                         line=dict(
                                             color=colors[channel %
                                                          len(colors)],
                                             width=2,
                                         )),
                              row=1,
                              col=1)
            fig.write_image('tmp.png', width=width, height=height)
            img = torch.Tensor(read_image('tmp.png'))
            os.remove('tmp.png')
            if indicator_thickness is not None:
                img = add_indicator_to_image(
                    img, (predictions - targets).pow(2).float().mean(),
                    indicator_thickness,
                    after=True)
            img = pad_image(img, [1.0, 1.0, 1.0], padding)
            col_output = torch.cat([col_output, img],
                                   dim=1) if col_output is not None else img
        output = torch.cat([output, col_output],
                           dim=2) if output is not None else col_output
    save_image(output, out_path, format='png')
    return output


def classifier2d(test_input: Tensor,
                 predictions: Tensor,
                 targets: Tensor,
                 classes: list,
                 out_path: str,
                 img_filter: Optional[str] = None,
                 background: List[float] = [0.7, 0.7, 0.7],
                 indicator_thickness: Optional[int] = None,
                 padding: Optional[int] = None,
                 font_size: int = 28):
    background = torch.Tensor(background)
    if test_input.shape[1] == 1:
        test_input = test_input.repeat(1, 3, 1, 1)
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
    for class_obj, examples, preds, targs in zip(classes, test_input,
                                                 predictions, targets):
        column = []
        labels = torch.Tensor(class_obj['labels']).int()
        for img, pred, targ in zip(examples, preds, targs):
            # Class relative accuracy
            class_rel_acc = torch.round(pred).int() == labels.int()
            class_rel_acc = class_rel_acc.float().mean()

            if img_filter is not None:
                img = run_img_filter(img, img_filter)

            img = add_indicator_to_image(img,
                                         class_rel_acc,
                                         indicator_thickness,
                                         after=False)

            # Average relative accuracy across all classes
            rel_acc = torch.round(pred).int() == targ.int()
            rel_acc = rel_acc.float().mean()
            img = add_indicator_to_image(img,
                                         rel_acc,
                                         indicator_thickness,
                                         after=True)

            img = pad_image(img, background, padding)
            column.append(img)
        column = torch.cat(column, dim=1)
        column = padv(column, background, font_size, bottom=False)
        column = add_label(column, class_obj['name'], font_size)
        columns.append(column)
    img = torch.cat(columns, dim=2)
    if not out_path.endswith('.png'):
        out_path += '.png'
    save_image(img, out_path)
    return read_image(out_path)


def add_indicator_to_image(img: Tensor,
                           rel_acc: float,
                           thickness: int = 4,
                           after: bool = True):
    max_hue = 0.42
    hue = np.clip(rel_acc * max_hue, 0.0, max_hue)
    color = hsv_to_rgb([hue, 1.0, 1.0])
    height = img.shape[1]
    colorbar = torch.Tensor(color).unsqueeze(1).unsqueeze(1).repeat(
        1, height, thickness)
    img = torch.cat([img, colorbar] if after else [colorbar, img], dim=2)
    return img


def padv(img, color, num_pixels, top=True, bottom=True):
    color = torch.Tensor(color).unsqueeze(1).unsqueeze(1)
    width = img.shape[2]
    hbar = color.repeat(1, num_pixels, width)
    if top:
        img = torch.cat([hbar, img], dim=1)
    if bottom:
        img = torch.cat([img, hbar], dim=1)
    return img


def padh(img, color, num_pixels, left=True, right=True):
    color = torch.Tensor(color).unsqueeze(1).unsqueeze(1)
    height = img.shape[1]
    vbar = color.repeat(1, height, num_pixels)
    if left:
        img = torch.cat([vbar, img], dim=2)
    if right:
        img = torch.cat([img, vbar], dim=2)
    return img


def pad_image(img, color, num_pixels):
    img = padh(img, color, num_pixels)
    img = padv(img, color, num_pixels)
    return img


def add_label(img: Tensor,
              label: str,
              size: int,
              fill=(0, 0, 0),
              margin: int = 6):
    img = ToPILImage()(img)
    font_path = os.path.join('data', 'fonts', 'arial.ttf')
    font = ImageFont.truetype(font_path, size)
    draw = ImageDraw.Draw(img)
    w, _ = draw.textsize(label, font=font)
    draw.text(xy=((img.width - w) / 2, margin),
              text=label,
              fill=fill,
              font=font)
    y = ToTensor()(img)
    return y


plot_fn = {
    'eeg': eeg,
    'plot2d': plot2d,
    'classifier1d': classifier1d,
    'classifier1d_multicolumn': classifier1d_multicolumn,
    'classifier2d': classifier2d,
    'dcm': plot2d_dcm,
    'localize_lesions': localize_lesions,
    'fmri_prob_atlas': fmri_prob_atlas,
    'fmri_stat_map_video': fmri_stat_map_video,
    'video': plot_video,
}


def get_plot_fn(name: str):
    if name not in plot_fn:
        raise ValueError(f'Plotting function "{name}" not found, '
                         f'valid options are {plot_fn}')
    return plot_fn[name]


def get_random_batch_with_label(ds,
                                labels: Tensor,
                                all_: bool,
                                batch_size: int,
                                end_idx: Optional[int] = None) -> List[int]:
    result = []
    for _ in range(batch_size):
        idx = get_random_example_with_label(ds,
                                            labels,
                                            all_=all_,
                                            exclude=result,
                                            end_idx=end_idx)
        result.append(idx)
    return result


def get_random_example_with_label(ds,
                                  labels: Tensor,
                                  all_: bool,
                                  exclude: List[int],
                                  end_idx: Optional[int] = None) -> int:
    labels = labels.int()
    n = len(ds)
    start_idx = 0 if end_idx is not None else np.random.randint(0, n)
    for i in range(n - start_idx):
        index = i + start_idx
        if index in exclude:
            continue
        eq = ds[index][1].int() == labels
        eq = eq.all() if all_ else eq.any()
        if eq:
            return index
    if end_idx is not None:
        raise ValueError(f'Unable to find example with labels {labels}')
    return get_random_example_with_label(ds,
                                         labels,
                                         all_=all_,
                                         exclude=exclude,
                                         end_idx=start_idx)


if __name__ == '__main__':
    import os
    from time import time

    import pydicom
    from skimage import exposure
    from skimage.transform import resize

    from dataset import (GraspAndLiftEEGDataset, RSNAIntracranialDataset,
                         TReNDSfMRIDataset)
    from dataset.dicom_util import normalized_dicom_pixels
    np.random.seed(int(time()))

    batch_size = 6
    ds = GraspAndLiftEEGDataset('C:/data/grasp-and-lift-eeg',
                                num_samples=32,
                                last_label_only=True)
    classes = [
        dict(name='Control', labels=[0, 0, 0, 0, 0, 0], all=True),
        dict(name='HandStart', labels=[1, 0, 0, 0, 0, 0], all=True),
        dict(name='FirstDigitTouch', labels=[0, 1, 0, 0, 0, 0], all=True),
        dict(name='BothStartLoadPhase', labels=[0, 0, 1, 0, 0, 0], all=True),
        dict(name='LiftOff', labels=[0, 0, 0, 1, 0, 0], all=True),
        dict(name='Replace', labels=[0, 0, 0, 0, 1, 0], all=True),
        dict(name='BothReleased', labels=[0, 0, 0, 0, 0, 1], all=True),
    ]
    example_indices = [
        get_random_batch_with_label(ds,
                                    torch.Tensor(klass['labels']),
                                    all_=klass['all'],
                                    batch_size=batch_size) for klass in classes
    ]
    test_input = torch.Tensor(
        np.array([[ds[i][0].numpy() for i in batch_indices]
                  for batch_indices in example_indices]))
    targets = torch.Tensor(
        np.array([[ds[i][1].numpy() for i in batch_indices]
                  for batch_indices in example_indices]))
    predictions = torch.Tensor(
        np.array(
            [[torch.rand(len(classes)).numpy() for _ in range(batch_size)]
             for _ in classes]))
    image = classifier1d_multicolumn(
        test_input=test_input,
        targets=targets,
        predictions=predictions,
        classes=classes,
        out_path='output.png',
        width=1024,
        height=256,
        indicator_thickness=12,
        line_opacity=0.3,
        layout_params=dict(showlegend=False,
                           margin={
                               't': 0,
                               'l': 0,
                               'b': 0,
                               'r': 0
                           }),
    )
    print('wrote image to output.png')

    #ds = RSNAIntracranialDataset(root='E:/rsna-intracranial', download=False)

    #path = os.path.join(ds.dcm_path, ds.files[10000])
    #img = pydicom.dcmread(path, stop_before_pixels=False)
    #img = normalized_dicom_pixels(img)
    #img = img.squeeze().numpy()
    #plt.imshow(img, cmap=plt.cm.bone)
    # plt.show()
    #img = plt.cm.bone(plt.Normalize()(img))
    # plt.imshow(img)
    # plt.show()
    def classobj(name, labels, all_):
        return {
            'name': name,
            'labels': labels,
            'all': all_,
        }

    classes = [
        classobj("Control", torch.Tensor([0, 0, 0, 0, 0, 0]), True),
        classobj("Epidural", torch.Tensor([1, 0, 0, 0, 0, 0]), False),
        classobj("Intraparenchymal", torch.Tensor([0, 1, 0, 0, 0, 0]), False),
        classobj("Intraventricular", torch.Tensor([0, 0, 1, 0, 0, 0]), False),
        classobj("Subarachnoid", torch.Tensor([0, 0, 0, 1, 0, 0]), False),
        classobj("Subdural", torch.Tensor([0, 0, 0, 0, 1, 0]), False),
        classobj("Any", torch.Tensor([0, 0, 0, 0, 0, 1]), False),
    ]

    batch_size = 5
    X = []
    Y = []
    for obj in classes:
        class_indices = []
        for _ in range(batch_size):
            idx = get_random_example_with_label(ds,
                                                obj['labels'],
                                                all_=obj['all'],
                                                exclude=class_indices)
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
    classifier2d(X, Y, Y.copy(), classes, out_path='img.png')

    img = torch.Tensor([0.0, 1.0,
                        0.0]).unsqueeze(1).unsqueeze(1).repeat(1, 16, 16)
    img = add_indicator_to_image(img, 0.9)
    img = pad_image(img, [1.0, 0.0, 0.0], 4)

    base_path = 'E:\\trends-fmri'
    ds = TReNDSfMRIDataset(os.path.join(base_path, 'fMRI_test'),
                           mask_path=os.path.join(base_path, 'fMRI_mask.nii'))
    x = torch.cat([ds[i].unsqueeze(0) for i in range(1)], dim=0)
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
        title=
        '${model}, fMRI Original (top) vs. Reconstruction (bottom), Epoch ${epoch}',
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
