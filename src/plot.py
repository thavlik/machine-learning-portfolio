import torch
from torch import Tensor
from plotly.subplots import make_subplots
from plotly.graph_objects import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.transforms import Resize, ToPILImage, ToTensor
from torchvision.io import write_video


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
               params: dict):
    rows = params['rows']
    cols = params['cols']
    fig = make_subplots(rows=rows, cols=cols)
    raise NotImplementedError
    fig.write_image(out_path,
                    width=params['width'],
                    height=params['height'])


def plot2d(orig: Tensor,
           recons: Tensor,
           model_name: str,
           epoch: int,
           out_path: str,
           params: dict,
           imshow_args: dict = {}):
    rows = params['rows']
    cols = params['cols']
    if 'thumbnail_size' in params:
        thumbnail_width = params['thumbnail_size']
        thumbnail_height = params['thumbnail_size']
    else:
        thumbnail_width = params.get('thumbnail_width', 256)
        thumbnail_height = params.get('thumbnail_height', 256)
    scaling = params.get('scaling', 1.0)
    fig = plt.figure(figsize=(cols * scaling, rows * scaling),
                     dpi=params.get('dpi', 110))
    grid = ImageGrid(fig,
                     111,  # similar to subplot(111)
                     nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
                     axes_pad=0.1)  # pad between axes in inch.
    i = 0
    n = min(rows * cols, orig.shape[0])
    to_pil = ToPILImage()
    resize = Resize((thumbnail_height, thumbnail_width * 2))
    for _ in range(rows):
        done = False
        for _ in range(cols):
            if i >= n:
                done = True
                break
            img = torch.cat([orig[i], recons[i]], dim=-1)
            img = to_pil(img)
            if img.size != (thumbnail_height, thumbnail_width):
                img = resize(img)
            ax = grid[i]
            ax.axis('off')
            ax.imshow(img, **imshow_args)
            i += 1
        if done:
            break
    if 'title' in params:
        interpolated = plot_title(params['title'],
                                  model_name,
                                  epoch)
        fig.suptitle(interpolated, **params.get('suptitle', {}))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')


def plot2d_dcm(orig: Tensor,
               recons: Tensor,
               model_name: str,
               epoch: int,
               out_path: str,
               params: dict):
    return plot2d(orig,
                  recons,
                  model_name,
                  epoch,
                  out_path,
                  params,
                  dict(cmap=plt.cm.bone))


def plot_video(orig: Tensor,
               recons: Tensor,
               model_name: str,
               epoch: int,
               out_path: str,
               params: dict):
    rows = params['rows']
    cols = params['cols']
    fps = params['fps']
    if 'thumbnail_size' in params:
        thumbnail_width = params['thumbnail_size']
        thumbnail_height = params['thumbnail_size']
    else:
        thumbnail_width = params.get('thumbnail_width', 256)
        thumbnail_height = params.get('thumbnail_height', 256)

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
    write_video(out_path, video_array, fps)


plot_fn = {
    'timeseries': timeseries,
    'plot2d': plot2d,
    'dcm': plot2d_dcm,
    'video': plot_video,
}


def get_plot_fn(name: str):
    if name not in plot_fn:
        raise ValueError(f'Plotting function "{name}" not found, '
                         f'valid options are {plot_fn}')
    return plot_fn[name]


if __name__ == '__main__':
    from dataset import RSNAIntracranialDataset
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

    # plot2d_dcm(batch,
    #           batch,
    #           'plot.png',
    #           dict(rows=4,
    #                cols=4))
