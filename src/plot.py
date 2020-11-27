import torch
from torch import Tensor
from plotly.subplots import make_subplots
from plotly.graph_objects import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.transforms import Resize, ToPILImage, ToTensor


def plot1d(recons: Tensor,
           orig: Tensor,
           out_path: str,
           params: dict):
    rows = params['rows']
    cols = params['cols']
    fig = make_subplots(rows=rows, cols=cols)
    raise NotImplementedError
    fig.write_image(out_path,
                    width=params['width'],
                    height=params['height'])


def plot2d(recons: Tensor,
           orig: Tensor,
           out_path: str,
           params: dict,
           imshow_args: dict = {}):
    rows = params['rows']
    cols = params['cols']
    if 'thumbnail_size' in params:
        thumbnail_width = params['thumbnail_size']
        thumbnail_height = params['thumbnail_size']
    else:
        thumbnail_width = params.get('thumbnail_width', 512)
        thumbnail_height = params.get('thumbnail_height', 256)
    scaling = params.get('scaling', 2.0)
    fig = plt.figure(figsize=(cols * scaling, rows * scaling))
    grid = ImageGrid(fig,
                     111,  # similar to subplot(111)
                     nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
                     axes_pad=0.1)  # pad between axes in inch.
    i = 0
    n = min(rows * cols, orig.shape[0])
    to_pil = ToPILImage()
    resize = Resize((thumbnail_height, thumbnail_width))
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
            grid[i].imshow(img, **imshow_args)
            i += 1
        if done:
            break
    fig.savefig(out_path)


def plot2d_dcm(recons: Tensor,
               orig: Tensor,
               out_path: str,
               params: dict):
    return plot2d(recons,
                  orig,
                  out_path,
                  params,
                  dict(cmap=plt.cm.bone))


plot_fn = {
    'plot1d': plot2d,
    'plot2d': plot2d,
    'plot2d_dcm': plot2d_dcm,
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
    batch = [ds[i][0].unsqueeze(dim=0) for i in range(16)]
    batch = torch.cat(batch, dim=0)
    plot2d_dcm(batch,
               batch,
               'plot.png',
               dict(rows=4,
                    cols=4))
