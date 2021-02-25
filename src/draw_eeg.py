from dataset import GraspAndLiftEEGDataset
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from torch import Tensor


def eeg(orig: Tensor,
        out_path: str,
        width: int,
        height: int,
        line_opacity: float = 0.7,
        layout_params: dict = {}):
    batch_size, num_channels, num_samples = orig.shape
    cols = batch_size
    fig = make_subplots(rows=num_channels, cols=cols)
    #if layout_params is not None:
    #    fig.update_layout(**layout_params)
    i = 0
    n = min(cols, batch_size)
    x = np.arange(num_samples) / num_samples
    for col in range(cols):
        if i >= n:
            break
        for channel in range(num_channels):
            yo = orig[i, channel, :]
            fig.add_trace(go.Scatter(
                x=x,
                y=yo,
                mode='lines',
                opacity=line_opacity,
                line=dict(
                    color='red',
                    width=2,
                ),
            ), row=channel+1, col=col+1)
        i += 1
    fig.write_image(out_path + '.png',
                    width=width,
                    height=height)


ds = GraspAndLiftEEGDataset('/data/grasp-and-lift-eeg-detection')
eeg(ds[0][0][:8].unsqueeze(0), 'out', 2048, 1024, layout_params=dict(
    xaxis=go.layout.XAxis(
        visible=False,
        showticklabels=False,
    ),
    yaxis=go.layout.YAxis(
        visible=False,
        showticklabels=False,
    ),
))
