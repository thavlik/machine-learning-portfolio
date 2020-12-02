import os
import numpy as np
from plot import plot_comparison
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

steps = 128
items = [
    ['sin(x)', [np.sin(2*np.pi*x/steps) * 0.5 + 0.5
                for x in range(steps)]],
    ['cos(x)',  [np.cos(2*np.pi*x/steps) * 0.5 + 0.5
                 for x in range(steps)]]
]
fig = go.Figure()
for name, y in items:
    x = np.arange(len(y))
    fig.add_trace(go.Scatter(x=x, y=y,
                             mode='lines',
                             name=name))
fig.update_layout(
    title='test'.title(),
    xaxis_title="Epoch",
    yaxis_title="loss".title(),
    font=dict(
        size=18,
    )
)
fig.write_image('out.png')
