import io
import os
import numpy as np
from plot import plot_comparison
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

img = np.array(Image.open('img.png'))

fig = plt.figure(figsize=(1, 3), dpi=80)
ax = fig.add_subplot(111)
ax.imshow(img)
ax.text(0, 0, str("Label"),
        horizontalalignment="left",
        verticalalignment="top")
buf = io.BytesIO()
data = fig.savefig(buf, format="png")
buf.seek(0)
img = Image.open(buf)
plt.show()

d1 = ImageDraw.Draw(img)
f = d1.getfont()

d1.text((0, 0), "Hello, TutorialsPoint!", fill=(255, 0, 0))
img.show()

"""
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
"""
