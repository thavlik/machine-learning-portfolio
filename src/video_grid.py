import os
import subprocess

import numpy as np
from decord import VideoReader, bridge
from PIL import Image

bridge.set_bridge('torch')

input = '/mnt/e/320x240'
rows = 4
cols = 4
fps = 12
num_seconds = 8
num_frames = fps * num_seconds
limit = rows * cols
files = sorted(f for f in os.listdir(input) if f.endswith('1.mp4'))
indices = np.random.randint(0, len(files), size=(1, limit)).squeeze()
files = [files[idx] for idx in indices]
print(f'Using videos {files}')
videos = [VideoReader(os.path.join(input, f)) for f in files]
frames = np.array([[vr.next().numpy() for _ in range(num_frames)]
                   for vr in videos])
for frame in range(num_frames):
    i = 0
    complete_frame = []
    for _ in range(rows):
        items = []
        for _ in range(cols):
            frame_data = frames[i][frame]
            items.append(frame_data)
            i += 1
        items = np.concatenate(items, axis=1)
        complete_frame.append(items)
    complete_frame = np.concatenate(complete_frame, axis=0)
    img = Image.fromarray(complete_frame)
    img = img.resize((img.width // 2, img.height // 2))
    out_path = 'frame_{0:04}.png'.format(frame)
    img.save(out_path)
    print(f'Wrote {out_path}')
out_path = 'output'
print(f'Encoding video to {out_path}.mp4')
cmd = f"ffmpeg -r {fps} -s {img.width}x{img.height} -i frame_%04d.png -crf 25 -pix_fmt yuv420p {out_path}.mp4"
print(f'Running {cmd}')
proc = subprocess.run(['bash', '-c', cmd], capture_output=True)
if proc.returncode != 0:
    msg = 'expected exit code 0 from ffmpeg, got exit code {}: {}'.format(
        proc.returncode, proc.stdout.decode('unicode_escape'))
    if proc.stderr:
        msg += ' ' + proc.stderr.decode('unicode_escape')
    raise ValueError(msg)
cmd = f"ffmpeg -i {out_path}.mp4 {out_path}.gif"
print(f'Running {cmd}')
proc = subprocess.run(['bash', '-c', cmd], capture_output=True)
if proc.returncode != 0:
    msg = 'expected exit code 0 from ffmpeg, got exit code {}: {}'.format(
        proc.returncode, proc.stdout.decode('unicode_escape'))
    if proc.stderr:
        msg += ' ' + proc.stderr.decode('unicode_escape')
    raise ValueError(msg)
print(f'Wrote gif to {out_path}.gif')
[os.remove(f'frame_{i}.png') for i in range(num_frames)]
