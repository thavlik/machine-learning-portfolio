import os
import numpy as np
import torch
import torch.utils.data as data
from math import floor
import youtube_dl
import cv2
from torchvision.transforms import Resize, ToPILImage, ToTensor
import json
import decord
from decord import VideoLoader, VideoReader
from decord import cpu, gpu

decord.bridge.set_bridge('torch')


class VideoDataset(data.Dataset):
    def __init__(self,
                 dir: str,
                 num_frames: int,
                 width: int,
                 height: int,
                 interval: int = 0,
                 skip: int = 0,
                 shuffle: int = 1,
                 limit: int = None):
        super(VideoDataset, self).__init__()
        videos = []
        for f in os.listdir(dir):
            if limit != None and len(videos) >= limit:
                break
            if f.endswith('.mp4'):
                videos.append(f)
        self.vr = [VideoReader(os.path.join(dir, f),
                               ctx=cpu(0),
                               width=width,
                               height=height)
                   for f in videos]
        n = 0
        for vr in self.vr:
            n += len(vr)
        self.n = n

    def __getitem__(self, index):
        cur = 0
        for vr in self.vr:
            end = cur + len(vr)
            if index >= end:
                cur = end
                continue
            index -= cur
            x = vr[index]
            x = torch.transpose(x, 0, -1)
            x = torch.transpose(x, 1, -1)
            x = x.float()
            return (x, [])
        raise ValueError(f'failed to seek index {index}')

    def __len__(self):
        return self.n


if __name__ == '__main__':
    import time
    num_frames = 1
    width = 320
    height = 240
    start = time.time()
    ds = VideoDataset(dir='E:/doom',
                      num_frames=num_frames,
                      width=width,
                      height=height,
                      limit=10)
    for x, y in ds:
        pass
    delta = time.time() - start
    print(f'Loaded in {delta} seconds')
    print('ok')
