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

class BatchVideoDataLoader(object):
    def __init__(self,
                 dir: str,
                 batch_size: int,
                 num_frames: int,
                 width: int,
                 height: int,
                 interval: int = 0,
                 skip: int = 0,
                 shuffle: int = 1,
                 limit: int = None):
        super(BatchVideoDataLoader, self).__init__()
        self.batch_size = batch_size
        videos = [os.path.join(dir, f)
                  for f in os.listdir(dir)
                  if f.endswith('.mp4')]
        if limit != None:
            videos = videos[:limit]
        self.vl = VideoLoader(videos,
                              ctx=[cpu(0)],
                              shape=(num_frames, width, height, 3),
                              interval=interval,
                              skip=skip,
                              shuffle=shuffle)

    def __next__(self):
        x, y = [], []
        for _ in range(self.batch_size):
            a, b = self.vl.next()
            # [F, W, H, C] -> [F, C, H, W]
            a = torch.transpose(a, 1, 3)
            x.append(a.unsqueeze(0))
            y.append(b.unsqueeze(0))
        x = torch.cat(x, dim=0).squeeze().float()
        y = torch.cat(y, dim=0).squeeze()
        return x, y

    def __len__(self):
        return len(self.vl)

    def __iter__(self):
        self.vl.__iter__()
        return self

if __name__ == '__main__':
    import time
    num_frames = 1
    width = 320
    height = 240
    start = time.time()
    ds = BatchVideoDataLoader(dir='E:/doom',
                              batch_size=8,
                              num_frames=num_frames,
                              width=width,
                              height=height,
                              limit=10)
    for x, y in ds:
        pass
    delta = time.time() - start
    print(f'Loaded in {delta} seconds')
    print('ok')
