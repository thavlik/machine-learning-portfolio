import os
import numpy as np
import torch
import torch.utils.data as data
from math import floor
import youtube_dl
import cv2
from torchvision.transforms import Resize, ToPILImage, ToTensor
import json
from decord import VideoLoader, VideoReader
from decord import cpu, gpu


class VideoDataset(data.Dataset):
    def __init__(self,
                 input_path: str,
                 dir: str,
                 width: int = 320,
                 height: int = 240):
        super(VideoDataset, self).__init__()
        with open(input_path, 'r') as f:
            self.videos = [video
                           for video in json.loads(f.read())
                           if os.path.exists(os.path.join(dir, video['id'] + '.mp4'))]
        self.vr = [VideoReader(os.path.join(dir, video['id'] + '.mp4'),
                               ctx=cpu(0))
                   for video in self.videos]
        self.n = sum(v['duration'] * v['fps']
                     for v in self.videos)

    def __getitem__(self, index):
        cur = 0
        for v, vr in zip(self.videos, self.vr):
            end = cur + v['duration'] * v['fps']
            if index >= end:
                cur = end
                continue
            return vr[index - cur]

    def __len__(self):
        return self.n


if __name__ == '__main__':
    width = 320
    height = 240
    ds = VideoDataset(input_path='data/doom/compiled.json',
                      dir='E:/doom',
                      width=width,
                      height=height)
    print('ok')
