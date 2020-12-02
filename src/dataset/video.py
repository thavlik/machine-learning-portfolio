import os
import numpy as np
import torch
import torch.utils.data as data
from math import floor
import youtube_dl
import cv2
from torchvision.transforms import Resize, ToPILImage, ToTensor
import json
from decord import VideoLoader
from decord import cpu, gpu


class VideoDataset(data.Dataset):
    def __init__(self,
                 input_path: str,
                 cache_path: str,
                 num_frames=1,
                 width=320,
                 height=240,
                 download=False):
        super(VideoDataset, self).__init__()
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.cache_path = cache_path
        self.download = download
        with open(input_path, 'r') as f:
            self.videos = json.loads(f.read())
        self.frames_per_example = num_frames * (skip_frames + 1) - skip_frames
        total_examples = 0
        for video in self.videos:
            total_frames = floor(video['duration'] * video['fps'])
            num_examples = total_frames - self.frames_per_example + 1
            video['num_frames'] = total_frames
            video['num_examples'] = num_examples
            total_examples += num_examples
        self.total_examples = total_examples

    def __getitem__(self, index):
        cur = 0
        for video in self.videos:
            end = cur + video['num_examples']
            if index >= end:
                cur = end
                continue
            start_example = index - cur
            start_frame = start_example * self.stride_frames
            if start_frame + self.frames_per_example >= video['num_frames']:
                raise ValueError(f"Unable to seek frame {start_frame} in video"
                                 f" with only {video['num_frames']} frames")
            return get_frames(id=video['id'],
                              ext=video['ext'],
                              total_frames=video['num_frames'],
                              start_frame=start_frame,
                              num_frames=self.num_frames,
                              skip_frames=self.skip_frames,
                              width=self.width,
                              height=self.height,
                              cache_path=self.cache_path,
                              download=self.download)
        raise ValueError(
            f"unable to seek example, was dataset length calculated incorrectly?")

    def __len__(self):
        return self.total_examples


if __name__ == '__main__':
    import youtube_dl
    import cv2
    import time
    width = 640
    height = 480
    num_frames = 9
    ds = VideoDataset(input_path='../dataset/doom/compiled.json',
                      cache_path='../dataset/doom/cache',
                      width=width,
                      height=height,
                      num_frames=num_frames)
    i = 0
    n = 30
    start = time.time()
    samples = []
    while True:
        idx = np.random.randint(0, ds.total_examples)
        example = ds[idx]
        assert example.shape == (num_frames, 3, height, width)
        i += 1
        if i % n == 0:
            delta = time.time() - start
            avg = delta / n
            samples.append(avg)
            print(
                f'{avg} seconds per example ({np.mean(samples)} seconds cumulative average)')
            start = time.time()
    print('done')
