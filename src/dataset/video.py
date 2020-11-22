import os
import numpy as np
import torch
import torch.utils.data as data
from math import floor
import youtube_dl
import cv2
from torchvision.transforms import Resize, ToPILImage, ToTensor
import json


def load_links(path):
    links = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith('#'):
                continue
            links.append(line)
    return links


def ensure_video_downloaded(id: str,
                            ext: str,
                            cache_path: str,
                            download: bool):
    path = os.path.join(cache_path, f'{id}.{ext}')
    if os.path.exists(path):
        return path
    elif not download:
        raise ValueError(f'video file {path} not found.'
                         'Hint: try again with download=True')
    if not cache_path.endswith('/'):
        cache_path += '/'
    with youtube_dl.YoutubeDL({
        'outtmpl': cache_path + '%(id)s.%(ext)s',
    }) as ydl:
        video = ydl.extract_info(
            f'https://www.youtube.com/watch?v={id}',
            download=True,
        )
        if 'entries' in video:
            raise ValueError(f'playlists are not supported by this function')
    if not os.path.exists(path):
        raise ValueError('video was not downloaded')
    return path


def get_raw_frames(id: str,
                   ext: str,
                   start_frame: int,
                   num_frames: int,
                   skip_frames: int,
                   cache_path: str,
                   download: bool):
    path = ensure_video_downloaded(id, ext, cache_path, download)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video stream or file")
    frames_per_example = num_frames * (skip_frames + 1) - skip_frames
    end_frame = start_frame + frames_per_example
    frames = []
    frame_no = 0  # global frame counter
    idx = 0  # counter that starts at the right frame
    while cap.isOpened() and frame_no < end_frame:
        ret, frame = cap.read()
        if ret != True:
            break
        if frame_no >= start_frame:
            if idx % (skip_frames + 1) == 0:
                frames.append(frame)
            idx += 1
        frame_no += 1
    cap.release()
    if len(frames) != num_frames:
        raise ValueError(f"expect {num_frames} frames, got {len(frames)}")
    return frames


def resize_frames(frames,
                  width: int,
                  height: int):
    to_pil = ToPILImage()
    resize = Resize((height, width))
    to_tensor = ToTensor()
    return [to_tensor(resize(to_pil(torch.Tensor(frame))))
            for frame in frames]


def get_frames(id: str,
               ext: str,
               start_frame: int,
               num_frames: int,
               skip_frames: int,
               width: int,
               height: int,
               cache_path: str,
               download: bool):
    frames = get_raw_frames(id=id,
                            ext=ext,
                            start_frame=start_frame,
                            num_frames=num_frames,
                            skip_frames=skip_frames,
                            cache_path=cache_path,
                            download=download)
    frames = resize_frames(frames, width, height)
    return torch.stack(frames)


def get_all_frames(id: str,
                   ext: str,
                   width: int,
                   height: int,
                   cache_path: str,
                   download: bool):
    path = ensure_video_downloaded(id, ext, cache_path, download)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video stream or file")
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            frames.append(frame)
            continue
        else:
            break
    cap.release()
    frames = resize_frames(frames, width, height)
    return frames


class VideoDataset(data.Dataset):
    """
    Arguments:
        input_path (str): path to compiled.json (output from compiled.py)

        cache_path (str): video download cache path

        num_frames (int): the number of sequential frames to
        include in each training example

        width (int): output resolution X

        height (int): output resolution Y

        skip_frames (int): number of frames to skip between each
        sampled frame. Default is 0, meaning no frames are skipped.

        stride_frames (int): number of frames to offset between
        examples. Default is 1, meaning no frames are skipped.

        download (bool): allow downloading of missing videos
    """

    def __init__(self,
                 input_path: str,
                 cache_path: str,
                 num_frames=1,
                 width=320,
                 height=240,
                 skip_frames=0,
                 stride_frames=1,
                 download=False):
        super(VideoDataset, self).__init__()
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.skip_frames = skip_frames
        self.stride_frames = stride_frames
        self.cache_path = cache_path
        self.download = download
        with open(input_path, 'r') as f:
            self.videos = json.loads(f.read())
        self.frames_per_example = num_frames * (skip_frames + 1) - skip_frames
        total_examples = 0
        for video in self.videos:
            total_frames = floor(video['duration'] * video['fps'])
            num_examples = total_frames - self.frames_per_example + 1
            num_examples = floor(num_examples / stride_frames)
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
    width = 640
    height = 480
    num_frames = 9
    ds = VideoDataset(input_path='../dataset/doom/compiled.json',
                      cache_path='../dataset/doom/cache',
                      width=width,
                      height=height,
                      num_frames=num_frames)
    ex = ds[0]
    assert ex.shape == (num_frames, 3, height, width)
    print('done')
"""    
    with open('../dataset/timothybrown.txt', 'r') as file:
        items = [line for line in file]

    def process_video(video):
        cap = cv2.VideoCapture(f'{video["id"]}.{video["ext"]}')
        # Check if camera opened successfully
        if not cap.isOpened():
            raise ValueError(f"Error opening video stream or file")
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                # frame is good
                pass
            else:
                break
        cap.release()

    with youtube_dl.YoutubeDL({'outtmpl': '%(id)s.%(ext)s'}) as ydl:
        for item in items:
            result = ydl.extract_info(
                item,
                download=False,
            )
            if 'entries' in result:
                # It is a playlist
                for video in result['entries']:
                    process_video(video)
            else:
                # Just a single video
                video = result
                process_video(video)
"""
