import os
import numpy as np
import torch
import torch.utils.data as data
from math import floor
import youtube_dl
import cv2
from torchvision.transforms import Resize, ToPILImage, ToTensor


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
                            cache_path: str):
    path = os.path.join(cache_path, f'{id}.{ext}')
    if os.path.exists(path):
        return path
    with youtube_dl.YoutubeDL({
        'outtmpl': '%(id)s.%(ext)s',
        'cachedir': cache_path,
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
                   cache_path: str):
    path = ensure_video_downloaded(id, ext, cache_path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video stream or file")
    frame_no = 0
    end_frame = start_frame + num_frames
    frames = []
    while cap.isOpened() and frame_no < end_frame:
        ret, frame = cap.read()
        if ret == True:
            if frame_no >= start_frame:
                frames.append(frame)
            frame_no += 1
        else:
            break
    cap.release()
    assert len(
        frames) == num_frames, f"expect {num_frames} frames, got {len(frames)}"
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
               width: int,
               height: int,
               cache_path: str):
    frames = get_raw_frames(id, ext, start_frame, num_frames, cache_path)
    frames = resize_frames(frames, width, height)
    return torch.stack(frames)


def get_all_frames(id: str,
                   ext: str,
                   width: int,
                   height: int,
                   cache_path: str):
    path = ensure_video_downloaded(id, ext, cache_path)
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


class DoomDataset(data.Dataset):
    """
    Arguments:
        path (str): path to links.txt

        cache_path (str): video download cache path

        num_frames (int): the number of sequential frames to
        include in each training example

        width (int): output resolution X

        height (int): output resolution Y

        skip_frames (int): number of frames to skip between each
        sampled frame
    """

    def __init__(self,
                 cache_path: str,
                 num_frames=1,
                 width=640,
                 height=480,
                 skip_frames=0):

        super(DoomDataset, self).__init__()
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.skip_frames = skip_frames
        self.cache_path = cache_path
        source_videos = [{
            'id': '4vkFdcbUtBU',
            'ext': 'mp4',
            'num_frames': 6000,
        }, {
            'id': '8FYSBH6K3xU',
            'ext': 'mp4',
            'num_frames': 6000,
        }, {
            'id': 'apo9Vb-5pWo',
            'ext': 'mp4',
            'num_frames': 9830,
        }]
        items = []
        total_examples = 0
        for video in source_videos:
            video['num_examples'] = floor(
                video['num_frames'] / (num_frames + skip_frames))
            total_examples += video['num_examples']
        self.source_videos = source_videos
        self.items = items
        self.total_examples = total_examples

    def __getitem__(self, index):
        cur = 0
        for video in self.source_videos:
            end = cur + video['num_examples']
            if index >= end:
                cur = end
                continue
            start_example = index - cur
            start_frame = start_example * (self.num_frames + self.skip_frames)
            if start_frame >= video['num_frames']:
                raise ValueError(f"Unable to seek frame {start_frame} in video"
                                 f" with only {video['num_frames']} frames")
            return get_frames(video['id'],
                              video['ext'],
                              start_frame,
                              self.num_frames,
                              width=self.width,
                              height=self.height,
                              cache_path=self.cache_path)
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
    ds = DoomDataset(cache_path='../dataset/cache',
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
