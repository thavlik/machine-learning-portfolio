import os
import numpy as np
import torch
import torch.utils.data as data


class DoomStillsDataset(data.Dataset):
    def __init__(self,
                 endpoint: str,
                 bucket: str,
                 cache=None):
        super(DoomStillsDataset, self).__init__()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


if __name__ == '__main__':
    import youtube_dl
    import cv2

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
    
    items = [
        "https://www.youtube.com/watch?v=4vkFdcbUtBU",
        "https://www.youtube.com/watch?v=apo9Vb-5pWo",
        "https://www.youtube.com/watch?v=c9oL3nCsILg",
        #"https://www.youtube.com/watch?v=apo9Vb-5pWo&list=PLm8AwdYOntbLy7vbWt42N555iuIN27XW0",
        #"https://www.youtube.com/watch?v=8kMHIzXzKSY&list=PLm8AwdYOntbLCyUVwimgxDL0azWx5EKAE",
        #"https://www.youtube.com/watch?v=86r_gPLSnPM&list=PLm8AwdYOntbLLr6z1pM4HW6znhVIxKxrn",
    ]
    with youtube_dl.YoutubeDL({'outtmpl': '%(id)s.%(ext)s'}) as ydl:
        for item in items:
            result = ydl.extract_info(
                item,
                download=True,
            )
            if 'entries' in result:
                # It is a playlist
                for video in result['entries']:
                    process_video(video)
            else:
                # Just a single video
                video = result
                process_video(video)
            
