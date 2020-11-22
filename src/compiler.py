import argparse
import json
import youtube_dl
import os
import sys

parser = argparse.ArgumentParser(
    description='Youtube dataset compiler')
parser.add_argument('--input',  '-i',
                    dest="input",
                    metavar='INPUT',
                    help='path to text file containing youtube video or playlist links',
                    default='../dataset/full.txt')
parser.add_argument('--output',
                    dest="output",
                    metavar='OUTPUT',
                    help='output file path',
                    default='../dataset/compiled.json')
parser.add_argument('--download',
                    dest="download",
                    metavar='DOWNLOAD',
                    help='download videos if true',
                    default=True)
parser.add_argument('--cache_dir',
                    dest="cache_dir",
                    metavar='CACHE_DIR',
                    help='video download path',
                    default='../dataset/cache')
args = parser.parse_args()


completed = []
videos = []

completed_path = os.path.join(os.path.dirname(args.output), '.completed.txt')

try:
    with open(completed_path) as f:
        completed = json.loads(f.read())
except:
    pass

if os.path.exists(args.output):
    with open(args.output) as f:
        videos = json.loads(f.read())


def write_completed():
    with open(completed_path, 'w') as f:
        f.write(json.dumps(completed))


def write_videos():
    with open(args.output, 'w') as f:
        f.write(json.dumps(videos))


def process_video(video, ydl, download):
    videos.append({
        k: video[k] for k in ['id',
                              'ext',
                              'vcodec',
                              'uploader_id',
                              'channel_id',
                              'duration',
                              'width',
                              'height',
                              'fps']
    })
    id = video['id']
    path = os.path.join(args.cache_dir, id + '.mp4')
    if download and not os.path.exists(path):
        try:
            ydl.extract_info(
                f'https://youtube.com/watch?v={id}',
                download=True,
            )
        except:
            print(f'Failed to download {id}: {sys.exc_info()}')


with open(args.input, "r") as f:
    lines = [line.strip() for line in f]

with youtube_dl.YoutubeDL({
    'verbose': True,
    'outtmpl': args.cache_dir + '/%(id)s.%(ext)s',
    # 'cachedir': args.cache_dir,
}) as ydl:
    for line in lines:
        if line in completed:
            print(f'Skipping {line}')
            continue
        result = ydl.extract_info(
            line,
            download=False,
        )
        if 'entries' in result:
            # It is a playlist
            for video in result['entries']:
                process_video(video, ydl, args.download)
        else:
            # Just a single video
            process_video(result, ydl, args.download)
        write_videos()
        completed.append(line)
        write_completed()
