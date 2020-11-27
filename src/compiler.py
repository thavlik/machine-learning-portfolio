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
                    default='../dataset/doom/full.txt')
parser.add_argument('--output', '-o',
                    dest="output",
                    metavar='OUTPUT',
                    help='output file path',
                    default='../dataset/doom/compiled.json')
parser.add_argument('--download',
                    dest="download",
                    metavar='DOWNLOAD',
                    help='download videos if true',
                    default=False)
parser.add_argument('--cache_dir',
                    dest="cache_dir",
                    metavar='CACHE_DIR',
                    help='video download path',
                    default='E:/cache')
parser.add_argument('--clean',
                    dest="clean",
                    metavar='CLEAN',
                    help='remove missing videos from output records (do not download)',
                    default=False)
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
    if os.path.exists(path):
        print(f'{id} already downloaded')
    elif download:
        try:
            ydl.extract_info(
                f'https://youtube.com/watch?v={id}',
                download=True,
            )
        except:
            print(f'Failed to download {id}: {sys.exc_info()}')


if args.clean:
    new_videos = [video
                  for video in videos
                  if os.path.exists(os.path.join(args.cache_dir,
                                                 video['id'] + '.mp4'))]
    print(f'Removed {len(videos)-len(new_videos)} videos')
    videos = new_videos
    write_videos()
    sys.exit(0)

with open(args.input, "r") as f:
    lines = [line.strip() for line in f]

with youtube_dl.YoutubeDL({
    'verbose': True,
    'outtmpl': args.cache_dir + '/%(id)s.%(ext)s',
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

# TODO: resample videos