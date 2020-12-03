import argparse
import json
import youtube_dl
import os
import sys
import subprocess
import time

parser = argparse.ArgumentParser(
    description='resize videos')
parser.add_argument('--input',  '-i',
                    dest="input",
                    metavar='INPUT',
                    help='path to dir with mp4 files',
                    default='E:/doom')
parser.add_argument('--output', '-o',
                    dest="output",
                    metavar='OUTPUT',
                    help='output dir',
                    default='E:/doom-processed')
parser.add_argument('--width',
                    dest="width",
                    metavar='WIDTH',
                    help='output resolution x',
                    default=320)
parser.add_argument('--height',
                    dest="height",
                    metavar='HEIGHT',
                    help='output resolution y',
                    default=240)
parser.add_argument('--skip-frames',
                    dest="skip_frames",
                    metavar='SKIP_FRAMES',
                    help='number of frames to skip',
                    default=1)
args = parser.parse_args()
denom = args.skip_frames + 1

if not os.path.exists(args.output):
    os.makedirs(args.output)

files = sorted([f for f in os.listdir(args.input)
                if f.endswith('.mp4')])
print(f'Processing {len(files)} files')
pcts = []
for i, file in enumerate(files):
    start = time.time()
    input = os.path.join(args.input, file).replace('\\', '/')
    output = os.path.join(args.output, file).replace('\\', '/')
    in_size = os.path.getsize(input)//1000//1000
    print(f'[{i+1}/{len(files)}] Processing {input} ({in_size} MiB)')
    cmd = f"ffmpeg -i $(wslpath {input}) -s {args.width}x{args.height} -y -c:a copy -an -vf select='not(mod(n\\,{denom})), setpts={1.0/denom}*PTS' $(wslpath {output})"
    proc = subprocess.run(
        ['bash', '-c', cmd], capture_output=True)
    if proc.returncode != 0:
        msg = 'expected exit code 0 from ffmpeg, got exit code {}: {}'.format(
            proc.returncode, proc.stdout.decode('unicode_escape'))
        if proc.stderr:
            msg += ' ' + proc.stderr.decode('unicode_escape')
        raise ValueError(msg)
    delta = time.time() - start
    out_size = os.path.getsize(output)//1000//1000
    pct = (1.0 - out_size / in_size) * 100
    pcts.append(pct)
    print(f'[{i+1}/{len(files)}] Wrote {output} in {delta} seconds ({out_size} MiB, {int(pct)}% reduction)')
avg = sum(pcts) / len(pcts)
print(f'Success, average reduction of {avg}%')
