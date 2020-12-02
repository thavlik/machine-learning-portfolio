import decord
from decord import VideoReader, VideoLoader
from decord import cpu, gpu
decord.bridge.set_bridge('torch')

path = 'E:/doom/_9zaLSmRgGc.mp4'

vl = VideoLoader([
    path,
    #'E:/doom/_BHunyDleDQ.mp4',
], ctx=[cpu(0)], shape=(10, 320, 240, 3), interval=0, skip=5, shuffle=1)
ex = vl.next()
vr = VideoReader(path, ctx=cpu(0))
# a file like object works as well, for in-memory decoding
with open(path, 'rb') as f:
    vr = VideoReader(f, ctx=cpu(0))
print('video frames:', len(vr))
# 1. the simplest way is to directly access frames
for i in range(len(vr)):
    # the video reader will handle seeking and skipping in the most efficient manner
    frame = vr[i]
    print(frame.shape)

# To get multiple frames at once, use get_batch
# this is the efficient way to obtain a long list of frames
frames = vr.get_batch([1, 3, 5, 7, 9])
print(frames.shape)
# (5, 240, 320, 3)
# duplicate frame indices will be accepted and handled internally to avoid duplicate decoding
frames2 = vr.get_batch([1, 2, 3, 2, 3, 4, 3, 4, 5])
print(frames2.shape)
# (9, 240, 320, 3)

# 2. you can do cv2 style reading as well
# skip 100 frames
vr.skip_frames(100)
# seek to start
vr.seek(0)
batch = vr.next()
print('frame shape:', batch.shape)
