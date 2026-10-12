# -*- coding: utf-8 -*-
"""冒烟：造 2s 带音轨测试片 → 跑 douyin_compose → 校验输出双流可读。"""
import os, subprocess, sys
import av, numpy as np
from av.video.frame import VideoFrame
from av.audio.frame import AudioFrame

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "_smoke_src.mp4")
OUT = os.path.join(HERE, "_smoke_out.mp4")

# ① 造源片：1440x1080 渐变视频 60 帧(30fps) + 440Hz 正弦波 AAC
o = av.open(SRC, "w")
ovs = o.add_stream("libx264", rate=30)
ovs.width, ovs.height, ovs.pix_fmt = 1440, 1080, "yuv420p"
oas = o.add_stream("aac", rate=44100)
oas.layout = "stereo"
for i in range(60):
    img = np.full((1080, 1440, 3), i * 4 % 255, np.uint8)
    for pkt in ovs.encode(VideoFrame.from_ndarray(img, format="bgr24")):
        o.mux(pkt)
t = np.arange(0, 2.0, 1 / 44100, dtype=np.float32)
tone = np.sin(2 * np.pi * 440 * t)
blk = 1024
for s in range(0, len(t), blk):
    seg = tone[s:s + blk]
    if len(seg) < blk:
        seg = np.pad(seg, (0, blk - len(seg)))
    data = np.vstack([seg, seg])
    fr = AudioFrame.from_ndarray(data, format="fltp", layout="stereo")
    fr.sample_rate = 44100
    for pkt in oas.encode(fr):
        o.mux(pkt)
for pkt in ovs.encode():
    o.mux(pkt)
for pkt in oas.encode():
    o.mux(pkt)
o.close()
print("源片 OK:", os.path.getsize(SRC), "bytes")

# ② 跑 douyin_compose
r = subprocess.run([sys.executable, os.path.join(HERE, "douyin_compose.py"),
                    SRC, OUT, "--title", "冒烟测试"],
                   capture_output=True, text=True, encoding="utf-8")
print(r.stdout, r.stderr)

# ③ 校验：重新打开输出，双流可读、帧数正确
v = av.open(OUT)
nv = na = 0
for f in v.decode():
    if isinstance(f, VideoFrame):
        nv += 1
    else:
        na += 1
dur = float(v.duration / av.time_base) if v.duration else 0
print(f"OK: {v.streams.video[0].width} x {v.streams.video[0].height} | "
      f"音频流 {len(v.streams.audio)} 条 | 视频帧 {nv} 音频块 {na} | 时长 {dur:.1f}s")
v.close()
os.remove(SRC)
