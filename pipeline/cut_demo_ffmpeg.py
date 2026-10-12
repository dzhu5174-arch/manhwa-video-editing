# -*- coding: utf-8 -*-
"""cut_demo_ffmpeg.py — 直接在用户的「分镜 demo」上剪辑（ffmpeg 执行）

分镜来源 = 用户 demo 的切点（帧内差≈0，任何变化即切点，不增删）
取景来源 = **用户 demo 自己的像素**（不回源图，保证画面不丢真、与 demo 完全对齐）
裁剪法则 = 单格化 v2：白像素/气泡/字幕框=死重出画；内容带居中；中间权重大；只放大到够关键帧移动
运镜     = 线性匀速（两关键帧）

用法:
  python cut_demo_ffmpeg.py --demo "<demo.mp4>" --out "<out.mp4>" [--t0 0 --t1 15] [--canvas 1440x1080]
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import cv2

WHITE = 235          # 白掩码阈值
ROW_DENS = 0.60      # 实内容行密度
GAP_FRAC = 0.05      # 内容带允许夹缝
BAND_LO = 0.30       # 内容带偏好：取带中段（上留 30% / 下留 30%）
TRAVEL_FRAC = 0.10   # 行程 = 画宽 10%（对标实测）


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        sys.exit("ffmpeg 失败: " + r.stderr.decode("utf-8", "replace")[-400:])
    return r


def cut_points(demo, fps_probe=30.0):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", demo, "-vf", "scale=306:204,format=gray",
                        "-f", "rawvideo", "-"], capture_output=True)
    a = np.frombuffer(r.stdout, dtype=np.uint8)
    n = len(a) // (306 * 204)
    A = a[:n * 306 * 204].reshape(n, 204, 306).astype(np.int16)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", demo], capture_output=True, text=True).stdout.strip())
    fps = n / dur
    d = np.abs(np.diff(A, axis=0)).mean(axis=(1, 2))
    cuts, i = [], 0
    while i < len(d):
        if d[i] > 1.5:
            t = (i + 1) / fps
            if not cuts or t - cuts[-1] > 0.8:
                cuts.append(round(t, 3))
            i += int(0.4 * fps)
        else:
            i += 1
    return [0.0] + cuts + [round(dur, 3)], dur


def longest_run(flags, gap=0):
    best, s, last = None, None, None
    for i, v in enumerate(flags):
        if v:
            if s is None:
                s = i
            last = i
        elif s is not None and i - last > gap:
            if best is None or last - s > best[1] - best[0]:
                best = (s, last)
            s = None
    if s is not None and (best is None or last - s > best[1] - best[0]):
        best = (s, last)
    return best


def window_in_strip(frame_bgr):
    """在 demo 帧里找页面条带 → 内容带 → 4:3 裁窗（带中段、避白避泡）。返回 (gx, gy, gw, gh, meta)"""
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mx = frame_bgr.max(2)
    cols = np.where(mx.mean(0) > 12)[0]
    rows = np.where(mx.mean(1) > 12)[0]
    if len(cols) < 20 or len(rows) < 20:
        return None
    sx0, sx1 = int(cols.min()), int(cols.max())
    sy0, sy1 = int(rows.min()), int(rows.max())
    strip = frame_bgr[sy0:sy1 + 1, sx0:sx1 + 1]
    sg = g[sy0:sy1 + 1, sx0:sx1 + 1]
    smn = strip.min(2)
    white = (smn > WHITE)
    dead = white.astype(np.float32)
    dens = 1.0 - dead.mean(1)
    H, W = sg.shape
    r = longest_run(list(dens > ROW_DENS), gap=int(GAP_FRAC * H))
    if not r:
        return None
    b0, b1 = int(r[0]), int(r[1])
    B = b1 - b0 + 1
    gh = W * 3 // 4                      # 4:3 窗口（条带全宽，最大化清晰度）
    if gh > B:                           # 带太短 → 窗口缩短（等比）
        gh = max(60, B)
    # 带中段：上留 30%、下留 30% 以内
    lo = b0 + int(BAND_LO * 0)           # 带内可放范围
    hi = b1 + 1 - gh
    if hi < lo:
        lo = hi = max(b0, min(b1 + 1 - gh, b0))
    gy = lo + int(0.5 * (hi - lo))       # 居中（中间权重高）
    gx = 0
    return (sx0 + gx, sy0 + gy, W, gh, dict(band=[b0, b1], white_in_band=float(white[b0:b1 + 1].mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=0.0)
    ap.add_argument("--canvas", default="1440x1080")
    a = ap.parse_args()
    CW, CH = [int(v) for v in a.canvas.lower().split("x")]
    bounds, dur = cut_points(a.demo)
    t1 = a.t1 if a.t1 > 0 else (min(15.0, dur))
    segs = [(x, y) for x, y in zip(bounds, bounds[1:]) if x >= a.t0 - 1e-6 and y <= t1 + 1e-6]
    if not segs:
        sys.exit("区间内没有分镜")
    if segs[0][0] > a.t0:
        segs[0] = (a.t0, segs[0][1])
    if segs[-1][1] < t1:
        segs[-1] = (segs[-1][0], t1)
    work = os.path.join(os.path.dirname(os.path.abspath(a.out)) or ".", "_cutwork")
    os.makedirs(work, exist_ok=True)
    print("分镜(%d 段): %s" % (len(segs), [(round(x, 2), round(y, 2)) for x, y in segs]))

    parts = []
    for k, (s0, s1) in enumerate(segs):
        d = s1 - s0
        fp = os.path.join(work, "f%02d.png" % k)
        sh(["ffmpeg", "-y", "-v", "error", "-ss", "%.3f" % ((s0 + s1) / 2), "-i", a.demo,
            "-frames:v", "1", fp])
        fr = cv2.imread(fp)
        w = window_in_strip(fr)
        if w is None:
            print("  shot%d 条带/内容带识别失败 → 居中原图" % k)
            gx, gy, gw, gh = 0, 0, fr.shape[1], fr.shape[0]
        else:
            gx, gy, gw, gh, meta = w
        # 关键帧：线性平移（行程 = 画宽 10% → 换算到裁窗像素）
        trav = max(2, int(round(TRAVEL_FRAC * gw)))
        y0 = max(0, gy - trav // 2)
        y1 = min(fr.shape[0] - gh, y0 + trav)
        if y1 <= y0:
            y1 = y0
        seg = os.path.join(work, "s%02d.mp4" % k)
        vf = ("crop=w=%d:h=%d:x=%d:y='%d+(%d-%d)*(t/%.3f)',scale=%d:%d:flags=lanczos,setsar=1,format=yuv420p"
              % (gw, gh, gx, y0, y1, y0, d, CW, CH))
        sh(["ffmpeg", "-y", "-v", "error", "-ss", "%.3f" % s0, "-t", "%.3f" % d, "-i", a.demo,
            "-vf", vf, "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-an", seg])
        parts.append(seg)
        print("  shot%d %.2f-%.2f %.2fs crop=%dx%d@(%d,%d) y:%d→%d %s" % (
            k, s0, s1, d, gw, gh, gx, gy, y0, y1, ("" if w is None else "白%.2f" % meta["white_in_band"])))
    lst = os.path.join(work, "list.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for p in parts:
            f.write("file '%s'\n" % p.replace("\\", "/"))
    tmp = os.path.join(work, "silent.mp4")
    sh(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", tmp])
    # 音轨：用户 demo 原音轨同区间
    sh(["ffmpeg", "-y", "-v", "error", "-ss", "%.3f" % a.t0, "-t", "%.3f" % (t1 - a.t0), "-i", a.demo,
        "-vn", "-c:a", "aac", "-b:a", "192k", os.path.join(work, "audio.m4a")])
    sh(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-i", os.path.join(work, "audio.m4a"), "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", a.out])
    print("OUT:", a.out, "%.1f MB" % (os.path.getsize(a.out) / 1048576))


if __name__ == "__main__":
    main()
