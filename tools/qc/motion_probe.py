# -*- coding: utf-8 -*-
"""motion_probe.py — 漫解成片运镜模式拆解：切镜 + 相位相关测全局位移 + 缓动曲线形态。
输出：每镜 方向/速度/缓动比，汇总分布。用法: motion_probe.py <video.mp4>
"""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import av, cv2, numpy as np

path = sys.argv[1]
c = av.open(path)
vs = c.streams.video[0]
fps = float(vs.average_rate or 30)
STEP = max(1, round(fps / 10))          # 10fps 采样
W0 = 360                                 # 分析用宽度
prev, frames, idx = None, [], 0
for fr in c.decode(vs):
    idx += 1
    if idx % STEP:
        continue
    img = cv2.resize(fr.to_ndarray(format="gray"), (W0, 10**6))
    img = cv2.resize(fr.to_ndarray(format="gray"),
                     (W0, int(W0 * fr.height / fr.width)))
    if prev is not None:
        frames.append(img)
    prev = img
c.close()
H, W = frames[0].shape
print(f"采样 {len(frames)} 帧 @10fps, 画面 {vs.width}x{vs.height} ({vs.width/vs.height:.2f}:1)")

# 切镜：直方图相关 < 0.85 或相位位移 > 画面 1/3（硬切）
shots, start = [], 0
h0 = cv2.calcHist([frames[0]], [0], None, [32], [0, 256])
for i in range(1, len(frames)):
    h1 = cv2.calcHist([frames[i]], [0], None, [32], [0, 256])
    corr = cv2.compareHist(h0, h1, cv2.HISTCMP_CORREL)
    if corr < 0.80:
        if i - start >= 8:               # ≥0.8s 才算镜头
            shots.append((start, i))
        start = i
        h0 = h1
if len(frames) - start >= 8:
    shots.append((start, len(frames)))
print(f"检出 {len(shots)} 个镜头（≥0.8s）")

def probe(a, b):
    """镜内逐帧相位相关 → 中位方向 + 缓动形态。"""
    dxs, dys, mags = [], [], []
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    for j in range(a, b - 1):
        f1 = frames[j].astype(np.float32)
        f2 = frames[j + 1].astype(np.float32)
        (dx, dy), resp = cv2.phaseCorrelate(f1, f2, win)
        if resp > 0.15:
            dxs.append(dx); dys.append(dy); mags.append((dx**2 + dy**2) ** .5)
    if not mags:
        return None
    med_dx, med_dy = float(np.median(dxs)), float(np.median(dys))
    k = max(2, len(mags) // 5)
    edge = (np.mean(mags[:k]) + np.mean(mags[-k:])) / 2
    mid = np.mean(mags[k:-k]) or 1e-6
    return dict(dur=round((b - a) / 10, 1), dx=round(med_dx, 2), dy=round(med_dy, 2),
                speed=round(float(np.median(mags)), 2), ease=round(float(edge / mid), 2))

rows = [probe(a, b) for a, b in shots]
rows = [r for r in rows if r]
for i, r in enumerate(rows[:40]):
    d = "静止" if r["speed"] < 0.15 else \
        ("↓纵移" if abs(r["dy"]) > abs(r["dx"]) * 1.5 and r["dy"] > 0 else
         "↑纵移" if abs(r["dy"]) > abs(r["dx"]) * 1.5 else
         "→横移" if r["dx"] > 0 else "←横移")
    print(f'镜{i:02d} {r["dur"]:4.1f}s  {d}  dy={r["dy"]:+6.2f} dx={r["dx"]:+6.2f} '
          f'速度={r["speed"]:5.2f}px/帧 缓动比={r["ease"]:.2f}')
mv = [r for r in rows if r["speed"] >= 0.15]
vert = sum(1 for r in mv if abs(r["dy"]) > abs(r["dx"]) * 1.5)
horz = sum(1 for r in mv if abs(r["dx"]) >= abs(r["dy"]) * 1.5)
eased = sum(1 for r in mv if r["ease"] < 0.65)
print(f"\n汇总: 总镜 {len(rows)} 静止 {len(rows)-len(mv)} 纵移 {vert} 横移 {horz} "
      f"对角 {len(mv)-vert-horz} | 缓动(头尾慢) {eased}/{len(mv)}")
print(f"镜长中位 {sorted(r['dur'] for r in rows)[len(rows)//2]}s")
