# -*- coding: utf-8 -*-
"""motion_survey.py — 批量扫描参考片运镜模式（零视觉token量化版）。
每镜输出：时长/方向/行程%画宽/速度/缓动比/累计缩放。断点续跑：已有JSON跳过。
用法: python motion_survey.py "<folder>" [--out "<outdir>"]
"""
import sys, io, os, json, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import av, cv2, numpy as np

folder = sys.argv[1]
outdir = os.path.join(folder, "_survey")
if "--out" in sys.argv:
    outdir = sys.argv[sys.argv.index("--out") + 1]
os.makedirs(outdir, exist_ok=True)
W0 = 360  # 分析宽度

def scan(path):
    c = av.open(path)
    vs = c.streams.video[0]
    fps = float(vs.average_rate or 30)
    step = max(1, round(fps / 10))
    frames, idx, prev = [], 0, None
    for fr in c.decode(vs):
        idx += 1
        if idx % step:
            continue
        img = cv2.resize(fr.to_ndarray(format="gray"), (W0, int(W0 * fr.height / fr.width)))
        if prev is not None:
            frames.append(img)
        prev = img
    W, H = vs.width, vs.height
    c.close()
    # 黑边检测（首帧上/下 2% 行均值）
    f0 = frames[0]
    bar = bool(np.mean(f0[:max(1, len(f0)//50)]) < 16 or np.mean(f0[-max(1, len(f0)//50):]) < 16)
    # 切镜
    shots, start = [], 0
    h0 = cv2.calcHist([frames[0]], [0], None, [32], [0, 256])
    for i in range(1, len(frames)):
        h1 = cv2.calcHist([frames[i]], [0], None, [32], [0, 256])
        corr = cv2.compareHist(h0, h1, cv2.HISTCMP_CORREL)
        if corr < 0.80:
            if i - start >= 8:
                shots.append((start, i))
            start = i
            h0 = h1
    if len(frames) - start >= 8:
        shots.append((start, len(frames)))
    # 镜内探测
    win = None
    small = [cv2.resize(f, (180, int(180 * f.shape[0] / f.shape[1]))) for f in frames]
    rows = []
    for (a, b) in shots:
        if win is None or win.shape != (frames[a].shape[0], frames[a].shape[1]):
            win = cv2.createHanningWindow((frames[a].shape[1], frames[a].shape[0]), cv2.CV_32F)
        dxs, dys, mags, scale_acc = [], [], [], 1.0
        for j in range(a, b - 1):
            f1 = frames[j].astype(np.float32); f2 = frames[j + 1].astype(np.float32)
            (dx, dy), resp = cv2.phaseCorrelate(f1, f2, win)
            if resp > 0.15:
                dxs.append(dx); dys.append(dy); mags.append((dx**2 + dy**2) ** .5)
            # ECC 仿射估计缩放（小图，失败跳过）
            try:
                _, wmat = cv2.findTransformECC(small[j].astype(np.float32), small[j + 1].astype(np.float32),
                                               np.eye(2, 3, dtype=np.float32), cv2.MOTION_AFFINE,
                                               (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 8, 1e-4))
                det = wmat[0, 0] * wmat[1, 1] - wmat[0, 1] * wmat[1, 0]
                if 0.8 < det < 1.25:
                    scale_acc *= float(det) ** .5
            except cv2.error:
                pass
        if not mags:
            rows.append(dict(dur=round((b - a) / 10, 1), dir="静止", travel=0.0,
                             speed=0.0, ease=1.0, scale=round(scale_acc, 4)))
            continue
        med_dx, med_dy = float(np.median(dxs)), float(np.median(dys))
        k = max(2, len(mags) // 5)
        edge = (np.mean(mags[:k]) + np.mean(mags[-k:])) / 2
        mid = np.mean(mags[k:-k]) or 1e-6
        speed = float(np.median(mags))
        d = ("静止" if speed < 0.15 else
             ("↓" if abs(med_dy) > abs(med_dx) * 1.5 and med_dy > 0 else
              "↑" if abs(med_dy) > abs(med_dx) * 1.5 else
              "→" if med_dx > 0 else "←"))
        travel = speed * (b - a) / W0  # 行程 = 中位速度×采样帧数 / 分析宽
        rows.append(dict(dur=round((b - a) / 10, 1), dir=d, travel=round(travel * 100, 1),
                         speed=round(speed, 2), ease=round(float(edge / mid), 2),
                         scale=round(scale_acc, 4)))
    # 汇总
    mv = [r for r in rows if r["dir"] != "静止"]
    vert_dirs = [r["dir"] for r in mv if r["dir"] in "↓↑"]
    flips = sum(1 for i in range(1, len(vert_dirs)) if vert_dirs[i] != vert_dirs[i - 1])
    zooms = [r for r in mv if abs(r["scale"] - 1) > 0.005]
    durs = sorted(r["dur"] for r in rows)
    summary = dict(
        n_shots=len(rows), static=len(rows) - len(mv),
        vert=sum(1 for r in mv if r["dir"] in "↓↑"),
        horz=sum(1 for r in mv if r["dir"] in "←→"),
        alt_rate=round(flips / max(1, len(vert_dirs) - 1), 2),
        dur_med=durs[len(durs) // 2],
        travel_med=sorted(r["travel"] for r in mv)[len(mv) // 2] if mv else 0,
        ease_med=sorted(r["ease"] for r in mv)[len(mv) // 2] if mv else 1,
        zoom_ratio=round(len(zooms) / max(1, len(mv)), 2),
        zoom_med=sorted(abs(r["scale"] - 1) for r in zooms)[len(zooms) // 2] if zooms else 0,
    )
    return dict(file=os.path.basename(path), w=W, h=H, blackbar=bar,
                shots=rows, summary=summary)

for p in sorted(glob.glob(os.path.join(folder, "*.mp4"))):
    name = os.path.basename(p)
    out = os.path.join(outdir, os.path.splitext(name)[0][:24] + ".json")
    if os.path.exists(out):
        print(f"跳过(已有): {name[:40]}")
        continue
    print(f"扫描: {name[:50]} ...", flush=True)
    try:
        r = scan(p)
        json.dump(r, open(out, "w", encoding="utf-8"), ensure_ascii=False)
        s = r["summary"]
        print(f'  {r["w"]}x{r["h"]} 黑边={r["blackbar"]} 镜{s["n_shots"]} 静{s["static"]} '
              f'纵{s["vert"]} 横{s["horz"]} 交替率{s["alt_rate"]} 镜长中位{s["dur_med"]}s '
              f'行程中位{s["travel_med"]}% 缓动比中位{s["ease_med"]} 缩放镜占比{s["zoom_ratio"]}(中位幅度{s["zoom_med"]})', flush=True)
    except Exception as e:
        print(f"  !! 失败: {e}", flush=True)
print("done")
