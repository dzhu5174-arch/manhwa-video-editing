# -*- coding: utf-8 -*-
"""parse_demo.py — 把用户的分镜 demo 视频解析成「每镜用哪张切片」。
原理：demo 里每段画面=某切片的局部放大。抽帧 → 内容掩码切镜 → 段中帧与全部切片
多尺度模板匹配 → 按时间重叠映射回 plan0 的镜头号。
用法: python parse_demo.py --config configs/ep02.json --demo "<demo.mp4>"
产出: work/ep02/demo_map.json + _demo_map.jpg（人工核对拼图）
"""
import argparse, json, os
import av, cv2, numpy as np
from PIL import Image, ImageDraw

FPS = 3
CUT_TH = 0.93
SLICE_W = 400  # 匹配用切片宽度


def imread_cn(path, gray=True):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)


def content_bbox(g, th=20):
    m = (g > th).astype(np.uint8)
    ys, xs = np.nonzero(m)
    if len(xs) < 500:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    if (x1 - x0) < 60 or (y1 - y0) < 60:
        return None
    return x0, y0, x1, y1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--demo", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8"))
    w = os.path.normpath(os.path.join(os.path.dirname(a.config), cfg["work"]))
    sdir = cfg["slices"][cfg["setkey"]]
    skip = {str(s) for s in cfg.get("pool_skip", [])}
    pool = sorted(f for f in os.listdir(sdir) if f.lower().endswith(".jpg")
                  and not f.startswith("_") and os.path.splitext(f)[0] not in skip)

    # 1) 抽帧 + 切镜（内容区域直方图）
    c = av.open(a.demo)
    vs = c.streams.video[0]; tb = float(vs.time_base)
    segs, prev_h, t0seg, mid, prev_crop = [], None, 0.0, None, None
    acc, t_last = 0.0, 0.0
    for f in c.decode(vs):
        t = float(f.pts * tb)
        if t < acc: continue
        acc = t + 1.0 / FPS; t_last = t
        g = cv2.cvtColor(f.to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY)
        bb = content_bbox(g)
        if bb is None:
            if prev_h is not None:  # 黑帧=段落结束
                segs.append([t0seg, t, mid]); prev_h = None; prev_crop = None
            continue
        crop = g[bb[1]:bb[3], bb[0]:bb[2]]
        small = cv2.resize(crop, (160, 120))
        h = cv2.normalize(cv2.calcHist([small], [0], None, [32], [0, 256]), None).flatten()
        # 直方图 + 均帧差双信号（暗场景 hist 不灵，帧差补；demo 是静帧，帧差>18 即换画面）
        diff = float(np.abs(small.astype(np.float32) - prev_crop.astype(np.float32)).mean()) if prev_crop is not None else 0.0
        prev_crop = small.copy()
        if prev_h is not None and (cv2.compareHist(prev_h, h, cv2.HISTCMP_CORREL) < CUT_TH or diff > 18):
            segs.append([t0seg, t, mid]); mid = None; t0seg = t
        if prev_h is None: t0seg = t
        if mid is None or abs(t - (t0seg + (t - t0seg) / 2)) < 0.2:
            mid = (crop.copy(), bb)
        prev_h = h
    if prev_h is not None: segs.append([t0seg, t_last, mid])
    segs = [s for s in segs if s[1] - s[0] > 0.5]
    # 长段二次细切（黑场/低对比场景 hist 不敏感，用逐帧差分补刀）
    fine = []
    for s0, s1, m in segs:
        if s1 - s0 <= 8.0:
            fine.append([s0, s1, m]); continue
        fine.append([s0, s1, m])  # 占位，细切在主循环外做不了，标记出来人工核
        print(f"  !! 长段 {s0:.1f}-{s1:.1f}s（{s1-s0:.0f}s）可能漏切，需人工核")
    segs = fine
    print(f"demo 切出 {len(segs)} 段")

    # 2) 切片库（匹配用小图）
    lib = []
    for fn in pool:
        im = imread_cn(os.path.join(sdir, fn))
        lib.append((fn, cv2.resize(im, (SLICE_W, int(SLICE_W * im.shape[0] / im.shape[1])))))

    # 3) 每段中帧 → 多尺度模板匹配
    results = []
    for i, (s0, s1, m) in enumerate(segs):
        crop, bb = m
        best = (0, None, None, None)
        for tw in (SLICE_W, 340, 280, 220):
            tt = cv2.resize(crop, (tw, int(tw * crop.shape[0] / crop.shape[1])))
            if tt.shape[0] < 40: continue
            for fn, g in lib:
                if tt.shape[0] > g.shape[0]: continue
                r = cv2.matchTemplate(g, tt, cv2.TM_CCOEFF_NORMED)
                _, mx, _, ml = cv2.minMaxLoc(r)
                if mx > best[0]:
                    best = (mx, fn, ml, (tw, tt.shape[0]))
        score, fn, ml, tsz = best
        # 匹配位置换算回切片源坐标（供取景参考）
        sc = None
        if fn:
            full = imread_cn(os.path.join(sdir, fn))
            k = full.shape[1] / SLICE_W
            sc = [int(ml[0] * k), int(ml[1] * k), int(tsz[0] * k), int(tsz[1] * k)]
        results.append(dict(seg=i, t0=round(s0, 2), t1=round(s1, 2),
                            img=fn, score=round(float(score), 3), rect=sc))
        print(f"seg{i:02d} {s0:6.1f}-{s1:6.1f}s -> {fn} score={score:.2f}")

    # 4) 映射到 plan0 镜头（归一化时间轴最近邻 + 单调约束；demo 与音频时长不同会累积漂移）
    plan0 = json.load(open(os.path.join(w, "plan0.json"), encoding="utf-8"))
    T_demo = max(r["t1"] for r in results); T_audio = max(p["t1"] for p in plan0)
    seg_frac = [((r["t0"] + r["t1"]) / 2) / T_demo for r in results]
    mp = {}; j = 0
    for p in plan0:
        f = ((p["t0"] + p["t1"]) / 2) / T_audio
        while j + 1 < len(results) and abs(seg_frac[j + 1] - f) < abs(seg_frac[j] - f):
            j += 1
        r = results[j]
        mp[str(p["shot"])] = dict(img=f'{cfg["setkey"]}_{r["img"]}',
                                  score=r["score"], rect=r["rect"], seg=r["seg"])
    json.dump(mp, open(os.path.join(w, "demo_map.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"demo_map: {len(mp)}/{len(plan0)} 镜匹配到切片 -> demo_map.json")

    # 5) 核对拼图：段中帧 | 匹配切片 并排
    tiles = []
    for r in results:
        crop = cv2.resize(results[r["seg"]]["crop"], (0, 0), fx=0.25, fy=0.25) if False else None
    rows = []
    for r in results:
        sgm = segs[r["seg"]]
        fr = cv2.cvtColor(sgm[2][0], cv2.COLOR_GRAY2BGR)
        fr = cv2.resize(fr, (160, int(160 * fr.shape[0] / fr.shape[1])))
        th = imread_cn(os.path.join(sdir, r["img"]), gray=False) if r["img"] else np.zeros((60, 120, 3), np.uint8)
        th = cv2.resize(th, (120, int(120 * th.shape[0] / th.shape[1])))
        h = max(fr.shape[0], th.shape[0], 50)
        row = np.zeros((h + 18, 292, 3), np.uint8)
        row[18:18 + fr.shape[0], :fr.shape[1]] = fr
        row[18:18 + th.shape[0], 166:166 + th.shape[1]] = th
        cv2.putText(row, f'{r["t0"]:.0f}s->{r["img"]} {r["score"]:.2f}', (4, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        rows.append(row)
    cols = 4
    per = (len(rows) + cols - 1) // cols
    ch = max(r.shape[0] for r in rows)
    canvas = np.zeros((per * ch, cols * 292, 3), np.uint8)
    for i, r in enumerate(rows):
        canvas[(i // cols) * ch:(i // cols) * ch + r.shape[0], (i % cols) * 292:(i % cols) * 292 + 292] = r
    cv2.imwrite(os.path.join(w, "_demo_map.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("核对拼图 -> _demo_map.jpg")


main()
