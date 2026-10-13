# -*- coding: utf-8 -*-
"""slice_bands.py v3 — 单格化窗口终版：死重掩码 + 窗口滑动打分
死重 = 大块白斑(气泡/白底) ∪ 平色大矩形(蓝/黑字幕框)；内容 = 其余。
每带窗口 = 滑动+变焦搜索 argmax(窗内内容密度 + 脸合规加分)，脸窗优先，阅读顺序补窗。
"""
import json, os, re
import cv2, numpy as np

SLICES = r"C:\Users\19150\Desktop\manhua video v2\勇者无法休息\源素材\第2话\切片超清2x"
FACES = r"C:\Users\19150\Desktop\manhua video v2\勇者无法休息\脚本_量产管线\work\ep02\faces.json"
YUNET = r"D:\WorkBuddy2\2026-10-12-15-24-12\_scratch\yunet.onnx"
OUT = r"C:\Users\19150\Desktop\manhua video v2\勇者无法休息\脚本_量产管线\work\ep02\切片窗口方案"
os.makedirs(OUT, exist_ok=True)
W_SLICE = 1800
CW, CH = 1440.0, 1080.0
S_MIN, S_MAX = 0.8, 2.6
MARGIN = 60
TRAVEL = 144.0
SS = 4          # 分析降采样倍数（1800→450）

def imread_cn(p): return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
DET = None
def yunet(img, score=0.35):
    global DET
    h, w = img.shape[:2]
    if DET is None: DET = cv2.FaceDetectorYN.create(YUNET, "", (w, h), score)
    DET.setInputSize((w, h))
    _, r = DET.detect(img)
    return [[float(v) for v in f[:4]] for f in r] if r is not None else []

def dead_mask(img):
    """死重掩码（原生分辨率→SS 倍降采样）：白斑大域 ∪ 平色大矩形。"""
    small = cv2.resize(img, (img.shape[1]//SS, img.shape[0]//SS), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    white = (g > 232).astype(np.uint8)
    dead = np.zeros_like(white)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 0.008 * h * w:
            dead[lab == i] = 1
    # 平色矩形：局部方差极低的大区域（字幕框底色）
    med = cv2.blur(g.astype(np.float32), (25, 25))
    var = cv2.blur((g.astype(np.float32) - med) ** 2, (25, 25))
    flat = ((var < 9) & (g < 232) & (g > 8)).astype(np.uint8)
    flat = cv2.morphologyEx(flat, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))  # 跨过文字连成整框
    n, lab, stats, _ = cv2.connectedComponentsWithStats(flat, 8)
    for i in range(1, n):
        x, y, ww, hh, a = stats[i]
        if a > 0.02 * h * w and min(ww, hh) > 8:   # 面积大且非细线
            dead[lab == i] = 1
    # 纯黑过渡帧（黑底白字 THAT'S RIGHT 类）：极暗平色大域
    blk = ((var < 9) & (g <= 25)).astype(np.uint8)
    blk = cv2.morphologyEx(blk, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(blk, 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 0.03 * h * w:
            dead[lab == i] = 1
    return dead.astype(np.float32), small.shape[0], small.shape[1]

def text_page_ratio(small, ys, xs, hs, ws):
    """主成分占比：文字页=许多同尺寸笔画（占比低），插画=一个大主体（占比高）。"""
    patch = small[ys:ys+hs, xs:xs+ws]
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    ink = (g < 225).astype(np.uint8)
    total = int(ink.sum())
    if total < 0.02 * hs * ws: return 1.0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    if n <= 2: return 1.0
    areas = sorted((stats[i, cv2.CC_STAT_AREA] for i in range(1, n)), reverse=True)
    return areas[0] / total

def bands_of(dead, sh):
    row_content = 1.0 - dead.mean(axis=1)
    k = np.ones(11) / 11
    prof = np.convolve(row_content, k, mode="same")
    solid = prof > 0.5
    bands, i = [], 0
    while i < sh:
        if solid[i]:
            j = i
            while j < sh and solid[j]: j += 1
            if (j - i) * SS >= 140: bands.append([i * SS, j * SS])
            i = j
        else: i += 1
    merged = []
    for b in bands:
        if merged and b[0] - merged[-1][1] < 320: merged[-1][1] = b[1]
        else: merged.append(b)
    return merged

def search_windows(bt, bb, dead, sh, faces_in, H, SS=4, small=None):
    """滑动+变焦打分（全原生域）。返回按 top 排序的窗口列表。"""
    cands = []
    s_list = [0.9, 1.1, 1.3, 1.6, 1.9, 2.2, 2.5]
    face_s = None
    if faces_in:
        fh = max(f[3] for f in faces_in)
        face_s = float(np.clip(0.42 * CH / fh, S_MIN, S_MAX))
        s_list = [face_s] + [s for s in s_list if abs(s - face_s) > 0.01]
    Ws_lim = int(W_SLICE / SS)
    for s in s_list:
        Ww, Hw = CW / s, CH / s
        wsm, hsm = max(1, int(Ww / SS)), max(1, int(Hw / SS))
        if hsm > sh: continue
        y_lo, y_hi = max(0, bt - MARGIN), min(H - Hw, bb + MARGIN)
        if y_hi < y_lo: y_lo = y_hi = max(0, min(y_lo, H - Hw))
        x_cands = [0.0] if Ww >= W_SLICE - 1 else (
            [float(np.clip(np.mean([f[0] + f[2] / 2 for f in faces_in]) - Ww / 2, 0, W_SLICE - Ww))] if faces_in
            else [0.0, (W_SLICE - Ww) / 2, W_SLICE - Ww])
        for yn in range(int(y_lo), int(y_hi) + 1, 120):
            for x0 in x_cands:
                xsm = int(x0 / SS)
                xsm = min(xsm, Ws_lim - wsm)
                ysm = yn // SS
                patch = dead[ysm:ysm + hsm, xsm:xsm + wsm]
                if patch.size == 0: continue
                fill = float(1.0 - patch.mean())
                if small is not None:
                    tpr = text_page_ratio(small, ysm, xsm, hsm, wsm)
                    if tpr < 0.35: fill *= 0.25        # 文字页重罚
                fb = 0.0
                if faces_in:
                    f = max(faces_in, key=lambda f: f[3])
                    fx, fy, fw2, fh2 = f[:4]
                    inside = (x0 <= fx and fx + fw2 <= x0 + Ww and yn <= fy and fy + fh2 <= yn + Hw)
                    ratio = fh2 * s / CH
                    if inside and 0.30 <= ratio <= 0.60: fb = 0.5
                    elif inside: fb = 0.15
                cands.append((fill + fb, s, x0, yn, "face" if (faces_in and s == face_s) else "search"))
    cands.sort(key=lambda c: (-c[0], c[3]))
    chosen = []
    for sc, s, x0, top, mode in cands:
        Hw = CH / s
        if any(not (top + Hw < t or top > t + CH / ss) for _, ss, _, t, _ in chosen): continue
        chosen.append((sc, s, x0, top, mode))
        if len(chosen) >= 5: break
    return [dict(s=round(s, 4), x0=round(x0, 1), top_start=round(top, 1),
                 top_end=round(top + TRAVEL / s, 1), mode=mode, band=[bt, bb],
                 score=round(sc, 3)) for sc, s, x0, top, mode in sorted(chosen, key=lambda c: c[3])]

def main():
    faces = json.load(open(FACES, encoding="utf-8"))
    result, redone = {}, []
    vis = os.path.join(OUT, "_vis"); os.makedirs(vis, exist_ok=True)
    for name in sorted(os.listdir(SLICES)):
        if not re.fullmatch(r"\d{3}\.jpg", name): continue
        tag = "c2_" + name[:3]
        img = imread_cn(os.path.join(SLICES, name))
        if img is None: continue
        H = img.shape[0]
        fl = faces.get(tag + ".jpg") or []
        if not fl:
            fl = yunet(img, 0.35)
            if fl: redone.append(tag)
        dead, sh, sw = dead_mask(img)
        bands = bands_of(dead, sh)
        wins = []
        for (bt, bb) in bands:
            fin = [f for f in fl if f[1] + f[3] > bt and f[1] < bb]
            wins += search_windows(bt, bb, dead, sh, fin, H)
        wins.sort(key=lambda w: w["top_start"])
        result[tag] = dict(h=H, bands=bands, faces=fl, windows=wins)
        scale = 240.0 / W_SLICE
        th = cv2.resize(img, (240, max(1, int(H * scale))))
        dm = (dead * 255).astype(np.uint8)
        dm3 = cv2.cvtColor(cv2.resize(dm, (240, max(1, int(sh * scale / 1)))), cv2.COLOR_GRAY2BGR)
        dm3 = cv2.resize(dm3, (240, th.shape[0]))
        ov = cv2.addWeighted(th, 0.75, dm3, 0.25, 0)
        for w in wins:
            cv2.rectangle(ov, (int(w["x0"]*scale), int(w["top_start"]*scale)),
                          (int((w["x0"] + CW/w["s"])*scale), int((w["top_start"] + CH/w["s"])*scale)), (0, 0, 255), 1)
        cv2.imencode(".jpg", ov, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(os.path.join(vis, name))
    json.dump(result, open(os.path.join(OUT, "切片窗口方案.json"), "w", encoding="utf-8"), ensure_ascii=False)
    nm = [len(v["windows"]) for v in result.values()]
    print(f"切片 {len(result)}，窗口 {sum(nm)}（均值 {np.mean(nm):.1f}），重检出脸 {len(redone)} 片")
    for t in ("c2_003", "c2_078", "c2_021", "c2_081"):
        v = result[t]
        print(f"{t}: 带={v['bands']} 脸={len(v['faces'])}")
        for w in v["windows"][:3]:
            print(f"   s={w['s']} x0={w['x0']} top={w['top_start']}→{w['top_end']} {w['mode']} score={w['score']}")

if __name__ == "__main__":
    main()
