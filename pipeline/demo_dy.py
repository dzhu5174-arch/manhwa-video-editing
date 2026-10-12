# -*- coding: utf-8 -*-
"""
demo_dy.py — 把用户 demo（10月7日 (1).mp4，3064x2048 黑底摆片）重做为抖音 9:16 满屏版。
规范源：skill\\jianying-editing\\SKILL.md（§5 气泡检测两档 + §6 对标数值卡）
流程：shots(镜头表+bbox) → 逐镜【气泡检测带→内容块→窗只落内容块】→ plan_dy.json
      → preview(同参渲染) → 全量帧阵复核 → film_qc 闸门 → 剪映草稿(原生关键帧)
用法：
  python demo_dy.py windows          # 取窗 → plan_dy.json + 全量帧阵
  python demo_dy.py preview          # 渲染预览片
  python demo_dy.py draft            # 生成剪映草稿
  python demo_dy.py all
"""
import json, os, subprocess, sys

import numpy as np
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(BASE)
SRC = os.path.join(PROJ, "demo", "dy_work", "demo_h264.mp4")
OUT = os.path.join(PROJ, "demo", "dy_work")
SHOTS = os.path.join(OUT, "shots2.json")
PLAN = os.path.join(OUT, "plan_dy.json")

CW, CH = 1080, 1920          # 抖音画布
SW, SH = 3064, 2048          # 源分辨率
AR = CW / CH                 # 0.5625
INSET = 8                    # bbox 内缩，防黑丝入画

# ---- §6 数值卡（对标 10 部实测基准）----
TRAVEL_MIN, TRAVEL_MAX = 90, 180   # 平移行程（画布 px）
SPEED = 70.0                       # 平移速度 px/s（小小 23-33 慢滑 / 白羊 83-123 快滑，取中）
ZOOM_RATE = 0.015                  # 微缩放速率 1.5%/s
ZOOM_MIN, ZOOM_MAX = 0.02, 0.06    # 幅度 2-6%/镜
LONG_SHOT = 7.5                    # >7.5s 拆镜（§6.2）
MIN_BLOCK = 160                    # 内容块最小高度（源 px）
BUBBLE_PENALTY = 3.0               # 窗内气泡带占比惩罚权重
BAND_TARGET = 0.10                 # 窗内气泡带占比达标线（≤ 即接受；达不到则逐级缩窗）


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"cmd fail: {' '.join(cmd)}\n{r.stderr[-500:]}")
    return r.stdout


def get_frame(t, cache={}):
    key = round(t, 2)
    if key not in cache:
        p = os.path.join(OUT, "frames", f"full_{key}.png")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if not os.path.exists(p):
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", SRC, "-frames:v", "1", p],
                           capture_output=True)
        cache[key] = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR) if os.path.exists(p) else None
    return cache[key]


# ---------- §5 气泡检测（彩色漫画版：连通域矩形块）----------
def bubble_mask(gray):
    """白气泡/文字框像素掩码：近白区域先做闭运算（把气泡内文字的缝隙填上，否则白底会被
    文字切成小连通域而漏检），再取满足 ①宽>10%W ②高>4%H ③填充率>0.5 ④内部有深色文字的块。"""
    h, w = gray.shape
    white = (gray > 215).astype(np.uint8)
    k = np.ones((11, 11), np.uint8)
    closed = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    mask = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        x, y, bw_, bh_, area = stats[i]
        if bw_ < w * 0.10 or bh_ < h * 0.04:
            continue
        if area / max(1, bw_ * bh_) < 0.5:
            continue
        sub = gray[y:y + bh_, x:x + bw_]
        if (sub < 120).mean() < 0.01:      # 无深色文字 → 不是气泡
            continue
        mask[y:y + bh_, x:x + bw_] = 1
    return mask


def region_ratio(ii, x0, y0, ww, wh):
    """2D 积分图求窗口内掩码占比。"""
    x1, y1 = x0 + ww, y0 + wh
    s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    return float(s) / max(1.0, ww * wh)


def band_mask(g):
    """行级检测带：白色气泡/大字 (frac>200)>0.33 且 std>12；深色气泡 mean<60 且 std>25。"""
    frac = (g > 200).mean(axis=1)
    mean = g.mean(axis=1)
    std = g.std(axis=1)
    white = (frac > 0.33) & (std > 12)
    dark = (mean < 60) & (std > 25)
    return white | dark, white, dark


def content_blocks(mask):
    """mask=True 为检测带；返回无检测带的连续区间 [(start,end),...]（end 不含）。"""
    blocks = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            j = i
            while j < n and not mask[j]:
                j += 1
            if j - i >= MIN_BLOCK:
                blocks.append((i, j))
            i = j
        else:
            i += 1
    return blocks


def ink_profile(g):
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    ink = (np.abs(gx) + np.abs(gy)).mean(axis=1)
    ink = ink / max(ink.max(), 1e-6)
    return ink


def clamp_block(win, bx0, bx1, by0, by1):
    """窗口钳进内容块（等比收缩保 9:16），块内可平移。"""
    x, y, ww, wh = win
    bw, bh = bx1 - bx0, by1 - by0
    if bw <= 0 or bh <= 0:
        return list(map(int, win))
    if ww > bw or wh > bh:
        ww = min(ww, bw, bh * AR)
        wh = ww / AR
    cx, cy = x + ww / 2.0, y + wh / 2.0
    cx = max(bx0 + ww / 2.0, min(cx, bx1 - ww / 2.0))
    cy = max(by0 + wh / 2.0, min(cy, by1 - wh / 2.0))
    return [int(cx - ww / 2), int(cy - wh / 2), int(ww), int(wh)]


def detect_faces(gray, cascade):
    """anime 人脸检测（宽容档 1.05/3，项目实测召回最高）。返回 [(x,y,w,h)]。"""
    if cascade is None or cascade.empty():
        return []
    try:
        boxes = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(24, 24))
        return [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes]
    except cv2.error:
        return []


def load_faces():
    """读 faces_dy.json（隔离 venv 的 cv2 4.10 跑出，本机 cv2 5.0 无 CascadeClassifier）。"""
    p = os.path.join(OUT, "faces_dy.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    print("⚠ 未找到 faces_dy.json（人脸偏好缺失）→ 先跑 faces_dy.py")
    return {}


def trim_bbox(g, thr=18, pad=14):
    """边缘黑带精修：按行/列平均亮度剔除黑边（demo 摆片边缘常有黑残）。
    返回精修后的 (x0, x1, y0, y1)（相对 g 的坐标）；失败返回 None。"""
    h, w = g.shape
    rowmean = g.mean(axis=1)
    colmean = g.mean(axis=0)
    rows = np.where(rowmean > thr)[0]
    cols = np.where(colmean > thr)[0]
    if len(rows) < 10 or len(cols) < 10:
        return None
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    x0, y0 = x0 + pad, y0 + pad
    x1, y1 = x1 - pad, y1 - pad
    if x1 - x0 < 60 or y1 - y0 < 100:
        return None
    return x0, x1, y0, y1


def edge_adjust(win, frame_gray, min_mean=26, max_iter=8):
    """边缘校验回退：四边 4px 带宽均值 < min_mean 则等比收紧窗口（保 9:16）。"""
    x, y, ww, wh = win
    for _ in range(max_iter):
        x = max(0, min(x, SW - ww)); y = max(0, min(y, SH - wh))
        c = frame_gray[int(y):int(y + wh), int(x):int(x + ww)]
        if c.size == 0:
            return [int(x), int(y), int(ww), int(wh)]
        ev = min(c[:4, :].mean(), c[-4:, :].mean(), c[:, :4].mean(), c[:, -4:].mean())
        if ev >= min_mean:
            break
        nw, nh = ww * 0.92, wh * 0.92
        cx, cy = x + ww / 2.0, y + wh / 2.0
        nw = max(120.0, nw)
        x, y, ww, wh = int(cx - nw / 2), int(cy - nh / 2), int(nw), int(nh)
    return [int(x), int(y), int(ww), int(wh)]


def plan_one_shot(rec, frame, idx, faces=None):
    """单镜取窗。优先级（skill 速查卡）：脸可见 → 满幅裁全 → 避气泡 → 运动。
    窗尺寸按内容满幅定死（不再为避气泡缩窗），气泡靠"选位置排到窗外"规避。"""
    x0b, x1b, y0b, y1b = rec["bb"]
    x0b += INSET; x1b -= INSET; y0b += INSET; y1b -= INSET
    if x1b - x0b <= 0 or y1b - y0b <= 0:
        return None
    crop = frame[y0b:y1b, x0b:x1b]
    g0 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    tb = trim_bbox(g0)                        # 精修：剔除 bbox 内的边缘黑带
    if tb:
        x0b, x1b, y0b, y1b = x0b + tb[0], x0b + tb[1], y0b + tb[2], y0b + tb[3]
        crop = frame[y0b:y1b, x0b:x1b]
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    else:
        g = g0
    bw, bh = x1b - x0b, y1b - y0b
    bmask = bubble_mask(g)                     # 像素级气泡掩码（连通域矩形块）
    ii = cv2.integral(bmask.astype(np.float64))  # 2D 积分图
    ink = ink_profile(g)
    cum = np.concatenate([[0.0], np.cumsum(ink)])
    faces = faces or []
    if tb:                                    # 脸框坐标随精修平移
        faces = [(fx - tb[0], fy - tb[2], fw, fh) for (fx, fy, fw, fh) in faces]

    dur = rec["t1"] - rec["t0"]
    amp = min(max(ZOOM_RATE * dur, ZOOM_MIN), ZOOM_MAX)
    travel_c = min(max(SPEED * dur, TRAVEL_MIN), TRAVEL_MAX)

    # ---- 取窗：满幅优先；气泡超标则逐级缩窗避泡（skill v7：气泡必须排出画外）----
    scales = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65]
    pick, fallback = None, None
    for sc in scales:
        if bh >= bw / AR:
            ww = int(round(bw * sc)); wh = int(round(ww / AR)); axis = "v"
        else:
            wh = int(round(bh * sc)); ww = int(round(wh * AR)); axis = "h"
        if wh > bh or ww > bw or ww < 60 or wh < 100:
            continue
        xmax, ymax = max(0, bw - ww), max(0, bh - wh)
        xs = {xmax // 2}
        for (fx, fy, fw, fh) in faces:
            xs.add(int(min(max(fx + fw / 2 - ww / 2, 0), xmax)))
        for t in range(0, xmax + 1, max(8, ww // 4)):
            xs.add(int(t))
        xs = sorted(x for x in xs if 0 <= x <= xmax)
        step_y = max(6, wh // 40)
        best_local = None
        for xo in xs:
            for yo in range(0, ymax + 1, step_y):
                s_ink = (cum[yo + wh] - cum[yo]) / wh
                band = region_ratio(ii, xo, yo, ww, wh)
                s_face = 0.0
                for (fx, fy, fw, fh) in faces:
                    fcx, fcy = fx + fw / 2.0, fy + fh / 2.0
                    if xo <= fcx <= xo + ww and yo <= fcy <= yo + wh:
                        s_face += 1.0 + 0.5 * (1 - abs(fcy - (yo + wh / 2.0)) / max(1, wh / 2.0))
                score = 2.2 * s_face + s_ink - BUBBLE_PENALTY * band
                if best_local is None or score > best_local[0]:
                    best_local = (score, xo, yo, band)
        if best_local is None:
            continue
        if fallback is None or best_local[3] < fallback[0]:
            fallback = (best_local[3], sc, best_local[1], best_local[2])
        if best_local[3] <= BAND_TARGET:      # 达标即停（保证"最大且达标"的窗优先）
            pick = (sc, best_local[1], best_local[2], ww, wh, best_local[3], axis)
            break
    if pick is None:                          # 全都不达标 → 取 band 最小的
        bandv, sc, xo, yo = fallback
        if bh >= bw / AR:
            ww = int(round(bw * sc)); wh = int(round(ww / AR)); axis = "v"
        else:
            wh = int(round(bh * sc)); ww = int(round(wh * AR)); axis = "h"
        pick = (sc, xo, yo, ww, wh, bandv, axis)
    sc, x0w, y0w, ww, wh, band_in, axis = pick

    # 平移：行程 clamp(70px/s×dur, 90, 180)，纵向优先（纵:横≈4:1），上下混排
    travel_src = min(travel_c * ww / float(CW), max(0, bh - wh) if axis == "v" else max(0, bw - ww))
    down = (idx % 3) != 2
    if axis == "v" and travel_src > 4:
        ts = min(y0w, (bh - wh) - travel_src) if down else max(y0w, travel_src)
        te = ts + travel_src if down else ts - travel_src
        w0 = [int(x0b + x0w), int(y0b + ts), int(ww), int(wh)]
        we = [int(x0b + x0w), int(y0b + te), int(ww), int(wh)]
    elif axis == "h" and (bw - ww) >= 4:
        right = down
        ts = min(x0w, (bw - ww) - travel_src) if right else max(x0w, travel_src)
        te = ts + travel_src if right else ts - travel_src
        w0 = [int(x0b + ts), int(y0b + y0w), int(ww), int(wh)]
        we = [int(x0b + te), int(y0b + y0w), int(ww), int(wh)]
    else:
        travel_src = 0.0
        w0 = [int(x0b + x0w), int(y0b + y0w), int(ww), int(wh)]
        we = list(w0)
    # 微缩放（绕终点窗中心，推近/拉远对称 49:51），钳在 bbox 内
    k = (1 + amp) if (idx % 2 == 0) else (1 - amp)
    cx = we[0] + ww / 2.0; cy = we[1] + wh / 2.0
    nw = min(ww * k, bw); nh = nw / AR
    ncx = max(x0b + nw / 2, min(cx, x1b - nw / 2))
    ncy = max(y0b + nh / 2, min(cy, y1b - nh / 2))
    wz = [int(ncx - nw / 2), int(ncy - nh / 2), int(nw), int(nh)]
    wz[0] = max(0, min(wz[0], SW - wz[2])); wz[1] = max(0, min(wz[1], SH - wz[3]))
    w0[0] = max(0, min(w0[0], SW - w0[2])); w0[1] = max(0, min(w0[1], SH - w0[3]))
    # 边缘校验回退（防黑边）
    fg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    w0 = edge_adjust(w0, fg)
    wz = edge_adjust(wz, fg)
    return {"win0": w0, "win1": wz,
            "travel_canvas": round(travel_c if travel_src > 0 else 0, 1),
            "zoom_pct": round((k - 1) * 100, 2), "scale_used": sc,
            "axis": axis, "faces": len(faces), "band_in_window": round(float(band_in), 3),
            "block": [0, int(bh)]}


def argmax_window(cum, ink, bs, be, wh, lo=None, hi=None):
    """在块内滑动窗高 wh，返回 ink 和最大的窗顶行（源坐标，含 y0b 偏移由调用方处理）。"""
    lo = bs if lo is None else max(bs, lo)
    hi = (be - wh) if hi is None else min(be - wh, hi)
    if hi <= lo:
        return int(bs)
    best_v, best_t = -1, lo
    step = max(4, int(wh // 60))
    for t in range(int(lo), int(hi) + 1, step):
        v = cum[t + int(wh)] - cum[t]
        if v > best_v:
            best_v, best_t = v, t
    return int(best_t)


def cmd_windows():
    faces_all = load_faces()
    print("人脸缓存: %d 镜有脸框" % sum(1 for v in faces_all.values() if v))
    shots = json.load(open(SHOTS, encoding="utf-8"))
    plan = []
    for rec in shots:
        if rec.get("empty") or not rec.get("bb"):
            continue
        mid = (rec["t0"] + rec["t1"]) / 2
        frame = get_frame(mid)
        if frame is None:
            continue
        dur = rec["t1"] - rec["t0"]
        nsub = 2 if dur > LONG_SHOT else 1     # >7.5s 拆镜
        for k in range(nsub):
            sub = {"i": rec["i"], "sub": k,
                   "t0": rec["t0"] + dur * k / nsub, "t1": rec["t0"] + dur * (k + 1) / nsub,
                   "bb": rec["bb"]}
            r = plan_one_shot(sub, frame, rec["i"] * 10 + k, faces_all.get(str(rec["i"]), []))
            if r is None:
                continue
            plan.append({"i": rec["i"], "sub": k, "t0": round(sub["t0"], 3), "t1": round(sub["t1"], 3),
                         "dur": round(sub["t1"] - sub["t0"], 3), "win0": r["win0"], "win1": r["win1"],
                         "travel": r["travel_canvas"], "zoom": r["zoom_pct"], "scale": r["scale_used"],
                         "axis": r["axis"], "faces": r["faces"], "band_in_window": r["band_in_window"]})
    json.dump(plan, open(PLAN, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("plan shots:", len(plan))
    import statistics as st
    print("镜时长中位 %.2fs | 平移镜 %d | 平均行程 %.0fpx | 微缩放中位 %.1f%%"
          % (st.median(p["dur"] for p in plan),
             sum(1 for p in plan if p["travel"] > 0),
             st.mean([p["travel"] for p in plan if p["travel"] > 0]) or 0,
             st.median([abs(p["zoom"]) for p in plan])))
    cmd_framesheet()


def cmd_framesheet():
    """全量帧阵：每镜起幅+落幅（§5 不允许抽查）。"""
    plan = json.load(open(PLAN, encoding="utf-8"))
    tiles = []
    for p in plan:
        fr = get_frame((p["t0"] + p["t1"]) / 2)
        imgs = []
        for w in (p["win0"], p["win1"]):
            x, y, ww, wh = w
            c = fr[y:y + wh, x:x + ww]
            c = cv2.resize(c, (216, 384), interpolation=cv2.INTER_AREA)
            imgs.append(c)
        tile = np.hstack(imgs)
        cv2.putText(tile, f"#{p['i']}.{p['sub']} t{p['dur']:.1f}s T{int(p['travel'])} Z{p['zoom']:+.0f}%",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        tiles.append(tile)
    # 分片输出（§5 全量复核，分片便于逐张目检）
    per = 6
    nparts = 0
    for start in range(0, len(tiles), 12):
        part = tiles[start:start + 12]
        rows = []
        for k in range(0, len(part), per):
            row = part[k:k + per]
            while len(row) < per:
                row.append(np.zeros_like(tiles[0]))
            rows.append(np.hstack(row))
        sheet = np.vstack(rows)
        nparts += 1
        out = os.path.join(OUT, f"framesheet_{nparts:02d}.png")
        cv2.imencode(".png", sheet)[1].tofile(out)
    print(f"framesheet: {nparts} 片（每片12镜，共{len(tiles)}镜）=> {OUT}")


def cmd_preview():
    plan = json.load(open(PLAN, encoding="utf-8"))
    prev = os.path.join(OUT, "preview_dy.mp4")
    vw = cv2.VideoWriter(prev, cv2.VideoWriter_fourcc(*"mp4v"), 30, (CW, CH))
    if not vw.isOpened():
        sys.exit("VideoWriter 打不开")
    nfrm = 0
    for k, p in enumerate(plan):
        cap = cv2.VideoCapture(SRC)
        cap.set(cv2.CAP_PROP_POS_MSEC, p["t0"] * 1000)
        nf = int(round(p["dur"] * 30))
        x0, y0, w0, h0 = p["win0"]
        x1, y1, w1, h1 = p["win1"]
        for i in range(nf):
            ok, fr = cap.read()
            if not ok:
                break
            u = i / max(1, nf - 1)
            e = u * u * (3 - 2 * u)
            x = x0 + (x1 - x0) * e; y = y0 + (y1 - y0) * e
            w = w0 + (w1 - w0) * e; h = h0 + (h1 - h0) * e
            x = max(0, min(x, SW - w)); y = max(0, min(y, SH - h))
            crop = fr[int(y):int(y + h), int(x):int(x + w)]
            if crop.size == 0:
                continue
            vw.write(cv2.resize(crop, (CW, CH), interpolation=cv2.INTER_AREA))
            nfrm += 1
        cap.release()
    vw.release()
    # 转 H.264 + 合入原 demo 音频（用户配音）→ 最终成品
    final = os.path.join(OUT, "demo_抖音版.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", prev, "-i", SRC,
                    "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "19", "-pix_fmt", "yuv420p", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                    "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart", final], capture_output=True)
    os.remove(prev)
    prev = final
    # 黑边检查（整条边均值<20 才算黑边）
    chk = subprocess.run(["ffmpeg", "-v", "error", "-i", prev, "-vf", "fps=2,scale=540:960,format=gray",
                          "-f", "rawvideo", "-"], capture_output=True)
    a = np.frombuffer(chk.stdout, np.uint8)
    n = len(a) // (540 * 960)
    a = a[:n * 540 * 960].reshape(n, 960, 540) if n else np.zeros((1, 960, 540), np.uint8)
    bad = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    for f in a:
        if f[:4, :].mean() < 20: bad["top"] += 1
        if f[-4:, :].mean() < 20: bad["bottom"] += 1
        if f[:, :4].mean() < 20: bad["left"] += 1
        if f[:, -4:].mean() < 20: bad["right"] += 1
    print(f"成品: {nfrm}帧 采样{n}帧 | 整条黑边帧 {bad}")
    print("成品 =>", prev)
    # 黑边检查（整条边均值<20 才算黑边）
    chk = subprocess.run(["ffmpeg", "-v", "error", "-i", prev, "-vf", "fps=2,scale=540:960,format=gray",
                          "-f", "rawvideo", "-"], capture_output=True)
    a = np.frombuffer(chk.stdout, np.uint8)
    n = len(a) // (540 * 960)
    a = a[:n * 540 * 960].reshape(n, 960, 540) if n else np.zeros((1, 960, 540), np.uint8)
    bad = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    for f in a:
        if f[:4, :].mean() < 20: bad["top"] += 1
        if f[-4:, :].mean() < 20: bad["bottom"] += 1
        if f[:, :4].mean() < 20: bad["left"] += 1
        if f[:, -4:].mean() < 20: bad["right"] += 1
    print(f"成品: {nfrm}帧 采样{n}帧 | 整条黑边帧 {bad}")
    print("成品 =>", prev)


def cmd_draft():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    pipeline = __import__("pipeline")
    pipeline._bootstrap_skill()
    from jy_wrapper import JyProject
    from pyJianYingDraft import KeyframeProperty as KP
    try:
        from pyJianYingDraft import ClipSettings
    except ImportError:
        from pyJianYingDraft.segment import ClipSettings
    from pyJianYingDraft.keyframe import Keyframe

    name = "勇者demo_抖音916_v2"
    droot = r"C:\Users\19150\AppData\Local\JianyingPro\User Data\Projects\com.lveditor.draft"
    if os.path.exists(os.path.join(droot, name)):
        sys.exit(f"草稿已存在：{name}")

    plan = json.load(open(PLAN, encoding="utf-8"))
    B = min(CW / float(SW), CH / float(SH))

    def geo(win):
        x, y, ww, wh = win
        s = CW / float(ww)
        sc = s / B
        tx = (s * (SW / 2.0 - x) - CW / 2.0) / (CW / 2.0)
        ty = (CH / 2.0 - s * (SH / 2.0 - y)) / (CH / 2.0)
        return sc, tx, ty

    proj = JyProject(name, width=CW, height=CH, overwrite=True)
    nseg = nmove = 0
    for idx, p in enumerate(plan):
        start_us = int(round(p["t0"] * 1e6))
        if idx + 1 < len(plan):
            dur_us = int(round((plan[idx + 1]["t0"] - p["t0"]) * 1e6))
        else:
            dur_us = int(round((p["t1"] - p["t0"]) * 1e6 + 1e6))
        seg = proj.add_media_safe(SRC, start_time=start_us, duration=dur_us, source_start=start_us)
        if seg is None:
            continue
        sc0, tx0, ty0 = geo(p["win0"])
        sc1, tx1, ty1 = geo(p["win1"])
        seg.clip_settings = ClipSettings(scale_x=sc0, scale_y=sc0, transform_x=tx0, transform_y=ty0)
        if abs(sc1 - sc0) > 1e-4:
            seg.add_keyframe(KP.uniform_scale, 0, sc0, **Keyframe.EASE_IN_OUT)
            seg.add_keyframe(KP.uniform_scale, dur_us, sc1, **Keyframe.EASE_IN_OUT)
            nmove += 1
        if abs(ty1 - ty0) > 1e-4:
            seg.add_keyframe(KP.position_y, 0, ty0, **Keyframe.EASE_IN_OUT)
            seg.add_keyframe(KP.position_y, dur_us, ty1, **Keyframe.EASE_IN_OUT)
            nmove += 1
        if abs(tx1 - tx0) > 1e-4:
            seg.add_keyframe(KP.position_x, 0, tx0, **Keyframe.EASE_IN_OUT)
            seg.add_keyframe(KP.position_x, dur_us, tx1, **Keyframe.EASE_IN_OUT)
            nmove += 1
        nseg += 1
    res = proj.save()
    print("segments:", nseg, "| motion kf:", nmove)
    print("save =>", res)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("windows", "all"):
        cmd_windows()
    if cmd in ("preview", "all"):
        cmd_preview()
    if cmd in ("draft", "all"):
        cmd_draft()
