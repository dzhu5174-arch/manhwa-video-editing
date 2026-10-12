# -*- coding: utf-8 -*-
"""pipeline.py — 勇者无法休息 · 量产管线（一条命令出一集）

从 全片v7 三个脚本参数化移植 + 草稿生成（build_draft_v5 泛化）：
  plan0 → plan（人脸感知窗口）→ qc（像素级规则+自动修窗）→ preview（ffmpeg 同参预览）→ draft（剪映草稿）

用法：
  python pipeline.py --config configs/ep02.json --steps plan0,plan,qc,preview,draft
  常用组合：
    python pipeline.py --config configs/ep02.json --steps plan0       # 生成分镜表（从句级时间轴）
    python pipeline.py --config configs/ep02.json --steps plan,qc     # 窗口规划 + QC 自动修窗
    python pipeline.py --config configs/ep02.json --steps preview     # 预览片（看效果/抽检）
    python pipeline.py --config configs/ep02.json --steps draft       # 剪映草稿（关键帧=剪映原生）

质量线 = v7 量产版（项目规则 v6/v7 子集）：无气泡/无文字框、脸完整可见、无白边黑边、
每镜缓动运镜（travel≈150px，方向混排）、字幕、关系钩开场。人工只保留三个把关点：
口播稿终审 / QC 报告抽检 / 成片过目（详见 README.md）。
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np
import cv2
from PIL import Image

# 控制台编码兜底：jy_wrapper 会打印 emoji（GBK 控制台会崩）
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcmod  # noqa: E402

# ---------------- 量产版规则常量（v7 质量线） ----------------
CANVAS_W, CANVAS_H = 1080, 1920
TRAVEL_PX = 150.0     # 常规行程（画布 px）
S_MIN = 0.74          # 超宽镜最低缩放（窗口高 ≤ 2595）
STATIC_MIN_PX = 40.0  # 行程 < 40px 记静止
MIN_SHOT_DUR = 1.2    # 短于 1.2s 的句子并入下一镜
SPLIT_SHOT_DUR = 3.5  # v5.1（2026-10-12 晚）：>3.5s 的句块拆两镜（同系列密度基准 0.379–0.409 镜/s）
DEFAULT_FACE_XML = r"C:\Users\19150\Desktop\manhua video v2\工具\模型\lbpcascade_animeface.xml"


# ---------------- 配置 ----------------
def load_cfg(path):
    cfg = json.load(open(path, encoding="utf-8"))
    cfg["_config_path"] = os.path.abspath(path)
    cfg.setdefault("canvas", [CANVAS_W, CANVAS_H])
    cfg.setdefault("steps_default", "plan,qc")
    return cfg


def resolve(cfg, p):
    """相对路径 → 相对配置文件目录；绝对路径原样。"""
    if not p:
        return None
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(os.path.dirname(cfg["_config_path"]), p))


def work_dir(cfg):
    w = resolve(cfg, cfg["work"])
    os.makedirs(w, exist_ok=True)
    return w


def slices_of(cfg):
    return {k: resolve(cfg, v) for k, v in cfg["slices"].items()}


def plan_path(cfg):
    return os.path.join(work_dir(cfg), "plan.json")


# ---------------- plan0：从句级时间轴生成分镜表 ----------------
def read_timeline(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            r = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
            rows.append(r)
    # 兼容列名
    def num(r, *keys):
        for k in keys:
            if k in r and r[k] != "":
                return float(r[k])
        raise KeyError(r)
    segs = [(num(r, "起始秒", "start"), num(r, "结束秒", "end"), r.get("文案", "")) for r in rows]
    return [s for s in segs if s[1] > s[0]]


def content_band(gray):
    """外层内容带：跳过上下纯白/纯黑边距（内部画格缝不处理，交给 QC）。"""
    H, W = gray.shape
    row_std = gray.std(axis=1)
    row_mean = gray.mean(axis=1)
    content = ~((row_std < 8.0) & ((row_mean > 240) | (row_mean < 15)))
    idx = np.where(content)[0]
    if len(idx) == 0:
        return 0, H
    return int(idx[0]), int(idx[-1]) + 1


def build_plan0(cfg):
    """把句级时间轴 + 切片目录 → plan0.json（顺序分镜：一句一镜；长句拆两镜；超短并入下一镜）。
    可在生成后人工/人工+AI 微调 plan0.csv/json（img 与窗口字段），plan0 是唯一的语义决策层。"""
    tl = resolve(cfg, cfg["timeline"])
    w = work_dir(cfg)
    rows = read_timeline(tl)

    # 合并超短句
    merged = []
    i = 0
    while i < len(rows):
        t0, t1, txt = rows[i]
        while t1 - t0 < MIN_SHOT_DUR and i + 1 < len(rows):
            i += 1
            t1 = rows[i][1]
            txt = txt + rows[i][2]
        merged.append([t0, t1, txt])
        i += 1

    # 长句拆两镜（v5：拆出的子镜继承父句块切片 + 沿用 use_cnt 错窗）
    segs = []
    for b, (t0, t1, txt) in enumerate(merged):
        if t1 - t0 > SPLIT_SHOT_DUR:
            mid = t0 + (t1 - t0) / 2.0
            segs.append((t0, mid, txt[: len(txt) // 2], b))
            segs.append((mid, t1, txt[len(txt) // 2:], b))
        else:
            segs.append((t0, t1, txt, b))

    # 切片池：按序号排序，跳过 000 封面（可配置 pool_skip）
    setkey = cfg["setkey"]  # 如 c2
    sdir = slices_of(cfg)[setkey]
    skip = {str(s) for s in cfg.get("pool_skip", ["000"])}
    files = sorted(f for f in os.listdir(sdir) if f.lower().endswith(".jpg"))
    pool = [f for f in files if os.path.splitext(f)[0] not in skip]
    if not pool:
        sys.exit(f"切片池为空: {sdir}")

    total = segs[-1][1] if segs else 1.0
    # 语义分镜层（2026-10-12 v5）：config.shot_map = {句块号: 切片号}（键=合并句块 ID，不是最终镜号；
    # 拆镜产生的子镜通过父句块查切片号）。主agent读全话拼图后人工排定，禁用时间比例猜测。
    shot_map = {int(k): str(v).zfill(3) for k, v in (cfg.get("shot_map") or {}).items()}
    if shot_map and len(shot_map) != len(merged):
        sys.exit(f"shot_map 数({len(shot_map)}) ≠ 句块数({len(merged)})，先核对时间轴/口播稿行数")
    demo_keys = {int(k) for k in (cfg.get("demo") or {})}
    try:
        faces_b0 = json.load(open(resolve(cfg, cfg.get("faces_json")) or os.path.join(w, "faces.json"),
                                  encoding="utf-8"))
    except Exception:
        faces_b0 = {}
    plan = []
    prev_idx = -10
    use_cnt = {}
    phase = 0        # v5：非 demo 镜的交替相位计数（demo 镜保留自带方向、不占相位索引）
    for k, (t0, t1, txt, blk_id) in enumerate(segs):
        if shot_map:
            fn = shot_map[blk_id] + ".jpg"
        else:
            # 时间比例映射：镜头在时间轴上的位置 → 切片序号（仅无 shot_map 时的兜底）
            frac = ((t0 + t1) / 2.0) / max(total, 1e-6)
            idx = int(round(frac * (len(pool) - 1)))
            idx = max(prev_idx + 1, idx) if prev_idx + 1 <= len(pool) - 1 else prev_idx
            idx = min(idx, len(pool) - 1)
            prev_idx = idx
            fn = pool[idx]
        img = f"{setkey}_{fn}"
        im = Image.open(os.path.join(sdir, fn)).convert("L")
        arr = np.asarray(im, dtype=np.uint8)
        W2, H2 = im.size
        blk0, blk1 = content_band(arr)
        blk_h = blk1 - blk0
        # 裁窗法则 v2 · 单格化（2026-10-12 夜 用户定律，取代全宽优先）
        if blk_id in demo_keys:
            up = (k % 2 == 0)        # demo 覆盖镜：方向由 demo 参数决定，不占相位索引
        else:
            up = (phase % 2 == 0)    # 按「非 demo 镜序列」交替，demo 段后重起相位
            phase += 1
        r2 = v2_pick_window(cfg, img, faces_b0.get(img, []), W2 / 2.0, blk0 + blk_h / 2.0)
        if r2 is None:
            s = max(V2["s_min"], CANVAS_W / float(W2)); H = CANVAS_H / s
            x0 = max(0.0, (W2 - CANVAS_W / s) / 2.0)
            center = blk0 + max(0.0, blk_h - H) / 2.0
            ts = te = center
            v2meta = dict(legal=False, why="no_legal_window")
        else:
            s = r2["s"]; H = CANVAS_H / s; x0 = r2["x0"]
            mid = r2["y_mid"]; trav = r2["trav"]
            uc = use_cnt.get(fn, 0)
            if uc and trav > 0:      # §3.4 同切片错窗 ±0.5H（带内）
                mid = min(max(mid + (0.5 * H if uc % 2 == 1 else -0.5 * H), r2["band"][0] + H / 2),
                          r2["band"][1] - H / 2)
            ts = mid + trav / 2.0 if up else mid - trav / 2.0
            te = mid - trav / 2.0 if up else mid + trav / 2.0
            v2meta = dict(white_area=round(r2["white_area"], 4),
                          face_ratio=(None if r2["face_ratio"] is None else round(r2["face_ratio"], 3)),
                          legal=r2["legal"], band=r2["band"])
        use_cnt[fn] = use_cnt.get(fn, 0) + 1
        plan.append(dict(shot=k, t0=round(t0, 3), t1=round(t1, 3), dur=round(t1 - t0, 3),
                         img=img, s=round(s, 5), x0=round(x0, 1),
                         top_start=round(ts, 1), top_end=round(te, 1),
                         blk=[blk0, blk1], block=blk_id, up=bool(up), v2=v2meta,
                         src="shot_map" if shot_map else "auto-plan0", text=txt))

    out = os.path.join(w, "plan0.json")
    json.dump(plan, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"plan0: {len(plan)} 镜 -> {out}")
    print("  （可以人工/AI 微调 img 与窗口字段后再跑 plan,qc）")
    return out


# ================= 裁窗法则 v2 · 单格化（2026-10-12 夜 用户定律，取代「全宽优先」） =================
# 用户基准：不丢真 > 裁满（长条→单格漫画） > 中间权重大·周边权重小 > 只放大到够关键帧移动
V2 = dict(
    white_thr=235,        # 白掩码：min(RGB) > thr
    row_density=0.60,     # 实内容行判据
    gap_frac=0.05,        # 内容带允许夹缝（占带高比例）
    white_area_max=0.02,  # 窗内白斑面积上限
    white_area_max_fb=0.05,   # 女性演出格（fullbody_ok）放宽
    face_lo=0.30, face_hi=0.60, face_target=0.45,   # 脸高 / 显示画高
    s_min=0.80, s_max=2.20,
    border_density=0.60,  # 窗口上/下边行内容密度下限（上下死重出画的机器判据）
    y_jump_max=1.00,      # 同切片相邻镜 y 跳变上限（窗高倍数）
)

_IMGC = {}


def _img_rgb_gray(cfg, img):
    key = (id(cfg), img)
    if key not in _IMGC:
        setkey, fn = img.split("_", 1)
        bgr = cv2.imdecode(np.fromfile(os.path.join(slices_of(cfg)[setkey], fn), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        _IMGC[key] = (np.ascontiguousarray(bgr[:, :, ::-1]),
                      cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    return _IMGC[key]


def _longest_run(flags, gap=0):
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


def v2_band(gray, rgb):
    """内容带：白∪文字 = 死重；实内容行密度 >0.6 的最长区段（允许 5% 夹缝）"""
    H, W = gray.shape
    white = (rgb.min(2) > V2["white_thr"])
    dead = white.astype(np.float32)
    try:
        _a = qcmod.detect(np.asarray(gray, dtype=np.uint8))
        boxes = list(_a.get("text_cells", [])) + list(_a.get("bubbles", []))
    except Exception:
        boxes = []
    for b in boxes:
        x, y, w, h = [int(v) for v in b[:4]]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 > x0 and y1 > y0:
            dead[y0:y1, x0:x1] = 1.0
    dens_full = 1.0 - dead.mean(1)
    cd_all = 1.0 - dead.mean(0)
    cr0 = _longest_run(list(cd_all > V2["border_density"]))
    bx0, bx1 = (int(cr0[0]), int(cr0[1])) if cr0 else (0, W - 1)
    # 行密度只在「内容列范围」内统计（法则：窗口边行密度，而非整行）
    dens = 1.0 - dead[:, bx0:bx1 + 1].mean(1)
    r = _longest_run(list(dens > V2["row_density"]), gap=int(V2["gap_frac"] * H))
    if not r:
        top = sorted(np.argsort(-dens)[:max(1, int(0.4 * H))])
        r = (int(top[0]), int(top[-1]))
    b0, b1 = int(r[0]), int(r[1])
    return dict(white=white, dens=dens, dens_full=dens_full, dead=dead,
                b0=b0, b1=b1, bx0=bx0, bx1=bx1)


def v2_pick_window(cfg, img, boxes, cx_hint, cy_hint, fullbody=False):
    """按法则 v2 定窗。返回 dict（含合法性与各量化指标）；无合法窗时 legal=False（上层报警，禁止静默 fallback）"""
    rgb, gray = _img_rgb_gray(cfg, img)
    H, W = gray.shape
    m = v2_band(gray, rgb)
    white, dens = m["white"], m["dens"]
    b0, b1, bx0, bx1 = m["b0"], m["b1"], m["bx0"], m["bx1"]
    B = max(1, b1 - b0 + 1)
    BW = max(1, bx1 - bx0 + 1)
    box = None
    fneed, giant = None, False
    if boxes:
        box = min(boxes, key=lambda b: (b[0] + b[2] / 2.0 - cx_hint) ** 2 + (b[1] + b[3] / 2.0 - cy_hint) ** 2)
    travel_n = TRAVEL_PX
    s = max(V2["s_min"], CANVAS_W / float(BW), (CANVAS_H + travel_n) / float(B))
    if box:
        s = max(s, min(V2["s_max"], V2["face_target"] * CANVAS_H / max(1.0, float(box[3]))))
    s = min(V2["s_max"], max(V2["s_min"], s))
    Wn, Hn = CANVAS_W / s, CANVAS_H / s
    if box:
        fx, fy, fw, fh = [float(v) for v in box[:4]]
        if fh <= Hn - 16.0:                      # 整脸装得下 → 完整保脸
            fneed = (fx, fy, fw, fh)
        else:                                     # 巨脸特写 → 核心区（眼鼻嘴）必须完整
            giant = True
            fneed = (fx, fy + 0.18 * fh, fw, 0.74 * fh)
    y_lo, y_hi = float(b0), float(b1 + 1) - Hn
    if fneed:
        _fx, _fy, _fw, _fh = fneed
        mm = 8.0
        y_lo, y_hi = max(y_lo, _fy + _fh + mm - Hn), min(y_hi, _fy - mm)
    if y_hi < y_lo:                                   # 脸与带冲突：以「不丢真」为先
        c = ((fneed[1] + fneed[3] / 2.0) if fneed else (b0 + b1) / 2.0) - Hn / 2.0
        y_lo = y_hi = max(0.0, min(H - Hn, c))
    want_x = ((fneed[0] + fneed[2] / 2.0) if fneed else (bx0 + bx1) / 2.0) - Wn / 2.0
    x0 = max(0.0, min(W - Wn, want_x))
    best = None
    step = max(2.0, (y_hi - y_lo) / 80.0)
    y = y_lo
    while y <= y_hi + 1e-6:
        wy, wx = int(round(y)), int(round(x0))
        hh, ww = int(round(Hn)), int(round(Wn))
        sub = white[wy:wy + hh, wx:wx + ww]
        if sub.size == 0:
            break
        wa = float(sub.mean())
        d_top = float(dens[min(H - 1, wy)])
        d_bot = float(dens[min(H - 1, wy + hh - 1)])
        fok, fr = True, None
        if fneed:
            fok = (y - 8.0 <= fneed[1]) and (fneed[1] + fneed[3] <= y + Hn + 8.0)
            fr = float(box[3]) * s / CANVAS_H
        lim = V2["white_area_max_fb"] if fullbody else V2["white_area_max"]
        ok = (wa <= lim) and (d_top > V2["border_density"]) and (d_bot > V2["border_density"]) and fok
        if (not fullbody) and (not giant) and fr is not None and not (V2["face_lo"] <= fr <= V2["face_hi"]):
            ok = False
        cen = abs((y + Hn / 2.0) - (b0 + b1 + 1) / 2.0) / float(B)
        sc = -4.0 * wa - 0.6 * cen - (0.0 if fr is None else abs(fr - V2["face_target"])) + (5.0 if ok else 0.0)
        if best is None or sc > best[0]:
            best = (sc, y, wa, d_top, d_bot, fok, fr, ok)
        y += step
    if best is None:
        return None
    _sc, yc, wa, d_top, d_bot, fok, fr, ok = best
    if fneed and not fok:      # 不丢真优先：有脸时只接受「脸（或巨脸核心区）完整」的落点
        best2 = None
        y = y_lo
        while y <= y_hi + 1e-6:
            wy, wx = int(round(y)), int(round(x0))
            hh, ww = int(round(Hn)), int(round(Wn))
            sub = white[wy:wy + hh, wx:wx + ww]
            if sub.size == 0:
                break
            fok2 = (y - 8.0 <= fneed[1]) and (fneed[1] + fneed[3] <= y + Hn + 8.0)
            if fok2:
                wa2 = float(sub.mean())
                fr2 = float(box[3]) * s / CANVAS_H
                sc2 = -4.0 * wa2 - abs(fr2 - V2["face_target"])
                if best2 is None or sc2 > best2[0]:
                    best2 = (sc2, y, wa2, float(dens[min(H - 1, wy)]), float(dens[min(H - 1, wy + hh - 1)]), True, fr2, False)
            y += step
        if best2 is not None:
            _sc, yc, wa, d_top, d_bot, fok, fr, ok = best2
    trav = min(travel_n / s, max(0.0, y_hi - y_lo))
    mid = min(max(yc, y_lo + trav / 2.0), y_hi - trav / 2.0) if trav > 0 else yc
    return dict(s=float(s), x0=float(x0), y_mid=float(mid), trav=float(trav),
                band=[b0, b1, bx0, bx1], white_area=wa, face_ratio=fr, giant=bool(giant),
                d_top=d_top, d_bot=d_bot, face_ok=bool(fok), legal=bool(ok))


# ---------------- plan：人脸感知窗口（build_plan_v7 泛化） ----------------
def step_plan(cfg):
    w = work_dir(cfg)
    p0 = resolve(cfg, cfg.get("plan0")) or os.path.join(w, "plan0.json")
    if not os.path.exists(p0):
        if cfg.get("timeline"):
            print("plan0 不存在 → 从 timeline 自动生成")
            build_plan0(cfg)
        else:
            sys.exit(f"没有 plan0（{p0}）也没有 timeline，无法规划")
    p5 = json.load(open(p0, encoding="utf-8"))
    p5.sort(key=lambda p: p["shot"])

    demo = {int(k): v for k, v in (cfg.get("demo") or {}).items()}
    face_xml = resolve(cfg, cfg.get("face_xml") or DEFAULT_FACE_XML)
    faces_json = resolve(cfg, cfg.get("faces_json")) or os.path.join(w, "faces.json")
    slices = slices_of(cfg)

    # 补 blk（老 plan 可能没有）
    for p in p5:
        if not p.get("blk"):
            setkey, fn = p["img"].split("_", 1)
            g = np.asarray(Image.open(os.path.join(slices[setkey], fn)).convert("L"), dtype=np.uint8)
            p["blk"] = list(content_band(g))

    faces = qcmod.load_faces(faces_json, face_xml, slices, sorted({p["img"] for p in p5}))

    out, notes = [], []
    use_cnt2 = {}          # v5.1：同片复用错窗计数（在真正定窗的 step_plan 里生效）
    for i, p in enumerate(p5):
        bid = p.get("block", p["shot"])   # v5：demo 键 = 句块号（shot_map 同层）
        q = dict(shot=p["shot"], t0=p["t0"], t1=p["t1"], dur=p["dur"], img=p["img"], block=bid)
        if bid in demo:
            q.update(demo[bid])
            q["src"] = "demo"
        else:
            # 裁窗法则 v2 · 单格化（2026-10-12 夜 用户定律）：内容带居中充满 + 上下死重出画 + 脸大
            s5 = p["s"]
            H5 = CANVAS_H / s5
            cx5 = p["x0"] + (CANVAS_W / s5) / 2
            cy5 = (p["top_start"] + p["top_end"]) / 2 + H5 / 2
            boxes = faces.get(p["img"], [])
            r2 = v2_pick_window(cfg, p["img"], boxes, cx5, cy5, fullbody=bool(p.get("fullbody_ok")))
            if r2 is None:
                s7 = max(S_MIN, s5)
                q.update(s=s7, x0=round(p["x0"], 1),
                         top_start=round(p["top_start"], 1), top_end=round(p["top_end"], 1))
                q["src"] = "v2-NOLEGAL"
                q["alarm"] = "v2_no_legal_window"
                notes.append("shot%d v2 无合法窗（报警，禁静默 fallback）" % p["shot"])
            else:
                s7 = r2["s"]
                H7 = CANVAS_H / s7
                trav = r2["trav"]
                mid = r2["y_mid"]
                up = bool(p.get("up", i % 2 == 0))
                uc2 = use_cnt2.get(p["img"], 0)
                use_cnt2[p["img"]] = uc2 + 1
                if uc2 and trav > 0:      # §3.4 同切片错窗 ±0.5H（带内）
                    mid = min(max(mid + (0.5 * H7 if uc2 % 2 == 1 else -0.5 * H7),
                                  r2["band"][0] + H7 / 2), r2["band"][1] - H7 / 2)
                ts = mid + trav / 2.0 if up else mid - trav / 2.0
                te = mid - trav / 2.0 if up else mid + trav / 2.0
                q.update(s=s7, x0=round(r2["x0"], 1), top_start=round(ts, 1), top_end=round(te, 1))
                q["src"] = "v2"
                q["v2"] = dict(white_area=round(r2["white_area"], 4),
                               face_ratio=(None if r2["face_ratio"] is None else round(r2["face_ratio"], 3)),
                               d_top=round(r2["d_top"], 3), d_bot=round(r2["d_bot"], 3),
                               band=r2["band"], legal=bool(r2["legal"]))
                if not r2["legal"]:
                    q["alarm"] = "v2_window_not_legal"
                elif abs(te - ts) * s7 < 60.0:
                    q["alarm"] = "v2_small_travel"
                if r2.get("giant"):
                    q["giant_face"] = True
        q["H"] = int(round(CANVAS_H / q["s"]))
        q["static"] = abs(q["top_end"] - q["top_start"]) * q["s"] < STATIC_MIN_PX
        q["dirn"] = "static" if q["static"] else ("up" if q["top_end"] < q["top_start"] else "down")
        # 纯平移（2026-10-12 v4：SIFT 全尺寸实测对标片 scale=1.0000，「推拉锁定」系 ECC 小图测量噪声，已证伪删除）。
        # plan0/demo 显式给的 s_end 才会生效；不主动加缩放。
        if p.get("fade"):
            q["fade"] = True
        if p.get("text"):
            q["text"] = p["text"]
        out.append(q)

    json.dump(out, open(plan_path(cfg), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    import collections
    print("shots:", len(out))
    print("dirn:", collections.Counter(q["dirn"] for q in out))
    tr = [abs(q["top_end"] - q["top_start"]) * q["s"] for q in out]
    print("travel: median %.0f | <60px: %s" % (sorted(tr)[len(tr) // 2], [q["shot"] for q, x in zip(out, tr) if x < 60]))
    print("s range: %.2f-%.2f" % (min(q["s"] for q in out), max(q["s"] for q in out)))
    for n in notes[:20]:
        print(" ", n)
    print("plan ->", plan_path(cfg))


# ---------------- qc：像素级规则执行 ----------------
def step_qc(cfg, check_only=False):
    w = work_dir(cfg)
    r = qcmod.run_qc(
        plan_path=plan_path(cfg),
        faces_json=resolve(cfg, cfg.get("faces_json")) or os.path.join(w, "faces.json"),
        face_xml=resolve(cfg, cfg.get("face_xml") or DEFAULT_FACE_XML),
        slices=slices_of(cfg),
        report_path=os.path.join(w, "qc_report.json"),
        sheet_path=os.path.join(w, "qc_sheet.jpg"),
        check_only=check_only,
        canvas_w=CANVAS_W,
    )
    # v5.1：plan 里 alarm 的镜（三级保脸后仍静止/冲突）强制 choice，进人工裁决回路
    try:
        _plan = json.load(open(plan_path(cfg), encoding="utf-8"))
        alarm_shots = sorted(p["shot"] for p in _plan if p.get("alarm"))
        giant_shots = sorted(p["shot"] for p in _plan if p.get("giant_face"))
        if alarm_shots or giant_shots:
            rp = os.path.join(w, "qc_report.json")
            rep = json.load(open(rp, encoding="utf-8"))
            for e in rep:
                if e.get("shot") in alarm_shots and e.get("status") != "risk":
                    e["status"] = "choice"
                    e["alarm"] = "v5.1 三级保脸后静止/冲突 → 人工裁决"
                if e.get("shot") in giant_shots:
                    # 巨脸特写：按法则只保「眼鼻嘴核心区」，qcmod 的整脸判据不再算 risk
                    e["giant_face"] = True
                    if e.get("status") == "risk":
                        e["status"] = "choice"
                        e["note"] = "巨脸特写：核心区(0.18-0.92fh)完整，允许裁发际线/下巴；整脸判据放宽"
            json.dump(rep, open(rp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            import collections as _c
            cnt = _c.Counter(e.get("status") for e in rep)
            r["clean"] = cnt.get("clean", 0); r["fixed"] = cnt.get("fixed", 0)
            r["choice"] = cnt.get("choice", 0); r["risk"] = cnt.get("risk", 0)
            print("v5.1 alarm->choice:", alarm_shots)
            print("v2 巨脸核心区放宽:", giant_shots)
    except Exception as _e:
        print("  !! v5.1 alarm 处理失败:", _e)
    if check_only and (r["risk"] or r["choice"]):
        sys.exit(1)
    return r


# ---------------- preview：PyAV 同参预览（2026-10-07 起；本机无 ffmpeg） ----------------
def step_preview(cfg):
    """与 draft 同参数渲染预览片：裁剪窗=plan 的 (x0,top_start)-(x0_end,top_end)，
    线性匀速（2026-10-11 蒸馏修正：对标片实测线性，非缓动）。支持横向运镜（x0_end）。"""
    import av
    from av.video.frame import VideoFrame
    w = work_dir(cfg)
    plan = json.load(open(plan_path(cfg), encoding="utf-8"))
    plan.sort(key=lambda p: p["shot"])
    slices = slices_of(cfg)
    out = resolve(cfg, cfg["preview_out"])
    os.makedirs(os.path.dirname(out), exist_ok=True)

    oc = av.open(out, "w")
    ovs = oc.add_stream("libx264", rate=30)
    ovs.width, ovs.height, ovs.pix_fmt = CANVAS_W, CANVAS_H, "yuv420p"
    ovs.options = {"crf": "18", "preset": "veryfast"}
    nf = 0
    last_hold = None       # v5：绝对时间铺帧时用于填句间缝隙的持帧
    for p in plan:
        setkey, fn = p["img"].split("_", 1)
        src = cv2.imdecode(np.fromfile(os.path.join(slices[setkey], fn), dtype=np.uint8),
                           cv2.IMREAD_COLOR)  # cv2.imread 不认中文路径
        if src is None:
            print("  !! 读图失败", p["img"]); continue
        H2, W2 = src.shape[:2]
        s0, s1 = p["s"], p.get("s_end", p["s"])   # s_end：微缩放终点（2026-10-08 加）
        x0, x1 = p["x0"], p.get("x0_end", p["x0"])
        y0, y1 = p["top_start"], p["top_end"]
        dur = p["dur"]
        # v5：按绝对时间铺帧——上一镜 t1 到本镜 t0 之间的缝隙持上一镜末帧（修 8.93s 累积漂移）
        target = int(round(p["t0"] * 30))
        while nf < target and last_hold is not None:
            for pkt in ovs.encode(VideoFrame.from_ndarray(last_hold, format="bgr24")):
                oc.mux(pkt)
            nf += 1
        n = max(1, int(round(dur * 30)))
        for k in range(n):
            u = min(k / 30.0 / dur, 1.0)
            e = u                              # 线性匀速（对标实测，2026-10-11）
            sk = s0 + (s1 - s0) * e                # 推近/拉远与平移同步缓动
            Ww, Hh = CANVAS_W / sk, CANVAS_H / sk
            cx = x0 + (x1 - x0) * e
            cy = y0 + (y1 - y0) * e
            cx = max(0.0, min(W2 - Ww, cx)); cy = max(0.0, min(H2 - Hh, cy))
            crop = src[int(cy):int(cy + Hh), int(cx):int(cx + Ww)]
            frame = cv2.resize(crop, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)
            if p.get("fade") and u > (dur - 0.9) / dur:  # 结尾淡出
                frame = (frame * max(0.0, (1 - u) * dur / 0.9)).astype(np.uint8)
            last_hold = frame
            for pkt in ovs.encode(VideoFrame.from_ndarray(frame, format="bgr24")):
                oc.mux(pkt)
            nf += 1
    for pkt in ovs.encode():
        oc.mux(pkt)

    oc.close()
    print("（预览为纯视频：取景/运镜过目用；音频在剪映草稿核对）")
    print(f"preview saved: {out}（{nf} 帧, {os.path.getsize(out)//1024}KB）")


# ---------------- draft：剪映草稿（关键帧=剪映原生） ----------------
def _bootstrap_skill():
    candidates = [
        os.getenv("JY_SKILL_ROOT", "").strip(),
        r"C:\Users\19150\.agents\skills\jianying-editor",
    ]
    for p in candidates:
        if p and os.path.exists(os.path.join(p, "scripts", "jy_wrapper.py")):
            sp = os.path.join(p, "scripts")
            if sp not in sys.path:
                sys.path.insert(0, sp)
            return sp
    raise ImportError("jianying-editor skill not found (set JY_SKILL_ROOT)")


def step_draft(cfg, force=False, draft_name=None):
    _bootstrap_skill()
    from jy_wrapper import JyProject
    from pyJianYingDraft import KeyframeProperty as KP
    try:
        from pyJianYingDraft import ClipSettings
    except ImportError:
        from pyJianYingDraft.segment import ClipSettings
    from pyJianYingDraft.keyframe import Keyframe

    name = draft_name or cfg.get("draft_name")
    if not name:
        sys.exit("draft_name 未配置")
    w = work_dir(cfg)
    plan = json.load(open(plan_path(cfg), encoding="utf-8"))
    plan.sort(key=lambda p: p["shot"])
    slices = slices_of(cfg)

    # 安全铁律：已被剪映打开过的草稿不可覆盖
    droot = r"C:\Users\19150\AppData\Local\JianyingPro\User Data\Projects\com.lveditor.draft"
    if os.path.exists(os.path.join(droot, name)) and not force:
        sys.exit(f"草稿已存在：{name}\n（铁律：已被剪映打开过的草稿不要覆盖；换新名字或用 --force-draft）")

    proj = JyProject(name, width=CANVAS_W, height=CANVAS_H, overwrite=True)
    nseg, nmove, nfail = 0, 0, []
    for idx, p in enumerate(plan):
        setkey, fn = p["img"].split("_", 1)
        img = os.path.join(slices[setkey], fn)
        W2, H2 = Image.open(img).size
        s0 = p["s"]
        s1 = p.get("s_end", s0)   # 终点缩放（2026-10-08 加：微推近/拉远，与平移同缓动）
        B = min(CANVAS_W / float(W2), CANVAS_H / float(H2))
        sc0, sc1 = s0 / B, s1 / B
        tx0 = (s0 * (W2 / 2.0 - p["x0"]) - CANVAS_W / 2.0) / (CANVAS_W / 2.0)
        x1 = p.get("x0_end", p["x0"])   # 可选：横向运镜终点（2026-10-07 加）
        tx1 = (s1 * (W2 / 2.0 - x1) - CANVAS_W / 2.0) / (CANVAS_W / 2.0)
        ty0 = (CANVAS_H / 2.0 - s0 * (H2 / 2.0 - p["top_start"])) / (CANVAS_H / 2.0)
        ty1 = (CANVAS_H / 2.0 - s1 * (H2 / 2.0 - p["top_end"])) / (CANVAS_H / 2.0)
        start_us = int(round(p["t0"] * 1e6))
        speech_us = int(round(p["dur"] * 1e6))
        # 画面时长覆盖到下一镜起点（含句间停顿），避免视频轨出现黑洞；
        # 运镜关键帧仍按口播时长计时——说完即停，停顿时画面定格（2026-10-07）
        if idx + 1 < len(plan):
            cover_us = int(round((plan[idx + 1]["t0"] - p["t0"]) * 1e6))
        else:
            cover_us = speech_us + 1000000  # 末镜多留 1s 托住尾部静音
        dur_us = max(speech_us, cover_us)
        seg = proj.add_media_safe(img, start_time=start_us, duration=dur_us)
        if seg is None:
            nfail.append(p["shot"])
            continue
        seg.clip_settings = ClipSettings(scale_x=sc0, scale_y=sc0, transform_x=tx0, transform_y=ty0)
        # 2026-10-11 蒸馏修正：对标片实测为线性匀速（缓动比≈1.00），丝滑来自
        # 运动中切镜+方向交替，不用 ease-in-out——关键帧一律线性（默认 Line）
        if abs(ty1 - ty0) > 1e-4:
            seg.add_keyframe(KP.position_y, 0, ty0)
            seg.add_keyframe(KP.position_y, speech_us, ty1)
            nmove += 1
        if abs(tx1 - tx0) > 1e-4:
            seg.add_keyframe(KP.position_x, 0, tx0)
            seg.add_keyframe(KP.position_x, speech_us, tx1)
            nmove += 1
        if abs(sc1 - sc0) > 1e-4:   # 微缩放关键帧（UNIFORM_SCALE，与剪映"缩放"滑轨同型）
            seg.add_keyframe(KP.uniform_scale, 0, sc0)
            seg.add_keyframe(KP.uniform_scale, speech_us, sc1)
            nmove += 1
        if p.get("fade"):
            seg.add_keyframe(KP.alpha, max(0, dur_us - 900000), 1.0)
            seg.add_keyframe(KP.alpha, dur_us, 0.0)
        nseg += 1

    vo = resolve(cfg, cfg.get("vo")) if cfg.get("vo") else None
    bgm = resolve(cfg, cfg.get("bgm")) if cfg.get("bgm") else None
    audio = resolve(cfg, cfg.get("audio")) if cfg.get("audio") else None
    if vo and os.path.exists(vo):
        proj.add_audio_safe(vo, start_time=0, duration=None, track_name="VO")
    if bgm and os.path.exists(bgm):
        proj.add_audio_safe(bgm, start_time=0, duration=None, track_name="BGM")
    if not (vo or bgm) and audio and os.path.exists(audio):
        proj.add_audio_safe(audio, start_time=0, duration=None, track_name="AudioMix")

    tl = resolve(cfg, cfg.get("timeline")) if cfg.get("timeline") else None
    if tl and cfg.get("draft_subs", True) and os.path.exists(tl):
        import pyJianYingDraft as draft
        style = draft.TextStyle(size=5.0, bold=True, color=(1.0, 1.0, 1.0), align=1,
                                auto_wrapping=True, max_line_width=0.92)
        border = draft.TextBorder(color=(0.0, 0.0, 0.0), alpha=1.0, width=40.0)
        cs = draft.ClipSettings(transform_y=-0.8)
        nsub = 0
        for t0, t1, txt in read_timeline(tl):
            t = proj.add_text_simple(txt, start_time=int(round(t0 * 1e6)),
                                     duration=int(round((t1 - t0) * 1e6)),
                                     track_name="Subtitles", style=style, border=border, clip_settings=cs)
            nsub += 1 if t is not None else 0
        print("subtitles added:", nsub)

    res = proj.save()
    print("segments:", nseg, "| motion keyframes:", nmove, "| failed:", nfail)
    print("save =>", res)


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="勇者无法休息 · 量产管线")
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", default=None, help="plan0,plan,qc,preview,draft（逗号分隔）")
    ap.add_argument("--check-only", action="store_true", help="qc：只检查不修改（门禁）")
    ap.add_argument("--draft-name", default=None)
    ap.add_argument("--force-draft", action="store_true", help="允许覆盖已存在草稿（慎用）")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    # 画布从 config 生效（全局常量在本进程内按配置重绑定；2026-10-07 修：此前 cfg['canvas'] 写了不生效）
    if cfg.get("canvas"):
        global CANVAS_W, CANVAS_H, TRAVEL_PX, STATIC_MIN_PX
        CANVAS_W, CANVAS_H = int(cfg["canvas"][0]), int(cfg["canvas"][1])
        # 2026-10-11 蒸馏：每镜行程 = 画宽 8–12%（对标片实测），静止阈值随画布归一
        TRAVEL_PX = 0.10 * CANVAS_W
        STATIC_MIN_PX = 0.03 * CANVAS_W
        print(f"canvas: {CANVAS_W}x{CANVAS_H} | 行程 {TRAVEL_PX:.0f}px | 静止阈值 {STATIC_MIN_PX:.0f}px")
    steps = [s.strip() for s in (args.steps or cfg["steps_default"]).split(",") if s.strip()]
    for s in steps:
        print(f"\n===== [{cfg.get('episode')}] step: {s} =====")
        if s == "plan0":
            build_plan0(cfg)
        elif s == "plan":
            step_plan(cfg)
        elif s == "qc":
            step_qc(cfg, check_only=args.check_only)
        elif s == "preview":
            step_preview(cfg)
        elif s == "draft":
            step_draft(cfg, force=args.force_draft, draft_name=args.draft_name)
        else:
            sys.exit(f"未知步骤: {s}")
    print("\ndone.")


if __name__ == "__main__":
    main()
