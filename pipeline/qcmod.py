# -*- coding: utf-8 -*-
"""qcmod.py — 像素级规则检测/自动修复（量产版）

从 全片v7/qc_pixels_v7.py 参数化移植（逻辑不改，质量线不变）：
  - 白缝/黑边（seam）、文字块（text_cells）、气泡（bubbles）检测
  - 人脸硬约束：源图有脸 ⇒ 窗口必须完整包含至少一张脸；窗口切脸 = 一票否决
  - 自动滑窗修复窗口 top；修不掉记"二选一/风险"，人工处理

用法（由 pipeline.py 调用）：
  run_qc(plan_path, faces_json, face_xml, slices, report_path, sheet_path, check_only)
"""
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

TUNE = dict(
    dark=65, bright=243, white=246,
    glyph_min_area=20, glyph_max_area=6000,
    glyph_max_dim=240, fill_max=0.45,
    cell=160, cell_min_glyphs=7, cell_min_peers=5, peer_ratio=2.0, cell_min_std=35.0,
    bubble_min_frac=0.02, bubble_min_glyphs=6, bubble_white_frac=0.55,
    seam_run=6, slide_range=300, slide_step=20,
    face_range=1600, face_step=50, face_margin=6,
)


def runs_1d(mask, minlen):
    out, y, n = [], 0, len(mask)
    while y < n:
        if mask[y]:
            y0 = y
            while y < n and mask[y]:
                y += 1
            if y - y0 >= minlen:
                out.append((y0, y))
        else:
            y += 1
    return out


def detect(g):
    H, W = g.shape
    res = {"seams": [], "text_cells": [], "glyphs": 0, "bubbles": []}

    def band_rows(band):
        st = band.std(axis=1)
        mn = band.mean(axis=1)
        return runs_1d((st < 8.0) & ((mn > 240) | (mn < 15)), TUNE["seam_run"])

    def band_cols(band):
        st = band.std(axis=0)
        mn = band.mean(axis=0)
        return runs_1d((st < 8.0) & ((mn > 240) | (mn < 15)), TUNE["seam_run"])

    if band_rows(g[:6]):
        res["seams"].append("top")
    if band_rows(g[-6:]):
        res["seams"].append("bottom")
    if band_cols(g[:, :6]):
        res["seams"].append("left")
    if band_cols(g[:, -6:]):
        res["seams"].append("right")

    glyphs = []
    for mask in ((g < TUNE["dark"]), (g > TUNE["bright"])):
        n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a < TUNE["glyph_min_area"] or a > TUNE["glyph_max_area"]:
                continue
            if w < 6 or h < 6 or w > TUNE["glyph_max_dim"] or h > TUNE["glyph_max_dim"]:
                continue
            if a / float(w * h) > TUNE["fill_max"]:
                continue
            ar = w / float(h)
            if ar < 0.12 or ar > 8.0:
                continue
            glyphs.append((int(cent[i][0]), int(cent[i][1]), int(a)))
    res["glyphs"] = len(glyphs)

    csz = TUNE["cell"]
    cellmap = {}
    for cx, cy, a in glyphs:
        cellmap.setdefault((cx // csz, cy // csz), []).append((cx, cy, a))
    for (ci, cj), pts in cellmap.items():
        if len(pts) < TUNE["cell_min_glyphs"]:
            continue
        areas = sorted(p[2] for p in pts)
        best = 0
        for s in areas:
            best = max(best, sum(1 for a in areas if s <= a <= s * TUNE["peer_ratio"]))
        if best < TUNE["cell_min_peers"]:
            continue
        y0, y1 = cj * csz, min(H, (cj + 1) * csz)
        x0, x1 = ci * csz, min(W, (ci + 1) * csz)
        if g[y0:y1, x0:x1].std() < TUNE["cell_min_std"]:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        res["text_cells"].append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)), len(pts)))

    wm = (g > TUNE["white"]).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(wm, 8)
    total = float(g.size)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < TUNE["bubble_min_frac"] * total:
            continue
        if a / float(w * h) < TUNE["bubble_white_frac"]:
            continue
        inner = [p for p in glyphs if x <= p[0] <= x + w and y <= p[1] <= y + h]
        if len(inner) >= TUNE["bubble_min_glyphs"]:
            res["bubbles"].append((int(x), int(y), int(w), int(h), len(inner)))
    return res


def score(res):
    return len(res["seams"]) * 3 + len(res["text_cells"]) * 2 + len(res["bubbles"]) * 3


def _cascade(xml):
    """加载级联分类器（**中文路径安全**）。

    2026-10-06 实测事故：本机 cv2（4.11/5.0）读不了含非 ASCII 字符的路径——
    `cv2.CascadeClassifier(r"...\\工具\\模型\\lbpcascade_animeface.xml")` 抛
    FileStorage 异常且 empty()==True；旧代码用 try/except 吞掉异常后 faces 全空，
    **人脸检测静默失效**（无任何报错，取景退化为纯像素判断）。
    这里优先用项目内的 imgio.load_cascade（复制到 ASCII 临时文件再加载）。
    """
    try:
        sys.path.insert(0, r"C:\Users\19150\Desktop\manhua video v2\工具\质检")
        from imgio import load_cascade  # type: ignore
        return load_cascade(xml)
    except Exception:
        pass
    cc = getattr(cv2, "CascadeClassifier", None)
    if cc is None:
        raise RuntimeError("当前 python 的 cv2 不支持 CascadeClassifier（需 opencv 4.x）")
    try:
        xml.encode("ascii")
        cas = cc(xml)
    except UnicodeEncodeError:
        import shutil
        import tempfile
        tmp = os.path.join(tempfile.gettempdir(), "cascade_ascii.xml")
        shutil.copy2(xml, tmp)
        cas = cc(tmp)
    if cas.empty():
        raise RuntimeError(f"级联模型加载失败: {xml}")
    return cas


def load_faces(faces_json, face_xml, slices, tags):
    faces = {}
    if os.path.exists(faces_json):
        try:
            faces = json.load(open(faces_json, encoding="utf-8"))
        except Exception:
            faces = {}
    missing = [t for t in tags if t not in faces]
    if missing and os.path.exists(face_xml):
        cascade = _cascade(face_xml)
        for tag in missing:
            setkey, fn = tag.split("_", 1)
            im = Image.open(os.path.join(slices[setkey], fn)).convert("RGB")
            W, H = im.size
            small = im.resize((max(1, W // 3), max(1, H // 3)))
            gray = cv2.cvtColor(np.asarray(small)[:, :, ::-1].copy(), cv2.COLOR_BGR2GRAY)
            # 参数实测（2026-10-06，12 张扫描）：sf=1.05/mn=3 召回最高（8/12 张、12 框），
            # 旧值 sf=1.1/mn=5 最低（6/12、6 框）。anime 级联召回本就有限，取宽容档。
            boxes = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3,
                                             minSize=(18, 18))
            faces[tag] = [[int(x * 3), int(y * 3), int(w * 3), int(h * 3)] for (x, y, w, h) in boxes]
        json.dump(faces, open(faces_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"faces.json refreshed (+{len(missing)} tags)")
    return faces


def face_eval(boxes, x0, tw, top, H):
    """returns (cut_count, inside_count) for the window [x0,x0+tw]x[top,top+H]."""
    m = TUNE["face_margin"]
    cut = inside = 0
    x1, y1 = x0 + tw, top + H
    for fx, fy, fw, fh in boxes:
        F1x, F1y = fx + fw, fy + fh
        inside_ = fx >= x0 + m and F1x <= x1 - m and fy >= top + m and F1y <= y1 - m
        inter = fx < x1 and F1x > x0 and fy < y1 and F1y > top
        if inside_:
            inside += 1
        elif inter:
            cut += 1
    return cut, inside


def run_qc(plan_path, faces_json, face_xml, slices, report_path, sheet_path, check_only=False,
           canvas_w=1080):
    plan = json.load(open(plan_path, encoding="utf-8"))
    plan.sort(key=lambda p: p["shot"])
    faces_all = load_faces(faces_json, face_xml, slices, sorted({p["img"] for p in plan}))
    report, risk, fixed, choice = [], [], [], []
    sheet_items = []

    for p in plan:
        setkey, fn = p["img"].split("_", 1)
        im = Image.open(os.path.join(slices[setkey], fn)).convert("L")
        W2, H2 = im.size
        arr = np.asarray(im, dtype=np.uint8)
        s = p["s"]
        H = int(p["H"])
        tw = int(round(float(canvas_w) / s))
        x0 = int(max(0, min(W2 - tw, p["x0"])))
        boxes = faces_all.get(p["img"], [])
        t0, t1 = float(p["top_start"]), float(p["top_end"])

        def ev_pixel(top):
            tt = int(max(0, min(H2 - H, top)))
            return score(detect(arr[tt:tt + H, x0:x0 + tw]))

        def ev_face(top):
            tt = int(max(0, min(H2 - H, top)))
            return face_eval(boxes, x0, tw, tt, H)

        tops = [t0, (t0 + t1) / 2.0, t1]
        face_ok = all(ev_face(t)[0] == 0 for t in tops) and (not boxes or any(ev_face(t)[1] >= 1 for t in tops))
        ssum = sum(ev_pixel(t) for t in tops)

        entry = dict(shot=p["shot"], img=p["img"], status="clean", score=ssum,
                     faces_total=len(boxes), face_ok=face_ok, fixed_top=None,
                     violations=[{"top": round(t, 1), "cut": ev_face(t)[0], "inside": ev_face(t)[1],
                                  **detect(arr[int(max(0, min(H2 - H, t))):int(max(0, min(H2 - H, t))) + H,
                                                 x0:x0 + tw])
                                  } for t in tops])

        # 人工/demo 指定的镜头：只报告不动窗（2026-10-07；QC 自动修窗曾把人工取景修坏）
        is_demo = p.get("src") == "demo"

        if (ssum > 0 or not face_ok) and not check_only and not is_demo:
            delta0 = 0.0

            def face_pass(cand):
                cut, ins = face_eval(boxes, x0, tw, cand, H)
                if cut == 0 and (not boxes or ins >= 1):
                    return True
                if boxes:  # fallback: 大脸/部分脸但覆盖率>=70%且脸心在窗内
                    for fx, fy, fw, fh in boxes:
                        ix = max(0, min(x0 + tw, fx + fw) - max(x0, fx))
                        iy = max(0, min(cand + H, fy + fh) - max(cand, fy))
                        cov = (ix * iy) / float(fw * fh)
                        ccx, ccy = fx + fw / 2.0, fy + fh / 2.0
                        if cov >= 0.70 and (x0 <= ccx <= x0 + tw) and (cand <= ccy <= cand + H):
                            return True
                return False

            safe = []
            lo = int(max(0, min(H2 - H, min(tops) - TUNE["face_range"])))
            hi = int(max(0, min(H2 - H, max(tops) + TUNE["face_range"])))
            for cand in range(lo, hi + 1, TUNE["face_step"]):
                if face_pass(cand):
                    safe.append(cand)
            if not safe and boxes:
                for cand in range(0, H2 - H + 1, 50):
                    if face_pass(cand):
                        safe.append(cand)
                if safe:
                    lo, hi = 0, H2 - H

            best = None
            if safe:
                coarse = safe[::max(1, len(safe) // 24)]
                scored = []
                for cand in coarse:
                    d = cand - min(tops)
                    sc = sum(ev_pixel(t + d) for t in tops)
                    scored.append((sc, abs(d), cand))
                scored.sort()
                for sc, _, cand in scored[:3]:
                    d = cand - min(tops)
                    for c2 in range(max(lo, cand - 40), min(hi, cand + 40) + 1, TUNE["slide_step"]):
                        if face_pass(c2):
                            sc2 = sum(ev_pixel(t + (c2 - min(tops))) for t in tops)
                            if best is None or sc2 < best[0]:
                                best = (sc2, c2 - min(tops))
                if best is None and scored:
                    best = (scored[0][0], scored[0][2] - min(tops))
            if best is not None:
                d = best[1]
                p["top_start"] = round(t0 + d, 1)
                p["top_end"] = round(t1 + d, 1)
                if best[0] == 0:
                    entry["status"] = "fixed"
                    fixed.append(p["shot"])
                else:
                    entry["status"] = "choice"
                    entry["remaining"] = best[0]
                    choice.append(p["shot"])
            else:
                entry["status"] = "risk"
                risk.append(p["shot"])
        elif is_demo and (ssum > 0 or not face_ok):
            entry["status"] = "choice"   # demo 镜有嫌疑 → 人工复核，但不动窗
            choice.append(p["shot"])
        elif ssum > 0 or not face_ok:
            entry["status"] = "risk"
            risk.append(p["shot"])

        # 静止镜报警（2026-10-11 蒸馏：对标片静止镜≈0）：平移行程 < 3% 画宽即记录；
        # 有微缩放(s_end)的只标 note，全无运动且非人工镜的升 choice 交人工
        pan_px = max(abs(p["top_end"] - p["top_start"]),
                     abs(p.get("x0_end", p["x0"]) - p["x0"])) * s
        entry["pan_px"] = round(pan_px, 1)
        if pan_px < 0.03 * canvas_w:
            if p.get("s_end") and abs(p["s_end"] - p["s"]) > 1e-4:
                entry["note"] = "平移静止(有微缩放)"
            elif p.get("src") != "demo" and entry["status"] == "clean":
                entry["status"] = "choice"
                entry["note"] = "静止镜：平移与缩放均无运动"
                choice.append(p["shot"])
        # 推拉错配校验已删除（2026-10-12：缩放定律系测量噪声，对标片纯平移）
        report.append(entry)

        mid = (p["top_start"] + p["top_end"]) / 2.0
        tt = int(max(0, min(H2 - H, mid)))
        sub = arr[tt:tt + H, x0:x0 + tw]
        rgb = Image.fromarray(sub).convert("RGB")
        w = 300
        rgb = rgb.resize((w, int(rgb.height * w / rgb.width)))
        d = ImageDraw.Draw(rgb)
        kx = w / float(tw)
        for (bx, by, bw, bh, cnt) in detect(sub)["bubbles"]:
            d.rectangle([bx * kx, by * kx, (bx + bw) * kx, (by + bh) * kx], outline=(255, 40, 40), width=3)
        for (bx, by, bw, bh, cnt) in detect(sub)["text_cells"]:
            d.rectangle([bx * kx, by * kx, (bx + bw) * kx, (by + bh) * kx], outline=(255, 120, 0), width=3)
        for fx, fy, fw, fh in boxes:
            cut, ins = face_eval([(fx, fy, fw, fh)], x0, tw, tt, H)
            if cut or ins:
                col = (60, 255, 60) if ins else (255, 0, 0)
                d.rectangle([(fx - x0) * kx, (fy - tt) * kx, (fx + fw - x0) * kx, (fy + fh - tt) * kx],
                            outline=col, width=2)
        sheet_items.append((p["shot"], p["img"], entry["status"], rgb))

    if not check_only:
        json.dump(plan, open(plan_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(report, open(report_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    cols = 5
    rows = (len(sheet_items) + cols - 1) // cols
    tile_h = max(it[3].height for it in sheet_items)
    sh = Image.new("RGB", (cols * 306 + 6, rows * (tile_h + 22) + 6), (12, 12, 12))
    dd = ImageDraw.Draw(sh)
    for k, (si, img, status, tile) in enumerate(sheet_items):
        x = 6 + (k % cols) * 306
        y = 6 + (k // cols) * (tile_h + 22)
        sh.paste(tile, (x, y + 16))
        color = {"clean": (120, 255, 120), "fixed": (255, 220, 120), "choice": (255, 170, 60),
                 "risk": (255, 80, 80)}.get(status, (200, 200, 200))
        dd.text((x + 3, y + 2), f"{si} {img} [{status}]", fill=color)
    sh.save(sheet_path)

    result = {"clean": sum(1 for e in report if e["status"] == "clean"),
              "fixed": len(fixed), "choice": len(choice), "risk": len(risk),
              "fixed_list": fixed, "choice_list": choice, "risk_list": risk,
              "report": report_path, "sheet": sheet_path, "plan": plan_path}
    print("=== QC PIXELS (量产版) ===")
    print(f"clean: {result['clean']} | fixed: {result['fixed']} | "
          f"choice: {result['choice']} | risk: {result['risk']}")
    if fixed:
        print("fixed:", fixed)
    if choice:
        print("二选一（默认保脸）:", choice)
    if risk:
        print("风险（需人工窗口/放大范围）:", risk)
        for e in report:
            if e["status"] == "risk":
                bad = [v for v in e["violations"] if score(v) > 0 or v["cut"] > 0 or (e["faces_total"] and v["inside"] == 0)]
                print(f"  shot {e['shot']} {e['img']} faces={e['faces_total']}: " +
                      "; ".join(f"top={v['top']} cut={v['cut']} in={v['inside']} seams={v['seams']} "
                                f"text={len(v['text_cells'])} bub={len(v['bubbles'])}" for v in bad))
    print("report:", report_path)
    print("sheet :", sheet_path)
    return result
