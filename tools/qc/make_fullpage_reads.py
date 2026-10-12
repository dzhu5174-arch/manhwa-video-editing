# -*- coding: utf-8 -*-
r"""make_fullpage_reads.py — 生成「整页通读」素材（读图唯一入口）

规则（2026-10-06 用户指定）：
  · 读图只看**整页**：页首读到页尾，图里必须包含这一格的完整画面
  · **保宽不缩**（默认 900），只纵向切高 —— 缩放会把气泡小字毁掉
  · 切点**尽量落在空白带**（纯色横条），不得切在人物/气泡上
  · 每张图带页号与页内 y 范围，便于引用「原图\NN.webp 第 a–b 处」

算法：先按 max_side 求最小段数 n → 把页均分 n 段得目标切点 →
      每个切点在 ±window 内吸附到最近的空白行（找不到就用目标值）。
输出：<源素材>\第N话\_整页通读\pNN_1.jpg / pNN_2.jpg …
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

MAX_SIDE = 7800
COVER_W = 900          # 宽 > 此值视为封面


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="某话的 原图 目录")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-side", type=int, default=MAX_SIDE)
    ap.add_argument("--window", type=int, default=900, help="切点吸附窗口(px)")
    ap.add_argument("--min-h", type=int, default=800, help="单段最小高度")
    args = ap.parse_args()

    src = os.path.abspath(args.dir)
    out = args.out or os.path.join(os.path.dirname(src), "_整页通读")
    os.makedirs(out, exist_ok=True)

    files = sorted(f for f in os.listdir(src)
                   if f.lower().endswith((".webp", ".jpg", ".jpeg", ".png")))
    print("原图 %d 张 → %s" % (len(files), out))
    total = 0
    for f in files:
        im = Image.open(os.path.join(src, f)).convert("RGB")
        W, H = im.size
        stem = os.path.splitext(f)[0]
        if W > COVER_W:
            im.save(os.path.join(out, "p%s.jpg" % stem), "JPEG", quality=92, subsampling=2)
            print("  %-10s %dx%d → 1 张（封面）" % (f, W, H))
            total += 1
            continue

        gray = np.asarray(im.convert("L")).astype(np.float32)
        rowstd = gray.std(axis=1)
        blank = rowstd < 6.0

        n = max(1, -(-H // args.max_side))          # ceil
        if n == 1:
            im.save(os.path.join(out, "p%s_1.jpg" % stem), "JPEG", quality=92, subsampling=2)
            print("  %-10s %dx%d → 1 张" % (f, W, H))
            total += 1
            continue

        # 均分目标切点
        targets = [round(H * k / n) for k in range(1, n)]
        cuts = [0]
        for t in targets:
            lo = max(cuts[-1] + args.min_h, t - args.window)
            hi = min(H - args.min_h, t + args.window)
            lo, hi = max(1, lo), min(H - 2, hi)
            if lo < hi:
                cand = [y for y in range(lo, hi) if blank[y]]
                c = min(cand, key=lambda y: abs(y - t)) if cand else t
            else:
                c = t
            cuts.append(c)
        cuts.append(H)

        segs = [(cuts[i], cuts[i + 1]) for i in range(n)]
        for i, (y0, y1) in enumerate(segs):
            im.crop((0, y0, W, y1)).save(
                os.path.join(out, "p%s_%d.jpg" % (stem, i + 1)), "JPEG",
                quality=92, subsampling=2)
            total += 1
        span = " ".join("%d-%d(%d)" % (a, b, b - a) for a, b in segs)
        snapped = all(blank[c] for c in cuts[1:-1])
        print("  %-10s %dx%d → %d 段  %s  切点全落空白:%s"
              % (f, W, H, n, span, "是" if snapped else "否"))
    print("共 %d 张" % total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
