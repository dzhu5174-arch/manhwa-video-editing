# -*- coding: utf-8 -*-
"""page_view.py — 原图（下拉式整页）分块阅读工具

背景（2026-10-06 用户指正）：
    **读图必须读原图，不要读切片。**
    切片是按空白带横向切出的条带，会把**气泡尾尖**与人物连接处切断，
    导致「尾尖不可判」的假性挂起（EP2 曾因此产生大量无谓疑问）。
    原图是下拉式整页（900×16000 长条），尾尖与画面上下文完整。

原图 16000px 高、超过读图上限（8192px），故本工具负责**分块降采样**输出。

用法：
  # 单页切成若干块（写到 工作区\_页\ 下）
  python page_view.py --page "<原图目录>\\02.webp" --tile 3600 --overlap 200 --width 900

  # 整话批量（默认只切有对白的页；--all 全切）
  python page_view.py --dir "<原图目录>" --out "<输出目录>" [--all]

输出：<输出目录>\p02_t1.jpg … 每块一张，可直接用 read_image 查看。
"""
import argparse
import os
import sys

from PIL import Image

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

MAX_SIDE = 8000          # 留余量给读图上限


def tiles_for(img, tile, overlap):
    W, H = img.size
    if H <= MAX_SIDE and W <= MAX_SIDE:
        yield 0, H
        return
    step = max(1, tile - overlap)
    y = 0
    while y < H:
        yield y, min(H, y + tile)
        if y + tile >= H:
            break
        y += step


def dump(path, outdir, tile, overlap, width, prefix=None):
    img = Image.open(path)
    W, H = img.size
    stem = prefix or os.path.splitext(os.path.basename(path))[0]
    made = []
    spans = list(tiles_for(img, tile, overlap))
    for i, (y0, y1) in enumerate(spans, start=1):
        crop = img.crop((0, y0, W, y1))
        if crop.width > width:
            h = int(crop.height * width / crop.width)
            crop = crop.resize((width, h), Image.LANCZOS)
        p = os.path.join(outdir, f"{stem}_t{i}.jpg")
        crop.convert("RGB").save(p, "JPEG", quality=88, subsampling=2)
        made.append(p)
    return W, H, made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tile", type=int, default=3600)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--all", action="store_true", help="批量时连小页也切")
    args = ap.parse_args()

    if args.page:
        outdir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.page)), "_页")
        os.makedirs(outdir, exist_ok=True)
        W, H, made = dump(args.page, outdir, args.tile, args.overlap, args.width)
        print(f"{os.path.basename(args.page)}: {W}x{H} → {len(made)} 块")
        for p in made:
            print("  " + os.path.basename(p))
        return 0

    if not args.dir:
        sys.exit("需要 --page 或 --dir")
    outdir = args.out or os.path.join(os.path.abspath(args.dir), "_页")
    os.makedirs(outdir, exist_ok=True)
    fs = sorted(f for f in os.listdir(args.dir)
                if f.lower().endswith((".webp", ".jpg", ".jpeg", ".png")))
    print(f"目录: {args.dir}（{len(fs)} 张）→ {outdir}")
    total = 0
    for f in fs:
        p = os.path.join(args.dir, f)
        img = Image.open(p)
        W, H = img.size
        if (not args.all) and max(W, H) <= MAX_SIDE:
            continue
        _, _, made = dump(p, outdir, args.tile, args.overlap, args.width)
        total += len(made)
        print(f"  {f:14} {W}x{H} → {len(made)} 块")
    print(f"共 {total} 块")
    return 0


if __name__ == "__main__":
    sys.exit(main())
