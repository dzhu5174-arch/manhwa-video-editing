# -*- coding: utf-8 -*-
r"""ocr_pages.py — 对某话「原图」全部整页跑 Windows OCR，产出带坐标的机械识别结果。

★ Windows OCR 引擎对超高图会静默返回 0 行 → 本工具自动**分块**（默认 4000px + 300px 重叠），
  再按坐标去重合并，保证长页（900×16000）也能被完整识别。

用法:
  python 工具\质检\ocr_pages.py --comic 勇者无法休息 --ch 1
  python 工具\质检\ocr_pages.py --comic 勇者无法休息 --ch 1 --lo 2 --hi 10
  python 工具\质检\ocr_pages.py --comic 勇者无法休息 --ch 2 --tile 4000 --overlap 300

产出（写进 源素材\第N话\OCR\）:
  pNN.json    —— [{text,x,y,w,h}]（y 为整页绝对坐标，按 y,x 排序）
  _汇总.txt   —— 逐页纯文本（带 y 坐标）
  _meta.json  —— 每页行数/分块数/耗时/失败记录

★ 桥接：webp→png 用 PIL；OCR 走 Windows PowerShell 5.1 + WinRT（见 win_ocr.ps1）。
★ 编码坑：win_ocr.ps1 必须纯 ASCII（PS 5.1 按 ANSI 读脚本）。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = r"C:\Users\19150\Desktop\manhua video v2"
PS51 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                    r"System32\WindowsPowerShell\v1.0\powershell.exe")
OCR_PS1 = os.path.join(ROOT, "工具", "质检", "win_ocr.ps1")
TMP = os.path.join(ROOT, ".piptmp", "ocrpng")


def ocr_png(png_path: str) -> list:
    p = subprocess.run(
        [PS51, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", OCR_PS1,
         "-Image", png_path],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (p.stdout or "").strip()
    if not out:
        raise RuntimeError("OCR 无输出: %s" % (p.stderr or "")[:200])
    d = json.loads(out)
    if isinstance(d, dict):
        d = [d]
    return d


def ocr_page(img: Image.Image, tile: int, overlap: int) -> tuple:
    """整页 OCR：按 tile 高分块，重叠 overlap，合并去重。返回 (rows, nchunks)。"""
    W, H = img.size
    rows, nchunks = [], 0
    y = 0
    while y < H:
        y2 = min(y + tile, H)
        nchunks += 1
        crop = img.crop((0, y, W, y2))
        png = os.path.join(TMP, "_tile.png")
        crop.save(png)
        for r in ocr_png(png):
            # 只收「非重叠区」的行，避免重复；首块全收，后续块丢掉顶部 overlap
            if nchunks > 1 and r["y"] < overlap:
                continue
            rr = dict(r)
            rr["y"] = int(r["y"]) + y
            rows.append(rr)
        if y2 >= H:
            break
        y = y2 - overlap
    rows.sort(key=lambda r: (r["y"], r["x"]))
    return rows, nchunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comic", required=True)
    ap.add_argument("--ch", required=True)
    ap.add_argument("--lo", type=int, default=1)
    ap.add_argument("--hi", type=int, default=999)
    ap.add_argument("--tile", type=int, default=4000)
    ap.add_argument("--overlap", type=int, default=300)
    a = ap.parse_args()

    src = os.path.join(ROOT, a.comic, "源素材", "第%s话" % a.ch, "原图")
    out = os.path.join(ROOT, a.comic, "源素材", "第%s话" % a.ch, "OCR")
    os.makedirs(out, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)

    files = sorted(os.listdir(src),
                   key=lambda f: int(re.match(r"(\d+)", f).group(1)))
    meta, summary = [], []
    for f in files:
        n = int(re.match(r"(\d+)", f).group(1))
        if not (a.lo <= n <= a.hi):
            continue
        t0 = time.time()
        try:
            img = Image.open(os.path.join(src, f)).convert("RGB")
            rows, nchunks = ocr_page(img, a.tile, a.overlap)
            with open(os.path.join(out, "p%03d.json" % n), "w", encoding="utf-8") as fh:
                json.dump(rows, fh, ensure_ascii=False, indent=1)
            summary.append("### p%03d  (%s, %dx%d)  %d 行 / %d 块"
                           % (n, f, img.width, img.height, len(rows), nchunks))
            for r in rows:
                summary.append("  y=%-6d x=%-5d %s" % (r["y"], r["x"], r["text"]))
            summary.append("")
            ok, err = True, ""
            print("  p%03d  %-10s %4dx%-6d %3d 行 / %d 块  %5.1fs"
                  % (n, f, img.width, img.height, len(rows), nchunks, time.time() - t0))
        except Exception as e:
            ok, err = False, str(e)[:300]
            summary.append("### p%03d  (%s)  FAIL %s\n" % (n, err))
            print("  p%03d  %-10s FAIL  %s" % (n, f, err[:80]))
        meta.append({"page": n, "file": f, "lines": len(rows) if ok else 0,
                     "ok": ok, "err": err, "sec": round(time.time() - t0, 1)})

    with open(os.path.join(out, "_汇总.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary))
    with open(os.path.join(out, "_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)

    okn = sum(1 for x in meta if x["ok"])
    print("\n完成：%d/%d 页成功；总行数 %d；输出 %s"
          % (okn, len(meta), sum(x["lines"] for x in meta), out))
    if okn < len(meta):
        sys.exit(1)


if __name__ == "__main__":
    main()
