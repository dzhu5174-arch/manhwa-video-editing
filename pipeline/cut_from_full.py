# -*- coding: utf-8 -*-
"""cut_from_full.py — 从整篇 mp3 按句级时间轴切出无缝分句（「整篇切割」）

为什么：逐句单独渲染的 mp3 每句自带 0.3~0.6s 空白，68 句可累计 +21s；
整篇切割的每句与时间轴逐行对齐（差 ≈ 0.00s），剪辑对轴不漂。
（第1话曾用此法：分句v3_整篇切割，逐句差 +0.00s；第2话起丢失，2026-10-06 恢复）

前置：时间轴必须先过 verify_tts（语速离群=0），离群时间轴切出来必然是错的。

用法：
  python cut_from_full.py --full ..\\配音\\第2话_口播_云希.mp3 \\
      --csv ..\\配音\\第2话_句级时间轴.csv --out ..\\配音\\第2话_分句_切割
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, help="整篇 mp3")
    ap.add_argument("--csv", required=True, help="句级时间轴 csv")
    ap.add_argument("--out", required=True, help="输出目录（建议 <第N话_分句_切割>）")
    ap.add_argument("--lead", type=float, default=0.05, help="句首保留少许（默认 0.05s）")
    ap.add_argument("--tail", type=float, default=0.10, help="句尾保留少许（默认 0.10s）")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("未找到 ffmpeg（不在 PATH 上）")
    os.makedirs(args.out, exist_ok=True)
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8-sig")))
    if not rows:
        sys.exit("时间轴为空")

    for r in rows:
        tag = r["句号"].strip()
        st = max(0.0, float(r["起始秒"]) - args.lead)
        en = float(r["结束秒"]) + args.tail
        out = os.path.join(args.out, f"L{tag}.mp3")
        cmd = ["ffmpeg", "-y", "-v", "error",
               "-ss", f"{st:.3f}", "-t", f"{en - st:.3f}",
               "-i", args.full, "-vn", "-c:a", "libmp3lame",
               "-b:a", "48k", "-ar", "24000", "-ac", "1", out]
        subprocess.run(cmd, check=True)

    print(f"整篇切割完成：{len(rows)} 句 -> {args.out}")
    print("校验：再跑 verify_tts 时请指向本目录核对（逐句差应 ≈ 0）")


if __name__ == "__main__":
    main()
