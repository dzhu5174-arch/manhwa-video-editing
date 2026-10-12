# -*- coding: utf-8 -*-
"""run_tts_batch.py — 勇者无法休息 批量配音（整篇+分句）+ 句级时间轴

用法：
  python run_tts_batch.py --chapters 2-9          # 逐话执行
  python run_tts_batch.py --chapters 2-9 --force  # 强制重跑

产出（<系列>\\配音\\）：
  第N话_口播_云希.mp3       整篇配音（语速自动校准到 汉字/5.4 秒）
  第N话_分句\\L01…LNN.mp3   分句（同一语速）
  第N话_句级时间轴.csv/.md   whisper 词级 + DP 对齐
  第N话_tts记录.json         校准记录
"""
import argparse
import os
import re
import subprocess
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
SERIES = os.path.dirname(HERE)
PY = sys.executable
VOICE = "zh-CN-YunxiNeural"


def parse_chapters(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    voicedir = os.path.join(SERIES, "配音")
    os.makedirs(voicedir, exist_ok=True)

    for ch in parse_chapters(args.chapters):
        doc = os.path.join(SERIES, "文案", f"第{ch}话", "04_口播稿.md")
        if not os.path.exists(doc):
            print(f"[第{ch}话] 未找到口播稿，跳过")
            continue
        text = open(doc, encoding="utf-8").read()
        body = "".join(re.findall(r"^L\d{2}\s+(.+)$", text, re.M))
        chars = len(re.findall(r"[\u4e00-\u9fff]", body))
        n_lines = len(re.findall(r"^L\d{2}\s+", text, re.M))
        target = max(60, round(chars / 5.4))
        full = os.path.join(voicedir, f"第{ch}话_口播_云希.mp3")
        splitdir = os.path.join(voicedir, f"第{ch}话_分句")
        tl = os.path.join(voicedir, f"第{ch}话_句级时间轴.csv")
        report = os.path.join(voicedir, f"第{ch}话_tts记录.json")
        os.makedirs(splitdir, exist_ok=True)

        # 清理旧版命名残留（数字命名 / probe）
        for f in list(os.listdir(splitdir)):
            if f.startswith("_probe") or re.fullmatch(r"\d{2}\.mp3", f):
                try:
                    os.remove(os.path.join(splitdir, f))
                except OSError:
                    pass

        # 断点续跑：分两段判断（TTS 段 / 对齐段），可独立补跑
        n_done = len([f for f in os.listdir(splitdir) if re.fullmatch(r"L\d{2}\.mp3", f)])
        need_tts = args.force or not (os.path.exists(full) and n_done == n_lines)
        need_align = args.force or not os.path.exists(tl)
        if not need_tts and not need_align:
            print(f"[第{ch}话] 三件套齐全，跳过")
            continue

        print(f"\n########## 第{ch}话 | {chars} 字 | 目标 {target}s | TTS={need_tts} 对齐={need_align} ##########", flush=True)
        t0 = time.time()
        if need_tts:
            r = subprocess.run([PY, os.path.join(HERE, "tts_episode.py"),
                                "--doc", doc, "--target", str(target),
                                "--out-full", full, "--out-split", splitdir,
                                "--voice", VOICE, "--report", report],
                               capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=2700)
            print((r.stdout or "")[-1200:])
            if r.returncode != 0:
                print("TTS 失败：", (r.stderr or "")[-800:])
                continue
        if need_align:
            r2 = subprocess.run([PY, os.path.join(HERE, "align_timeline.py"),
                                 "--audio", full, "--doc", doc, "--out-csv", tl],
                                capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=2700)
            print((r2.stdout or "")[-1200:])
            if r2.returncode != 0:
                print("时间轴失败：", (r2.stderr or "")[-800:])
        print(f"########## 第{ch}话 完成 {((time.time() - t0) / 60):.1f}min ##########", flush=True)

    print("\nALL DONE")


if __name__ == "__main__":
    main()
