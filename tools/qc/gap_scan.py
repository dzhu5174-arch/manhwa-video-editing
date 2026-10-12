# -*- coding: utf-8 -*-
"""gap_scan.py — 分句缝隙扫描：找「落在切割件之外」的语音（防丢字，2026-10-07）

背景：`audit_splits.py` 逐件比对只能查「件内」内容；若语音掉进件与件的**缝隙**里，
单件比对永远查不出（EP2 实测：L19 的句首「啊！」落在 L18/L19 的缝隙中）。
本工具专查缝隙：把所有未覆盖区间（头、句间缝、尾）中「有能量」的段落微转写，
出现中文字符即报警。

用法：
  python gap_scan.py --full "<配音>\\第N话_口播_云希.mp3" --csv "<配音>\\第N话_句级时间轴.csv"
         [--model <path>] [--th -45] [--min 0.08] [--lead 0.05] [--tail 0.10]

  · 覆盖口径与 cut_from_full.py 对齐：分句件实际覆盖 [起始−lead, 结束+tail]；
    缝隙 = 「上一句 结束+tail」到「下一句 起始−lead」之间的未覆盖段。

退出码：0=缝隙无语音；1=有缝隙语音（需处理）
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
import wave

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

MODEL_DEFAULT = r"C:\Users\19150\Desktop\manhua video v2\工具\模型\faster-whisper-large-v3"
CJK = re.compile(r"[\u4e00-\u9fff]")


def dur(p):
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", p], capture_output=True, text=True)
        return float(r.stdout.strip())
    except FileNotFoundError:
        from mutagen.mp3 import MP3
        return MP3(p).info.length


def read_spans(csv_path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    spans = []
    for r in rows:
        spans.append((r["句号"], float(r["起始秒"]), float(r["结束秒"])))
    return spans


def gaps_of(spans, total, lead=0.05, tail=0.10):
    out = []
    prev_end = 0.0
    for tag, st, en in spans:
        a = max(0.0, st - lead)
        if a - prev_end >= 0.0:
            out.append((f"L{tag}(前)", prev_end, a))
        prev_end = max(prev_end, min(total, en + tail))
    if total - prev_end >= 0.0:
        out.append(("尾部", prev_end, total))
    return out


def extract_wav(full, a, b, path):
    try:
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", full,
                        "-ar", "16000", "-ac", "1", path], check=True)
        w = wave.open(path)
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        return x
    except FileNotFoundError:
        # 无 ffmpeg 环境：PyAV 直接解码切片为 16k 单声道 float
        import av
        c = av.open(full)
        rs = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=16000)
        chunks = []
        for fr in c.decode(c.streams.audio[0]):
            for r in rs.resample(fr):
                chunks.append(r.to_ndarray()[0])
        for r in rs.resample(None):
            chunks.append(r.to_ndarray()[0])
        x = np.concatenate(chunks)
        return x[int(a * 16000):int(b * 16000)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--th", type=float, default=-45.0, help="能量阈值 dB（超过则微转写）")
    ap.add_argument("--min", type=float, default=0.08, help="最短检查时长 s")
    ap.add_argument("--lead", type=float, default=0.05, help="与 cut_from_full 对齐的句首保留")
    ap.add_argument("--tail", type=float, default=0.10, help="与 cut_from_full 对齐的句尾保留")
    args = ap.parse_args()

    total = dur(args.full)
    spans = read_spans(args.csv)
    gaps = [(n, a, b) for n, a, b in gaps_of(spans, total, args.lead, args.tail)
            if b - a >= args.min]
    print(f"整篇 {total:.2f}s | 缝隙 {len(gaps)} 段（≥{args.min}s）")

    model = None
    hits = []
    tmp = tempfile.mkdtemp(prefix="gapscan_")
    for name, a, b in gaps:
        wav = os.path.join(tmp, "g.wav")
        x = extract_wav(args.full, a, b, wav)
        peak = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9) if len(x) else -180.0
        note = ""
        if peak > args.th:
            if model is None:
                from faster_whisper import WhisperModel
                model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=16)
            segs, _ = model.transcribe(x, language="zh", beam_size=5,
                                       vad_filter=False, condition_on_previous_text=False)
            txt = "".join(s.text for s in segs).strip()
            note = txt
            if CJK.search(txt):
                hits.append((name, a, b, peak, txt))
        print(f"  {name:<10} {a:7.2f}-{b:7.2f} ({b - a:.2f}s) 峰值 {peak:6.1f} dB  {note[:40]}")

    print("=" * 60)
    if hits:
        print(f"❌ 发现 {len(hits)} 段缝隙语音（可能丢字）：")
        for name, a, b, peak, txt in hits:
            print(f"  {name} {a:.2f}-{b:.2f}s  {peak:.1f}dB  「{txt}」")
        return 1
    print("✅ 缝隙内无语音")
    return 0


if __name__ == "__main__":
    sys.exit(main())
