# -*- coding: utf-8 -*-
"""align_timeline.py — 口播稿句级时间轴（faster-whisper 词级 + 字符 DP 对齐）

用法：
  python align_timeline.py --audio ..\配音\第2话_口播_云希.mp3 --doc ..\文案\第2话\04_口播稿.md \
      --out-csv ..\配音\第2话_句级时间轴.csv

产出：
  - 句级时间轴 csv（句号,起始秒,结束秒,时长秒,时间码,文案）
  - 同名 .md 表格
  - 控制台打印对齐质量（匹配率/边界抽查）
"""
import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import time

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

MODEL_DEFAULT = r"C:\Users\19150\Desktop\manhua video v2\工具\模型\faster-whisper-large-v3"


def parse_lines(doc):
    text = open(doc, encoding="utf-8").read()
    lines = [(m.group(1), m.group(2).strip())
             for m in re.finditer(r"^L(\d{2})\s+(.+)$", text, re.M)]
    if not lines:
        sys.exit(f"未找到 L## 行: {doc}")
    return lines


def norm(ch):
    """归一化：仅保留中日韩字符与字母数字（小写）。"""
    o = ord(ch)
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:  # CJK
        return ch
    if ch.isascii() and ch.isalnum():
        return ch.lower()
    return None


def dp_align(a, b):
    """全局对齐（NW）：返回配对列表 [(i, j)]（含错配；gap 不返回）。"""
    n, m = len(a), len(b)
    INF = 1 << 30
    prev = np.arange(m + 1, dtype=np.int32)
    dirs = np.zeros((n + 1, m + 1), dtype=np.uint8)  # 0=diag 1=up 2=left
    if m >= 0:
        dirs[0, 1:] = 2
    for i in range(1, n + 1):
        cur = np.empty(m + 1, dtype=np.int32)
        cur[0] = i
        dirs[i, 0] = 1
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            d = prev[j - 1] + cost
            u = prev[j] + 1
            l = cur[j - 1] + 1
            if d <= u and d <= l:
                cur[j] = d
                dirs[i, j] = 0
            elif u <= l:
                cur[j] = u
                dirs[i, j] = 1
            else:
                cur[j] = l
                dirs[i, j] = 2
        prev = cur
    i, j = n, m
    pairs = []
    while i > 0 or j > 0:
        d = dirs[i, j]
        if d == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif d == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def _write_out(args, sents, spans, adur):
    """单调化 + 写 csv/md + 语速抽查（whisper 与 proportional 两种模式共用）。"""
    out = []
    prev_end = 0.0
    for st, en, _src in spans:
        st = max(st, prev_end)
        en = max(en, st + 0.05)
        en = min(en, adur)
        out.append((st, en))
        prev_end = en

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["句号", "起始秒", "结束秒", "时长秒", "时间码", "文案"])
        for (tag, txt, _), (st, en) in zip(sents, out):
            mm = int(st // 60)
            ss = st - mm * 60
            w.writerow([tag, f"{st:.2f}", f"{en:.2f}", f"{en - st:.2f}",
                        f"{mm:02d}:{ss:05.2f}", txt])

    md = args.out_csv[:-4] + ".md"
    with open(md, "w", encoding="utf-8") as f:
        f.write(f"# 句级时间轴（{os.path.basename(args.audio)}）"
                f"　模式: {args.mode}\n\n")
        f.write("| 句号 | 起始 | 结束 | 时长 | 文案 |\n|---|---|---|---|---|\n")
        for (tag, txt, _), (st, en) in zip(sents, out):
            f.write(f"| {tag} | {st:.2f} | {en:.2f} | {en - st:.2f} | {txt} |\n")

    print("csv ->", args.out_csv)
    print("md  ->", md)
    print(f"总时长: {out[-1][1]:.1f}s | 抽查: L01 {out[0][0]:.2f}-{out[0][1]:.2f} | "
          f"L{len(out):02d} {out[-1][0]:.2f}-{out[-1][1]:.2f}")
    long_ones = [(sents[i][0], round(en - st, 1), len(sents[i][1]))
                 for i, (st, en) in enumerate(out) if en - st > 12]
    if long_ones:
        print("⚠ 超长句段（>12s）:", long_ones)

    zh = {tag: len(re.findall(r"[\u4e00-\u9fff]", txt)) for tag, txt, _ in sents}
    cps = [(tag, zh[tag] / max(en - st, 0.01)) for (tag, _, _), (st, en) in zip(sents, out)]
    med = statistics.median(v for _, v in cps)
    bad = [(tag, round(v, 1)) for tag, v in cps if v > med * 1.45 or v < med * 0.55]
    if bad:
        print(f"语速抽查：中位 {med:.1f} 字/秒 | ⚠ 离群 {bad}（切点可能不准，核对后再用）")
    else:
        print(f"语速抽查：中位 {med:.1f} 字/秒 | 无离群")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--mode", choices=["whisper", "proportional"], default="whisper",
                    help="whisper=词级对齐（默认，最准）；"
                         "proportional=按字数比例分配（**降级方案**：无需模型，"
                         "误差 ±0.3–0.8s，仅供无模型时出大致时间轴，切点须人工核对）")
    args = ap.parse_args()

    lines = parse_lines(args.doc)
    sents = [(tag, txt, [norm(c) for c in txt if norm(c)])
             for tag, txt in lines]
    script_chars = [c for _, _, cc in sents for c in cc]
    sent_of = np.array([i for i, (_, _, cc) in enumerate(sents) for _ in cc], dtype=np.int32)
    print(f"句数: {len(sents)} | 脚本字符: {len(script_chars)}")

    # ---------------- 降级方案：按字数比例分配 ----------------
    if args.mode == "proportional":
        try:
            adur = float(subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", args.audio],
                capture_output=True, text=True).stdout.strip())
        except FileNotFoundError:
            from mutagen.mp3 import MP3
            adur = MP3(args.audio).info.length
        zh = [len(re.findall(r"[\u4e00-\u9fff]", txt)) or 1 for _, txt, _ in sents]
        total_zh = sum(zh)
        # 句间留 0.15s 气口（从总时长里先扣掉）
        gap = 0.15
        usable = max(1.0, adur - gap * (len(sents) - 1))
        spans, t = [], 0.0
        for n in zh:
            d = usable * n / total_zh
            spans.append((t, min(adur, t + d), "proportional"))
            t += d + gap
        print(f"⚠ 降级模式（proportional）：音频 {adur:.1f}s，按字数比例分配，"
              f"误差 ±0.3–0.8s —— 切点用于剪辑前须人工核对")
        _write_out(args, sents, spans, adur)
        return

    from faster_whisper import WhisperModel

    def transcribe(model):
        segs, info = model.transcribe(args.audio, language="zh", beam_size=1, vad_filter=True,
                                      word_timestamps=True, condition_on_previous_text=False)
        words = []
        for s in segs:
            for w in (s.words or []):
                words.append((w.word, w.start, w.end))
        return words, info.duration

    model = None
    words = None
    adur = None
    for dev, ct in [("cuda", "float16"), ("cpu", "int8")]:
        try:
            t0 = time.time()
            model = WhisperModel(args.model, device=dev, compute_type=ct)
            words, adur = transcribe(model)
            print(f"whisper device={dev} ok ({time.time() - t0:.0f}s)")
            break
        except Exception as e:
            print(f"device={dev} 失败: {str(e)[:120]}")
            model = None
    if words is None:
        sys.exit("whisper 转写失败")
    print(f"词数: {len(words)} | 音频: {adur:.1f}s")

    # 字符级时间流
    chars = []
    for wtext, ws, we in words:
        cc = [norm(c) for c in wtext if norm(c)]
        n = len(cc)
        for j, c in enumerate(cc):
            t0 = ws + (we - ws) * j / n
            t1 = ws + (we - ws) * (j + 1) / n
            chars.append((c, t0, t1))
    whisper_chars = np.array([c for c, _, _ in chars], dtype=object)

    pairs = dp_align(script_chars, [c for c, _, _ in chars])
    matched = sum(1 for i, j in pairs if script_chars[i] == whisper_chars[j])
    hit_rate = matched / max(1, len(script_chars))
    print(f"对齐: 配对 {len(pairs)} | 字符命中率 {hit_rate:.1%}")

    # 每句收集对齐到的 whisper 字符时间
    sent_times = [[] for _ in sents]
    for si, sj in pairs:
        if sj < len(chars):
            sent_times[sent_of[si]].append((chars[sj][1], chars[sj][2]))

    spans = []
    for i, (_, _, cc) in enumerate(sents):
        ts = sent_times[i]
        if len(ts) >= max(3, int(0.35 * len(cc))):
            st = min(t[0] for t in ts)
            en = max(t[1] for t in ts)
            # 收一点边界（词级对齐的天然外扩）
            st = max(0.0, st - 0.02)
            en = min(adur, en + 0.05)
            spans.append((st, en, "aligned"))
        else:
            spans.append(None)

    # 插值兜底
    for i, sp in enumerate(spans):
        if sp is not None:
            continue
        prev = next((spans[k] for k in range(i - 1, -1, -1) if spans[k]), (0.0, 0.0, "x"))
        nxt = next((spans[k] for k in range(i + 1, len(spans)) if spans[k]), (adur, adur, "x"))
        st = prev[1]
        en = max(nxt[0], st + 0.1)
        spans[i] = (st, en, "interp")

    _write_out(args, sents, spans, adur)


if __name__ == "__main__":
    main()
