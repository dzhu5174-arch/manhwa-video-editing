# -*- coding: utf-8 -*-
"""verify_tts.py — 配音三件套深度验收（2026-10-06 升级）

用法：
  python verify_tts.py --chapters 2-9 [--quick]

检查项：
  ✓ 整篇时长 vs 目标（汉字/5.4s）；分句数、时间轴行数 vs 稿行数
  ✓ 时间轴单调、收尾对齐整篇
  ④ 语速离群：每句 汉字/时长 偏离中位数（>1.45x 或 <0.55x）→ 切点可疑，
     不得直接用于剪辑对轴（历史事故：EP2 L07 差 4s、EP9 L44）
  ⑤ 逐句交叉（--quick 跳过）：分句 mp3 与时间轴行时长差 >0.9s
  ⑥ 空白统计：分句合计 - 整篇（逐句渲染件会累计 +~20s；剪辑请用
     cut_from_full.py 的「整篇切割」件）

退出码：0=通过；1=有离群或不合格项（附清单）
"""
import argparse
import csv
import os
import re
import statistics
import subprocess
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SERIES = r"C:\Users\19150\Desktop\manhua video v2\勇者无法休息"


def dur(p):
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", p], capture_output=True, text=True)
        return float(r.stdout.strip())
    except FileNotFoundError:
        from mutagen.mp3 import MP3
        return MP3(p).info.length


def parse_chapters(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def resolve_doc(ch):
    """稿件来源优先级：工作区工作版（当前在写/在用的稿）> 文案归档稿。

    配音是按「定的那一版稿」渲染的；归档稿可能与配音不同步，
    因此只要工作版存在就以它为准，避免拿旧稿校验新配音（曾导致 64≠68 假报警）。
    """
    cands = [os.path.join(SERIES, "工作区", f"第{ch}话", "口播_工作版.md"),
             os.path.join(SERIES, "口播工作区", f"第{ch}话", "口播_工作版.md")]
    work = next((c for c in cands if os.path.exists(c)), cands[0])
    arch = os.path.join(SERIES, "文案", f"第{ch}话", "04_口播稿.md")
    if os.path.exists(work):
        return work
    return arch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", default="2-9")
    ap.add_argument("--quick", action="store_true", help="跳过逐句时长交叉（省时间）")
    args = ap.parse_args()

    all_ok = True
    detail = []
    print(f"{'话':<4}{'整篇':<10}{'目标':<8}{'分句':<9}{'离群':<6}判定")
    for ch in parse_chapters(args.chapters):
        doc = resolve_doc(ch)
        full = os.path.join(SERIES, "配音", f"第{ch}话_口播_云希.mp3")
        splitdir = os.path.join(SERIES, "配音", f"第{ch}话_分句")
        tl = os.path.join(SERIES, "配音", f"第{ch}话_句级时间轴.csv")
        issues = []

        text = open(doc, encoding="utf-8-sig").read() if doc and os.path.exists(doc) else ""
        n_lines = len(re.findall(r"^L\d{2,3}\s+", text, re.M))
        body = "".join(re.findall(r"^L\d{2,3}\s+(.+)$", text, re.M))
        target = round(len(re.findall(r"[\u4e00-\u9fff]", body)) / 5.4)

        d = dur(full) if os.path.exists(full) else None
        # 逐句时长交叉优先核对「分句_切割」（整篇无缝切割版 = 剪辑真正使用的件）；
        # 退化时才核对「分句」（逐句渲染件，每句自带首尾留白，天然比时间轴长）。
        cutdir = os.path.join(SERIES, "配音", f"第{ch}话_分句_切割")
        xdir = cutdir if os.path.isdir(cutdir) else splitdir
        pieces = [f for f in os.listdir(xdir) if re.fullmatch(r"L\d{2,3}\.mp3", f)] \
            if os.path.isdir(xdir) else []
        n_pieces = len(pieces)

        n_off, n_cross, pad = 0, 0, None
        if os.path.exists(tl):
            rows = list(csv.DictReader(open(tl, encoding="utf-8-sig")))
            parsed = []
            try:
                starts = [float(r["起始秒"]) for r in rows]
                ends = [float(r["结束秒"]) for r in rows]
                mono = all(starts[i] >= starts[i - 1] for i in range(1, len(starts))) and \
                    all(starts[i] >= ends[i - 1] - 0.08 for i in range(1, len(starts)))
                if not mono:
                    issues.append("时间轴非单调/重叠")
                if len(rows) != n_lines:
                    issues.append(f"对轴行数{len(rows)}≠{n_lines}")
                if d is not None and ends and abs(ends[-1] - d) > max(6, 0.06 * d):
                    issues.append(f"对轴收尾{ends[-1]:.1f}s≠整篇{d:.1f}s")
                for r in rows:
                    c = len(re.findall(r"[\u4e00-\u9fff]", r["文案"]))
                    dd = float(r["时长秒"])
                    parsed.append((r["句号"], dd, c, c / dd if dd > 0 else 0))
            except Exception as e:
                issues.append(f"对轴解析失败:{str(e)[:40]}")

            if parsed:
                med = statistics.median(v for _, _, _, v in parsed)
                cps_off = [(t, round(v, 1)) for t, _, _, v in parsed
                           if v > med * 1.45 or v < med * 0.55]
                n_off = len(cps_off)
                if cps_off:
                    issues.append(f"语速离群{n_off}句")
                    detail.append(f"  EP{ch} 语速离群（中位 {med:.1f} 字/秒）: " +
                                  ", ".join(f"L{t}={v}" for t, v in cps_off[:8]) +
                                  ("…" if len(cps_off) > 8 else ""))

                if not args.quick and os.path.isdir(xdir):
                    cross, total_pieces, missing = [], 0.0, 0
                    for t, dd, c, v in parsed:
                        p = os.path.join(xdir, f"L{t}.mp3")
                        if not os.path.exists(p):
                            missing += 1
                            continue
                        dp = dur(p)
                        total_pieces += dp
                        diff = dp - dd
                        if diff > 0.9 or diff < -0.9:
                            cross.append((t, round(diff, 2)))
                    if missing:
                        issues.append(f"缺分句{missing}个")
                    n_cross = len(cross)
                    src = "切割件" if xdir == cutdir else "逐句渲染件"
                    if cross:
                        issues.append(f"{src}-对轴差异常{n_cross}句")
                        detail.append(f"  EP{ch} {src}vs对轴 差>0.9s: " +
                                      ", ".join(f"L{t}{dd:+.2f}s" for t, dd in cross[:8]))
                    if d and xdir == splitdir:
                        pad = total_pieces - d
                        if pad > 4:
                            detail.append(f"  EP{ch} 分句合计比整篇多 {pad:+.1f}s"
                                          f"（逐句渲染空白，剪辑请用整篇或切割件）")
                    elif d and xdir == cutdir:
                        pad = total_pieces - d
                        detail.append(f"  EP{ch} 切割件合计 vs 整篇差 {pad:+.1f}s"
                                      f"（无缝切割，应≈0）")
        else:
            issues.append("缺时间轴")

        if d is None:
            issues.append("缺mp3")
        elif abs(d - target) > max(15, 0.15 * target):
            issues.append(f"时长偏差{d - target:+.0f}s")
        if n_pieces != n_lines:
            issues.append(f"分句{n_pieces}≠{n_lines}")

        verdict = "✅" if not issues else "❌ " + "; ".join(issues)
        if issues:
            all_ok = False
        print(f"{ch:<4}{(f'{d:.1f}s' if d else '-'):<10}{f'{target}s':<8}"
              f"{f'{n_pieces}/{n_lines}':<9}{n_off:<6}{verdict}")

    if detail:
        print("\n明细：")
        for line in detail:
            print(line)
    print("\n" + ("=== 全部通过 ===" if all_ok else "=== 存在不合格项（离群句不得直接用于剪辑）==="))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
