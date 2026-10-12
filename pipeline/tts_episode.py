# -*- coding: utf-8 -*-
"""tts_episode.py — 量产管线：整篇 TTS（自动校准语速到目标时长）+ 分句渲染

用法：
  python tts_episode.py --doc ..\文案\第2话\04_口播稿.md --target 238 \
      --out-full ..\配音\第2话_口播_云希.mp3 --out-split ..\配音\第2话_分句

产出：
  - 整篇 mp3（校准后的语速）
  - 分句 mp3（L01..LNN，同一语速）
  - 校准记录 json（--report）
"""
import argparse
import asyncio
import json
import os
import re
import subprocess
import sys

import edge_tts

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def parse_lines(doc):
    text = open(doc, encoding="utf-8").read()
    lines = [(m.group(1), m.group(2).strip())
             for m in re.finditer(r"^L(\d{2})\s+(.+)$", text, re.M)]
    if not lines:
        sys.exit(f"未找到 L## 行: {doc}")
    return lines


def dur(path):
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", path], capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip())
    except FileNotFoundError:
        from mutagen.mp3 import MP3
        return MP3(path).info.length


async def render(txt, path, voice, rate, retries=4):
    last = None
    for i in range(retries):
        try:
            await asyncio.wait_for(
                edge_tts.Communicate(txt, voice, rate=rate).save(path),
                timeout=120)
            return
        except Exception as e:
            last = e
            print(f"  retry {i + 1} ({type(e).__name__}: {str(e)[:80]})")
            # 边 TTS 对「同一长文本短时间重复请求」会返回 NoAudioReceived，
            # 退避必须够长（实测 2/4/6s 不够，改为 5/12/25/40s）
            await asyncio.sleep([5, 12, 25, 40][min(i, 3)])
    raise last


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True)
    ap.add_argument("--target", type=float, required=True, help="目标秒数")
    ap.add_argument("--out-full", required=True)
    ap.add_argument("--out-split", required=True)
    ap.add_argument("--voice", default="zh-CN-YunxiNeural")
    ap.add_argument("--report", default=None)
    ap.add_argument("--rate", default=None,
                    help="直接指定语速（如 +40%%），跳过自校准迭代。"
                         "边 TTS 对同一长文本短时间重复请求会返回 NoAudioReceived，"
                         "已知语速时务必用本参数（单次渲染即可）")
    ap.add_argument("--iters", type=int, default=4, help="自校准最大迭代轮数")
    ap.add_argument("--freeze", action="store_true",
                    help="定稿锁：只把 --doc 的 sha256 记入 --report，不跑 TTS")
    ap.add_argument("--force", action="store_true",
                    help="定稿锁不匹配时仍强制跑（慎用）")
    args = ap.parse_args()

    # ---- 定稿哈希锁（2026-10-08，T1）：拿错版本的稿子直接拒跑 ----
    import hashlib
    doc_hash = hashlib.sha256(open(args.doc, "rb").read()).hexdigest()
    if args.freeze:
        if not args.report:
            sys.exit("--freeze 需要 --report 指定记录文件")
        rec = {}
        if os.path.exists(args.report):
            rec = json.load(open(args.report, encoding="utf-8"))
        rec.update({"doc": args.doc, "doc_sha256": doc_hash})
        json.dump(rec, open(args.report, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"定稿锁已记录: {args.report}\n  doc={args.doc}\n  sha256={doc_hash[:16]}…")
        return
    if args.report and os.path.exists(args.report):
        rec = json.load(open(args.report, encoding="utf-8"))
        frozen = rec.get("doc_sha256")
        if frozen and frozen != doc_hash and not args.force:
            sys.exit(
                "✗ 定稿锁拒跑：--doc 与已冻结定稿不一致\n"
                f"  冻结时 doc: {rec.get('doc')}\n"
                f"  本次 doc:   {args.doc}\n"
                f"  冻结 hash: {frozen[:16]}…  本次 hash: {doc_hash[:16]}…\n"
                "  若定稿已更新：先 --freeze 重新冻结；确需强行跑加 --force")

    lines = parse_lines(args.doc)
    full_text = "\n".join(t for _, t in lines)
    zh = len(re.findall(r"[\u4e00-\u9fff]", full_text))
    print(f"句数: {len(lines)} | 汉字: {zh} | 目标: {args.target:.0f}s")

    os.makedirs(args.out_split, exist_ok=True)
    probe = os.path.join(args.out_split, "_probe.mp3")
    probe_used = False

    if args.rate:
        # 快速路径：已知语速 → 单次渲染（避免边 TTS 对同文本重复请求的限流）
        r = args.rate if args.rate.startswith(("+", "-")) else f"+{args.rate}"
        rate_num = float(r.rstrip("%")) / 100.0
        print(f"指定语速 rate={r}（跳过自校准）")
    else:
        probe_used = True
        rate_num = 0.40
        best = None
        for i in range(args.iters):
            r = f"{rate_num:+.0%}"
            await render(full_text, probe, args.voice, r)
            d = dur(probe)
            print(f"try{i + 1}: rate={r} dur={d:.1f}s (误差 {d - args.target:+.1f}s)")
            if best is None or abs(d - args.target) < abs(best[1] - args.target):
                best = (rate_num, d)
            if abs(d - args.target) <= 4:
                break
            if i < args.iters - 1:
                await asyncio.sleep(8)      # 轮间必须留冷却，否则触发 NoAudioReceived
            rate_num = max(0.0, min((1 + rate_num) * (d / args.target) - 1, 1.5))
        rate_num, probe_d = best
        r = f"{rate_num:+.0%}"
        print(f"BEST rate={r} -> {probe_d:.1f}s")

    await render(full_text, args.out_full, args.voice, r)
    total = dur(args.out_full)
    print(f"full: {args.out_full}  {total:.1f}s")

    # 分句并发：边 TTS 对突发并发敏感（实测并发 4 会零星 NoAudioReceived），
    # 用 2 + 每句前 0.4s 抖动，稳过优先。
    sem = asyncio.Semaphore(2)

    async def one(tag, txt):
        async with sem:
            await asyncio.sleep(0.4)
            await render(txt, os.path.join(args.out_split, f"L{tag}.mp3"), args.voice, r)

    await asyncio.gather(*[one(t, x) for t, x in lines])
    split_total = sum(dur(os.path.join(args.out_split, f"L{t}.mp3")) for t, _ in lines)
    print(f"split done: {len(lines)} files, total {split_total:.1f}s")
    if abs(total - args.target) > 0.15 * args.target:
        print(f"⚠ 时间轴前置警告：整篇时长偏差 >15%（{total:.1f}s vs 目标 {args.target:.0f}s）")

    if probe_used:
        try:
            os.remove(probe)
        except OSError:
            pass

    rep = dict(doc=args.doc, doc_sha256=doc_hash, voice=args.voice, rate=r,
               target=args.target, full=args.out_full, full_dur=total,
               split_total=split_total, lines=len(lines))
    if args.report:
        json.dump(rep, open(args.report, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("done.")


if __name__ == "__main__":
    asyncio.run(main())
