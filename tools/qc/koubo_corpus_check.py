# -*- coding: utf-8 -*-
"""koubo_corpus_check.py — 语料基准闸门：口播稿 vs 文案分析语料统计。
硬拒（exit 1）：语速越界、句长中位/p90 越界、strict 模式下台词缺溯源标记。
警告（不拒）：引语比、问号密度、开场钩子。
用法:
  koubo_corpus_check.py <口播稿.md> --target <秒数> [--strict-source]
"""
import argparse, json, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = r"C:\Users\19150\Desktop\manhua video v2"
STATS = ROOT + r"\文案分析\蒸馏_语料统计.json"

ap = argparse.ArgumentParser()
ap.add_argument("doc")
ap.add_argument("--target", type=float, required=True, help="目标秒数")
ap.add_argument("--strict-source", action="store_true", help="台词行必须带[源:...]标记")
a = ap.parse_args()

S = json.load(open(STATS, encoding="utf-8"))
corpus_rate = S["汉字总量"] / S["总时长秒"]          # 字/秒
med0, p900 = S["句法"]["句长中位"], S["句法"]["句长p90"]
short0 = S["句法"]["短句占比_le15"]
quote0 = S["引语比_引号内汉字占全文"]

cjk = lambda s: re.findall(r"[一-鿿]", s)

lines = []
for raw in open(a.doc, encoding="utf-8"):
    raw = raw.strip()
    m = re.match(r"^L\d+\s*(.*)", raw)
    if m:
        lines.append(m.group(1))

total = sum(len(cjk(t)) for t in lines)
lens = sorted(len(cjk(t)) for t in lines if cjk(t))
rate = total / a.target
med = lens[len(lens) // 2]
p90 = lens[int(len(lens) * 0.9)]
short = sum(1 for L in lens if L <= 15) / len(lens)
q_marks = sum(t.count("？") + t.count("?") for t in lines)
q_per_1k = q_marks / max(total, 1) * 1000
quote_chars = sum(len(cjk(m)) for t in lines for m in re.findall(r"「([^」]*)」", t))
quote_ratio = quote_chars / max(total, 1)

fails, warns = [], []

def check(name, val, lo, hi, unit=""):
    ok = lo <= val <= hi
    status = "✓" if ok else "✗"
    print(f"{status} {name}: {val:.2f}{unit}（语料区间 {lo:.2f}~{hi:.2f}）")
    if not ok:
        fails.append(f"{name} {val:.2f} 越界 [{lo:.2f},{hi:.2f}]")

print(f"正文 {len(lines)} 行 / {total} 字 / 目标 {a.target:.0f}s\n—— 硬校验 ——")
check("语速", rate, corpus_rate * 0.75, corpus_rate * 1.25, " 字/秒")
check("句长中位", med, med0 - 6, med0 + 6, " 字")
check("句长p90", p90, 0, p900 + 15, " 字")
check("短句占比", short, short0 - 0.15, short0 + 0.15)

if a.strict_source:
    no_src = [t for t in lines if "「" in t and "[源:" not in t]
    if no_src:
        fails.append(f"{len(no_src)} 行台词缺溯源标记")
    print(f"{'✓' if not no_src else '✗'} 溯源标记: {len(lines) - len(no_src)}/{len(lines)} 行齐全")

print("—— 软提示 ——")
hook_words = "没想到|居然|突然|竟然|结果|原来|但|却|？|\\?"
head = "".join(lines[:3])
hook = bool(re.search(hook_words, head))
print(f"{'✓' if hook else '⚠'} 开场钩子: 前3行{'含' if hook else '不含'}悬念/疑问信号")
print(f"  问号密度: {q_per_1k:.1f}/千字（语料 {S['问号']['千字']:.1f}）")
print(f"  引语比: {quote_ratio:.2f}（语料 {quote0:.2f}；用户排版层允许「」直引）")

if fails:
    print("\n✗ 拒稿:", "; ".join(fails))
    sys.exit(1)
print("\n✓ 放行" + (f"（{len(warns)} 条警告）" if warns else ""))
