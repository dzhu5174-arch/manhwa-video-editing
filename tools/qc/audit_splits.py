# -*- coding: utf-8 -*-
"""audit_splits.py — 分句切割件「内容完整性」审查（转写回对，2026-10-07）

用途：确认 `<漫画>\\配音\\第N话_分句_切割\\L##.mp3`（剪辑实际使用的无缝切割件）
每一句 ① 有声音、时长合理；② 转写内容与稿内对应行一致（不缺字/不多词/不串句）。

用法：
  python audit_splits.py --dir "<配音>\\第N话_分句_切割" --doc "<工作区>\\第N话\\口播_工作版.md"
         [--model <path>] [--only L01,L07] [--json <out.json>]

判定：
  OK       相似度 ≥ 0.72，且无串句嫌疑
  可疑     0.50 ≤ 相似度 < 0.72；或转写明显偏长（串入邻句）；或疑似截断
  不合格   相似度 < 0.50；或转写为空（疑似静音/切空）；或时长 < 0.3s
退出码：0=全 OK；1=有可疑/不合格
"""
import argparse
import difflib
import json
import os
import re
import subprocess
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

MODEL_DEFAULT = r"C:\Users\19150\Desktop\manhua video v2\工具\模型\faster-whisper-large-v3"


def dur(p):
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", p], capture_output=True, text=True)
    except FileNotFoundError:
        from mutagen.mp3 import MP3
        return MP3(p).info.length
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def norm(s):
    out = []
    for ch in s or "":
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            out.append(ch)
        elif ch.isascii() and ch.isalnum():
            out.append(ch.lower())
    return "".join(out)


def parse_doc(path):
    t = open(path, encoding="utf-8").read()
    return {m.group(1): m.group(2).strip()
            for m in re.finditer(r"^L(\d{2,3})\s+(.+)$", t, re.M)}


def load_model(model_dir):
    from faster_whisper import WhisperModel
    last = None
    for dev, ct in [("cuda", "float16"), ("cpu", "int8")]:
        try:
            m = WhisperModel(model_dir, device=dev, compute_type=ct,
                             cpu_threads=(16 if dev == "cpu" else 0) or None)
            print(f"model ok: device={dev}, compute={ct}")
            return m
        except Exception as e:
            last = e
            print(f"device={dev} 失败: {str(e)[:100]}")
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--only", default=None, help="只查指定行，逗号分隔（如 L01,L07）")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    doc = parse_doc(args.doc)
    only = set(x.strip().upper() for x in args.only.split(",")) if args.only else None
    tags = sorted(doc.keys(), key=lambda x: int(x))
    if only:
        tags = [t for t in tags if (f"L{t}" in only or t in only)]

    files = [(t, os.path.join(args.dir, f"L{t}.mp3")) for t in tags]
    missing = [f"L{t}" for t, p in files if not os.path.exists(p)]
    ok_files = [(t, p) for t, p in files if os.path.exists(p)]

    model = load_model(args.model)
    results = []
    print(f"审查 {len(ok_files)} 个文件" + (f"；缺文件 {missing}" if missing else ""))
    print(f"{'句':<5}{'判定':<12}{'时长':<8}{'相似度':<8}转写（前 34 字）")
    for t, p in ok_files:
        d = dur(p)
        try:
            segs, _ = model.transcribe(p, language="zh", beam_size=5,
                                       vad_filter=False, condition_on_previous_text=False)
            heard = "".join(s.text for s in segs)
        except Exception as e:
            heard = ""
            print(f"  [warn] L{t} transcribe 异常: {str(e)[:60]}")
        exp, h = norm(doc[t]), norm(heard)
        ratio = difflib.SequenceMatcher(None, exp, h).ratio() if (exp and h) else 0.0

        if not h:
            verdict = "不合格·空"
        elif d < 0.3:
            verdict = "不合格·过短"
        elif ratio < 0.50:
            verdict = "不合格"
        elif ratio < 0.72:
            verdict = "可疑"
        elif len(h) > len(exp) * 1.7 and len(h) - len(exp) >= 8:
            verdict = "可疑·串句"
        else:
            verdict = "OK"
        results.append({"句号": f"L{t}", "判定": verdict, "时长秒": round(d, 2),
                        "相似度": round(ratio, 3), "转写": heard.strip(),
                        "原稿": doc[t]})
        print(f"L{t:<4}{verdict:<12}{d:<8.2f}{ratio:<8.3f}{heard.strip()[:34]}")

    bad = [r for r in results if r["判定"] != "OK"]
    print("=" * 60)
    print(f"合计 {len(results)} 句：OK {len(results) - len(bad)} | 可疑/不合格 {len(bad)}"
          + (f" | 缺文件 {len(missing)}" if missing else ""))
    if bad:
        print("需复核：" + ", ".join(f"{r['句号']}({r['判定']} r={r['相似度']})" for r in bad))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"dir": args.dir, "doc": args.doc, "缺文件": missing,
                       "结果": results}, f, ensure_ascii=False, indent=2)
        print("json ->", args.json)

    return 1 if (bad or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
