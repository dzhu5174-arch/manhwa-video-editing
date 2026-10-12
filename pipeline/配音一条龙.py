# -*- coding: utf-8 -*-
"""配音一条龙.py — 口播稿 → 整篇/分句/时间轴/切割件 + 三验，一条命令跑完。

用法（在本机任意目录都行）：
    D:\\develop\\python\\python.exe 配音一条龙.py --doc <口播稿.md> --target 241 [--rate +40%]

示例：
    D:\\develop\\python\\python.exe 配音一条龙.py ^
        --doc "..\\工作区\\第2话\\口播_v24.md" --target 241 --rate +40%

流程：tts_episode → align_timeline → verify_tts → cut_from_full_av → gap_scan → audit_splits
任何一步失败立即停下并打印原因，不会带病往下走（退出码非 0）。
已知语速时一定要带 --rate，跳过 edge-tts 自校准（省 3-4 轮重复请求，避免限流）。
"""
import argparse, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # 脚本_量产管线
SERIES = HERE.parent                             # 漫画目录（如 勇者无法休息\）
ROOT = SERIES.parent                             # 项目根
MODEL = ROOT / "工具" / "模型" / "faster-whisper-large-v3"


def run(step, cmd):
    print(f"\n===== {step} =====", flush=True)
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"\n[失败] {step} 退出码 {r.returncode}。修好后重跑同一条命令即可（已完成的步骤产物还在，"
              f"但保险起见整链会重跑）。", file=sys.stderr)
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True, help="口播稿 md（定稿，路径含 第N话）")
    ap.add_argument("--target", type=float, required=True, help="目标秒数")
    ap.add_argument("--rate", default="+40%", help="edge-tts 语速，默认 +40%%")
    ap.add_argument("--skip-audit", action="store_true", help="跳过最慢的逐句内容审查（不推荐）")
    a = ap.parse_args()

    doc = Path(a.doc).resolve()
    if not doc.exists():
        sys.exit(f"[失败] 找不到口播稿：{doc}")
    m = re.search(r"第(\d+)话", str(doc))
    if not m:
        sys.exit("[失败] 口播稿路径里必须含「第N话」，否则推不出章节号")
    ch = int(m.group(1))
    PY = sys.executable
    full = SERIES / "配音" / f"第{ch}话_口播_云希.mp3"
    split = SERIES / "配音" / f"第{ch}话_分句"
    csvf = SERIES / "配音" / f"第{ch}话_句级时间轴.csv"
    cut = SERIES / "配音" / f"第{ch}话_分句_切割"

    run("① 渲染整篇+分句", [PY, str(HERE / "tts_episode.py"), "--doc", str(doc),
        "--target", str(a.target), "--rate", a.rate,
        "--out-full", str(full), "--out-split", str(split)])
    run("② whisper 词级对轴", [PY, str(HERE / "align_timeline.py"),
        "--audio", str(full), "--doc", str(doc), "--out-csv", str(csvf)])
    run("③ verify_tts 三件套验收", [PY, str(HERE / "verify_tts.py"), "--chapters", str(ch)])
    run("④ 整篇无缝切割", [PY, str(HERE / "cut_from_full_av.py"),
        "--full", str(full), "--csv", str(csvf), "--out", str(cut)])
    run("⑤ gap_scan 缝隙扫描", [PY, str(ROOT / "工具" / "质检" / "gap_scan.py"),
        "--full", str(full), "--csv", str(csvf)])
    if not a.skip_audit:
        # audit_splits 逐句重转写最慢，按 20 句一段跑，防止单次跑太久
        import csv as _csv
        with open(csvf, encoding="utf-8-sig") as f:
            lids = [r[0] for r in _csv.reader(f) if r and r[0].startswith("L")]
        for i in range(0, len(lids), 20):
            run(f"⑥ audit_splits 逐句审查 {i + 1}-{min(i + 20, len(lids))}/{len(lids)}",
                [PY, str(ROOT / "工具" / "质检" / "audit_splits.py"),
                 "--dir", str(split), "--doc", str(doc), "--model", str(MODEL),
                 "--only", ",".join(lids[i:i + 20])])
    print(f"\n[完成] 第{ch}话配音四件套齐、三验全过。产物在 {SERIES / '配音'}")


if __name__ == "__main__":
    main()
