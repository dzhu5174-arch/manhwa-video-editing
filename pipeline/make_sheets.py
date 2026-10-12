# -*- coding: utf-8 -*-
"""make_sheets.py — 为一个章节的 切片 目录生成双列阅读拼图（供 AI/人 阅读取证）

用法：
  python make_sheets.py "C:/Users/19150/Desktop/manhua video v2/勇者无法休息/源素材/第2话"

产出：<series_dir>/阅读拼图/sheet_000.jpg（封面）+ sheet_001..jpg（每张两块）+ map.json
"""
import json
import os
import sys

from PIL import Image

GRAY = (85, 85, 85)


def main():
    series_dir = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\Users\19150\Desktop\manhua video v2\勇者无法休息\源素材\第2话"
    tiles_dir = os.path.join(series_dir, "切片")
    out_dir = os.path.join(series_dir, "阅读拼图")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(tiles_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
    if not files:
        sys.exit(f"切片目录为空: {tiles_dir}")

    mapping = {}
    # sheet_000 = 封面
    cover = Image.open(os.path.join(tiles_dir, files[0])).convert("RGB")
    cover.save(os.path.join(out_dir, "sheet_000.jpg"), quality=90)
    mapping["sheet_000.jpg"] = [files[0]]

    rest = files[1:]
    idx = 1
    for i in range(0, len(rest), 2):
        a = Image.open(os.path.join(tiles_dir, rest[i])).convert("RGB")
        b = Image.open(os.path.join(tiles_dir, rest[i + 1])).convert("RGB") if i + 1 < len(rest) else None
        tw = a.width
        h = max(a.height, b.height) if b else a.height
        sheet = Image.new("RGB", (tw * 2, h), GRAY)
        sheet.paste(a, (0, 0))
        if b:
            sheet.paste(b, (tw, 0))
        name = f"sheet_{idx:03d}.jpg"
        sheet.save(os.path.join(out_dir, name), quality=90)
        mapping[name] = [rest[i]] + ([rest[i + 1]] if b else [])
        idx += 1

    json.dump(mapping, open(os.path.join(out_dir, "map.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"sheets: {len(mapping)} -> {out_dir}")


if __name__ == "__main__":
    main()
