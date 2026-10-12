# -*- coding: utf-8 -*-
"""
faces_dy.py — 用隔离 venv 的 cv2 4.10（含 CascadeClassifier）做 anime 人脸检测。
本机主环境 cv2 5.0 已移除 CascadeClassifier，故本脚本必须用
C:\\Users\\19150\\.workbuddy\\binaries\\python\\envs\\default\\Scripts\\python.exe 运行。
输出：dy_work/faces_dy.json  {shot_i: [[x,y,w,h]...]}（bbox crop 内坐标）
"""
import json, os, subprocess
import numpy as np
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(BASE)
ROOT = os.path.dirname(PROJ)
SRC = os.path.join(PROJ, "demo", "dy_work", "demo_h264.mp4")
OUT = os.path.join(PROJ, "demo", "dy_work")
SHOTS = os.path.join(OUT, "shots2.json")
CAS = os.path.join(ROOT, "工具", "模型", "lbpcascade_animeface.xml")
INSET = 8

# 中文路径安全：复制到 ASCII 临时文件再加载
tmp = os.path.join(os.environ.get("TEMP", "/tmp"), "lbpcascade_animeface.xml")
if not os.path.exists(tmp):
    with open(CAS, "rb") as f:
        open(tmp, "wb").write(f.read())
cas = cv2.CascadeClassifier(tmp)
if cas.empty():
    raise SystemExit("级联加载失败（不要静默吞掉）：" + tmp)


def imread_zh(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


shots = json.load(open(SHOTS, encoding="utf-8"))
res = {}
tmpdir = os.path.join(OUT, "frames")
os.makedirs(tmpdir, exist_ok=True)
for rec in shots:
    if rec.get("empty") or not rec.get("bb"):
        continue
    mid = (rec["t0"] + rec["t1"]) / 2
    p = os.path.join(tmpdir, f"full_{round(mid,2)}.png")
    if not os.path.exists(p):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{mid:.3f}", "-i", SRC, "-frames:v", "1", p],
                       capture_output=True)
    fr = imread_zh(p)
    x0b, x1b, y0b, y1b = rec["bb"]
    crop = fr[y0b + INSET:y1b - INSET, x0b + INSET:x1b - INSET]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    boxes = cas.detectMultiScale(g, scaleFactor=1.05, minNeighbors=3, minSize=(24, 24))
    res[str(rec["i"])] = [[int(v) for v in b] for b in boxes]

json.dump(res, open(os.path.join(OUT, "faces_dy.json"), "w", encoding="utf-8"), ensure_ascii=False)
n = sum(1 for v in res.values() if v)
print(f"faces: {n}/{len(res)} 镜检出人脸 | 文件 => " + os.path.join(OUT, "faces_dy.json"))
