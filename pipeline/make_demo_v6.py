# -*- coding: utf-8 -*-
"""v6 15s demo + 自检指标（从 plan.json 渲染前 15s；音频取 VO 同区间）"""
import os, sys, json, subprocess
import numpy as np, cv2

PJ = r"C:\Users\19150\Desktop\manhua video v2\勇者无法休息\脚本_量产管线"
sys.path.insert(0, PJ)
import pipeline as P

cfg = P.load_cfg(os.path.join(PJ, "configs", "ep02.json"))
W, H = P.CANVAS_W, P.CANVAS_H
work = P.work_dir(cfg)
plan = json.load(open(os.path.join(work, "plan.json"), encoding="utf-8"))
plan.sort(key=lambda x: x["shot"])
faces = json.load(open(P.resolve(cfg, cfg["faces_json"]), encoding="utf-8"))

# ---- 自检指标 ----
s_face = [p["s"] for p in plan if p.get("v2", {}).get("mode") == "face"]
jmps = []
for a, b in zip(plan, plan[1:]):
    if a["img"] == b["img"]:
        jmps.append(abs(b["top_start"] - a["top_start"]) / (H / b["s"]))
mono = all(x <= 1.0 + 1e-6 for x in jmps)
print("v6 自检: 镜数 %d | face窗 s 中位 %s (带 1.9-2.5) | face窗数 %d" % (
    len(plan), (round(float(np.median(s_face)), 2) if s_face else "无"), len(s_face)))
print("       同切片相邻镜 y 跳变>1窗高 的个数 %d (应0) | 静止镜 %d | 缩放镜 %d" % (
    sum(1 for x in jmps if x > 1.0), sum(1 for p in plan if p.get("static")),
    sum(1 for p in plan if p.get("s_end", p["s"]) != p["s"])))

# ---- 前 15s 渲染 ----
segs = [p for p in plan if p["t0"] < 15.0]
print("前 15s 镜:", [(p["shot"], round(p["t0"], 2), p["img"], p["s"]) for p in segs])
wd = r"C:\Users\19150\AppData\Local\Temp\opencode\demo15\v6"
os.makedirs(wd, exist_ok=True)
_H = {}
def sh_of(img):
    if img not in _H:
        setkey, fn = img.split("_", 1)
        _H[img] = cv2.imdecode(np.fromfile(os.path.join(P.slices_of(cfg)[setkey], fn), dtype=np.uint8),
                               cv2.IMREAD_GRAYSCALE).shape[0]
    return _H[img]

parts = []
for k, p in enumerate(segs):
    setkey, fn = p["img"].split("_", 1)
    src = os.path.join(P.slices_of(cfg)[setkey], fn)
    s = p["s"]
    Wn, Hn = W / s, H / s
    SH = sh_of(p["img"])
    x0 = max(0.0, min(1800 - Wn, p["x0"]))
    y0 = max(0.0, min(SH - Hn, p["top_start"]))
    y1 = max(0.0, min(SH - Hn, p["top_end"]))
    dur = min(p["dur"], 15.0 - p["t0"])
    if dur <= 0.05:
        continue
    seg = os.path.join(wd, "s%02d.mp4" % k)
    vf = ("crop=w=%d:h=%d:x=%d:y='%d+(%d-%d)*(t/%.3f)',scale=%d:%d:flags=lanczos,setsar=1,format=yuv420p"
          % (int(Wn), int(Hn), int(x0), int(y0), int(y1), int(y0), dur, W, H))
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "30", "-loop", "1", "-i", src,
                        "-t", "%.3f" % dur, "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "18", "-pix_fmt", "yuv420p", "-an", seg], capture_output=True)
    if r.returncode != 0:
        print("FAIL", k, r.stderr.decode("utf-8", "replace")[-200:]); raise SystemExit(1)
    parts.append(seg)
lst = os.path.join(wd, "list.txt")
with open(lst, "w", encoding="utf-8") as f:
    for x in parts:
        f.write("file '%s'\n" % x.replace("\\", "/"))
sil = os.path.join(wd, "silent.mp4")
subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", sil], check=True)
subprocess.run(["ffmpeg", "-y", "-v", "error", "-t", "15", "-i", P.resolve(cfg, cfg["vo"]),
                "-c:a", "aac", "-b:a", "192k", os.path.join(wd, "a.m4a")], check=True)
OUT = r"C:\Users\19150\Videos\EP2_15s_demo_v6.mp4"
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", sil, "-i", os.path.join(wd, "a.m4a"),
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", OUT], check=True)
print("DEMO:", OUT, "%.1f MB" % (os.path.getsize(OUT) / 1048576))
