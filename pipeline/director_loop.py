# -*- coding: utf-8 -*-
"""director_loop.py — 剪辑裁决闭环（AI 子代理专用闸口）。
定位：pipeline 干全部参数活；AI 只做「看候选图 → 选 A/B/C/HUMAN」的离散裁决。

  scan   跑 plan+qc → 对 risk/非人工锁定的 choice 镜生成候选窗位图 + 待裁决.json
  apply  严格校验 裁决结果.json → 写入 config demo 覆盖 → 自动重跑 plan+qc

反幻觉设计：AI 不能发明文件名/窗位/代码，只能从脚本给出的候选里选；schema 之外一律拒收。
"""
import sys, io, os, json, shutil, subprocess, argparse, datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CW, CH = 1440, 1080
TRAVEL = 0.10 * CW          # v2 蒸馏：行程=画宽10%（纯平移，零缩放）


def run_pipeline(cfg_path, steps):
    r = subprocess.run([sys.executable, os.path.join(HERE, "pipeline.py"),
                        "--config", cfg_path, "--steps", steps],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = "\n".join((r.stdout or "").splitlines()[-6:])
    if r.returncode != 0:
        sys.exit(f"pipeline 失败（{steps}）:\n{tail}\n{r.stderr[-800:] if r.stderr else ''}")
    return tail


def load_cfg(cfg_path):
    return json.load(open(cfg_path, encoding="utf-8"))


def workdir(cfg, cfg_path):
    w = cfg["work"]
    if os.path.isabs(w):
        return w
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(cfg_path)), w))


def qc_entries(w):
    d = json.load(open(os.path.join(w, "qc_report.json"), encoding="utf-8"))
    return d["entries"] if isinstance(d, dict) else d


def slice_path(cfg, img):
    setkey, fn = img.split("_", 1)
    p = os.path.join(cfg["slices"][setkey], fn)
    return p


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def candidates(q, box, H2):
    """A=当前窗 B=人脸中心(眼位40%)按镜号奇偶定向 C=反方向。纯平移（v4：推拉锁定已证伪删除）。"""
    s, x0 = q["s"], q["x0"]
    H = CH / s
    down = q["shot"] % 2 == 1
    half = TRAVEL / s / 2

    def mk(mid_top, go_down):
        ts = clamp(mid_top - half, 0, H2 - H)
        te = clamp(mid_top + half, 0, H2 - H)
        if not go_down:
            ts, te = te, ts
        return dict(s=s, x0=x0, top_start=round(ts, 1), top_end=round(te, 1))

    opts = {"A": dict(s=s, x0=x0, top_start=q["top_start"], top_end=q["top_end"],
                      **({"s_end": q["s_end"]} if q.get("s_end") else {}))}
    if box:
        fx, fy, fw, fh = box
        eye_cy = fy + (0.40 * fh if fh > H else fh / 2)   # 巨脸对眼位，小脸对中心
        mid = clamp(eye_cy - H / 2, 0, H2 - H)
    else:
        mid = clamp((q["top_start"] + q["top_end"]) / 2, 0, H2 - H)
    opts["B"] = mk(mid, down)
    opts["C"] = mk(mid, not down)
    return opts


def render_option(cfg, img, o):
    arr = cv2.imdecode(np.fromfile(slice_path(cfg, img), dtype=np.uint8), cv2.IMREAD_COLOR)
    H2 = arr.shape[0]
    tiles = []
    for top, sk in ((o["top_start"], o["s"]), (o["top_end"], o.get("s_end", o["s"]))):
        w, h = int(CW / sk), int(CH / sk)
        top = int(clamp(top, 0, max(0, H2 - h)))
        crop = arr[top:top + h, int(o["x0"]):int(o["x0"]) + w]
        tiles.append(cv2.resize(crop, (360, 270)))
    return tiles  # [起幅, 落幅]


def cmd_scan(cfg_path):
    cfg = load_cfg(cfg_path)
    w = workdir(cfg, cfg_path)
    run_pipeline(cfg_path, "plan,qc")
    ents = qc_entries(w)
    demo_locked = {int(k) for k in (cfg.get("demo") or {})}
    flagged = [e["shot"] for e in ents
               if e.get("status") == "risk" or (e.get("status") == "choice" and e["shot"] not in demo_locked)]
    import collections
    print("QC:", dict(collections.Counter(e.get("status") for e in ents)))
    if not flagged:
        print("CLEAN —— 无待裁决镜头，可直接 preview,draft")
        return
    plan = {q["shot"]: q for q in json.load(open(os.path.join(w, "plan.json"), encoding="utf-8"))}
    faces = json.load(open(os.path.join(w, "faces.json"), encoding="utf-8"))
    quest = {"episode": cfg["episode"],
             "rule": "每镜选取景最好的候选：脸完整可见/气泡与文字框不入画/主体居中。拿不准填 HUMAN。禁止发明候选之外的窗位。",
             "shots": []}
    for sid in flagged:
        q = plan[sid]
        boxes = faces.get(q["img"], [])
        box = boxes[0] if boxes else None
        arr = cv2.imdecode(np.fromfile(slice_path(cfg, q["img"]), dtype=np.uint8), cv2.IMREAD_COLOR)
        opts = candidates(q, box, arr.shape[0])
        rows = []
        for label in ("A", "B", "C"):
            tiles = render_option(cfg, q["img"], opts[label])
            row = np.hstack(tiles)
            cv2.putText(row, label, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            rows.append(row)
        sheet = np.vstack(rows)
        fn = f"ajud_shot{sid:02d}.jpg"
        cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(os.path.join(w, fn))
        quest["shots"].append({"shot": sid, "img": q["img"], "image": fn,
                               "note": "上行=起幅 下行=落幅；A=当前 B/C=备选",
                               "options": opts})
    out = os.path.join(w, "待裁决.json")
    json.dump(quest, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"待裁决 {len(flagged)} 镜: {flagged}")
    print(f"问卷: {out}")
    print("→ AI 逐镜看图（work 目录 ajud_shot*.jpg），写 裁决结果.json 后跑 apply")


def cmd_apply(cfg_path):
    cfg = load_cfg(cfg_path)
    w = workdir(cfg, cfg_path)
    qp = os.path.join(w, "待裁决.json")
    ap = os.path.join(w, "裁决结果.json")
    if not os.path.exists(qp):
        sys.exit("没有 待裁决.json —— 先跑 scan")
    if not os.path.exists(ap):
        sys.exit("没有 裁决结果.json —— AI 还没写裁决")
    quest = json.load(open(qp, encoding="utf-8"))
    ans = json.load(open(ap, encoding="utf-8"))
    # ---- 强约束校验：schema 之外一律拒收 ----
    if set(ans) - {"answers"}:
        sys.exit("拒收：顶层只允许 answers 键")
    if not isinstance(ans.get("answers"), list):
        sys.exit("拒收：answers 必须是数组")
    qmap = {s["shot"]: s for s in quest["shots"]}
    human, applied = [], []
    for a in ans["answers"]:
        if set(a) - {"shot", "choice", "note"}:
            sys.exit(f"拒收：答案含非法键 {a}")
        sid, ch = a.get("shot"), a.get("choice")
        if sid not in qmap:
            sys.exit(f"拒收：shot {sid} 不在问卷里（禁止发明镜头）")
        if ch == "HUMAN":
            human.append(sid); continue
        if ch not in qmap[sid]["options"]:
            sys.exit(f"拒收：shot {sid} 的 choice={ch} 不是候选之一（{list(qmap[sid]['options'])}/HUMAN）")
        applied.append((sid, ch, a.get("note", "")))
    if not applied and human:
        sys.exit(f"全部标记 HUMAN：{human} —— 停工，交人工")
    # ---- 写 demo 覆盖（先备份）----
    bak = cfg_path + ".bak_" + datetime.datetime.now().strftime("%m%d_%H%M%S")
    shutil.copy(cfg_path, bak)
    demo = cfg.setdefault("demo", {})
    for sid, ch, note in applied:
        entry = dict(qmap[sid]["options"][ch])
        entry["_note"] = f"DS裁决{ch} {datetime.date.today().isoformat()}" + (f" {note}" if note else "")
        demo[str(sid)] = entry
    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"已写入 demo 覆盖 {len(applied)} 镜: {[s for s, _, _ in applied]}（备份 {os.path.basename(bak)}）")
    if human:
        print(f"⚠ 剩余 HUMAN 项 {human} —— 需人工处理后再跑 scan")
    run_pipeline(cfg_path, "plan,qc")
    ents = qc_entries(w)
    import collections
    print("重跑 QC:", dict(collections.Counter(e.get("status") for e in ents)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "apply"])
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    {"scan": cmd_scan, "apply": cmd_apply}[a.cmd](a.config)


if __name__ == "__main__":
    main()
