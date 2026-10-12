# -*- coding: utf-8 -*-
"""push_to_github.py — 改完代码后一键上传（可覆盖），并自动更新 README（短、直观、含变更日志与对标差距）

用法:
  python push_to_github.py --msg "本次改了什么"
"""
import argparse
import datetime
import os
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/dzhu5174-arch/manhwa-video-editing.git"
REPO_DIR = r"C:\Users\19150\github\manhwa-video-editing"
SRC = r"C:\Users\19150\Desktop\manhua video v2"
PJ = os.path.join(SRC, "勇者无法休息", "脚本_量产管线")
CHANGELOG = "CHANGELOG.md"

README = """# manhwa-video-editing · 漫解剪辑流水线

**一句话**：分镜按客户的 demo 切点走；取景按「单格化」law 把长条裁成单格；运镜线性匀速；一切结论可复算。

## 流水线
```
cut_demo_ffmpeg.py  在你的 demo 上直接剪（ffmpeg）→ 分镜=你的切点，画面=你的像素
pipeline.py         plan0 分镜 → plan 定窗 → qc 像素质检 → preview 预览 → draft 剪映草稿
director_loop.py    scan 出候选图 → 人工/AI 裁决 → apply 写回 config 并复跑
tools/qc            motion_survey 等独立复核
```

## 现行法则
1. **单格化裁窗 v2**（用户定律）：不丢真 > 裁满 > 中间权重高·周边低 > 只放大到够关键帧移动
   - 白像素/气泡/字幕框 = 死重，出画；窗口上下边必须贴内容（行密度>0.6）
   - 脸高占显示画高 30–60%；巨脸特写保「眼鼻嘴核心区」(0.18–0.92fh)；窗内白斑 ≤2%
   - 无合法窗 → 报警（禁止静默 fallback）
2. **运动层**：线性匀速、行程=画宽 10%、↑↓ 交替、绝对时间铺帧

## 与对标账号的差距（诚实版）
基准 = 实测对标片《勇士请不要休息 10-11》(1440×1080, 315s)：119 镜 / 0.379 镜每秒 / 线性匀速 / 行程≈画宽 10%
| 项 | 我们 | 对标 | 差距 |
|---|---|---|---|
| 镜密度 | 87 镜 / 225s = 0.387/s | 0.379/s | ✅ 持平 |
| 运镜曲线 | 线性匀速 | 线性匀速 | ✅ 持平（S 型缓动已证伪） |
| 单格化取景 | 34 clean / 24 fixed / 25 choice / 4 risk | 全部单格满幅 | ❌ **主要差距**：多脸格、巨脸格、静止镜未收敛 |
| 阅读顺序推进 | 未实现跨格单调推进 | 同格多镜按阅读顺序 | ❌ 待做 |
| 转场/音效层次 | 无（仅线性位移） | 有淡入淡出/强调处理 | ❌ 待做 |
| 小小谈漫 / 白羊爱漫画 | — | 未逐一实测 | ⚠️ 需要各自的量化抽样才能下结论 |

## 文件地图
`pipeline/` 主流水线+config+过程文档+samples ｜ `tools/qc/` 独立质检 ｜ `skill/` 运镜蒸馏

## 约定与坑
cv2 读不了中文路径（先落 ASCII 临时路径）｜cv2 5.0 无 Haar，人脸走 YuNet｜已被剪映打开的草稿不得覆盖｜任何验收结论必须能从 plan.json / qc_report.json 独立复算
"""


def sh(cmd, cwd=None, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=cwd)
    if check and r.returncode != 0:
        print(r.stdout, r.stderr)
        sys.exit("命令失败: %s" % " ".join(cmd))
    return r


def ensure_repo():
    if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
        os.makedirs(os.path.dirname(REPO_DIR), exist_ok=True)
        sh(["git", "clone", REPO_URL, REPO_DIR])
        sh(["git", "config", "user.name", "zds5174"], cwd=REPO_DIR)
        sh(["git", "config", "user.email", "1915075083@qq.com"], cwd=REPO_DIR)
    else:
        sh(["git", "pull", "--rebase", "--autostash"], cwd=REPO_DIR, check=False)


def copy_code():
    pairs = [
        (os.path.join(PJ, "*.py"), os.path.join(REPO_DIR, "pipeline")),
        (os.path.join(PJ, "configs", "*.json"), os.path.join(REPO_DIR, "pipeline", "configs")),
        (os.path.join(PJ, "*.md"), os.path.join(REPO_DIR, "pipeline", "docs")),
        (os.path.join(PJ, "work", "ep02", "*.json"), os.path.join(REPO_DIR, "pipeline", "samples")),
        (os.path.join(SRC, "工具", "质检", "*.py"), os.path.join(REPO_DIR, "tools", "qc")),
        (os.path.join(SRC, "skill", "manhwa-pan-pipeline", "SKILL.md"),
         os.path.join(REPO_DIR, "skill", "manhwa-pan-pipeline")),
    ]
    import glob
    for pat, dst in pairs:
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(pat):
            shutil.copy2(f, dst)
    with open(os.path.join(REPO_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(README)


def log_change(msg):
    p = os.path.join(REPO_DIR, CHANGELOG)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    old = open(p, encoding="utf-8").read() if os.path.exists(p) else "# 变更日志（新→旧）\n\n"
    head, _, body = old.partition("\n\n")
    with open(p, "w", encoding="utf-8") as f:
        f.write(head + "\n\n" + "- **%s** %s\n" % (ts, msg) + body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True)
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()
    ensure_repo()
    copy_code()
    log_change(a.msg)
    sh(["git", "add", "-A"], cwd=REPO_DIR)
    r = sh(["git", "status", "--porcelain"], cwd=REPO_DIR)
    if not r.stdout.strip():
        print("无变化，跳过提交"); return
    sh(["git", "commit", "-m", a.msg], cwd=REPO_DIR)
    if not a.no_push:
        sh(["git", "push", "origin", "HEAD:main"], cwd=REPO_DIR)
        print("已推送 →", REPO_URL)
    print("完成：", a.msg)


if __name__ == "__main__":
    main()
