# 脚本_量产管线 · 一条命令出一集

> 2026-10-04 建立。把 全片v7 的人工迭代管线参数化为「量产版」：**质量线 = v7 子集**（无气泡/无文字框、脸完整可见、无白边黑边、缓动运镜 travel≈150px、方向混排、烧字幕、关系钩开场）。
> 量产取舍：不再做逐镜人工复核与美化；自动 QC + 抽检 + 只修 QC 报警镜。目标 **4–6 小时/集**（其中人工 ~2 小时：口播稿终审 / QC 抽检 / BGM + 成片过目）。

## 目录

```
脚本_量产管线\
  pipeline.py        ← 主入口：plan0 / plan / qc / preview / draft
  qcmod.py           ← 像素级规则检测+自动修窗（从 qc_pixels_v7 参数化移植，逻辑不变）
  make_sheets.py     ← 章节切片 → 双列阅读拼图（写文案用）
  configs\
    ep01.json        ← 第1话（回归验证用）
    ep02.json        ← 第2话（试跑）
  work\
    ep01\  ep02\     ← 每集独立工作区（plan.json / qc_report.json / qc_sheet.jpg / segments / …）
```

## 一集完整流程

| 步骤 | 命令/工具 | 人工把关 |
|---|---|---|
| 1. 素材 | 已有：下载→切片→2x 超清（`脚本\hero_*.py` + Real-ESRGAN） | — |
| 2. 阅读拼图 | `python make_sheets.py <章节目录>` | — |
| 3. 文案 | 阅读笔记 → 口播稿（skill: manhwa-koubo-script） | **① 口播稿终审** |
| 4. 配音 | `脚本\hero_tts.py`（整篇+分句）；时间轴 CSV（句级） | — |
| 5. 分镜表 | `python pipeline.py --config configs/ep02.json --steps plan0`（一句一镜自动分配；可人工微调 `work/ep02/plan0.json` 的 `img`/窗口） | — |
| 6. 规划+QC | `--steps plan,qc` → 自动修窗 + `qc_report.json` + `qc_sheet.jpg` | **② QC 抽检**（看 sheet，只修 risk/choice 里的问题镜） |
| 7. 预览 | `--steps preview` → `C:\Users\19150\Videos\<集>_预览.mp4` | 过目取景 |
| 8. 草稿 | `--steps draft` → 剪映草稿（原生关键帧 + 字幕轨 + VO/BGM 轨） | **③ 剪映导出 + 成片过目** |

质量线细节（v7 子集）：窗口高 ≤2595（s≥0.74）、常规行程 150px、`i%2` 上下混排、行程<40px 记静止、EASE_IN_OUT 关键帧、单行字幕。

## 常用命令

```powershell
# 全流程（第2话）
D:\develop\python\python.exe pipeline.py --config configs/ep02.json --steps plan0,plan,qc,preview,draft

# 单步
D:\develop\python\python.exe pipeline.py --config configs/ep02.json --steps plan,qc
D:\develop\python\python.exe pipeline.py --config configs/ep02.json --steps draft --draft-name 勇者无法休息_第2话_v1_量产

# QC 门禁（不修改，有 risk/choice 时 exit 1）
D:\develop\python\python.exe pipeline.py --config configs/ep02.json --steps qc --check-only
```

## 环境与坑

- Python：`D:\develop\python\python.exe`（cv2 5.0.0 已装）；人脸模型 `C:\Users\19150\AppData\Local\Temp\opencode\lbpcascade_animeface.xml`（faces.json 缓存按集独立）。
- **草稿铁律**：已被剪映打开过的草稿**永不覆盖**；脚本默认检测同名草稿并拒绝，如需重建请换名（`--draft-name`）。
- 控制台编码：脚本已内置 `utf-8` 兜底（jy_wrapper 的 emoji 打印不会再把 GBK 控制台打崩）。
- `plan0.json` 是**唯一语义决策层**（哪句话配哪张图）：自动顺序分配只是初稿，可人工微调；plan/qc 全自动。
- 与 全片v7 的关系：plan/qc 自动层逻辑一致（2026-10-04 回归：QC 状态 32/13/16/0 与 v7 完全一致；61 镜 43 镜像素级相同，差异全部来自 v7 的人工后修镜 12/19/23/27/28/30/36/42/43/46/48/49/52 与 demo 恢复）。

## 验证记录（2026-10-04）

- ep01 回归：`--steps plan,qc` → clean 32 / fixed 13 / choice 16 / risk 0（与 v7 一致）
- ep01 冒烟：`--steps preview` → `C:\Users\19150\Videos\_量产回归_ep01预览.mp4`（45MB，61 段全成功）
- 草稿冒烟：61 段 + 56 运动关键帧 + 音频轨，生成成功（测试草稿已删）
- 第2话：素材 82 切片 / 42 张阅读拼图已生成，等待 文案+配音 后试跑
