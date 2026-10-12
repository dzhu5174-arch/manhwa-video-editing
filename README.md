# manhwa-video-editing · 勇者无法休息 漫解剪辑流水线

> 本仓库 = 该项目的**剪辑流水线全部代码 + 数据契约 + 裁决记录**。目的：让剪辑不再黑箱，所有参数可审计、可复算、可互相校验。
> 导出时间：2026-10-12（EP2 v5.1 之后；v2 单格化裁窗改造进行中）

---

## 一、这条流水线在干什么

**输入**：切片超清池（1800px 宽长条）＋ 句级时间轴（配音分句的绝对时间）＋ 人脸框 + 语义分镜表（shot_map：句块→切片号，人工排定）
**输出**：① 预览片（ffmpeg/PyAV 同参渲染）② 剪映草稿（关键帧=剪映原生，可直接终审）

```
plan0  分镜：合并短句 / 拆长句（阈值 SPLIT_SHOT_DUR）/ 按 shot_map 绑切片 → plan0.json
  ↓
plan   定窗：内容带 + 保脸 + 自适应缩放 + 运镜起止 → plan.json
  ↓
qc     像素级规则校验 + 自动修窗（气泡/文字框/白缝/脸被切）→ qc_report.json + qc_sheet.jpg
  ↓
preview 同参渲染预览（线性匀速运镜 / 绝对时间铺帧）
  ↓
draft   生成剪映草稿（每镜独立片段 + 运动关键帧 + 字幕）
```

## 二、现行法则（按优先级）

1. **裁窗法则 v2 · 单格化（2026-10-12 夜，用户定律，取代「全宽优先」）**
   - 不丢真 > 裁满（长条→单格漫画）> 中间权重大·周边权重小 > 只放大到够关键帧移动
   - 白像素（=空白）与气泡/字幕框 = 死重，出画；窗口上下边必须贴内容
   - 脸高占显示画高 30–60%；窗内白斑 ≤2%（女性演出格 ≤5%）
   - 无合法窗 → 报警（禁止静默 fallback 中心）
2. **运动层**：线性匀速（对标片实测线性拟合残差 1%）、行程 = 画宽 10%、方向 ↑↓ 交替、demo 镜不占相位、绝对时间铺帧
3. **硬围栏**：分镜切点/语义映射是人工资产；草稿只新建不覆盖；参数改动必须留下可复算证据

## 三、文件地图

| 目录/文件 | 说明 |
|---|---|
| `pipeline/pipeline.py` | 主流水线：build_plan0 / step_plan / step_qc / step_preview / step_draft |
| `pipeline/qcmod.py` | 像素级 QC 内核：接缝/文字框/气泡检测 + 人脸评估 |
| `pipeline/director_loop.py` | 裁决回路：scan（出候选图）→ apply（校验裁决结果→写 config→复跑） |
| `pipeline/configs/ep02.json` | 第2话配置：切片池/时间轴/人脸表/画布/shot_map/demo 覆盖 |
| `pipeline/docs/` | 任务卡、验收裁决、裁窗法则、审核口径（全部过程证据） |
| `pipeline/samples/` | 关键中间产物样例：plan0 / plan / qc_report / 待裁决 / 裁决结果 / 镜号映射 / 自检报告 |
| `tools/qc/` | 独立质检：motion_survey（运镜量化普查）、chashou_lint、make_chain、gap_scan、audit_splits… |
| `skill/manhwa-pan-pipeline/SKILL.md` | 运镜蒸馏：对标片实测的线性匀速模型与复现配方 |

## 四、怎么跑

```bash
PY=D:/develop/python/python.exe
cd pipeline
$PY pipeline.py --config configs/ep02.json --steps plan0,plan,qc      # 分镜→定窗→质检
$PY pipeline.py --config configs/ep02.json --steps preview,draft      # 出预览 + 剪映草稿
$PY director_loop.py scan --config configs/ep02.json                  # 需要裁决时出候选图
$PY director_loop.py apply --config configs/ep02.json                 # 应用裁决并复跑
$PY ../tools/qc/motion_survey.py "<预览目录>" --out <outdir>           # 独立运镜复核
```

## 五、约定与坑

- **cv2 读不了中文路径**：任何交给 cv2 的文件先落 ASCII 临时路径（faces.json 流程已内置约定）
- **cv2 5.0 无 CascadeClassifier**：人脸走 OpenCV YuNet（`face_detection_yunet_2023mar.onnx`），Haar 路线已废弃
- **草稿铁律**：已被剪映打开过的草稿不得覆盖，一律新建名字
- **人脸表**：`faces.json` 为 `{切片tag: [[x,y,w,h],...]}`（切片原生像素）
- **可复算原则**：任何验收结论必须能从 `plan.json` / `qc_report.json` 独立算出
