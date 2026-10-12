# EP2 重剪任务卡（opencode + DeepSeek 执行版）

> 你是剪辑执行员，不是剪辑师。所有参数已由脚本按对标定律算好，
> 你的工作 = 跑命令 + 看候选图做选择题 + 写 JSON。除此之外一概禁止。

## 环境（逐字使用，不得更改）

- 工作目录：`C:\Users\19150\Desktop\manhua video v2\勇者无法休息\脚本_量产管线`
- Python：`D:\develop\python\python.exe`
- 配置：`configs\ep02.json`（画布 1440×1080，v2 运镜定律已内置：线性匀速 / 行程=画宽10% / ↓↑交替 / ↑推↓拉2.8% / 静止镜禁止）

## 铁律（违反任意一条 = 任务失败，立即停工）

1. **禁止编写、修改任何 .py 文件**；禁止修改 config 的任何字段（config 只能由 `director_loop.py apply` 写入）。
2. **禁止自己找图、拼图、裁图、发明切片文件名**——你只能使用 `待裁决.json` 里列出的 img 和候选窗位。
3. 裁决结果只能写进 `work\ep02\裁决结果.json`，且必须通过 schema 校验（被脚本拒收就修正格式，**不得绕过校验改脚本**）。
4. 拿不准的镜头一律填 `"HUMAN"`，禁止猜。
5. 草稿名固定用 `勇者无法休息_第2话_v3`，**禁止覆盖任何已存在的剪映草稿**，禁止使用 `--force-draft`。
6. 全流程最多 2 轮裁决循环；仍有 risk/HUMAN 未清 → 停工，把清单报告给人类。

## 执行流程（按序，逐字跑命令）

### Step 1 扫描
```powershell
cd "C:\Users\19150\Desktop\manhua video v2\勇者无法休息\脚本_量产管线"
D:\develop\python\python.exe director_loop.py scan --config configs/ep02.json
```
- 打印 `CLEAN` → 直接跳到 Step 4。
- 否则它会生成：`work\ep02\待裁决.json` + 每镜一张候选图 `work\ep02\ajud_shotNN.jpg`。

### Step 2 裁决（你唯一需要动脑的地方）
读 `work\ep02\待裁决.json`，逐镜打开对应 `ajud_shotNN.jpg` 看图：
- 每镜 3 个候选 A/B/C（每候选上行=起幅、下行=落幅）。
- 选景标准：**脸完整可见（巨脸特写允许裁发际线/下巴，但眼鼻嘴必须在画内）＞ 气泡/文字框不入画 ＞ 主体居中**。
- A = 当前窗位；B/C = 脚本生成的备选。都不满意 → `"HUMAN"`。

把答案写入 `work\ep02\裁决结果.json`，格式（一个字都不许多）：
```json
{"answers": [{"shot": 28, "choice": "B"}, {"shot": 38, "choice": "HUMAN"}, {"shot": 58, "choice": "C", "note": "可选备注"}]}
```

### Step 3 应用并复扫
```powershell
D:\develop\python\python.exe director_loop.py apply --config configs/ep02.json
```
脚本会自动校验 → 写入 demo 覆盖 → 重跑 plan+qc 并打印新状态。
- 还有非人工锁定的 risk/choice → 回到 Step 1（第 2 轮）。
- 有 HUMAN 项 → 停工报告，等人类处理。

### Step 4 出片
```powershell
D:\develop\python\python.exe pipeline.py --config configs/ep02.json --steps preview,draft --draft-name 勇者无法休息_第2话_v3
```
完成后报告：QC 状态分布、预览视频路径、剪映草稿名。**不要自己看预览下结论**，预览是给人类终审的。

## 完成报告格式（最后输出）
```
QC: clean=X fixed=X choice=X risk=0
裁决: shot NN→X, ...（HUMAN: [...]）
预览: <preview_out 路径>
草稿: 勇者无法休息_第2话_v3
遗留问题: <有就列，没有写"无">
```
