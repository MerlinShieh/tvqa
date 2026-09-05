# tvqa —— 电视音画质量自动化检测工程

针对电视画面流的 **黑屏 / 白屏 / 闪烁 / 卡顿丢帧 / 花屏 / 撕裂 / 音画不同步** 自动化检测与评测工程。
设计文档见 [docs/电视音画质量自动化检测方案.md](docs/电视音画质量自动化检测方案.md)（v1.1+），
数据集的真值规格与生成方法见 [docs/DATASETS.md](docs/DATASETS.md)。

> ⚠️ **重要说明：本工程无法开箱直接运行。**
> 它是一个以「AI Agent 辅助开发与驱动」为前提的检测框架与评测工程：
> 1. 仓库内数据集只含 **manifest 真值 + 抽样帧**，完整帧序列需按 [docs/DATASETS.md](docs/DATASETS.md) 本地再生成后才能复现完整评测；
> 2. 采集卡 / 串口 / ADB 等真实硬件通道默认走 **mock 仿真后端**，接入真实设备需自行配置并验证；
> 3. 运行环境（虚拟环境、依赖、设备名/端口等配置项）需按实际环境搭建调整；
> 4. 若需在无硬件条件下**模拟运行**（mock 通道、数据集生成、评测编排等），建议通过 **AI Agent** 辅助处理——本工程的模拟链路正是在 AI Agent 会话中构建与验证的。

```
├── config/
│   ├── default.yaml          # 工程参数与设备选择（backend/设备名全在这里改）
│   └── profiles/{eval,field}.yaml  # 检测阈值档（全部阈值可配置）
├── tvqa/                     # 工程源码
│   ├── sources/              # 输入后端：帧目录/视频文件/音频文件/采集卡（同接口可切换）
│   ├── detectors/            # 检测器：luma(黑/白/闪)、stutter、corruption(撕裂/花屏)、avsync
│   ├── channels/             # 串口 + ADB 通道（真实/mock 双实现）
│   ├── evaluate/             # manifest 真值解析 + 帧级匹配记分卡
│   └── ...                   # probe 归因信号 / archive 归档 / report HTML / inject 花屏生成
├── tests/                    # pytest（23 用例：检测器行为 + 归档 + 匹配 + L3 链路）
├── docs/                     # 设计方案 + 数据集生成文档
└── input/                    # 故障数据集（manifest 真值全量 + 抽样帧 + 音画 mp4，见 docs/DATASETS.md）
```

## 快速开始

```bash
# 环境（已有 .venv 可跳过）
uv venv .venv && uv pip install --python .venv/Scripts/python.exe -r requirements.txt

# 生成花屏合成数据集（首次）
.venv/Scripts/python.exe -m tvqa inject-corruption

# 全量评测：6 类视觉故障 + 音画 L1/L3，产出归档 + HTML 报告
.venv/Scripts/python.exe -m tvqa eval --dataset input --profile eval

# 只跑某个数据集
.venv/Scripts/python.exe -m tvqa eval --dataset input --only 黑屏

# 重新生成某个会话的 HTML 报告
.venv/Scripts/python.exe -m tvqa report --session output/session_bjtime_xxx

# 现场模式（需采集卡/串口/ADB 硬件或 mock 后端）
.venv/Scripts/python.exe -m tvqa run --profile field
```

## 硬件切换（零代码改动）

全部在 `config/default.yaml` 改 backend 与设备标识即可：

| 通道 | 当前（无硬件） | 硬件到位后 |
|---|---|---|
| 视频 | `video.backend: frames_dir` / `video_file` | `video.backend: capture_card`，`index` 改采集卡编号（0/1/2/3） |
| 音频 | `audio.backend: audio_file` | `audio.backend: sounddevice`，`device_name` 填输入设备名 |
| 串口 | `serial.backend: mock` | `serial.backend: pyserial`，`port`/`baudrate` 按机型配置 |
| ADB | `adb.backend: mock` | `adb.backend: real`，`device` 填序列号（可空） |

命令行临时覆盖：`--set video.backend=capture_card --set video.index=1`

## 归档与追溯

每次运行产出 `output/session_bjtime_*/`：

- `run_meta.json`：时间、主机、代码版本、**完整配置快照**（复现实验用）
- `events.jsonl`：事件流，每条含 `event_id` / 帧区间 / metrics / `log_line`（对应日志行）
- `<type>_events_summary.csv`：分类事件汇总（表头带 BOM，追加无 BOM）
- `evidence/<数据集>/<事件目录>/`：代表帧截图（start/mid/end）+ `event.json` + 音视频/系统日志切片
- `logs/tvqa.log`：北京时间毫秒统一日志
- `scorecard.json` + `index.html`：记分卡与可点击回溯的 HTML 报告

追溯链：**报告行 → event_id → evidence 目录（event.json+截图）→ logs/tvqa.log 行号**。

## 检测器一览

| 检测器 | 故障 | 方法 |
|---|---|---|
| `luma` | 黑屏/白屏/闪烁 | 逐帧亮度：镜像极性状态机（候选→确认→闭合）；闪烁=滑窗亮度跳变率 |
| `stutter` | 卡顿/冻结/丢帧 | 原脚本状态机移植（运动前置 arming→重复帧触发→静默恢复→结束时间反推）|
| `corruption` | 撕裂/局部冻结/块效应 | 行带位移差（撕裂）；块网格冻结矛盾图（局部花屏）；网格梯度比（块效应）|
| `avsync` | 音画不同步 | L1 被动（切点×onset 直方图投票）；L3 主动（闪光+蜂鸣测试片，误差 ≤0.4 帧）|
| `probe`+`correlate` | 归因 | 串口/ADB 系统信号与图像事件同窗关联（HDMI 断链/渲染 jank/解码错误/underrun）|

## 阈值调整速查（全部走配置，代码零改动）

**改配置文件**（`config/profiles/eval.yaml` / `field.yaml`，按档位长期生效）或**命令行临时覆盖**（单次生效，可叠加）：

```bash
# 例：黑屏确认时长从 5 帧放宽到 30 帧（1 秒）
python -m tvqa eval --only 黑屏 --set luma.black_confirm_frames=30
```

| 想调什么 | 配置键（所在档位文件） | 默认 |
|---|---|---|
| 黑屏确认时长 | `luma.black_confirm_frames` | eval 4 / field 150 |
| 黑屏恢复判定 | `luma.black_end_tolerance_frames` | 2 / 6 |
| 白屏确认/恢复 | `luma.white_*` | 同黑屏 |
| 闪烁跳变幅度/频率阈值 | `luma.flicker_min_amplitude` / `luma.flicker_min_rate` | 10 / 3.0 |
| 卡顿触发时长 | `stutter.duplicate_trigger_frames` | 4 / 8 |
| 卡顿恢复静默 | `stutter.recovery_quiet_frames` | 45 / 300 |
| 运动前置/武装超时 | `stutter.required_motion_frames` / `motion_arm_timeout_frames` | 6/45、15/180 |
| 撕裂位移差阈值/事件最短帧数 | `corruption.tear_min_shift_diff` / `tear_min_event_frames` | 6 / 3 |
| 局部冻结块数/持续 | `corruption.frozen_min_blocks` / `frozen_min_event_frames` | 10 / 6 |
| 音画事件搜索窗 | `avsync.search_window_ms` | 1500 |

**评测对账口径**（`config/default.yaml` 的 `evaluate` 节）：

| 想调什么 | 配置键 | 默认 |
|---|---|---|
| 事件与真值差多少帧内算命中 | `evaluate.match_tolerance_frames`（30≈1 秒） | 10 |
| 最小重叠度 | `evaluate.match_min_iou` | 0.05 |
| 过滤多少帧以下的噪声事件 | `evaluate.ignore_events_below_frames` | 0 |

## 关于 input/ 数据集

仓库内 `input/` 只包含 **manifest 真值（全量）+ 每类数据集的抽样帧 + 音画测试 mp4**；
完整帧序列约 2.9GB 不入库。每个数据集的注入规则（黑/白/闪/卡顿/撕裂/花屏/音画的生成参数与通用布局规则）
在 [docs/DATASETS.md](docs/DATASETS.md) 中有完整说明，可按文档本地还原；花屏类型内置生成器：

```bash
python -m tvqa inject-corruption --base input/frames --out input/花屏
```

抽样帧保留原始命名，`frames_dir` 后端可直接读入跑通「检测 → 匹配 → 记分」全链路。

> **数据来源声明**：数据集使用的原视频来源于哔哩哔哩
> [BV1kp4R6FEHL](https://www.bilibili.com/video/BV1kp4R6FEHL/)，版权归原作者所有，仅用于本项目的功能测试与演示，不作任何商业用途。

## License

[MIT](LICENSE) © 2026 MerlinShieh

## 已知边界

- 静止场景的像素级冻结与真卡顿不可分（检测器用"运动新鲜度封顶"抑制跨度，最终归因靠系统信号）。
- 纯 BGM 内容无语义声画锚点，L1 被动测音画偏移不可测（报告会标低置信）；测量链路由 L3 验证。
- 30fps 采集只能分辨 ≲10Hz 的可见闪烁；背光 PWM 类不可见闪烁需另案（光电二极管）。
- 串口 console 协议各机型不一（登录/prompt/波特率），真实通道接入时按机型固化配置。
