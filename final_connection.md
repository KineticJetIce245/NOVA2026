# final_connection.md — 从现有 EEG 流式管线到「听觉注意」现场演示

> 状态：**v1.1（设计已锁定，可开工）**。本文件是唯一目标锚点。
> 每个步骤实施前先读本文件；实施后把「状态」列改成 `DONE` 并附证据路径。
> 任何偏离本文件的改动，先改本文件。

---

## 0. 一句话目标

让一个人同时听两段语音（候选 A / 候选 B），系统用他的 EEG 实时判断**他正在注意哪一段**，
把结果显示在 [attune-ui](https://github.com/yut31/attune-ui) 界面上，并按判断对两路音量做衰减（**只衰减，不放大**）。

最终验收方式：**用已有数据集，以尽可能接近实机的方式，完整跑通一次 demo。**

---

## 1. 现状（`audio_v2` @ `9fb82d8`）

| 已有 | 位置 | 状态 |
| --- | --- | --- |
| 实时 EEG 流式内核（含 `TimeBase`、`TaskOffloader`） | `src/nova2026/streaming/` | 稳定、有测试 |
| 听觉注意解码（包络 + Ridge + 控制器 + 混音 + `TimestampedAudio`） | `src/nova2026/auditory/` | 合成数据端到端跑通 |
| CLI（train/replay/live/evaluate/convert） | `scripts/auditory/` | 可用 |
| 设备接入（eego / LSL / 录制回放 / CNT 对比） | `scripts/getlive/` | 可用（硬件未实测） |
| KU Leuven AAD 数据集（16 受试 × 20 trial × 64ch@128Hz + 两路语音 + 标签） | `datasets/AAD-KULeuven/` | 已在，未转换、未训练 |
| 全量测试入口 | `scripts/run_tests.py` | 已存在 |

**环境**（步骤 1 实测冻结，见 `results/test_baseline_20260913-005823.txt`）：
Python 3.13.15 / numpy 2.5.2 / scipy 1.18.0 / mne 1.12.1 / torch 2.13.0+cu132（CUDA 可用）/
scikit-learn 1.9.0 / pandas 3.0.5 / matplotlib 3.11.1；Node v26.7.0；
缺 `fastapi`/`uvicorn`/`websockets`/`httpx`；`sounddevice` 不装。
**基线测试**：5 suites / 523 tests / 0 skipped / 92.1s，exit 0，全绿。

### 已实测的四条硬事实

1. `S1.mat` 的 `trials` 是 20 个 struct，EEG `49792×64 @128Hz`；`stimuli` 字段是 **hrtf** 文件名，
   而官方 `preprocess_data.m` 取包络用的是 **dry** 文件。
2. 现有 `load_kuleuven` 的 `rglob(name)` 要求唯一匹配 —— 同 part 重复试次会让同一文件名出现 2~3 次
   （`rep_part1_track1_hrtf.wav` 出现 3 次），朴素转换会抛错。
3. 数据集**不含通道位置文件**；`mne.channels.make_standard_montage('biosemi64')` 第 48 个（1-based）正是
   `Cz`，与官方 `rereference='Cz'` 一致 → 通道命名口径据此钉死，并在 `metadata.json` 写明依据。
4. attune-ui 是团队成员在另一仓库的子模块，**无需许可**；只读克隆在 `C:\Files\git\_research\attune-ui`。
5. **音频素材实测**（主 agent 侦察，`datasets/AAD-KULeuven/stimuli/*_dry.wav`）：
   **44 100 Hz / 单声道 / int16**；完整段约 **394.0–395.3 s**，`rep_` 段 **125.0 s**；
   peak 约 3297–4886（int16 满量程的 10–15%，故归一化后幅度 ≈0.10–0.15）。
   16 个 dry + 16 个 hrtf。**包络体积可忽略**：单个候选中位约 389 s × 64 Hz ≈ 2.5 万点 float32 ≈ 100 KB。
6. **素材比 EEG 长**：`S1` trial 1 的 EEG 是 49 792 点 @128 Hz = **389.0 s**，而对应素材 394.0 s。
   音频**必须按 EEG 长度截断**（与数据集 README 的建议一致），否则 `AuditoryTrial` 的两列长度会超出可对齐范围。
7. **回放加速**：完整 trial 是 6.5 分钟，1× 回放对开发迭代太慢。步骤 10.5 的 `demorun` 需要
   `--clip <秒>`（默认取一段短切片），演示用完整段。切片只影响播放长度，不影响对齐全链路。
8. EEG 侧采样率 128 Hz、模型侧 64 Hz，链内重采样比正好为 2。

---

## 2. 架构

```text
[源 A] KU Leuven trial（EEG 128Hz + 两路语音 + 标签）   ← 保底，无设备可跑
[源 B] eego 放大器 / LSL 实时 EEG                        ← 最终目标，同一条链路
                     |
                     v
   acquisition 线程 → AuditoryProcessor.feed → EEGWindow
                    → TimestampedAudio.align（参考包络，来自 datasets/audio）
                    → TaskOffloader.submit  ──→ worker: RidgeDecoder.score
                    → LatestEstimate（单槽覆盖）
                     |
        AttentionSession（无 FastAPI 依赖，回放/真人/FastAPI 共用）
                     |
        AttentionProducer → FastAPI 传输层（REST /api/* + WS /ws/live，包协议 v1）
                     |
        attune-ui frontend（React 19 + Vite，原样移植；Web Audio 双路衰减）
```

### 2.1 铁律

1. 决策只在 Python 侧产生；前端不重算、不平滑、不推断任何测量值。
2. **失败显式**：无效窗 / 证据断裂 / 同步超阈 → `uncertain` 或 `unavailable`，绝不假装有结论。
3. 标签只用于事后评分，**绝不进入解码路径**。
4. 同源单端口 loopback，不开 CORS。
5. 新模块必须带测试与文档，沿用仓库既有风格（`VALIDATION.md` / 模块 README）。
6. 会话启动时**选定音频必须已有对应包络**，缺包络 → 启动即失败，绝不在运行期偷偷现算。

### 2.2 时间权威与参考系（关键设计）

| 时钟 | 对什么有权威 | 谁拥有 |
| --- | --- | --- |
| EEG 采样时钟（LSL） | 「这块电压何时被采到」 | 放大器；回放模式为确定性虚拟时钟 |
| 浏览器音频时钟 | 「音频播到哪、何时真的发声」 | `AudioContext.getOutputTimestamp()` |
| 会话相对时钟 | 仅决策的顺序与新鲜度 | `AttentionEstimate` |

- **参考系 = EEG 时钟。** 理由：EEG 不可再生且是相关运算的独立变量；解码器 lag 建模在 EEG 空间；
  `TimeBase` / `TimestampedAudio.audible_at` 已按此实现。
- **播放位置的定义权在浏览器，不在后端。** 后端对 `media_time_s` / `media_revision` 只有**抄写权，没有发明权**
  （前端 `mediaAudio.js:19-23` 会逐字段校验，不一致即回中性增益）。
- 两层之间靠**实测量** `audio_start_lsl` 相连：`LSL 时刻 = audio_start_lsl + media_time_s`。
  回放模式该偏移**构造保证为 0**，并在测试中断言。

### 2.3 同步守卫（发布每帧前逐条检查，任一不满足 → `uncertain` + 中性增益）

1. 帧的 `media_time_s` 在浏览器上报窗口内，且 `|Δt| ≤ 0.75 s`；
2. 参考包络与播放位置同源同锚点（同一 `media_revision`）；
3. 证据连续（无 `evidence_gap`）；
4. 时钟拟合残差在阈值内（`sync.status == "observed"`）。

---

## 3. 已锁定的设计决定

### 3.1 数据与包络（B 组，你的新增要求）

| 项 | 决定 |
| --- | --- |
| 包络存放 | **`datasets/audio/`**（无设备也能拿到包络；体积小，`.npz` 走 git；原始音频仍在 `datasets/AAD-KULeuven/stimuli/`） |
| 计算脚本 | `python -B -m scripts.auditory.envelopes --audio <file...> --out datasets/audio [--force]` |
| 库位置 | `src/nova2026/auditory/envelopes.py` 增加**离线全量**入口（现有 `EnvelopeExtractor` 保持不变） |
| 文件格式 | `<audio_stem>.npz`：`envelope`（float32，N×1 @64Hz）、`timestamps`、`metadata`(JSON) |
| 元数据必须含 | 源文件路径 + **源 SHA256** + `audio_rate` + `sample_rate` + `band` + `envelope_method` + 生成时间 + 脚本版本 |
| 缺失即失败 | 会话启动时校验：包络存在 + `source_sha256` 与磁盘文件一致 + 参数与会话 `AuditoryConfig` 一致；任一不符**启动失败**，不允许运行期现算 |
| 通道约定 | 单声道；立体声先按现有 `read_audio` 语义下混；>2 通道**拒绝**（不静默合并） |
| 必备测试 | **分块喂入 ↔ 整段喂入产出一致**（若不一致，先修 `EnvelopeExtractor`，因为它会改变特征契约 → 必须在训练前修完） |

### 3.2 线程与进程（A 组）

| 项 | 决定 |
| --- | --- |
| 服务形态 | **单 FastAPI 进程**同源提供 `/api/*` + `/ws/live` + `dist/` 静态文件 |
| 线程 | acquisition（主循环）／offload worker **×1**／audio producer（仅真人模式）／uvicorn 事件循环（只推送，不计算） |
| Offloader | **保留**：workers=1，capacity=2，overflow=`drop_oldest`；丢弃必须变成显式 `evidence_gap`，控制器要求证据连续 |
| 包络 | 会话启动前**全量加载自 `datasets/audio/`**；运行期只做插值，采集线程零重活 |
| 共享核心 | `AttentionSession` 抽到 `src/nova2026/auditory/session.py`，回放/真人/FastAPI 共用同一份时序逻辑 |

### 3.3 参考与播放（B 组）

- 参考 = **原始 wav 的预计算包络**（干净、可复现）；另做一次回环录制**仅用于校验**，不参与解码。
- 立体声：**左声道 = 候选 A，右声道 = 候选 B**；现场**必须用耳机**（已确认有耳机）。
  扬声器串音不在本系统处理范围内，写进 `VALIDATION.md`。
- 不额外加后端总电平层；前端 0 dB 时用素材原始电平。
- 拼接好的立体声 WAV 落 `output/`，按 trial 缓存并 git-ignore。

### 3.4 同步与降级（C 组）

| 项 | 决定 |
| --- | --- |
| 上报通道 | **复用 `POST /api/media/control`**（前端已实现，不改契约） |
| 时钟拟合 | 最近 **60 s** 滑动窗口的仿射拟合（offset + drift），旧样本衰减 |
| 卡顿判定 | `report.sequence` 不连续 → 记事件并**拒绝该窗口的增益**；`acquire.max_lag`/`gaps` 超阈 → 整帧降级 |
| 降级粒度 | **整帧降级**（`decision=uncertain` + 中性增益），阈值参数化 |

### 3.5 控制器与证据（D 组）

- 发布字段：`correlation_a/b`（原始分数，**绝不平滑**）+ `decision ∈ {A,B,uncertain,unavailable}`（字段名必须精确，前端按 `decision` 优先）。
- 起始衰减 **6 dB**，参数化；改增益须重跑评估（不改解码）。
- 保留 `min_switch_windows=3`、`max_age=3.0`。
- 无证据 → 两路均 0 dB；**绝不在不确定时随便衰减一路**。
- 会话开始前 2 s 发 `decision=unavailable`（预热），不发 `uncertain`。

### 3.6 前端与协议（E 组）

- 移植 `frontend/` 到 **`apps/attune-ui/`**，记录来源 commit；依赖版本原样锁死（React 19.2.0 / Vite 7.3.1）。
- 必要偏离仅两处，均记录在移植说明：①测试改用 `ATTUNE_PYTHON` 指向 `.venv\Scripts\python.exe`；
  ②**允许显示层小改**——把 `state.rejected`（协议违规计数）放在显眼处，让"静默丢弃"变可见。不改协议、不改解码。
- 文案：保留 `RESEARCH PROTOTYPE · DEVELOPMENT ONLY` 与 `simulated` 横幅逻辑；真实数据**绝不**带 `simulated`。
- 一键入口默认用 **build 后的 `dist/`**；Vite dev server 保留为开发命令。
- Node v26.7.0 需实测 `npm ci + build + test`；若 26 不兼容则记录并要求 24 LTS。

### 3.7 依赖（F 组）

- `fastapi` / `uvicorn` / `websockets` / `httpx` 进 `pyproject.toml` 的 `[project.optional-dependencies] transport`，
  版本锁到 attune-ui 验证过的组合，**装前按 `secure-import` 走审批**。
- **不装 `sounddevice`**：浏览器 Web Audio 是唯一播放路径。

### 3.8 记录与评估（G 组）

- 原始运行 → `records/`；评估报告 → `results/`。
- 每场 demo 自动产出四项：①窗口级准确率 + 覆盖率 + 误切换；②同步质量（块误差 / 漂移 ppm / 拟合残差）；
  ③拒帧统计（质量 / 同步 / 证据断裂）；④留出故事与留一被试两种口径。
- `VALIDATION.md` 明确**不声称**：听力改善、理解力提升、真人被试有效（除非真做）、跨被试泛化（除非数字在）、
  声学延迟已补偿（只能报测量值）。

### 3.9 兜底与交付线（H 组）

| # | 决定 |
| --- | --- |
| H1 | **无论如何都做 demo**；若留一被试接近随机，UI 顶部横幅写明，主展示用留出故事口径 |
| H2 | **提供一个无设备 `demorun` 演示脚本**，用真实 KU Leuven 数据以 1× 实时回放跑完整链路（含真实前端、真实音频、真实传输层） |
| H3 | 允许剧本化（指定 trial / 指定切换时刻），但脚本必须可见，不得伪装成实时被试结果 |
| H4 | `AttentionSession` 抽为 `src/nova2026/auditory/session.py` |
| H5 | 前端放 `apps/attune-ui/` |
| H6 | attune-ui 是团队子模块，无需许可；仍保留来源 commit 与 README |

### 3.10 最坏情况下必须实测的三个量

| 量 | 怎么得到 |
| --- | --- |
| 播放链路到 EEG 的绝对偏移 | 回环测量（音频分流进 EEG 输入或扬声器+麦克风），写入 `profile` |
| 解码器对音频偏移的容差 | **偏移扫描实验（步骤 5.5）**：包络滑 ±300 ms，看相关系数衰减 → 决定窗长/margin 是否可信 |
| 浏览器输出延迟 | `AudioContext.outputLatency` / `baseLatency`，记入 `profile` |

---

## 4. 最终验收（本计划的完成定义）

**用已有数据集、以尽可能真实模拟实机的方式，完整跑一次 demo。** 具体判据：

| # | 判据 | 证据 |
| --- | --- | --- |
| V1 | 无设备 `demorun` 一条命令跑通：FastAPI 起服务 + 提供 `dist/` + 1× 回放真实 KU Leuven trial | 完整运行日志（exit 0） |
| V2 | 前端真实渲染：`attention`/`gain`/`eeg_display`/`session` 包被解码显示，`rejected` 计数为 0 | 截图 + 诊断面板 JSON |
| V3 | 增益真的激活：`mediaFocusReady` 全条件满足，`attune` 模式观察到每路 dB 变化 | 前端状态导出的时间序列 |
| V4 | 决策与真值可比：窗口级准确率 + 覆盖率 + 误切换写入 `results/` | 指标 JSON |
| V5 | 同步质量可查：块时间误差、漂移 ppm、拟合残差 | `timing.json` / 报告 |
| V6 | 离线回归全绿：`scripts/run_tests.py` + 前端 `npm test` | 测试输出 |
| V7 | `VALIDATION.md` 写明声称/不声称/失败模式 | 文档 |

**注意**：V1–V5 是**无设备**路径，用真实数据 + 真实链路 + 真实前端达成；真人硬件路径（步骤 11）只交付
「代码与校准流程就绪 + 实测记录（硬件在场时）」，硬件不在场则在 `VALIDATION.md` 标注未验证。

---

## 5. 步骤计划（每步 = 一个子 agent 任务）

约定：子 agent 必须给出**可复现命令 + 真实输出**；不得越界改本步骤之外的模块；跨模块需求先回报。
每个子 agent 开工前必须加载 §6.1 列出的 skills，并在交付里确认已加载。预算见 §6.2。

| # | 步骤 | 交付物 | 证据 | 预算（步/分钟） | 状态 |
| --- | --- | --- | --- | --- | --- |
| 1 | **基线固化** | `results/test_baseline_<date>.txt` + 依赖清单 | `python -B scripts/run_tests.py` 全绿 | 14 / 15 | **DONE** — `results/test_baseline_20260913-005823.txt`：523 tests 全绿，commit `21a637e` |
| 2 | **KU Leuven 转换** | `metadata.json` + `converted/S*/trial_*.npz`；解决 hrtf/dry 与重复文件名 | 转换 exit 0；抽查形状/时长/对齐 | 45 / 50 | TODO |
| 3 | **包络预计算**（B1） | `scripts/auditory/envelopes.py` + `src` 离线入口 + `datasets/audio/*.npz` + 一致性测试 | 生成全部 16 个 KU Leuven 素材包络；分块↔整段一致 | 45 / 50 | TODO |
| 4 | **数据体检 + 契约冻结** | `results/kuleuven_audit.md`；冻结 `AuditoryConfig` 与特征契约 | 审计脚本 + 报告 | 35 / 40 | TODO |
| 5 | **解码器训练与评估** | `models/auditory_kuleuven.npz` + `results/aad_<date>.json`（留出故事 **+** 留一被试 + 窗长曲线） | 训练命令 + 指标 JSON | 35 / 75 | TODO |
| 5.5 | **偏移扫描实验** | `results/aad_shift_sweep_<date>.json`：包络滑 ±300 ms 的相关衰减曲线 | 扫描脚本 + 曲线 | 30 / 40 | TODO |
| 6 | **传输层移植**（须读 `secure-web-dev`） | `src/nova2026/transport/{protocol,publisher,sessions,server}.py` + 契约测试 | 单测全绿：包校验/快照/1013/生命周期 | 50 / 60 | TODO |
| 7 | **前端移植** | `apps/attune-ui/`（来源 commit 记录）+ `npm ci/build/test` 通过 | 构建产物 + 测试输出 | 35 / 60 | TODO |
| 8 | **`AttentionSession` + 生产者** | `src/nova2026/auditory/session.py` + `AttentionProducer`；合成 trial 先打通 | 端到端日志 + 前端截图 | 50 / 60 | TODO |
| 9 | **媒体时间轴与音频** | 立体声 WAV（L=A,R=B）经 `/api/media/file`；`media/control` 握手 + 250 ms `report` | `|Δt| ≤ 0.75 s` 证据 + 增益激活证据 | 50 / 60 | TODO |
| 10 | **真实 trial 全链路 + 一键入口** | `python -B -m scripts.auditory_ui.demo`：起服务 + 1× 回放 + 开浏览器 | V1–V5 全部证据 | 50 / 70 | TODO |
| 10.5 | **无设备 demorun**（H2） | `python -B -m scripts.auditory_ui.demorun`：无人值守跑完整场并出报告 | 运行日志 + `results/` 报告 | 30 / 45 | TODO |
| 11 | **真人实时模式** | eego/LSL 接同一 `AttentionSession`；校准流程（含回环测量） | 硬件实测记录（在场时） | 45 / 75 | TODO |
| 12 | **评估收口 + 文档** | `VALIDATION.md`、`documents/auditory_ui_protocol.md`、根 `README.md` | 文档 + 评估命令 | 35 / 50 | TODO |

### 依赖关系

```text
1 → 2 → 3 → 4 → 5 → 5.5 ─┐
                         ├→ 8 → 9 → 10 → 10.5 → 12
        6 → 7 ───────────┘              └→ 11（硬件在场）
```

---

## 6. 子 agent 执行协议

派发提示词必须包含：

1. 本文件路径 + 步骤编号 + 该步骤「交付物 / 证据 / 预算」原文。
2. 仓库约定：可复用代码进 `src/`，实验/脚本进 `scripts/`（按主题分子目录），raw 数据不改，文档同步更新，引用进 `refs.bib`。
3. 红线：不改原始数据集；标签不进解码路径；不静默吞异常；不用 pickle；不新增未审批依赖。
4. 触发规程（见 §6.1）：装依赖前读 `secure-import`；写网络/服务代码前读 `secure-web-dev`；破坏性操作前读 `secure-action`；阶段完成读 `structure-dev`。
5. 交付格式：改动文件清单 + **实际执行过的命令 + 真实输出** + 未解决项 + 是否偏离本文件。
6. 自查：本步骤是否让「判断他在听 A 还是 B」更接近一步？

### 6.1 强制 skill 加载

子 agent 开工前**必须**调用 `skill` 工具加载下列 skills，并在交付里报告「已加载 XX」。主 agent 在验收时核对；未加载即视为该步未完成。

| 步骤 | 必须加载 | 原因 |
| --- | --- | --- |
| 1 | — | 纯执行 |
| 2, 3, 4, 5, 5.5 | `structure-dev`（阶段完成时） | 代码/测试/审查规范 |
| 6, 8, 9, 10, 10.5, 11 | **`secure-web-dev`** + `structure-dev` | 涉及 HTTP 服务、WebSocket、LSL/网络通信 |
| 7 | `secure-import` + `structure-dev` | 会执行 `npm ci` 安装第三方依赖 |
| 任何 `pip install` / `uv add` | **`secure-import`**（装之前） | 依赖安装审批流程 |
| 任何删除/覆盖已有产物 | **`secure-action`**（执行之前） | 破坏性操作审批 |

### 6.2 子 agent 预算与防死循环（主 agent 强制执行）

**事实前提**：本 harness 没有给子 agent 设置步数/上下文 token 上限的配置项，也没有 token 用量查询接口。可执行的抓手只有：`interrupt_agent`、`send_message`、`list_agents`，以及超时杀进程。
因此预算以**可核查的代理指标**执行：**工具调用步数**（子 agent 自己报，主 agent 用工作区产物核对）+ **墙钟时间**（主 agent 对进程计时）。

**三条硬规则**

1. **预算**：每步的「步数 / 分钟」上限见 §5 表格。到 80% 时子 agent 必须停止扩张范围，只做收尾。
2. **回滚闸**：同一条失败命令或同一个失败测试**最多重试 2 次**。第 3 次之前必须停止并回报「我卡在 X，试过 Y/Z，假设 H 可能是错的」。禁止「改测试让它过」——改测试必须单独说明理由并提供独立证据。
3. **越界即熔断**：超预算 → 主 agent 立即 `interrupt_agent`（附原因），记录「超预算熔断」，并**派新子 agent**，新 agent 的提示词必须写明：
   - 前一个 agent 在**第 N 步 / 第 M 分钟**被熔断；
   - **实现方式很可能有问题**，并给出主 agent 从会话记录里读到的**具体诊断**（例如：反复修改同一个测试、同一命令重复执行、契约理解错误、环境假设错误）；
   - **不要从断点继续硬接**：先独立核对工作区现状（`git status`、已有产物、失败测试的真实输出），再提出不同的实现路径。
4. **限额升级**：若**连续 2 个**子 agent 都在**明显正常推进**（每步都有新产物、无重复命令）的情况下撞上限，把该步上限提高 50%（步数与分钟同时），并把升级理由、新上限写进本文件 §7 变更记录。**正常推进 ≠ 时间花得多**：判据是新产物 + 命令多样性。
5. **熔断不是失败**：允许同一步最多 2 次熔断；第 3 次熔断说明**步骤划分本身有问题**，此时必须停下向用户报告，而不是继续派第 4 个 agent。

**主 agent 的核查动作**（每步固定执行）

1. 派发前登记：步骤号、预算、预登记命令（见 §6.3）。
2. 子 agent 结束后：`git status --short` + 产物存在性 + 抽样读文件，确认不是"报告完成但没落地"。
3. 亲自执行该步的证据命令；核对输出与子 agent 声称一致。
4. 写入本文件 §5 状态列：`DONE` / `BLOCKED(原因)` / `BUDGET-STOP(第几次)`。

**验收陷阱（步骤 1 实测发现，后续步骤沿用）**：`git check-ignore -v <path>` 在「命中 `.gitignore` 的 `!` 取反规则」与「真的被忽略」两种情况下**都返回退出码 0**，因此不能单独作为"该文件可被提交"的判据。判据改用 `git add --dry-run <path>`（可提交 exit 0，被忽略 exit 1）+ `git status --short <path>`（被忽略的文件根本不出现）。

### 6.3 预登记命令与预期（防"试到过为止"）

每一步在派发前先在本表登记「我要跑什么、预期看到什么」。实际输出与预期不符时，子 agent 必须**停下来判断**，不允许连续盲试。

| 步骤 | 预登记命令 | 预期 |
| --- | --- | --- |
| 1 | `python -B scripts/run_tests.py` | 全绿；记录通过数与耗时 |
| 2 | `python -B -m scripts.auditory.convert --kind kuleuven ...` | exit 0；`trial_*.npz` 数量 = trial 数；`eeg.shape[0] == len(timestamps) == len(labels)` |
| 3 | `python -B -m scripts.auditory.envelopes --audio <16 文件>` | 生成 16 个 `.npz`；分块↔整段一致性测试通过 |
| 4 | 审计脚本 | 每受试 trial 数 = 20；时长落在合理区间；标签分布两种 |
| 5 | 训练 + 评估 | 留出故事准确率显著 > 50%；留一被试数字如实记录 |
| 5.5 | 偏移扫描 | 相关峰在 0 附近；±300 ms 内衰减曲线单调下降 |
| 6 | `python -B -m unittest discover -s tests/transport -v` | 全绿 |
| 7 | `npm --prefix apps/attune-ui ci && npm --prefix apps/attune-ui test` | CI 与测试通过；Node 26 若有问题如实记录 |
| 8 | `python -B -m scripts.auditory_ui.session --trial ...`（合成） | 产生合法包；前端显示 attention 卡片 |
| 9 | 同上 + 音频 | `|Δt| ≤ 0.75 s`；`attune` 模式观察到 dB 变化 |
| 10/10.5 | `python -B -m scripts.auditory_ui.demorun` | exit 0；`results/` 出现指标 JSON |

### 主 agent（我）的职责

1. 维护本文件为唯一真相；任何设计变更先改这里再改代码。
2. 逐步派发子 agent，每步验收证据（命令 + 真实输出），拒绝"声称完成"。
3. 守住铁律：标签不进解码、失败必须显式、后端不发明时间戳、缺包络即失败。
4. 跨步骤一致性：`AttentionSession` 只有一份时序逻辑；契约冻结后才训练。
5. 收尾产出**完整报告**：做了什么、证据在哪、验证了什么/没验证什么、各子 agent 分工与偏离。

---

## 7. 变更记录

| 日期 | 变更 | 作者 |
| --- | --- | --- |
| 2026-09-12 | 创建草案 v0.1（骨架 + 步骤表） | 主 agent |
| 2026-09-12 | v0.2：补 attune-ui 侦察结论、后端契约表、移植约束 | 主 agent |
| 2026-09-12 | v1.0：锁定 A–H 全部设计决定；新增 `datasets/audio` 包络要求；新增偏移扫描（5.5）与无设备 demorun（10.5）；写入最终验收 V1–V7 与主 agent 职责 | 主 agent |
| 2026-09-12 | v1.1：新增 §6.1 强制 skill 加载表、§6.2 子 agent 预算与防死循环（熔断、新 agent 接手、限额升级）、§6.3 预登记命令表；步骤表加入每步预算 | 主 agent |
| 2026-09-13 | v1.2：步骤 1 **DONE**（523 tests 全绿，commit `21a637e`）；按 §6.2 规则 4 将全部步骤预算重校准（步骤 1 实测 14 步，名义 6 步属于预算设定错误，非 agent 失控）；§6.2 增补 `git check-ignore` 的验收陷阱与替代判据 | 主 agent |
