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

### 已实测的硬事实

1. `S1.mat` 的 `trials` 是 20 个 struct，EEG `49792×64 @128Hz`；`stimuli` 字段是 **hrtf** 文件名，
   而官方 `preprocess_data.m` 取包络用的是 **dry** 文件。
   **→ 步骤 2 修正**：其中一半 trial 的 `stimuli` 本身就是 `_dry.wav`（part3/part4），所以映射必须**幂等**；
   `attended_track` 是 **int**（1/2）不是字符串。
2. ~~现有 `load_kuleuven` 的 `rglob(name)` 要求唯一匹配 → 同 part 重复试次会让同一文件名出现 2~3 次，朴素转换会抛错。~~
   **→ 步骤 2 证伪（主 agent 的原始判断是错的）**：`stimuli/` 是扁平目录、每个文件只出现一次，
   `rglob` 又是**按 trial 独立调用**的，所以旧代码**不会抛错**（实测 S1：`OK n = 20`，exit 0，5.4 s）。
   真正的缺陷是**静默的**：旧代码用 **hrtf** 渲染版算参考包络，与数据集作者 recipe（dry）矛盾——
   hrtf 里的头相关滤波正是解码要解释的东西，却混进了参考包络。这比崩溃更危险。
3. 数据集**不含通道位置文件**；`biosemi64` 第 48 个（1-based）正是 `Cz`，与官方 `rereference='Cz'` 一致
   → 通道命名口径据此钉死，并在 `metadata.json` 写明依据（已落地）。
4. attune-ui 是团队成员在另一仓库的子模块，**无需许可**；只读克隆在 `C:\Files\git\_research\attune-ui`。
5. **音频素材实测**：**44 100 Hz / 单声道 / int16**；完整段约 **394.0–395.3 s**，`rep_` 段 **125.0 s**；
   peak 约 3297–4886（int16 满量程的 10–15%，归一化后 ≈0.10–0.15）。包络体积可忽略（单个 ≈100 KB）。
6. **素材比 EEG 长**：每个 trial 都如此，差 **+1.0 s 到 +72.9 s**。必须按 EEG 长度截断音频
   （数据集 README 与 `preprocess_data.m` 都这么做）。**步骤 4 需冻结此口径**。
7. **`rep_part{N}` 与 `part{N}` 是同一段音频**（主 agent 实测）：`rep_part1_track1_dry.wav` 与
   `part1_track1_dry.wav` 的**前 125.00 s 逐样本完全相同**（max abs diff = 0；两者 SHA1 不同是因为长度不同）。
   → **故事泄漏风险确认存在**：留出故事/受试者的划分必须把 `rep_*` 与其本体视为**同一组**，
   否则同一段语音会同时出现在训练与验证里，准确率会虚高。这是步骤 5 的硬约束。
8. **回放加速**：完整 trial 6.5 分钟，1× 回放对开发迭代太慢 → `demorun` 需要 `--clip <秒>`。
9. EEG 侧 128 Hz、模型侧 64 Hz，链内重采样比正好为 2。
10. **KU Leuven 的物理单位是 µV**（峰值 ≈325.6），而现场 ANT 数据是**伏特**（峰值 ≈0.0833 V）。
    两者的 `source_unit_exponent` 必须参数化（见 §3.11 第 6 条）。
11. **KU Leuven 素材是 44 100 Hz**，而现场实验音频是 **48 000 Hz**（实测）。包络生成按每个文件的真实采样率处理，
    元数据记录 `source_rate`，因此两类素材不会互相污染；但任何假设"素材都是 44.1 kHz"的代码都会错。
12. **步骤 3 的规模**：`datasets/audio/` 现有 **18 个包络**（16 个 KU Leuven dry + `left_mono` + `right_mono`），
    合计约 2.1 MB（其中两个 900 s 的测试集包络各 ≈278 KB）。`--verify` 全部通过。
13. **标签在一个 trial 内恒定，且类别严重不平衡**（主 agent 实测 320 个 trial）：
    标签集合只有两种——`{0}` 出现在 **256** 个 trial，`{1}` 出现在 **64** 个 trial，**没有任何 trial 混合两种标签**。
    即 **80% / 20% 不平衡**。直接后果：
    - 窗口级"准确率"会被多数类拉高，**必须同时报每类准确率或平衡准确率**，否则数字没有意义（数据集 README 警告的正是这类虚高）。
    - 步骤 5 的留出划分必须**分层**：按故事分组后仍要保证验证集里有 class 1，否则报告的是一个只见过 class 0 的模型的成绩。
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

### 3.11 真人硬件的通道数与跨设备口径（2026-09-13 实测更新）

**实测参数（读自 `tmp/antneurodata/audio/*.cnt`，用 `mne.io.read_raw_ant`）**：

| 项 | 值 |
| --- | --- |
| 采样率 | **500 Hz** |
| 通道数 | **24 个 EEG**（无独立参考通道出现在 `.cnt` 里） |
| 通道名（列序） | `Fp1, Fp2, F9, F7, F3, Fz, F4, F8, F10, M1, T7, C3, C4, T8, M2, Cz, P7, P3, Pz, P4, P8, Oz, O1, O2` |
| 时长 | 会话 1 = 310.19 s；会话 2 = 328.63 s |
| 单位 | **伏特**（MNE 语义）：5 s 内峰值 ≈ 0.0833 V = 83333 µV，正是 24 位放大器的典型饱和量级（±83.9 mV） |
| 标记 | `.evt` 由 `read_raw_ant` 自动读入 annotations：`1004/Start`、`1001/Left side`、`1006/Custom Annotation`；会话 1 有 26 个、会话 2 有 28 个 |

**跨设备公共通道**：KU Leuven(64) ∩ 现场 cap(24) = **20 个**
`Fp1 Fp2 F7 F3 Fz F4 F8 T7 C3 C4 T8 Cz P7 P3 Pz P4 P8 Oz O1 O2`；
现场独有 4 个：`F9 F10 M1 M2`（前颞与乳突），KU Leuven 独有 44 个。
KU Leuven 列索引：`Fp1=0 Fp2=33 F7=6 F3=4 Fz=37 F4=39 F8=41 T7=14 C3=12 C4=49 T8=51 Cz=47 P7=22 P3=20 Pz=30 P4=57 P8=59 Oz=28 O1=26 O2=63`。

**这条为什么重要**：`AuditoryProcessor` / `prepare()` 会把源通道与模型契约通道逐一比对，
不一致就**拒绝启动**（`scripts/auditory/live.py:66-69`，`src/nova2026/streaming/preflight.py` 的 `ChannelContract`）。
用 64 通道训练的解码器**无法**直接接这台 24 通道设备。

**选定口径：公共通道子集（方案 A），外加保底的 64 通道路径**

1. **主口径**：解码器在 **20 个公共通道**上训练与评估（两个数据集都能提供），现场契约即这 20 个通道；现场多出的 `F9 F10 M1 M2` 不参与解码（`ChannelContract` 支持按名选择，多出的列被忽略）。
2. **为什么不用 KU Leuven 独有的 44 个**：现场拿不到这些电极，用它们训练等于在演示时喂常数。
3. **为什么不做空间投影（方案 B）**：现有 `spatial.py` 的算子需要电极坐标，而现场 cap 的布点文件尚未确认；且投影会引入近似误差，20 个同名同位的公共电极不存在这个问题。
4. **保底**：无设备 `demorun`（V1–V5）用 **64 通道**，保证与数据集原论文口径可比；真人模式用 20 通道契约。
5. **评估必须两种契约都报**：20 通道 vs 64 通道的准确率差，是"能不能上真人"的直接判据。
6. **单位口径**：现场数据是**伏特**（需 `source_unit_exponent=-6`），KU Leuven 是 **µV**（≈325 峰值，即 `0`）。
   两者**都不需要**在链内做数值换算，因为 `Repair` 的 `source_unit_exponent` 只用于端点安全检查、不改值
   （`src/nova2026/streaming/preprocess/repair.py:86-88, 164-172`）；但 `AuditoryProcessor` 目前把它**硬编码为 -6**
   （`src/nova2026/auditory/streaming.py:71`），对 KU Leuven 的 µV 数据是错的量纲假设。**步骤 5 之前必须参数化**，
   否则安全检查会误判（325 µV 被读成 3.25e8 µV，远超 75 000 µV 的饱和线）。

### 3.13 扰动用例矩阵（用户 2026-09-14 要求：**只有一项是"正常"，其余都必须带故障**）

最终演示验证必须跑**一整组**用例：1 项正常路径 + N 项注入故障。每一项都要有明确的
**预期行为**（失败必须显式，不得崩溃、不得静默给结论）、**可观测证据**与**判据**。
故障注入优先复用仓库既有能力：`scripts/auditory/evaluate.py:inject_fault`、
`nova2026.streaming.offload.TaskOffloader` 的 overflow 策略、`Recovery`、`Repair`、
`Acquire` 的 `max_lag`/`gaps` 计数、`TimestampedAudio` 的块误差/漂移诊断。

| # | 用例 | 注入方式 | 预期行为（必须显式） | 证据 |
| --- | --- | --- | --- | --- |
| C1 | **正常运行**（唯一"完美"项） | 无 | 会话跑完，增益激活，指标齐全 | 运行日志 + 指标 JSON |
| C2 | EEG 中途断流 | 中途停止上游 LSL 源 | 生产者停止并发布 `session.status=error`；UI 转 stale，增益回中性；**不挂起** | 事件时间线 + UI 状态 |
| C3 | 大 lag（数据迟到数秒） | 人为延迟上游时间戳 | 超阈时帧降级为 `uncertain`，增益中性；`acquire_max_lag_seconds` 记录实际值 | 帧序列 + timing.json |
| C4 | 时间戳跳跃/大 gap | 在流中插入秒级空洞 | 小 gap 按 `Repair` 合成并标 `interpolated`；超限则恢复或停止，**不得静默跳过** | 窗口 reasons + recovery 事件 |
| C5 | **EEG 数据重复** | 同一样本块重复送入 | 由 `Acquire`/`TimeBase` 检出并计数；不得把重复数据当新证据推进决策 | `gaps`/`max_lag` + 决策时间线 |
| C6 | EEG 含 NaN/Inf 段 | 注入非有限值 | `Repair` 修复并标 `interpolated`；超 `max_seconds` 抛 `UnrepairableError` → 恢复或停止 | `repaired_samples` + reasons |
| C7 | 坏通道 | 置某通道为死电极 | 按 `max_bad_channels`/`exclude_channels` 策略处理并**记录到 run record** | `bad_channel_windows` |
| C8 | UI↔后端断连 | 拔掉 WebSocket | 前端按既有语义暂停播放、增益中性、退避重连并取新快照；重连后不显示旧测量为当前 | 前端状态导出 + 重连日志 |
| C9 | 后端进程重启 | 杀进程后重启 | 前端 `sequence` 归零被接受（新快照），不因序号回退而卡死 | 重连日志 |
| C10 | 会话中途重启 | stop → start | 旧流被清空，新 `session_id`，增益不残留 | 前端状态 |
| C11 | 音频时钟漂移 | 人为偏移 `report` 时间 | 超 0.75 s 窗口即不激活增益（回中性）；拟合残差记录 | gain 时间序列 + 拟合残差 |
| C12 | 播放欠载/卡顿 | 故意拖慢 `report` | 记为事件；不把陈旧决策当作当前 | 事件计数 |
| C13 | 包协议违规 | 发送非法包（坏 version/深嵌套/超大） | 前端**拒绝并计数**，`rejected` 告警可见（步骤 7 的偏离 2） | `state.rejected` 渲染 |
| C14 | 缺包络启动 | 删除某素材的 `datasets/audio/*.npz` | **会话启动即失败**并指名缺哪个（§3.1 铁律 6） | 启动错误输出 |
| C15 | 候选长度不匹配 | 两路音频长度不等 | 启动即失败，不得静默截断 | 启动错误输出 |
| C16 | 标签越界/缺失 | 标注文件缺段 | 标为 `-1`（unknown）并计入覆盖率分母之外的统计 | 覆盖率报告 |

**判据的统一要求**：每个用例都要回答三个问题——①系统是否**崩溃**（不允许）？②失败是否**显式**
（必须出现在指标/状态里，不允许静默给结论）？③是否**污染了其它用例**（每个用例独立运行、独立记录）？
结果写入 `results/perturbation_<YYYYMMDD-HHMMSS>.json`，并作为 V9 的验收对象。

### 3.14 传入 EEG 数据的对抗性用例（用户特别要求）

除 C3–C6 外，导入器与流式链还必须对以下**输入形状**做测试（可用合成数据，不需硬件）：

| # | 输入 | 期望 |
| --- | --- | --- |
| E1 | 同一块数据重复送入 | 决策不因重复而更快提交（不产生虚假的连续证据） |
| E2 | 时间戳非单调 | 明确拒绝（`irregular_timestamps`），不得接受 |
| E3 | 采样率与声明不符 | 与模型契约比对后拒绝启动 |
| E4 | 声道数多于/少于契约 | 按名选择或拒绝；**绝不按列数截断** |
| E5 | 单位差 10 的幂（V 当 µV 用） | 由 `source_unit_exponent` 参数化后一致性检查能识别（§3.11 第 6 条） |
| E6 | 巨大 gap（> `max_seconds`） | 触发恢复或停止，记录 `large_gap` |
| E7 | 全零/饱和信号 | 质量判定拒绝窗口，增益保持中性 |

### 3.12 真实测试数据集（用户指定的最终 test dataset）

**位置**：`tmp/antneurodata/`。**这是用户实机录制的数据，最终验证以它为准。**

| 文件 | 作用 |
| --- | --- |
| `audio/*.cnt` + `*.evt` | 两个会话的 EEG（500 Hz / 24 通道，见 §3.11）与事件标记 |
| `audio_files/experiment/dichotic_15min.wav` | 双声道刺激：**左声道 = 左耳候选，右声道 = 右耳候选** |
| `audio_files/experiment/left_raw.wav` / `right_raw.wav` | 两路候选的原始单声道素材（包络来源） |
| `audio_files/experiment/left_cut.wav` / `right_cut.wav` / `left_mono.wav` / `right_mono.wav` | 用户已做的切片/单声道版本 |
| `audio_files/experiment/check_left.wav` / `check_right.wav` | 通道检查用 |
| `audio_files/experiment/experiment/…` | 注意：存在同名嵌套层，导入器必须显式定位，不得靠猜 |

**标记语义（用户 2026-09-13 明确确认，不再有歧义）**：

| 标记 | 含义 |
| --- | --- |
| `1001/Left side` | 提示参与者**注意左耳**（= 候选左声道） |
| `1006/Custom Annotation` | 提示参与者**注意右耳**（= 候选右声道）；两个会话各 12 个，与 `1001` 严格交替 |
| `1004/Start` | 会话/录音起点锚 |
| `1007/Saying-YES`、`impedance` | 非注意标记 |

**误触排除清单（用户 2026-09-13 确认"就这几个"）**——导入器必须按此显式排除，不得靠启发式：

| 会话 | EEG t | 标记 | 处理 |
| --- | --- | --- | --- |
| 会话 2 | 148.354 s | `1007/Saying-YES` | 排除 |
| 会话 2 | 148.402 s | `1006/Custom Annotation` | 排除（误触，与上一行相隔 0.05 s） |
| 会话 2 | 326.692 s | `1004/Start`（第二个） | 排除（停止录音时的误触） |

排除后每个会话的有效切换标记均为 **12 个**（`1001` 12 个、`1006` 12 个），切换间隔 8–17 s（均值 ≈12.6 s）。

**音频文件关系（主 agent 实测，逐位比对）**——**只有 `left_mono.wav`/`right_mono.wav` 是实际播出的信号**：

| 文件 | 采样率 | 声道 | 时长 | 与 `dichotic_15min.wav` 的关系 |
| --- | --- | --- | --- | --- |
| `dichotic_15min.wav` | 48 000 | 2 | 900.00 s | 参考本体；两声道互不相关（corr≈-0.0005），是真双耳分离 |
| `left_mono.wav` | 48 000 | 1 | 900.00 s | 与**左声道逐位相同**（max diff = 0） |
| `right_mono.wav` | 48 000 | 1 | 900.00 s | 与**右声道逐位相同** |
| `check_left.wav` / `check_right.wav` | 48 000 | 1 | 900.00 s | 分别与左右声道逐位相同（通道检查副本） |
| `left_cut.wav` | 48 000 | 1 | 900.00 s | corr=1.0 但**幅度被改过**（max diff = 10 308）→ 不可作为参考 |
| `right_cut.wav` | 48 000 | 1 | 900.00 s | 与右声道逐位相同 |
| `left_raw.wav` / `right_raw.wav` | 48 000 | 1 | 1130.43 / 1091.51 s | **比刺激更长、未播过的原始素材** → 不可作为参考 |

**→ 步骤 3 修正（主 agent 执行，2026-09-13）**：原先按本节旧稿把包络建在 `left_raw`/`right_raw` 上，
而那两段**没有播给参与者**；参考包络必须取自实际播出的声道。已删除 `datasets/audio/left_raw.npz` 与 `right_raw.npz`，
改为从 `left_mono.wav` / `right_mono.wav` 生成（各 57 600 samples @ 64 Hz = 900.0 s），
`--verify` 全 18 个包络通过（含逐文件 SHA256 重算）。

**时间轴（用户给定，必须严格执行）**：

- **会话 1**（`…19-34-06`）：音频从 **0 s** 开始播放；EEG 时长 310.19 s。
- **会话 2**（`…19-41-11`）：音频从 **4:27 = 267.0 s** 开始播放；EEG 时长 328.63 s。
- **会话内以 `1004/Start` 标记为锚**（会话 1 在 t=5.006 s，会话 2 在 t=12.838 s）——
  即 `音频位置 = 起点 + (EEG 时间 − Start 标记时间)`，不是直接用 EEG 时间。
  会话 2 因此覆盖刺激的 279.8–595.3 s；两个会话合计覆盖约 595 s / 900 s。

**标注口径（用户给定）**：

- 每次切换后取 **0.5–1 s 作为 buffer**：该区间标 `label = -1`（unknown）。
- 有效窗口持续到**下一次切换前 0.5 s**：该区间同样标 `-1`。
- 用户提示「有的时候可能会有很大的干扰」：这些段落在审计里必须可见（质量指标 + 拒帧统计），不得静默剔除。

**定位**：`tmp/antneurodata/` 是**测试集**（评估用），不是训练集——训练仍在 KU Leuven 上做。
`tmp/` 已被 git-ignore，所以导入器与标签清单必须能把结果写到 `datasets/`（派生物，不入库）
与 `results/`（评估 JSON，入库例外见 `.gitignore`）。

**新增依赖**：读取 `.cnt` 需要 `antio`（已装，`0.7.1`，MNE 核心团队维护，LGPL-3.0，win_amd64 wheel）；
`mne.io.read_raw_cnt` 是给 NeuroScan 的，读 ANT 会报 `Event table offset … larger than file size` 这类**误导性错误**，
必须用 `mne.io.read_raw_ant`。`antio` 需要进 `pyproject.toml` 的 `eeg-ant` 可选依赖组。

---

## 4. 最终验收（本计划的完成定义）

**用已有数据集、以尽可能真实模拟实机的方式，完整跑一次 demo。** 具体判据：
**"尽可能真实模拟实机" = 用用户实机录制的 ANT 数据（§3.12）作为测试输入，走真实流式链路、真实前端、真实音频增益通路。**

| # | 判据 | 证据 |
| --- | --- | --- |
| V1 | 无设备 `demorun` 一条命令跑通：FastAPI 起服务 + 提供 `dist/` + 1× 回放一段**真实 trial**（KU Leuven 与/或 ANT） | 完整运行日志（exit 0） |
| V2 | 前端真实渲染：`attention`/`gain`/`eeg_display`/`session` 包被解码显示，`rejected` 计数为 0 | 截图 + 诊断面板 JSON |
| V3 | 增益真的激活：`mediaFocusReady` 全条件满足，`attune` 模式观察到每路 dB 变化 | 前端状态导出的时间序列 |
| V4 | 决策与真值可比：窗口级准确率 + 覆盖率 + 误切换写入 `results/`；**ANT 测试集上必须报 20 通道契约的结果** | 指标 JSON |
| V5 | 同步质量可查：块时间误差、漂移 ppm、拟合残差 | `timing.json` / 报告 |
| V6 | 离线回归全绿：`scripts/run_tests.py` + 前端 `npm test` | 测试输出 |
| V7 | `VALIDATION.md` 写明声称/不声称/失败模式 | 文档 |
| V8 | **ANT 真实数据导入可核对**：标记表原样导出 + 误触清单 + 20 通道公共子集映射被断言 | 导入报告 JSON + 用户可核对的标记表 |
| V9 | **扰动矩阵跑通**（§3.13/§3.14）：**1 项正常 + 15 项注入故障**，每项都有显式结果，且没有任何一项以"崩溃"或"静默给结论"收场 | `results/perturbation_<date>.json` + 逐用例判据 |

**注意**：V1–V5、V8 是**无设备**路径，用真实数据 + 真实链路 + 真实前端达成；真人硬件路径（步骤 11）只交付
「代码与校准流程就绪 + 实测记录（硬件在场时）」，硬件不在场则在 `VALIDATION.md` 标注未验证。

---

## 5. 步骤计划（每步 = 一个子 agent 任务）

约定：子 agent 必须给出**可复现命令 + 真实输出**；不得越界改本步骤之外的模块；跨模块需求先回报。
每个子 agent 开工前必须加载 §6.1 列出的 skills，并在交付里确认已加载。预算见 §6.2。

| # | 步骤 | 交付物 | 证据 | 预算（步/分钟） | 状态 |
| --- | --- | --- | --- | --- | --- |
| 1 | **基线固化** | `results/test_baseline_<date>.txt` + 依赖清单 | `python -B scripts/run_tests.py` 全绿 | 14 / 15 | **DONE** — `results/test_baseline_20260913-005823.txt`：523 tests 全绿，commit `21a637e` |
| 2 | **KU Leuven 转换** | `metadata.json` + `converted/S*/trial_*.npz`；解决 hrtf/dry 与重复文件名 | 转换 exit 0；抽查形状/时长/对齐 | 45 / 50 | **DONE** — commit `3e0ba15`；320 trials / 15.15 GB，dry 映射逐位可验证；auditory 套件 53 tests 全绿。**证伪了计划原事实 2**（见 §1） |
| 3 | **包络预计算**（B1） | `scripts/auditory/envelopes.py` + `src` 离线入口 + `datasets/audio/*.npz` + 一致性测试 | 生成全部 16 个 KU Leuven 素材包络；分块↔整段一致 | 45 / 50 | **DONE** — commit `ab63ede`；离线=运行时用**精确相等**断言，分块↔整段 max diff = 0.0；18 个包络 `--verify` 通过。主 agent 修正了测试集包络来源（`*_mono` 而非 `*_raw`，见 §3.12）。**例外**：`src/nova2026/auditory/envelopes.py` 365 行，超 structure-dev 的 300 行建议，经主 agent 知情批准（拆分会把「离线=运行时」契约分散到两个文件） |
| 4 | **数据体检 + 契约冻结** | `results/kuleuven_audit.md`；冻结 `AuditoryConfig` 与特征契约 | 审计脚本 + 报告 | 35 / 40 | TODO |
| 5 | **解码器训练与评估** | `models/auditory_kuleuven.npz` + `results/aad_<date>.json`（留出故事 **+** 留一被试 + 窗长曲线） | 训练命令 + 指标 JSON | 35 / 75 | TODO |
| 5.5 | **偏移扫描实验** | `results/aad_shift_sweep_<date>.json`：包络滑 ±300 ms 的相关衰减曲线 | 扫描脚本 + 曲线 | 30 / 40 | TODO |
| 6 | **传输层移植**（须读 `secure-web-dev`） | `src/nova2026/transport/{protocol,publisher,sessions,server}.py` + 契约测试 | 单测全绿：包校验/快照/1013/生命周期 | 50 / 60 | TODO |
| 7 | **前端移植** | `apps/attune-ui/`（来源 commit 记录）+ `npm ci/build/test` 通过 | 构建产物 + 测试输出 | 35 / 60 | TODO |
| 8 | **`AttentionSession` + 生产者** | `src/nova2026/auditory/session.py` + `AttentionProducer`；合成 trial 先打通 | 端到端日志 + 前端截图 | 50 / 60 | TODO |
| 9 | **媒体时间轴与音频** | 立体声 WAV（L=A,R=B）经 `/api/media/file`；`media/control` 握手 + 250 ms `report` | `|Δt| ≤ 0.75 s` 证据 + 增益激活证据 | 50 / 60 | TODO |
| 9.5 | **ANT 真实数据集导入**（§3.12） | `scripts/auditory/antneuro.py`：读 `.cnt`（`read_raw_ant`）+ 会话音频起点（0 s / 267 s）+ 标记清单 + 用户标注口径（切换 ±buffer → `-1`） | 标记表导出供人工核对；每个会话的 trial 形状/时长/标签分布；误触清单显式记录 | 45 / 60 | TODO |
| 10 | **真实 trial 全链路 + 一键入口** | `python -B -m scripts.auditory_ui.demo`：起服务 + 1× 回放 + 开浏览器 | V1–V5 全部证据 | 50 / 70 | TODO |
| 10.5 | **无设备 demorun**（H2） | `python -B -m scripts.auditory_ui.demorun`：无人值守跑完整场并出报告 | 运行日志 + `results/` 报告 | 30 / 45 | TODO |
| 11 | **真人实时模式** | eego/LSL 接同一 `AttentionSession`；校准流程（含回环测量） | 硬件实测记录（在场时） | 45 / 75 | TODO |
| 12 | **评估收口 + 文档** | `VALIDATION.md`、`documents/auditory_ui_protocol.md`、根 `README.md` | 文档 + 评估命令 | 35 / 50 | TODO |

### 步骤 3–5 的口径修订（2026-09-13，依据 §3.11/§3.12 实测）

| 步骤 | 原口径 | 修订后 |
| --- | --- | --- |
| 3 | 对 KU Leuven 16 个 dry 素材算包络 | **不变**；另需对 §3.12 的 `left_raw.wav`/`right_raw.wav` 也算（测试集需要参考包络） |
| 4 | 审计 KU Leuven + 冻结契约 | **增加**：①参数化 `Repair` 的 `source_unit_exponent`（µV vs V，见 §3.11 第 6 条）；②审计必须验「trial 内标签恒定」；③`rep_*` 与本体同组（事实 7） |
| 5 | 训练 64 通道解码器 | **双契约**：64 通道（保底，与论文可比）**和** 20 公共通道（真人模式用）各训一个；留出故事与留一被试两种口径都要跑；`group` 必须把 `rep_*` 与本体合并；**必须报每类准确率/平衡准确率**（事实 13：标签 80/20 不平衡，单一"准确率"会误导）；留出划分必须分层，保证验证集含 class 1 |

### 依赖关系

```text
1 → 2 → 3 → 4 → 5 → 5.5 ─┐
                         ├→ 8 → 9 → 9.5 → 10 → 10.5 → 12
        6 → 7 ───────────┘               └→ 11（硬件在场；用 20 公共通道契约）
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
| 2, 3, 4, 5, 5.5, 9.5 | `structure-dev`（阶段完成时） | 代码/测试/审查规范 |
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

## 7. 主 agent 变更登记规则（用户 2026-09-13 要求，硬规则）

**本文件的任何改动都必须登记，没有例外。**

1. 每次修改 `final_connection.md`，必须在 §8 变更记录追加一行：版本号、日期、**本次实际改了什么**、以及该改动对应的 commit（若当次提交尚未产生，则在下一次提交时回填）。
2. 设计层面的改动（新增/删除约束、改口径、改验收判据）必须同时更新受影响的正文小节，不允许只改变更记录或只改正文。
3. 子 agent **无权**修改本文件；发现计划有问题时向主 agent 报告，由主 agent 改并登记。
4. 每个功能步骤的 commit 里，若该步导致本文件变化，该变化必须包含在**同一次提交**中，保证「代码与它所依据的计划」在同一版本下可追溯。
5. 判定标准：事后只看 `git log -p final_connection.md`，应当能完整重建计划的演化过程，不需要任何口头补充。

---

## 8. 决策记录（用户 2026-09-14 要求，硬规则）

**任何可能影响最终结果的决定都必须写在这里**——不只是代码改动，还包括：口径选择、
阈值设定、数据取舍、把某物排除在验证之外、以及"我选择不修某个问题"。
判定标准：**如果三个月后有人问"为什么是这个数/这条路"，只读本表就能答出来。**

格式：编号 / 日期 / 决定 / 理由 / 被否掉的替代方案 / 对最终结果的影响 / 相关 commit。

| # | 日期 | 决定 | 理由 | 否掉的替代 | 影响 | commit |
| --- | --- | --- | --- | --- | --- | --- |
| D-01 | 09-12 | 参考系 = EEG 采样时钟；播放位置权威 = 浏览器 | EEG 不可再生且是相关运算的独立变量；解码器 lag 建在 EEG 空间 | 以后端时钟为权威 | 所有对齐与拟合都绕 EEG 时基 | — |
| D-02 | 09-12 | 后端对 `media_time_s`/`media_revision` 只有抄写权 | 前端逐字段校验，不一致即回中性 | 后端自行推算更"准"的时间 | 杜绝了"看起来更准但永久不激活增益" | — |
| D-03 | 09-12 | Offloader 保留，workers=1 / capacity=2 / `drop_oldest` + 显式 `evidence_gap` | 结构上保证采集线程不做重活；丢帧必须显式 | 去掉 offload；或 capacity 加大 | 决策延迟有上界；丢帧会降级而非静默 | — |
| D-04 | 09-12 | 包络在会话启动前从 `datasets/audio/` 加载并校验，缺失即启动失败 | 避免运行期现算引入的不确定性 | 运行期按需计算 | 演示启动更脆但行为可预测 | — |
| D-05 | 09-13 | 通道口径选**公共子集**（20 通道）用于真人模式，64 通道保底 | 现场拿不到 KU Leuven 独有的 44 个电极 | 空间投影；双份独立布点 | 真人模式输入维度更小，性能需实测 | — |
| D-06 | 09-13 | 测试集参考包络改用 `left_mono`/`right_mono`（实际播出声道） | 与 `dichotic_15min.wav` 逐位相同；`*_raw` 未播过 | 继续用 `*_raw` | **修掉了一个会毁掉整个测试集评估的错误** | `7982c5e` |
| D-07 | 09-13 | 排除三处误触标记（1007/Saying-YES、紧随的 1006、第二个 1004/Start） | 用户确认「就这几个」 | 启发式过滤全部可疑标记 | 标注边界明确，可复核 | — |
| D-08 | 09-13 | `rep_*` 与其本体合并为同一故事组 | 实测前 125 s 逐样本相同，否则故事泄漏 | 按文件名分组 | 留出故事划分不再虚高 | `9dd4b4a` |
| D-09 | 09-13 | 必须报每类/平衡准确率（标签 80/20 不平衡） | 单一"准确率"会被多数类拉高 | 只报总体准确率 | 报告的结论更保守也更可信 | `9dd4b4a` |
| D-10 | 09-13 | 前端原样移植，仅两处偏离（`ATTUNE_PYTHON`、`rejected` 可见化） | 保留上游协议校验与测试价值 | 重写前端；或改协议 | 前端行为可追溯到上游 commit | `61ca57c` |
| D-11 | 09-13 | 不升级 `vite`/`esbuild`（npm audit 2 条 dev-only 公告） | 运行路径是 FastAPI 提供 `dist/`，不暴露 dev server；§3.6 要求锁版本 | 升到 vite 7.3.6 | dev server 有已知公告，运行路径不受影响 | — |
| D-12 | 09-14 | 为 3 个失败的前端测试补**标注为测试替身**的 `apps/backend/` | 协议校验测试必须真跑，否则"全绿"是假的 | 接受 3 个失败；或 vendor 同侪整个 `backend/` | 夹具与真实现分叉风险，故明确标注为替身并由 `src/nova2026/transport` 取代 | 本次提交 |
| D-13 | 09-14 | 验证必须是**1 项正常 + 15 项扰动**（§3.13/§3.14） | 用户要求：只有一项完美，其余都要带故障 | 只跑正常路径 | 验收强度显著提高，工作量大增 | 本次提交 |
| D-14 | 09-14 | 主 agent **不写代码/不写测试**，全部派子 agent；主 agent 只做定义、派发、核验、登记 | 用户要求；避免主 agent 的上下文与精力被实现细节吃掉，也避免"自己写自己验" | 主 agent 顺手修小问题 | 派发开销上升，但验收独立性更强 | 本规则提交 |

---

## 9. 主 agent 的纪律（用户 2026-09-14 要求，硬规则）

**主 agent 不写代码、不写测试。**

1. **一切实现与测试都发给子 agent。** 包括但不限于：写脚本、改库代码、写测试、跑修复、
   造夹具、改文档里的实现细节。主 agent 的职责是**定义任务、派发、独立验收、登记与提交计划**。
2. **主 agent 允许做的只有**：
   - 读代码、读产物、读报告（理解与判断）；
   - 运行**只读**的核验命令（如 `git status`、`git log`、复跑既有测试、查看文件）——
     目的是**验收**，而不是替子 agent 干活；
   - 编辑 `final_connection.md`（计划、决策记录、变更记录）；
   - 亲手提交（`git add` / `git commit`）与派发 / 熔断子 agent。
3. **"顺手修一下"是被禁止的**：发现小错误时，派一个子 agent 去修，或把修复要求写进下一步的 brief，
   而不是自己改。唯一例外是**紧急与安全**（例如正在破坏仓库的操作）。
4. **追溯承认**：`bb45634`（`apps/backend/` 测试替身）是主 agent 自己写的，**违反本条**。
   当时未立规则，事后补记于此，作为反例保留。
5. 判定：`git log` 里除计划文件外，**不应出现主 agent 署名的实现提交**。
   主 agent 的提交只应包含 `final_connection.md` 与主 agent 自己的核验记录。

---

## 10. 变更记录

| 日期 | 变更 | 作者 |
| --- | --- | --- |
| 2026-09-12 | 创建草案 v0.1（骨架 + 步骤表） | 主 agent |
| 2026-09-12 | v0.2：补 attune-ui 侦察结论、后端契约表、移植约束 | 主 agent |
| 2026-09-12 | v1.0：锁定 A–H 全部设计决定；新增 `datasets/audio` 包络要求；新增偏移扫描（5.5）与无设备 demorun（10.5）；写入最终验收 V1–V7 与主 agent 职责 | 主 agent |
| 2026-09-12 | v1.1：新增 §6.1 强制 skill 加载表、§6.2 子 agent 预算与防死循环（熔断、新 agent 接手、限额升级）、§6.3 预登记命令表；步骤表加入每步预算 | 主 agent |
| 2026-09-13 | v1.2：步骤 1 **DONE**（523 tests 全绿，commit `21a637e`）；按 §6.2 规则 4 将全部步骤预算重校准（步骤 1 实测 14 步，名义 6 步属于预算设定错误，非 agent 失控）；§6.2 增补 `git check-ignore` 的验收陷阱与替代判据 | 主 agent |
| 2026-09-13 | v1.3（commit `0e9b7fa`）：§1 新增音频素材实测（44.1 kHz 单声道 int16、394.0–395.3 s、`rep_` 段 125.0 s、包络约 100 KB）；素材比 EEG 长故必须按 EEG 长度截断；demorun 需 `--clip`；链内重采样比 2 | 主 agent |
| 2026-09-13 | v1.4：新增 §3.11 真人硬件 24+1 通道与跨设备契约口径（含 A/B/C 三个候选方案与暂定口径）；新增 §7「主 agent 变更登记规则」 | 主 agent |
| 2026-09-13 | v1.5：**求真而非求顺**——§1 事实 2 被步骤 2 证伪（旧 loader 不抛错，真正缺陷是静默用 hrtf 当参考包络），新增事实 7（`rep_*` 与本体育**逐样本相同** → 故事泄漏是真实风险）、事实 10（µV vs V）；步骤 2 标 DONE（commit `3e0ba15`） | 主 agent |
| 2026-09-13 | v1.6：新增 §3.12 真实测试数据集（`tmp/antneurodata/`，含两会话音频起点 0 s / 267 s、标记语义与误触警告、用户给定的 ±buffer 标注口径）；§3.11 由实测替换为确定参数（500 Hz / 24 通道 / 伏特 / 20 个公共通道）；新增步骤 9.5（ANT 导入器）、步骤 3–5 口径修订表、验收 V8；依赖图更新 | 主 agent |
| 2026-09-13 | v1.7：步骤 3 **DONE**（commit `ab63ede`）；用户确认标记语义（1001=左、1006=右）与误触清单；实测音频文件关系并**修正测试集包络来源**（`*_mono` 逐位等于实际播出声道，`*_raw` 未播过）；§1 新增事实 11（44.1 k vs 48 k）与事实 12（18 个包络）；登记 `envelopes.py` 365 行的结构例外 | 主 agent |
