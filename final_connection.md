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
10. **KU Leuven 的物理单位是 µV**（峰值 ≈325.6），而现场 ANT 数据是**伏特**（峰值 ≈0.0833 V），
    经 `unit_scaler` 后进入链的同样是 µV。
    两者的 `source_unit_exponent` 已参数化（见 §3.11 第 6 条）。
    **修正**：主 agent 原判断"`AuditoryProcessor` 的 `-6` 默认值是错的、应为 `0`"**是错的**；
    按 `_to_uv = 10 ** (exponent + 6)`，µV 源正确值就是 `-6`。上游默认值一直是对的。
11. **KU Leuven 素材是 44 100 Hz**，而现场实验音频是 **48 000 Hz**（实测）。包络生成按每个文件的真实采样率处理，
    元数据记录 `source_rate`，因此两类素材不会互相污染；但任何假设"素材都是 44.1 kHz"的代码都会错。
12. **步骤 3 的规模**：`datasets/audio/` 现有 **18 个包络**（16 个 KU Leuven dry + `left_mono` + `right_mono`），
    合计约 2.1 MB（其中两个 900 s 的测试集包络各 ≈278 KB）。`--verify` 全部通过。
13. **标签在一个 trial 内恒定，且类别严重不平衡**（主 agent 实测 320 个 trial，步骤 4 复核）：
    标签集合只有两种——`{0}` 出现在 **256** 个 trial，`{1}` 出现在 **64** 个 trial，**没有任何 trial 混合两种标签**。
    按时间算：**A = 48 894.0 s，B = 25 087.5 s，即 66.1% / 33.9%**。直接后果：
    - **分类器的 null（全猜多数类）是 66.1%，不是 50%**。任何"准确率 > 50% 即成功"的说法都是错的。
    - **必须同时报每类准确率或平衡准确率**，否则数字没有意义（数据集 README 警告的正是这类虚高）。
    - 步骤 5 的留出划分必须**分层**：按故事分组后仍要保证验证集里有 class 1。
14. **两个分组守卫存在泄漏路径**（步骤 4 发现，尚未修）：`nova2026/auditory/evaluation.py` 的
    `check_split` / `assert_held_out` 用 `trial.group.split("|")` **字面比较**，而转换产物的 `group`
    仍带 `rep_` 前缀，于是 `{part1_track1, part1_track2}` 与 `{rep_part1_track1, rep_part1_track2}`
    看起来互不相交——**训练 trial 0、验证 trial 8 会被接受，尽管两者含同一段 125 s 音频**。
    修法：在进入 `evaluation` 之前用 `scripts/auditory/kuleuven_contract.canonical_story` / `group_key`
    归一化。**步骤 5 的评估必须使用归一化后的 group**，否则留出故事成绩是泄漏出来的。
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
4. ~~时钟拟合残差在阈值内（`sync.status == "observed"`）~~ → **修正（步骤 12 核查，2026-09-14）**：本实现**没有时钟拟合**，`sync` 恒为 `unobserved / offset_ms=null`（V9 的 C11/C12 已确认），**这条不可能满足**；前端门控的第 9 条只排除 `desynchronized`/`invalid`，门控照常打开。**本条从守卫表删除**，改为如实记录：`observed` 需要浏览器时钟拟合，该机制尚未实现（D-35 第 ② 条）。

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
6. **单位口径**：现场数据是**伏特**（`source_unit_exponent=0`），KU Leuven 是 **µV**。
   两者都**不需要**在链内做数值换算，因为 `Repair` 的 exponent 只用于端点安全检查、不改值
   （`src/nova2026/streaming/preprocess/repair.py:86-88, 164-172`）。
   **修正（步骤 4 实测推翻主 agent 原判断）**：`_to_uv = 10 ** (exponent + 6)`，所以
   **µV 源的正确 exponent 是 `-6`**（`_to_uv = 1.0`，325 µV 被当作 325 µV 比较，正确）；
   若误用 `0`，325 µV 会被当成 3.25e8 µV，越过 75 000 µV 饱和线。
   即**上游默认值 `-6` 本来就是对的**，主 agent 原先"默认值错了"的判断是错的。
   真正需要的是**让调用点能显式声明**：现场 ANT 数据经 `unit_scaler` 后喂进链的已是 µV，
   因此其调用点的语义与 KU Leuven 相同（都是 µV 链输入），参数化的意义在于**可声明、可校验**，
   而不是修正默认值。

### 3.15 呈现方式的目标变更（用户 2026-09-14 明确）：从"两耳不同"到"两耳同混合"

**现状（已录数据与 KU Leuven 都是这个形态）**：左耳 = 候选 A、右耳 = 候选 B（**空间/耳间分离**）。
被试靠"注意哪只耳"完成任务，解码器可以捡到耳间与空间线索。

**目标形态**：**两只耳朵同时听到 A 与 B 的同一混合**，被试注意其中一路。这才是真正的
选择性注意（cocktail-party），**没有耳间线索可捡，可解码差异显著更小**。

**受影响的四处，逐条记明**：

| # | 位置 | 现状 | 目标形态下必须做的改动 |
| --- | --- | --- | --- |
| 1 | 解码器判别余量 | 双耳分离下 `MIN_MARGIN = 0.5` 可用 | margin 与窗长**必须在新呈现条件下重新实测**，不得沿用旧数字；预期判别余量下降 |
| 2 | 前端增益通路 | `<audio>` → `ChannelSplitter(2)` → 两个 `GainNode`（对左右耳各自衰减） | 需要"对混合中两路各自增益"的通路。**短期兼容做法**：后端提供「左声道=偏向 A 的混合、右声道=偏向 B 的混合」，前端**零改动**；**正确做法**：前端改 Web Audio 内部两路混音（改动 vendored 前端，需用户批准） |
| 3 | 参考包络 | 取自两路独立素材 | 不变——包络仍取自**未衰减的原始候选**，绝不能被混音或增益污染（否则自激，见 §2.1 铁律） |
| 4 | 已录 ANT 数据的定位 | —— | 它是**双耳分离形态**，因此是该任务的**预演**而非最终形态验证。此点**必须写进 `VALIDATION.md`**，否则会声称过头 |

**暂定推进顺序（不阻塞当前步骤）**：先按现有形态把整条链路跑通（V1–V9），
再在步骤 11 之后加一步"同混合呈现"的改造与重新评估。**代价与风险已登记为 D-20。**

**回退选项（用户 2026-09-14 补充）**：若"两耳同混合"效果不好（解码余量太小、演示看不出差别），
**允许回退到"每耳不同音频"**（即现状形态），代价是**可能要改 UI 代码**（若已按混音方案改过前端）。
两种模式必须**由配置选择**，不是二选一删掉另一种——因为已录的 ANT 数据本身就是"每耳不同"形态，
它是回退路线的现成验证集。

**由此产生的代码组织决定（D-21）**：**在 `scripts/` 下新开一个文件夹专门放不同模式的 demo**
（而非把模式分支塞进同一套脚本）。理由：两种模式的差异不止一个开关——
参考包络来源、增益通路语义（对耳 vs 对声源）、前端是否需要改造、以及评估口径都不同；
塞进同一套脚本会让 `scripts/auditory_ui/` 同时承担两条路线的分支，**体积与职责都会膨胀**，
而用户明确点到这两个风险。分文件夹后：共享逻辑留在 `src/nova2026/`（`AttentionSession`、
生产者、传输层都是模式无关的），每个模式文件夹只放"该模式特有的装配与参数"。

### 3.17 步骤 5 暴露的三条跨步硬约束（必须传给 5.5 / 8 / 10 / 11）

1. **默认质量策略会让 demorun 中途停机。** 在 `check_channels=True, max_bad_channels=0` 下，
   KU Leuven 里幅度超 500 µV 的 fault 很常见；`S1/trial_003` 直接抛
   `EEG quality faults persisted beyond the allowed duration (122s)`（恢复上限 = `max(15, 2*history+2)`）。
   步骤 5 的缓存因此显式使用 `check_channels=False`（这是文档化的 run policy：**信号不变，只改判决**）。
   **步骤 10 的 demorun 若沿用默认策略会在中途停下，V1 直接跑不通** → demorun 必须显式选择放宽策略，
   并且这个选择要写进运行记录与 `VALIDATION.md`，而不是悄悄改默认值。
2. **链条会丢掉每个 trial 最后约 1 s。** 重采样器的 reserve 使 64 Hz 流提前结束
   （124 s trial → 7872 而非 7936 点）。缓存按**实测窗口长度**而非公式推导，实现里有注释。
   任何"用公式算窗数"的代码都会错一格。
3. **`evaluation.assert_held_out` 会拒绝 LOSO。** 它按 story token 比对，而每个被试都有同样 4 个故事，
   留一被试的训练侧必然包含验证侧的故事。步骤 5 只对"留出故事"口径调用该守卫，LOSO 改用自有 trial 级检查。
   这是既有库语义，不是 bug——但步骤 8/10 若复用该守卫必须先看清这一点。
4. **对齐预算压倒通道预算（步骤 5.5 实测，必须传给 8/9/10/11）。** 5 s 窗、留出故事口径下：
   峰值在 **−25 ms**（64ch 平衡 0.6256；0 点 0.6195 距峰 0.006），**half-depth 宽度 104 ms**，
   ±250 ms 处已回落到掷硬币；**每 100 ms 残余错位代价 = 0.1191（早）/ 0.1496（晚）**平衡准确率。
   对比：整个"64 通道 → 20 通道"的差距只有 **0.022** —— 即 **100 ms 错位的代价是换掉 44 个电极的 5–7 倍**。
   直接推论：
   - 前端那个 `|Δt| ≤ 0.75 s` 的门槛**界定的是"报告陈旧度"，不是对齐规格**——它约为解码器全部容差的 7 倍；
     若补偿后的**残余**错位真能接近 0.75 s，解码器贡献为零（那时准确率就是掷硬币）。
     **步骤 9 必须在协议文档与 `VALIDATION.md` 里写明这一点**，避免把 0.75 s 当成对齐指标。
   - **对齐预算应写成 ±100 ms 量级**；步骤 11 的回环测量（`residual_offset_seconds`，容差 ±30 ms）
     因此是对的方向，且比门槛要求更严——这点要在步骤 11 报告里对上。
   - 步骤 5 的"30–60 s 窗更优"是**零偏移下**的测量；长窗在非零偏移下是否更耐受**尚未测量**。
     若时间允许，8/10 之后补一次"长窗 × 非零偏移"的小扫描，否则在 `VALIDATION.md` 标注为未验证预测。
   - `MIN_MARGIN` 是相关**差值**的门槛，本实验没有测量它；本曲线对它**不构成**任何结论。
5. **`MIN_MARGIN = 0.5` 从未被标定，而它现在决定演示能不能看**（步骤 8 实测，必须传给 9/10）：
   真实 trial（S1/trial_008，5 s 窗）跑出 **119 窗 / 117 评分 / 决策分布 unavailable 19、uncertain 465、A 12**
   ——即 **94% 的帧没有结论**（12/12 已决策帧与真值一致，但只有 12 帧）。这是铁律 2 的**显式降级**，不是 bug，
   但若照此演示，UI 几乎全程显示"不确定"。步骤 5.5 只标定了**偏移代价**，从未标定 margin 本身。
   **步骤 9/10 必须显式处理，且不得偷偷调参**：先在**留出数据**上标定 margin（报出 margin 与
   coverage/accuracy 的权衡），把标定过程与数据划分写进运行记录，再据此选窗长与 margin；
   若最终仍选择保持 margin=0.5（宁可不给结论），那也必须写成**记录在案的取舍**而不是默认值。
6. **放宽质量策略会让 `signal_quality` 失真**（步骤 8 发现，宜在 9/10 修）：`check_channels=False` 下
   `QualityMonitor.reasons()` 按策略返回空，`signal_quality.quality` **恒为 1.0**、死电极显示 `artifact=false`，
   因为 `EEGWindow.bad_channels` 是"policy aside"的证据却没被 `AuditoryWindow` 携带。
   修法：给 `AuditoryWindow` 加字段并据此判决 `artifact`——**不许用增加 reject 的方式绕过**（那会重新引入停机）。

### 3.16 扰动用例矩阵（用户 2026-09-14 要求：**只有一项是"正常"，其余都必须带故障**）

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

**误触排除清单（2026-09-14 修订：**只排除 2 个**；第三个改为"保留并标歧义"）**

用户原话是"就这几个"，但独立核验（`results/antneuro_testset_report.md` §4）与本仓导入器实测发现：
若连 `1006@148.402` 一起排除，则只剩 **11 个右 cue vs 12 个左 cue**，并出现 **25.670 s** 的"左"段；
而保留它时，`135.554→148.402` = 12.848 s 与 `148.402→161.224` = 12.822 s **都落在本会话正常区间**（10.922–16.692 s）。
证据指向**该 `1006` 才是真正的右提示**，而 2.000 s 长的 `1007/Saying-YES` 才是异常事件。故：

| 会话 | EEG t | 标记 | 处理 |
| --- | --- | --- | --- |
| 会话 2 | 148.354 s | `1007/Saying-YES`（时长 2.000 s） | **排除** |
| 会话 2 | 326.692 s | 第二个 `1004/Start` | **排除**（停止录音误触） |
| 会话 2 | 148.402 s | `1006/Custom Annotation` | **保留并标 `ambiguous-cue`**（修正 D-07 的"三处全排除"） |

排除后每个会话的有效切换标记均为 **12 个**（`1001` 12 个、`1006` 12 个），严格交替，间隔 8.004–16.692 s。

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
| V4 | 决策与真值可比：窗口级准确率 + 覆盖率 + 误切换写入 `results/`；**ANT 测试集上必须报 20 通道契约的结果** | 指标 JSON | **部分** — KU Leuven 侧达标（`results/demo_run_*`、`results/aad_*`）；**ANT 侧由步骤 10.7 补齐**（步骤 12 核查发现 `results/` 内**当时没有任何 ANT 解码数字**，已在 `VALIDATION.md` §3.9 如实标为缺口） || V5 | 同步质量可查：块时间误差、漂移 ppm、拟合残差 | `timing.json` / 报告 |
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
| 4 | **数据体检 + 契约冻结** | `results/kuleuven_audit.md`；冻结 `AuditoryConfig` 与特征契约 | 审计脚本 + 报告 | 35 / 40 | **DONE** — commit `48c412d`；585 行报告 + 19264 行逐 trial JSON；`source_unit_exponent` 已参数化（`DEFAULT_SOURCE_UNIT_EXPONENT = -6` 带白名单校验，且**默认行为未变**）；分组键已折 `rep_`；契约冻结列出 7 项不可变内容。**并纠正了主 agent 关于单位的一处错误判断**（见 §3.11 第 6 条） |
| 5 | **解码器训练与评估** | `models/auditory_kuleuven.npz` + `results/aad_<date>.json`（留出故事 **+** 留一被试 + 窗长曲线） | 训练命令 + 指标 JSON | 35 / 75 | **DONE** — commit `1a6bc3c`；320 trials、双契约、两口径、窗长曲线。**关键结论：原始准确率（0.617/0.609/0.594/0.586）低于多数类率 0.6586，只有平衡准确率可信**（64ch 0.6195/0.6115，20ch 0.5974/0.5891）。**20 通道比 64 通道低约 2.2 个百分点 → 上真人的代价已量化**。窗长曲线单调上升：64ch 平衡 0.637(5s)→0.698(10s)→0.796(30s)→0.875(60s)，**步骤 5.5 与 demo 应偏向 30–60 s 窗**（60 s 点仅 288 窗/4 被试）。**例外**：`feature_cache.py` 530 行 / `train_kuleuven.py` 584 行超 300 行建议，理由同步骤 3。**预算超支 2.4×**（85 步 vs 35） |
| 5.5 | **偏移扫描实验** | `results/aad_shift_sweep_<date>.json`：包络滑 ±300 ms 的相关衰减曲线 | 扫描脚本 + 曲线 | 30 / 40 | **DONE（第 2 次尝试 `a859f40`）** — 第 1 次 `6ee0ba5` 的曲线**已撤回**（其 `.md` 带 RETRACTED 横幅）。根因不是"1 个样本错位"而是**纯符号反转 + trim 假象**：`shift_window` 读 `+shift`、`trial_scores` 又减一次（`low = max(first, shift, 0)`），零点不可见、非零点全镜像。配对现已**逐窗口对齐原始链路**：`max|Δenvelope|` 5.3e-10…8.6e-10（镜像对照 2.4e-02…3.0e-02），`max|Δscore|` 3.3e-09…8.5e-08，**两路径每个偏移的平衡准确率完全相同**；零点仍**逐位复现**步骤 5。另修：亚采样偏移原先被 `round(δ*64)` 静默取整（12.5 ms 步长产生重复点）；越过录音边界的窗口现在**弃权**而非被截断打分。**额外发现并修掉** `_crossing` 向内游走导致 `width_of` 返回**负宽度**（即撤回报告里的 −109/−447 ms）。22 个测试全绿、无 skip、**两个 `expectedFailure` 已移除并转为真通过的测试**，5 个变异（符号/取整/截断/网格点/向内游走）全部致红。全仓 **663 tests 全绿** |
| 6 | **传输层移植**（须读 `secure-web-dev`） | `src/nova2026/transport/{protocol,publisher,sessions,server}.py` + 契约测试 | 单测全绿：包校验/快照/1013/生命周期 | 50 / 60 | **DONE** — commit `7f2ab43`；63 个契约测试全绿；`tests/transport` 已加入 `scripts/run_tests.py`（全仓 628 tests）；依赖 5 个 pin 经 `secure-import` 逐个核验。**例外**：`server.py` 364 行 / `create_app` ~170 行，经批准作为已记录例外。**预算超支约 2 倍**（103 步 / 90 分钟），未触发重试闸 |
| 7 | **前端移植** | `apps/attune-ui/`（来源 commit 记录）+ `npm ci/build/test` 通过 | 构建产物 + 测试输出 | 35 / 60 | **DONE** — commit `61ca57c`（来源 `4a523956`，逐文件 SHA256 记录在 `PROVENANCE.md`）；`npm ci` / `build` 通过；测试 **53/53**（配合 `apps/backend/` 测试替身与 `ATTUNE_PYTHON`，见 D-12/D-16） |
| 8 | **`AttentionSession` + 生产者** | `src/nova2026/auditory/session.py` + `AttentionProducer`；合成 trial 先打通 | 端到端日志 + 前端截图 | 50 / 60 | **DONE** — commit `d92a5d6`（15 文件 / +4463）。`session.py`(687) + `sources.py`(378) + `producer.py`(289) + `scripts/auditory_ui/`（预登记 CLI + 前端证据 mjs），**session 无任何 FastAPI/transport import**。合成 20 s → 263 包 14/14 断言通过；**真实 trial（S1/trial_008，1× 实时 124 s）→ 1616 包 14/14 通过**，`output/auditory_ui/packets_trial008.jsonl` 留证。快照 3 包先于 1613 增量；1616/1616 过本仓 validator；增益 992 个全部 ≤0 dB；`session` 生命周期包全为 `source=server`。**前端证据（无浏览器，如实标注）**：真实包流经 vendored 前端自身的 `protocol.js`/`state.js`/`decoders.js`（rejected=0）+ `Dashboard.js` 用 `react-dom/server` 渲染出 **"Focused on Speaker A"、FOCUSED、B 路 −6 dB** → `results/auditory_frontend_trial008.html`；**不证明**布局/交互/Web Audio 通路（属步骤 9/10）。5 条降级路径各有真实包（warmup / audio_unavailable / evidence_gap / evidence_stale / processing_failed），缺包络 → HTTP **409** 并指名文件。12/12 变异被捕获。全仓 **700 tests 全绿** + 前端 53/53。**例外**：`session.py` 687 行、CLI 579、单测 589 超 300 行建议（同步骤 3/5/5.5/6 先例）。**发现两条硬约束**（见 §3.17 第 5、6 条：margin 未标定致 94% 帧无结论；放宽策略下 `signal_quality` 恒为 1.0） |
| 9 | **媒体时间轴与音频** | 立体声 WAV（L=A,R=B）经 `/api/media/file` 提供；`media/control` 握手 + 250 ms `report` | 时间差 ≤ 0.75 s 的证据 + 增益激活证据 | 50 / 60 | **DONE** — commit `9202290`（14 文件 / +3845）。新增 `src/nova2026/auditory/render.py`（`render_stereo`：L=A / R=B、int16、按 EEG 长度截断；**呈现方式是参数**，`dichotic` 已接线、`crossmix` 已实现未接线）；`transport/media.py` 增 `media_reference()`（**回抄**客户端最后上报值；未 prepare / 被作废 / 超 1.5 s → `None`）与 `MediaBroadcaster`（250 ms 一个 `media` 包，仅 `running`，`source=server`）；`producer.py` **每帧只读一次引用**并同时戳 attention 与 gain（原先两次读取会让 `playbackGains` 判定位置不等而回中性）。**前端门控 14/14 条全真**（t=40 s 首次开启），`playbackGains('attune')` = **[1.0, 0.5011872336272722]** = 0 dB / −6.0 dB，正是包内 `a_db`/`b_db` 的 `dbToLinear`。361 个增益包时间差 **max 0.3003 / mean 0.1492 s**，全部 ≤0.75。12/12 变异被捕获。全仓 **740 tests 全绿** + 前端 53/53。**未做**：真实浏览器（门控在 Node 里执行 vendored `mediaAudio.js`）；§3.17 第 6 条如实留给步骤 10。**例外**：`render.py` 350 / `transport/media.py` 458 / `auditory_ui/media.py` 341 / `auditory_ui/session.py` 1047 行 |
| 9.6 | **决策 margin 标定**（§3.17 第 5 条要求） | `results/aad_margin_calibration_<date>.json` + 接线规格 | 留出数据上的 accuracy/coverage 曲线 + 推荐工作点 | 40 / 60 | **DONE** — commit `bd2da11`（`scripts/auditory/margin_calibration.py` 946 行、15 测试、`config.py` 纯增量）。**留出证据**：4 折故事留出 × 80 trial，每折**重新拟合**且 `assert_held_out` 通过；已部署的 320-trial 模型**从不被评分**；快速评分器与 `model.score` 差 4.6e-16。**结论**：现行 `margin=0.5` 在 5 s 窗只有 **0.15% 覆盖率**、30/60 s 窗为 **0**——步骤 8 的"94% 无结论"是 margin 所致，不是链路问题。**推荐（现有 5 s 模型即可用）**：`margin 0.05` → 覆盖率 **0.524**、准确率 **0.684**、平衡 **0.688**（4/4 故事高于随机）；代价：全帧召回 0.348/0.380、首次决策中位 10 s、误切换 2.43 次/分。栅格最优为 `0.05 @ 60 s`（覆盖 0.384、平衡 0.948、0.11 次/分）**但 60 s 模型不存在**（窗长属解码器契约，不能作 RunPolicy 字段）。**默认值未改**（`MIN_MARGIN=0.5` 有 3 个测试守着），新增 `CALIBRATED_MARGIN=0.05` / `CALIBRATED_HISTORY_SECONDS=5.0`，**无任何东西默认为它们**。8/8 变异被捕获。**未接线**：`RunPolicy.margin` 等 4 步规格已写入报告，按并行纪律延后 |
| 9.5 | **ANT 真实数据集导入**（§3.12） | `scripts/auditory/antneuro.py`：读 `.cnt`（`read_raw_ant`）+ 会话音频起点（0 s / 267 s）+ 标记清单 + 用户标注口径（切换 ±buffer → `-1`） | 标记表导出供人工核对；每个会话的 trial 形状/时长/标签分布；误触清单显式记录 | 45 / 60 | **DONE** — commit `fda17fc`（7 文件 / +3166）。两会话均导入：**155093×20 @500 Hz / 310.19 s** 与 **164317×20 @500 Hz / 328.63 s**，各 24 个 cue，`-1` 41.3/43.3 s、A 138.8/144.9 s、B 130.1/140.4 s，**存活 86.7 % / 86.8 %**（buffer 0.5 s；1.0 s 口径 245.4/261.8 s 也报）。**只排除 2 个标记**，`1006@148.402` **保留并标 ambiguous**（见 §3.12 修订），排除后 12/12 严格交替；完整标记表（含 non-cue）入 `results/antneuro_import_*.md` 与可核对的 CSV。独立核验：由标记表重算的 labels 与 npz 内**逐样本一致**；`trial.audio` 与 `left_mono.npz` **逐值一致**。25/25 变异被捕获。全仓 **785 tests 全绿**。**偏差**：`trial.audio` 存的是**参考包络（64 Hz）而非波形**（约 8 MB/会话而非 ~250 MB），代价是不能直接渲染可听 WAV——已记录 `playable_source_slice_samples` 供步骤 10/12 切片。**例外**：`antneuro.py` 864 / 测试 554 行 |
| 10 | **真实 trial 全链路 + 一键入口** | `python -B -m scripts.auditory_ui.demo`：起服务 + 1× 回放 + 开浏览器 | V1–V5 全部证据 | 50 / 70 | **DONE** — commit `3765202`（11 文件 / +370）。**一条命令**（125.5 s，exit 0）：校验两个包络 → 渲染立体声 → 环回起服务（`dist/` 同源挂 `/`）→ 真实 `POST /api/session/start` → 打印 URL → 1× 回放 → 干净收尾。**V1** ✅ · **V2** ✅ 2123 包、`rejected=0`、7 种流全部经 vendored 前端自身模块解码并渲染（**Node 侧证据，明确不声称浏览器/截图**；该证据脚本 19/21，2 项失败是"渲染哪一帧"的瞬时采样问题，已如实留档）· **V3** ✅ 门控 14/14、**9.0 s 开启**、`playbackGains` = 0/−6.0 dB；**前后对比同 trial：margin 0.5 → 12 个非中性增益帧；margin 0.05 → 284 of 496，首次决策从 40 s 提前到 8.6 s** · **V4** ✅ 但**如实标为不可用作成绩**：本 trial 全程单一标签，其多数类 null = 1.0000，窗口级 0.9296 **打不赢任何东西**；可信数字仍来自 9.6 的留出标定 · **V5** ✅ 496 个时间差 max 0.2997 / mean 0.1529 s、握手 470/471 个 200 + `observed` + revision 2；块时序**不可得**并写明原因（无真实音频设备）· **V6** ✅ 全仓 **760 tests 全绿** + 前端 53/53。同时落地 **D-29 接线**（`RunPolicy.margin`、`AttentionController` 显式传入、`--margin`，默认未变有 5 组测试 + 变异守着）与 **§3.17 第 6 条修复**（`bad_channels` 贯通到 `artifact`/census，**未新增任何 reject**）。**例外**：`demo.py` 904 / `session.py` 776 行 |
| 10.5 | **无设备 demorun**（H2） | `python -B -m scripts.auditory_ui.demorun`：无人值守跑完整场并出报告 | 运行日志 + `results/` 报告 | 30 / 45 | **DONE** — commit `b10f916`（`demorun.py` ≈2.3k 行、`rejection_evidence.mjs`、10 测试；`session.py` 加法式补 `recovery_events`/`recovery_segment`/`repaired_samples`）。一条命令 exit 0、239 s、**23/23 用例、18 PASS + 5 FINDING**；每例独立进程 + 独立目录 + 输入 SHA256 前后不变。**三问全答：崩溃 0/23、显式 23/23、污染 0/23**。全仓 **795 tests 全绿** |
| **V9** | **扰动验收**（用户点名要求） | 1 项正常 + 15 项注入故障 + 7 项对抗性 EEG 输入，每项独立运行独立记录 | `results/perturbation_20260913-062335.json` + 逐用例判据 | — | **DONE** — 与步骤 10.5 同一次运行。亮点：C2 断流 `terminal=error` 且不挂起；C3 真实 `Acquire.max_lag=4.758 s` 且 53 个 stale 帧里 **0 个自信决策、0 个非中性增益**；C4 小洞 `interpolated`、2.008 s 洞 `processing_failed` **指名 gap**；C6 `nonfinite_run`；C8 断连后增益全中性、重连恢复；C13 4 个非法包全被前端计数拒绝；C14 真实 **HTTP 409 指名缺哪个包络**；C16 越界标签构造即拒；E2 链与容器**双双拒绝**非单调时间戳；E4 **拒绝而非按列截断**；E5 同一损坏在单位声明错/对时分别得 `unsafe_endpoints` / `interpolated`。**5 条 FINDING 已登记为 D-35** |
| 11 | **真人实时模式** | eego/LSL 接同一 `AttentionSession`；校准流程（含回环测量） | 硬件实测记录（在场时） | 45 / 75 | TODO |
| 12 | **评估收口 + 文档** | `VALIDATION.md`、`documents/auditory_ui_protocol.md`、根 `README.md` | 文档 + 评估命令 | 35 / 50 | **DONE** — commits `1e69f47` + `7133c20`。`VALIDATION.md`（339 行）逐条把**每个数字与它的口径写在同一句里**（例如"64ch 留出 0.6195 平衡，而原始 0.6170 **低于** 逐窗 null 0.6586 / 按时间 0.6609，故只有平衡数有意义"），并命名 **D-35 的四条失败模式**；`documents/auditory_ui_protocol.md`（237 行）写明**0.75 s 界定的是报告陈旧度而非对齐**；根 README 指向新入口并把 5.5/8/9/9.5/10/10.5 状态改对。**核查发现三处计划与证据不符**（见 D-36），已由主 agent 修正计划 |

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

**并行纪律（D-18 的教训）**：**同一时刻不得有两个子 agent 修改同一个文件**。
派发前主 agent 必须确认目标文件集互不相交；`pyproject.toml`、`scripts/run_tests.py`、`.gitignore`
这类共享文件一次只允许一个 agent 触碰，否则会出现提交污染或互相覆盖。

### 6.4 测试不得依赖"环境里装没装某个包"，且必须能被证伪

`fb0e9ed` 的教训（步骤 6 的 antio 修复）值得单独立规：

1. **前提要自己造，不要继承环境。** 需要"某依赖缺失"的分支，必须在测试内用
   `sys.modules` / `find_spec` 打桩造出该前提并在结束时还原——不能靠"这台机器恰好没装"。
   反之，需要依赖在场的行为，也要在名称/文档里写明该前提。
2. **断言必须能被证伪。** 该测试原本断言报错信息里含 `antio`，而 **MNE 自己的兜底信息里就含
   `antio` 与 `pip install antio`**，所以把被测模块的解释分支整段删掉，测试**依然通过**——
   它不是"脆弱"，是**空洞**。改成断言 `--cnt-npy`（只有本模块的信息里才有）后才真正生效。
3. **提交前做一次变异验证**：临时破坏被测行为，确认测试会红，再还原。
   这是唯一能区分"测试通过"与"测试没用"的办法。

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
| 5 | 训练 + 评估 | **分类器 null 是 66.1%（按时间 A 48 894 s / B 25 087.5 s），不是 50%**；留出故事准确率必须**显著高于 66.1%** 才算有效，且必须报每类/平衡准确率 |
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
| D-15 | 09-14 | 主 agent 可读子 agent 报告并判断对错；认为不对则**再派一个子 agent**，但**同一问题最多 3 个** | 用户要求：既允许纠错，又防止无限套娃 | 主 agent 自己改；或无限重派 | 每个交付物最多 3 次尝试，第 3 次失败即上报用户定夺 | 本规则提交 |
| D-16 | 09-14 | `apps/backend/` 测试替身**暂时保留**（不立刻改接 `nova2026.transport`） | 三个 vendored 前端测试需要 `backend.adapters.{results,contracts,legacy,mock}`，而真传输层不提供适配器/mock；改接等于改写 vendored 测试，须单独授权 | 立刻重写那三个测试 | 仓库内短期存在两份包构造器，故替身已明确标注为 test double 并在 `__init__.py` 写明由 `src/nova2026/transport` 取代 | — |
| D-17 | 09-14 | 修复 `tests/streaming/test_compare_cnt.py` 的 `antio` 前提（把"环境里没有 antio"改成测试内可控前提） | `antio` 是读真实 ANT 数据所必需，装上后该测试的假设失效，全仓从绿变红 | 卸载 `antio`（会让 .cnt 读不了）；或改断言迁就现状 | 全仓回归恢复到基线口径（628 tests 全绿） | 待提交 |
| D-18 | 09-14 | 登记**提交污染**：`pyproject.toml` 的 `transport` extra 由步骤 4 的 commit `48c412d` 一并带入 | 两个子 agent 同时改同一文件、步骤 4 后提交；内容正确无需改写历史 | `git rebase`/改写历史 | 归属记录在案：内容属于步骤 6，提交归属步骤 4。**教训：并行子 agent 不得同时改同一文件** | — |
| D-19 | 09-14 | 修复 `tests/streaming/test_compare_cnt.py`：把"环境里没有 antio"改为测试内可控前提，并把断言从 `antio`/`pip install antio` 收紧为 `--cnt-npy` | 原断言**空洞**——MNE 兜底信息本身就含这两个串，删掉被测分支测试仍通过；变异测试才暴露 | 保留原断言；或只改前提不收紧断言 | 全仓恢复 640 tests 全绿，且该测试现在真的能捕获回归；新增 §6.4 立规 | `fb0e9ed` |
| D-20 | 09-14 | 呈现方式目标改为**两耳同混合**（§3.15），但**先按现有双耳分离形态跑通全链路**，之后再改造并重评 | 用户明确最终形态；但改造会同时动摇解码余量、前端增益通路与已录数据的定位，在链路未跑通前改造会让"卡在哪一步"无法定位 | 立刻改造前端混音与重新录制 | margin/窗长需在新条件下重测；短期用「左=偏A混合、右=偏B混合」实现前端零改动；`VALIDATION.md` 必须写明已录数据是预演 | §3.15 已登记 |
| D-21 | 09-14 | **两种呈现模式各占一个脚本文件夹**（`scripts/` 下按模式分子目录），共享逻辑留在 `src/`；模式由配置选择，不删除任一路线 | 用户指出"塞进同一套脚本会让最后的 script 体积与职责都膨胀"；两模式差异涉及包络来源、增益语义、前端改造与评估口径 | 单套脚本内加模式分支；或只保留一种模式 | `scripts/auditory_ui/` 只放当前模式的装配；回退路线（每耳不同）有独立目录；已录 ANT 数据天然是回退路线的验证集 | §3.15 已登记 |
| D-22 | 09-14 | 在继续功能步骤前，先做一次**全仓库大整理**（派专职子 agent）：① 删除非测试的路线验证类与 legacy 代码（**含 `scripts/*/legacy/**`，用户明确要求"legacy 也删掉"**；只留 git 历史）② 阅读并分类 `documents/` 下所有 AI 生成的 md，统一命名、跨目录整合，并重写根 README 写明各脚本用法 ③ 对 `tmp/`/`results/`/`records/` 的派生物**统一重命名而非删除** ④ 其余非源码部分由子 agent 自行判断 | 用户明确要求；整理先行可避免后续步骤继续往混乱目录里加东西 | 边做功能边整理 | 整理会成为独立提交；**删除判据 = 已跟踪 + 无任何 live 代码/测试引用**，满足即删（用 `git rm`），不留"待定"；删除类操作走 `secure-action`；测试与夹具不属 legacy | **DONE** `4f4a9e5`(删 26 文件/−4384 行) `32ee369`(文档统一命名+根 README) `c95d075`(派生物重命名 119 路径+映射清单)；删后 641+53 全绿 |
| D-23 | 09-14 | 步骤 5.5 第 1 次尝试**判定为不可采信**，标记"未完成"并派第 2 个子 agent；替补 brief 必须携带第 1 次的诊断（约 1 个样本的系统错位；缓存路径与原始链路在非零偏移处差 +0.09~+0.20）与其建议（**先用 `replay_windows(audio_offset=δ)` 逐窗口钉死配对，再放大扫描**） | 用户第 4 条要求：失败/卡死时先查清原因并告知替补，避免其掉进同一陷阱；且 §6.4 宁可报 BLOCKED 也不给编造数字 | 直接重派不带诊断的替补；或接受这条不可信的曲线 | `6ee0ba5` 产物保留（脚本+测试+存档曲线 + 2 个 `expectedFailure` 自证缺陷），但 §5 状态为"未完成"；**步骤 9/10 不得引用本步数字**；§3.10「解码器对音频偏移的容差」仍是未测量量 | `6ee0ba5` |
| D-24 | 09-14 | 步骤 5.5 **第 2 次尝试判定为可信，标 DONE**；第 1 次的**曲线撤回但文件保留**（`.md` 加 RETRACTED 横幅，`.json` 不动，不删除） | 第 2 次把配对逐窗口钉到原始链路（1e-10 包络 / 1e-8 分数 / 决策完全一致），并证明第 1 次的根因是**符号反转 + trim 假象**而非错位一格；撤回文件保留可让后人看到"错过的方向长什么样" | 删除撤回产物（需 secure-action）；或把两次曲线混在一起 | 决策：**对齐预算压倒通道预算**（100 ms 错位 ≈ 0.10–0.15 平衡准确率，是 64→20 通道差距 0.022 的 5–7 倍）；`0.75 s` 门槛被明确界定为**陈旧度**而非对齐规格 | `a859f40` |
| D-25 | 09-14 | 登记 `scripts/auditory/shift_sweep.py` **880 行**超过 structure-dev 的 300 行建议，作为已批准例外 | 该文件由三块互锁内容组成（配对/位移机制、曲线统计、链一致性审计），拆开会把"缓存 = 链"这一被测试断言的契约分散到多个文件 | 强行拆分；或压缩注释 | 与步骤 3/5/6 的长文件例外同类，已在计划内留痕 | `a859f40` |
| D-26 | 09-14 | 步骤 8 **DONE**；登记两条新硬约束（§3.17 第 5、6 条） | 真实 trial 端到端跑通，且暴露"margin 未标定 → 94% 帧无结论"与"放宽策略下 signal_quality 恒为 1.0"两个会直接影响演示真实性的问题 | 把 1616 包当成完全成功而不看决策分布 | 步骤 9/10 必须**显式标定 margin 并记录数据划分**，或把"保持 0.5"写成记录在案的取舍；`signal_quality` 的失真要在 9/10 修（给 `AuditoryWindow` 加 `bad_channels` 字段） | `d92a5d6` |
| D-27 | 09-14 | 长文件例外再登记：`session.py` 687 / CLI 579 / 单测 589 行 | 与 D-25 同因（互锁内容拆分会割裂被测试断言的契约） | 强行拆分 | 结构例外的清单继续增长，`VALIDATION.md` 收口时应统一说明 | `d92a5d6` |
| D-28 | 09-14 | 步骤 9 **DONE**；呈现方式接线选 **`dichotic`（L=A / R=B）**，`crossmix` 实现但不接线；**不为 `crossmix` 另建 `scripts/` 目录** | KU Leuven 录制本身是耳分离、步骤 5 的解码器正是按它训练与评分的、已录 ANT 集同形，且现有 `ChannelSplitter → 2×GainNode` 无需改动；而"两耳同混合"需要改动前端（按声源混音而非按耳衰减）——那是 §3.15 的目标形态，不是当前形态 | 立刻接线 `crossmix`；或为它建第二个 `scripts/` 目录 | 演示在现有形态下可跑；D-21 的"两模式分目录"改为**按需再建**（装配已参数化，第二个目录现在只会重复代码而非区分职责）——计划文本据此澄清 | `9202290` |
| D-29 | 09-14 | **采纳 margin 标定结论作为演示工作点**（§3.17 第 5 条闭环）：演练用 `margin 0.05`（覆盖率 0.524 / 平衡 0.688，4/4 故事高于随机），并**把选择写进运行记录**；`MIN_MARGIN=0.5` 默认值不动 | 0.5 在真实 trial 上只给 0.15% 覆盖率、30/60 s 窗为 0，"演示全程显示不确定"无法验收；标定在**留出故事**上做、每折重新拟合、部署模型从不被评分 | 保持 0.5 并接受 94% 无结论；或偷偷把默认值改掉 | 演示可看；代价与召回率公开（全帧召回 0.348/0.380、误切换 2.43 次/分）；**接线（RunPolicy.margin 等 4 步）由步骤 10 落地并在运行记录里写出生效值**；30/60 s 更优点需**新训模型**（窗长属解码器契约） | `bd2da11` |
| D-30 | 09-14 | 登记 5.5/9/9.6 的长文件例外与"栅格下限被顶到"的未测量项：`shift_sweep.py` 880、`margin_calibration.py` 946、`render.py` 350、`transport/media.py` 458、`auditory_ui/session.py` 1047 | 同 D-25/D-27 的理由 | 强行拆分 | `VALIDATION.md` 需统一列出全部结构例外；`margin` 阶梯下限 0.05 已"顶到"，**低于 0.05 何时不值得要仍未测量**（已列为未测量项） | 多处 |
| D-31 | 09-14 | 步骤 10 **DONE**；两个 CLI 的 `--margin` **默认值故意不同**：`demo.py` 默认 **0.05（标定点）**，证据 CLI `session.py` 默认 **0.5（`MIN_MARGIN`，保守）** | demo 的目的是展示能力，必须用标定工作点，否则门控几乎不开；证据 CLI 的用途是可复现的保守证据，不应替用户做激进选择 | 两边统一成 0.5（演示看不到效果）；或统一成 0.05（证据 CLI 变得激进） | 两者都在运行记录里写出**生效值**，所以在任何报告里都能追溯到到底用了哪个 margin；`MIN_MARGIN=0.5` 本身仍未改 | `3765202` |
| D-32 | 09-14 | 步骤 9.5 **DONE**；**修正 D-07**：误触只排除 **2 个**（`1007@148.354`、第二个 `1004@326.692`），`1006@148.402` **保留并标 ambiguous-cue** | 排除它会只剩 11 个右 cue、出现 25.670 s 的"左"段；保留则相邻两段 12.848/12.822 s 均在本会话正常区间——证据指向该 `1006` 才是真右提示 | 按用户原话"三处全排除"（会造成标签时间线错 12.8 s，且破坏 12/12 交替） | 标签时间线在 135.6–161.2 s 段与用户初判相反；已在 §3.12 与标记表 `reason` 字段写明，供人工最终裁决 | `fda17fc` |
| D-33 | 09-14 | ANT trial 的 `audio` 列存**参考包络（64 Hz）**而非音频波形 | 链从不读 `trial.audio`；参考包络必须与解码所用**逐位一致**（可复现），且体积 8 MB/会话而非 ~250 MB | 存波形列（+250 MB/会话，且与包络参考重复） | ANT trial 不能直接渲染可听 WAV；已记录 `playable_source_slice_samples`（精确切片），步骤 10/12 若要真播按此切片一次命令即可 | `fda17fc` |
| D-34 | 09-14 | 步骤 10.5 + V9 **DONE**；矩阵定为 **23 用例**（C1–C16 + E1–E7），即"1 正常 + 15 故障 + 7 对抗输入"，比 §3.16 原写的"1+15"更全 | 用户点名要求"对 incoming EEG 做各种测试"（重复、大 lag 等），§3.14 的 7 项 E 用例正是这半张表；全部实现比抽样更能暴露机制缺口 | 只做 C 组 16 例 | 18 PASS + **5 FINDING**（见 D-35）；`results/perturbation_*` 成为 V9 的验收对象 | `b10f916` |
| D-35 | 09-14 | **登记 5 条 FINDING 为已知机制缺口**（不变量都成立，缺的是"把缺口说出来"的机制）：① `Acquire` 只统计迟到间隔，**不计数重复/重叠块**（C5：链以 `irregular_timestamps` 拒绝并消耗证据，但采集层 `gaps==0`）② **整条链没有时钟拟合/漂移残差**（C11：注入 1.25× 漂移后 \|Δt\| 达 2.63 s、0 增益被应用——不变量守住，但 `sync` 恒为 `unobserved/offset_ms=null`，**再次确认 §3.17-4：0.75 s 界定的是陈旧度而非对齐**）③ 传输层**无 underrun/stall 事件计数器**（C12：26 次扣留报告、陈旧决策未被应用，但计数只能由扣留日志导出）④ **C15/E3 期望的"启动即拒绝"在 replay 路径上不存在**：候选长度不等（18 s vs 24 s）与采样率不符（256 vs 128）都表现为**逐窗显式降级**（`audio_unavailable` / `scoring_failed`，0 自信决策），而非启动失败 | 按"宁可报告不要抢修"处理：四处都属机制缺失而非不变量破坏，且补做会牵动 `Acquire`/传输层/会话启动契约，风险高于收益 | 强行补做四者；或把 FINDING 改写成 PASS（掩盖缺口） | **`VALIDATION.md` 必须写明这四条**，尤其是 ④：契约不符在本实现里以逐窗显式降级体现，**不是**启动拒绝——这直接改变使用者对失败模式的预期 | `b10f916` |
| D-36 | 09-14 | 步骤 12 **DONE**；**核查并修正计划里三处与证据不符之处**：① **V4 当时未达成**——`results/` 内没有任何 ANT 解码数字（V4 状态改为"部分"，ANT 侧交步骤 10.7）② §5 步骤 9.5 的"锚点残差 9.6/5.8 ms"**查无实据**（JSON 只有 `start_anchor_seconds` 与 `start_offset_seconds`），已从计划删除并标未验证 ③ §2.3 同步守卫第 4 条（`sync.status == "observed"`）**在本实现中不可能满足**（无时钟拟合），已从守卫表删除并改为如实记录 | 步骤 12 按"每个数字必须能追溯到打开过的文件或跑过的命令"逐条取证，因此发现了计划自身的三处不实；这三处若留着，会让后续读者以为有 ANT 成绩、有锚点残差、有同步守卫 | 保留计划原文（读者会被误导）；或让步骤 12 去实现缺失的机制（越界） | `VALIDATION.md` 按**证据**写而不是按计划写；计划的守卫表、V4 行、9.5 行同步修正 | `1e69f47` |
| D-37 | 09-14 | **真人路径的按通道排除**：给音频链的 `Repair` 安全检查加**按通道排除**通路，让**已声明**的贴轨/死电极不能停机，同时其余通道保持完整保护；被排除电极**仍被记录**（census + 运行记录）。**不采用**"抬高全局阈值直到贴轨通过"的办法 | 步骤 10.7 实测：你的录制里 **F8 全程贴轨（83 333.3 µV）、F3 贴轨 39.8%**，2 个坏电极**否决了整个会话**（26–40 s 停机，`unsafe_endpoints`，决策 0、门控从未开启）。把"坏"放大成"全废"不是安全。抬全局阈值会同时**弄瞎**其余 18 个通道的保护，属于用掩盖换通过 | 抬高 `saturation_limit_uv` 直到通过（牺牲所有通道的检查）；或把那两个电极写死进契约（改契约会作废所有已训模型） | 真人模式能在 18 个可用电极上跑完；坏电极的证据不丢；**契约不变**，所以已训模型仍然有效。若排除后**仍然**无决策，那也是如实结论——ANT 是 CPz 参考、贴轨 2 个、解码器在另一套数据（KU Leuven/Cz/128 Hz）上训练 | 待提交 |
| D-38 | 09-14 | 步骤 10.7 的定位：**live 路径传输已验证，解码被机制挡住**；`RidgeDecoder.validate` 的 `input_sfreq` 不匹配靠**显式、打印、可选**的 500→128 Hz 适配级解决（录制与契约不动） | 模型契约记录训练时的 128 Hz，live 链是 500 Hz；不匹配就拒绝是**正确**行为，不能用改契约来绕过 | 改模型契约里的 `input_sfreq`（作废模型的可追溯性）；或忽略契约（静默错误） | 适配级是**显式的运行期选择**，不是隐式转换；未验证项：真实放大器、真人解码、音频回环 `residual_offset_seconds`（该路径无音频设备）。**新测得一条对齐预算分量**：链自身 `Resampler.startup_delay_seconds = 0.014 s`（128→64 Hz），replay 与 live 两条路径都**从未补偿**它 | `a52ea9b` |
| D-39 | 09-14 | 按通道排除的落地（`8885612`）：`Repair` 与 `QualityMonitor` **本来就是按通道**的，缺的是 live 运行器**没有办法声明**某电极——所以上一步才去抬全局阈值（现已移除，`--saturation-limit-uv` 恢复为 `None`=链自身 75 000）。新增 `--exclude-channels` + **`--allow-unrailed-exclusion` 闸**：`check_excluded_channels()` 要求被排除电极在**已发布窗口**里实测到贴轨（重复极值中位保持 ≥50 % 窗口；实测 F8 100 %、F3 90.9 %）才准排除；排除项经 `RunPolicy` 传递（进 policy 与运行记录，**绝不进 `AuditoryProcessor.contract`**）。**不丢列**：列仍在、损坏时held、仍出现在 census 与 `bad_channels()` | 2/20 个坏电极不该否决整场；抬全局会**弄瞎**其余 18 个通道（有测试：F8 贴轨但未声明、F3 已声明 → 仍以 `unsafe_endpoints` 停机 5 次） | 抬全局阈值；或把电极写死进契约（作废模型） | 真人模式在 18 个可用电极上能跑完；坏电极证据不丢；契约不变 | `8885612` |
| D-40 | 09-14 | **计划修正（重要）**：§3.11「两个贴轨电极导致停机」**属实但不是约束点**——真正的停机原因是**适配级改了数据率却没同步声明 `input_sfreq`**：`Repair` 把 128 Hz 的一步读成 500 Hz 下的 3.9 样本，**每步合成 3 行**，把所有窗口判为 `interpolated`（7608 次幻影修复，26.4 s 停机）。A/B 实证：声明 500 → 27.2 s 停、7 窗全 `interpolated`；声明 128 → **299 窗、0 修复、0 拒绝**。修法：链**声明它实际被喂的速率**，并加 `_PassThroughResampler`（`Resampler` 拒绝相等速率） | 这条比贴轨更隐蔽：阈值全没动，却因为**声明的速率与实际不符**而把健康数据全判为插值；若只盯贴轨，会一直在错的地方修 | 继续怀疑贴轨电极（会持续误诊） | 任何"在链前做重采样"的接线都**必须同时声明速率**——已写成计划级教训，后续步骤与文档都要带 | `8885612` |
| D-41 | 09-14 | **V4 的 ANT 侧仍未达成，且原因是测量到的**：修好速率后 ANT 跑满 130 s、121 窗、0 修复/0 恢复/0 无效、F8/F3 入 census，但**决策 0**（uncertain 495、unavailable 8）。测量到的原因：**20 个电极里 13 个被 `QualityMonitor` 判 `amplitude` 故障**——该录制电平从锚点漂移 1.5–2.0 mV（保留蒙太奇峰值 41.5 mV），而链的偏移上限是 **500 µV**，于是**每个窗口都是 artifact**、`signal_quality` 恒 0、`_verdict` 拒绝提交 | 这是**同类结论的第三次**：质量策略按 KU Leuven 那类数据标定，换到这台设备的录制上把**慢漂移**当成了 artifact。按 §6.4 与 D-35 的口径，先如实登记而不是调参凑数 | 把 500 µV 默认值抬到通过（弄瞎所有通道）；或直接宣布 ANT 不可用而不诊断 | 已派下一个子 agent 把质量阈值做成**可声明、可记录**的运行策略（同 D-37 的模式），并要求**先测漂移统计、再据此声明、且记录默认未变**；若声明后仍无决策，则如实写"V4 的 ANT 侧未达成"并给出新原因 | `8885612` |
| D-43 | 09-14 | **可声明质量阈值落地**（commit `55eb9b8`）：`RunPolicy.amplitude_limit_uv` → `AuditoryProcessor` **同时**交给 `QualityMonitor` 与 `Repair`（一个值一个 owner；链默认 500 µV 不动、契约不加键、policy 字段默认 `None` 而非抄一个 500）。**关键方法学**：阈值必须在**监视器被喂的那个点**测量，而不是原始录制上——原始窗口 13/20 超限、最大 3 637.7 µV；经前置适配级后 **20/20 超限、最大 41 178 µV**（F3/F8 贴轨），差 5.4 倍；首次用原始数字 4 000 µV 声明后**仍有 10 个电极故障**而日志看起来像成功。录制运行声明 **20 000 µV**（高于最大被放行电极的 19 522 µV，低于放大器轨 83 333 µV 的 4.2 倍）。效果：坏通道 census 由 **18 个降到恰好 {F3, F8}** | 用户/计划要的是"能声明、能记录"而不是抬全局；且测量点选错会让声明值看起来有效实则无效（本轮实测过） | 抬全局默认；或按原始窗口的统计声明 | **代价已记录**：在该录制上这个阈值**丧失了选择性**（`signal_quality==1.0` 只意味着"没有移动超过 20 mV"，不再是"这一窗无 artifact"）——这是**录制本身**的问题（它的正常漂移就超过旧阈值），不是数字选错；另两条准则（饱和 75 000 µV、flatline）不受影响且仍然独立生效，census 仍逐电极记录 | `55eb9b8` |
| D-44 | 09-14 | **变异测试第二个陷阱立规**：`.venv` 的 `__editable__.nova2026-0.1.0.pth` 指向**原始 `src`**，因此在树副本里跑子进程**仍会导入未变异模块**——本轮因此把 **5 个变异中的 4 个误报为"存活"**。修法：在子进程里把副本的根与 `src` **前置到 `sys.path`**。另：harness 自身还有一个缺陷使 5 个变异"因错误原因而看起来全部被捕获"（注入的 `unittest.skip` 弄出语法错误），**打印子进程 stderr** 才发现 | 上一轮已立"清 bytecode"规则（D-42），但那只解决缓存、不解决**导入路径**；两者叠加会让变异报告整体失真 | 信任红/绿判定；或只看"是否存活" | **§6.4 追加要求**：变异必须在**隔离副本 + 前置 `sys.path` + 清 `__pycache__` + 打印子进程 stderr** 四个条件下运行；不满足时"存活/捕获"的结论无效 | `55eb9b8` |
| D-45 | 09-14 | **demo 的 `--serve-seconds`/`--browser` 判定为真实缺陷并修复**：`drive()` 在 `finally` 里无条件 `stop.set()`，而留守循环是 `while not stop.is_set()` → **回放结束的同秒关掉 transport**，页面随之死亡。`serve.ps1` 曾用"退出即重启"绕过（代价：换端口 + 一段死页面） | 用户的实际用法就是"点开页面看"，而这个标志位正是为此存在；绕过方案不能算修复 | 接受重启绕过 | 修好后页面在回放结束后继续可访问；**须同时确认回放结束后页面显示的是"已结束/陈旧"而不是把旧值当实时**（铁律 2）；README 中"这些 trial 回放 124 s"一并纠正为实测 389 s | 待提交 |
| D-46 | 09-14 | **移动硬盘交付方案**：`G:\NOVA2026` 同步到 `integration`，并按"只带 demo 所需"复制（模型、包络、S1+S2 的 40 个 trial、素材、ANT 会话与音频、`dist/`）；**训练用的特征缓存（1.6 GB）不拷**，代价是那台机器**能跑 demo、不能训练** | 用户要"另一个设备上跑出 demo"；全量 35 GB 里大部分与 demo 无关 | 整仓复制（多出十几 GB 无用数据） | 盘上现状与缺口写入 `documents/syncing_to_another_machine.md`（Mac 优先） | `3e320dc` |
| D-47 | 09-14 | **包络的 `source_path` 改为仓库相对路径，并修掉解析器**。修前 `parents[1]` 对 `datasets/audio/x.npz` 是 **`datasets/`** 而非仓库根 → 旧回退对**绝对路径是空操作、对仓库相对路径偏一级**，两者都失效；于是目标机器上 SHA256 校验**静默关闭**而 demo 照常启动 | 用户明确该盘要在 **Mac** 上用，绝对路径在那边是 `/Volumes/...`，形状完全不同；静默关闭的抗陈旧校验等于没有 | 在目标机重新生成（盘符变了仍会坏） | 18/18 已按相对路径重生成，**`max|Δenvelope| = 0`、`max|Δtimestamps| = 0`**、`source_sha256` 不变；`--verify` 换工作目录也通过；解析器改为逐级尝试并用"把修复前代码放回去"做变异验证 | `3e320dc` |
| D-48 | 09-14 | **界面改为"看得懂"**：新增决策时间线条带（按后端自己的 decision 着色，含 playhead 与各结论计数）、共享坐标轴的双条（因为决策用的是**差值**）、未平滑的 A−B 差值轨迹与 ±0.05 判定带、以及用文字给出结论与可听后果（"Source B turned down to −6 dB"）；精确数值降为小字凭据行 | 用户原话：**"我不希望结果是只有两个数字在闪"** | 保留裸数字（观众无法判断 0.198 是强还是噪声） | 真实无头 Chromium 验证：Play 点击成功、Web Audio 时钟真实、门控打开、42.93 s 处截图；前端测试 53 → **59 全绿**；`reasons` 字段此前被解码器丢弃，现已透传 | `6358cd8` |
| D-49 | 09-14 | **接通 `eeg_display`，并同时画两条轨迹（原始 + 带通后的）**。此前该包类型是"**契约存在、无人发布**"：`documents/auditory_ui_protocol.md:87` 原文写着 `eeg_display` …「**no** - nothing publishes it today」，前端有解码器、有面板，于是面板永远显示 `Awaiting EEG display data`。用户看到后问"kul 是否看不到 eegdata"。实测确认**数据一直在跑**：`audio_sources` 包的 `input_type = kuleuven_replay`、`simulated = false`、候选为 `part1_track1_dry.wav`/`part1_track2_dry.wav`，试次 `S1/trial_004.npz` 的 EEG 为 `(49792, 64) float64`（64 通道 × 389 s @128 Hz），每 0.25 s 出一次结论——**看不到的是画面，不是数据** | 用户选择"两个都画"（原话选项：接上、两个都画）。理由：原始轨迹回答"电极收到了什么"，带通轨迹回答"**解码器实际用的是哪一段**"——后者才是决策的输入，只画原始信号只能证明"有电"，不能证明"决策从哪来" | ① 只画原始（不能对应决策）② 只画带通（看不出信号被处理过）③ 不接、只在文档里写"不显示" | **包格式由此冻结**（`sample_rate`/`samples`=带通前、`filtered_sample_rate`/`filtered_samples`=带通后、两个单位标签、两个**逐窗**量程提示、`lag_seconds`、`channel_source`）。**两个必须守住的约束**：① `AuditoryProcessor.contract` **一个键都不能加**（加了会作废全部已训模型）；② 原始信号只在 `feed()` 内、`bandpass` 调用**之前**短暂存在，且 `CircularBuffer.push` 返回的是复用存储的视图——**保存引用会被静默改写**，必须显式拷贝且只拷显示通道 | 待提交 |
| D-50 | 09-14 | **浏览器证据的复核结论：两份运行记录质量不同，必须分清，且两条轨迹必须同区间**。① `results/demo_browser_run.json`（09:06，端口 51231）**不足以支持"可见演示被验证"**：`position_advanced=false`、`position_at_last_sample=0`、71 个采样点里 `<audio>.currentTime` **恒为 0**、`paused` 恒为 true，另有 **3 次 `POST /api/media/control` 409**（服务端答"Media command rejected; stop and prepare again"）——即那次音频**在无头浏览器里从未真正播放**，恰好与用户先前"没有声音"的现象同因。② `results/attune_story_browser_run.json`（09:09，端口 57126）才是合格证据：`position_advanced=true`、`bad_responses=0`、`focus_values=["Unavailable","Source A","Source B"]`，配套截图 `attune_story_shot_01_..._55s.png` 读出 `Media Time 00:57.02 / 01:20.00`、大字 `Source A`、`Source B turned down to -6 dB`、时间线计数 `A 118 / B 35 / Uncertain 17 / No data 11`、差值 `+0.3019` 对判定带 `±0.05`。**结论：只有 ② 能作为"看得见"的证据，① 只能作为失败模式的证据**。③ 截图同时暴露一个待解释瑕疵：**`REJECTED PACKETS: 3`**（前端校验拒绝了 3 个包）。④ 两条 EEG 轨迹**必须覆盖同一时间区间**——带通窗口取最后 5 s 而原始只取最新 chunk（0.25 s）会让长度差 20 倍且时间轴不可比，等于让面板的核心主张变成假的；已据此拦下实现并要求测试钉死"两条点数相同、跨度相同" | 用户此前报告的"没有声音/有声音"正是 ① 与 ② 的差别；把 ① 当作成功证据会让交付建立在假前提上。④ 是"两条轨迹可比"这一主张成立的前提，不是风格问题 | ① 把 09:06 那份当作通过（会掩盖真实的音频未播放缺陷）② 接受两轨迹不同区间（面板会说谎）③ 只报"测试全绿"而不看 `position_advanced` | **新增硬规则**：凡引用浏览器证据，必须同时报 `position_advanced`、`bad_responses`、`<audio>.currentTime` 三个字段，缺一不可；`ever_paused_after_click=true` 本身不算失败（回放结束会重置），但 `position_advanced=false` 必须当作失败处理。`REJECTED PACKETS: 3` 记为**未解释项**，需单独追因 | 本次核查 |
| D-51 | 09-14 | **`.gitignore` 自相矛盾之处已修**：`results/antneuro_live_*.jsonl` 这一条否定规则放行了 **4 个原始包流共 2.6 MB / 6883 包**（`declared` 763 KB、`excluded_probe` 228 KB、`run` 838 KB、`split` 787 KB），而同文件 172 行相邻注释自己写着「The packet stream stays out of git under results/ **alongside every other packet stream**」，且该文件开头 results/ 段落的规则是"**运行记录**入库（含测量值与结论词）、**包流**是派生物应留在 `output/`"。规则与自己的注释直接冲突，故删除该否定规则 | 仓库对"什么算可引用的证据"有明确而一致的口径，包流不属于它；留着会让 Mac 那台机器多拉 2.6 MB 无用原始流，并给后来者一个错误的先例 | ① 保留规则、承认包流入库（与全仓口径冲突）② 改动已跟踪文件（需 `git rm`，属破坏性操作，须用户批准） | **已跟踪**的 2 个包流 `antneuro_live_20260913_packets.jsonl`（133 KB）与 `antneuro_live_excluded_20260913_packets.jsonl`（816 KB）**保持不动**——它们已被提交、任何文档都未引用它们，删它们属破坏性操作且收益很小，故**列为待用户决定项**而非自行处理。新增的 3 项证据否定规则（`attune_story_*` 运行记录与截图、`demo_serve_mutation.json`）另计 | 本次核查 |
| D-52 | 09-14 | **发现并登记"媒体控制槽"竞态——它是用户"没有声音"的真正根因，且当前 demo 的成败取决于谁先抢到槽**。链：`apps/attune-ui/src/mediaController.js:63-72` 的 `play()` **先 `await send('prepare')`，成功后才 `await element.play()`**；而 `send()` 在 `rest.mediaControl` 收到非 2xx 时抛错，于是 409 让 `play()` 直接走 `catch → fail()`，**`element.play()` 永不执行**。服务端槽位是粘性的：`src/nova2026/transport/media.py:256-259` 只要 `self.client_id is not None and client != self.client_id` 就 `ValueError`（→ HTTP 409），且 `client_id` **只在 `stop()`（L167）里清空**，而 `stop()` 只由 `bind()` 在 **session_id 变化**时调用（L149-155）。`demo.py::drive` 又**无条件**创建 `SimulatedMediaClient(client_id="demo-runner")`（L283-288）并在 `client.configure` 后立即开始报告（L300-304）。**因此浏览器与模拟客户端争同一个独占槽，先 prepare 者独占全程。** 实测对照：`results/demo_browser_run.json`（09:06，端口 51231）浏览器**抢输**——仅 3 次 `POST /api/media/control` 且**全 409**、71 个采样点里 `currentTime` 恒 0、`paused` 恒 true；`results/attune_story_browser_run.json`（09:09，端口 57126）浏览器**抢赢**——`media_control_posts=318`、`bad_responses=0`、`position_advanced=true`。**用户在 62245 上"最后有声音"属于抢赢的那一类**，其页面数据可自证槽位属于浏览器：`media.payback_state=playing`、`sync_status=observed`、`attention.media_time_s` 达 142.25 s | 用户先前报告的"没有声音"就是抢输；而"能听见"被误当作"demo 可用"，实际它**依赖竞态**。不写清这一点，后续任何人都无法解释"同一份代码为何一次响一次不响"，也会把 09:06 的成功误判为环境问题（`9382f54e` 已正确地把它从"无头假象"升级为真实缺陷） | ① 把它当作无头浏览器伪影（会掩盖真实缺陷）② 直接改前端让页面"抢槽"（服务端逻辑禁止：`client != self.client_id` 一律拒，页面**无法**抢占，只能靠对方先 `stop`）③ 现在就去改（根因受两个在跑 agent 的 `demo.py` 争用，且未验证修复方向） | **未修，原因已记录**：修复方向只有两条，且都需先脱离竞态——(a) 给 `demo.py` 加"页面自己当媒体拥有者"的模式（不启动 `SimulatedMediaClient`，代价是**门控依赖真人真的在播放**），(b) 让演示**先**让页面 prepare、模拟客户端仅在无浏览器时补位。两者都要改 `demo.py`（`f411b9b7` 正在改它）故延后。**对交付的影响必须如实写明**：`live_demo_two_commands.md` 与 `where_the_demo_stands.md` 都要写出这条竞态，并给出"页面抢输时页面自己会显示 *Playback synchronization unavailable. Stop, then Play to reconnect.*"这一可观察症状（`mediaController.js:14`） | 本次核查 |
| D-53 | 09-14 | **`REJECTED PACKETS: n` 是假警报，根因在客户端的连接握手而非数据丢失**。截图 `results/attune_story_shot_01_gate_should_be_open_at_55s.png` 上那条红色告警（当时 n=3）不是丢包。链：`src/nova2026/transport/server.py:88-111` **故意**向每个新 socket 重放保留快照（为关闭订阅竞态，这是**正确**的服务端行为）；`apps/attune-ui/src/transport.js:30-33` **先**应用 REST `/api/state` 快照并据此抬高 `state.sequence`；`state.js:10` 又拒绝一切 `sequence <= state.sequence` 的包。于是**服务端自己的握手掌势必然被计成"被拒"**。用前端自己的模块实测（真实快照）：从空闲态应用一次 → `rejected=0`；REST 先、socket 后（客户端的真实顺序）→ `rejected=8`。被丢的包是**已经通过 REST 应用过的完全重复**，所以页面其余部分看起来正常 | 这条红色告警对操作者说"Measurements may be missing or stale"，而实际什么都没丢——**假警报比没有告警更糟**，它会训练使用者忽略真警报。同时它长期冒充"真实缺陷"，使 09:09 那次合格运行被误读为有瑕疵 | ① 直接删掉 `state.js:10` 的守卫（会同时放过**流中**真正的重复/乱序包，多条既有测试依赖它）② 改服务端不重放（会重新打开它专门关闭的订阅竞态）③ 判为"设计如此"不管（继续误导） | **须在传输层做记账**，不动 `state.js` 的全局不变量：连接期的快照重放按"状态替换"处理而非事件流。**遗留证据缺口**：截图里的 `3` 已**不可复现**——浏览器驱动从不记录 `state.rejected`，所以 3 这个具体数字无法从运行记录反查（`60bccc72` 明确指出）。修复前必须补上驱动对该字段的记录。另记：`HEAD /api/media/file` 返回 **404** 而同路径的 range GET 返回 206（FastAPI 的 `@app.get` 只注册 GET），**影响为零**（浏览器与 `<audio>` 用 range GET），仅登记 | 本次核查 |
| D-54 | 09-14 | **`serve.ps1` 会把 demo 的渲染音频写成 `demo_stereo.wav`——一个「跑启动器就会覆盖用户正在听的文件」的地雷**。`serve.ps1` 的 `New-DemoArgs` 只传 `--trial --model --out --stream-out`（外加 `--seconds`/`--margin`/`--serve-seconds`/`--open-browser`），**从不传 `--media-out`**，于是 `demo.py` 用它自己的默认值 `DEFAULT_MEDIA_OUT = output/auditory_ui/demo_stereo.wav`（`demo.py:101`）。后果有两层：① 跑 `serve.ps1` 会重渲染并覆盖那个文件，若它**正被浏览器串流播放**，播放会读到半截文件；② 本会话早些时候我正是因为发现这个风险，才命令后端 agent 必须自传 `--media-out`——**同一条地雷在启动器里一直没被处理**。独立理由：两个 demo 实例共用同一 `--media-out` 会互相覆盖，而 `duration` 是页面从该文件读的，换文件等于把正在服务的会话重新调音 | 这是用户会直接踩到的类别（`serve.ps1` 就是无设备 demo 的推荐入口），触发条件是「正常使用」而非误操作 | ① 只依赖「使用者记得自传 `--media-out`」（与默认值冲突，必然有人踩）② 改 `demo.py` 的默认值（牵动所有既有脚本与运行记录的既有口径，代价大于收益） | 修法：给 `serve.ps1` 自己的 `--media-out`（每次运行独立命名，`-Smoke` 用固定名），与 `serve.sh` 已做的保持一致；**同批把 `--render-out`/`--media-record`/`--gate-out` 也改成每次运行独立命名**（并发 demo 交错写证据文件会让运行记录失去证据效力）。本条同时纠正我两个转述错的前提：**本机 `bash` 可用**（WSL bash 5.2.21），故 `serve.sh` 是真通过 `bash -n` 而非类比验证；`apps/attune-ui` 的 npm 基线是 **70 测试 / 67 通过 / 3 失败**（3 个均为本机无 `python3`，属既有失败），不是我先前记的 56/7 | 待提交 |
| D-55 | 09-14 | **D-52 的媒体槽竞态已修**（commit `f8c6fb8`）：新增 `StandbyMediaClient`——替身**在发出任何命令之前**就读服务端自己的记录（`MediaTimeline.claimed`，一个新加的只读探针，首次 `prepare` 被接受后在本会话内**保持为真**）来决定谁该拿槽，于是**输的一方从不下达会被拒的命令**：没有 409、没有 `fail()`、没有沉默的死播放器。它不是猜时序，读的是服务端「握手已完成」这一事实，窗口是显式的、会写进运行记录。两种模式是**声明**而非推断：无人值守时给页面一个有界窗口后接管；自己开了浏览器时**无期限等待**（那里页面不是可能性而是事实）。实测日志：`media owner: standby (unattended: the page is offered the slot, then the stand-in takes it)` 与 `media slot: demo owns it (no controller claimed the slot within 10s)`。5 个测试跑在**真实协议**而非 mock 上；2 个变异都被捕获（去掉仲裁 → 1 fail + 2 error；让 `claimed` 不粘 → 2 fail + 2 error） | 用户最初的抱怨「没声音」就是抢输；而抢赢/抢输由延迟决定（替身在会话启动瞬间就 prepare，页面要等人点击），**demo 的成败因此取决于竞态**，这不是一个 demo 该有的属性 | ①「等 3 秒」式启发（是对人类行为的猜测，人点慢了就复现同一 bug）② 让页面去抢槽（服务端逻辑禁止：`client != self.client_id` 一律拒，页面**无法**抢占）③ 继续不修（用户的原始诉求未达成） | 演示从「赌竞态」变为确定性：有人看时页面拥有槽，无人值守时替身拥有。**仍未验证**：真实浏览器里点 Play 的实际结果（本机没有跑有头浏览器去确认），故 `live_demo_two_commands.md` 与 Mac 指引里仍把「没声音」列为需要报告的可观察症状 | `f8c6fb8` |
| D-56 | 09-14 | **变异脚本的行动名必须读、不能猜——本会话因此把两个变异留在了工作树里**。`mutate_media_ownership.py` 的复原动作叫 **`revert`**，我按习惯传了 `undo`；脚本对未知动作的处理是**打印用法并返回 2**（`main()` 末尾的兜底分支），于是一次 `apply race` 之后两次 `undo` 都**静默地什么也没做**，而 `apply` 已经把变异写进了 `demo.py` 与 `transport/media.py`。是**我自己重跑测试**才发现的（红签名 `failures=2, errors=2` 与 `probe` 变异完全一致）。修法：用正确动作 `revert` 复原，并确认三处都没有 `MUTATION` 标记、测试回绿 | 这是本仓库**第四个**变异测试陷阱，且性质不同：D-42 是字节码缓存、D-44 是 `.pth` 导入路径、`0462240a` 报的是 CRLF/LF 导致查找落空——**这三个都会让「变异没生效」看起来像「变异被抓」**；而这次相反，是**变异生效了却没被复原**，会把被篡改的代码提交出去 | ① 相信脚本的退出码（`apply` 返回 0，`undo` 也返回 2 但被我忽略）② 只跑一次测试看红（看到红就以为「变异被抓」，不会想到它还在） | **§6.4 追加要求**：变异脚本的**动作名必须从脚本自身读出**（或先跑无参看用法），复原后必须**独立复核**——`git status` 对被跟踪文件是最强的检查（本次 `demo.py` 复原后 git 不再报告它有改动，即逐字节精确），未跟踪文件则需查标记串 + 重跑测试确认回绿。**不得**仅凭脚本自报 sha256 就认定已复原 | 本次核查 |
| D-57 | 09-14 | **我把一个「正处于变异状态」的文件提交了出去**——这是 D-56 的镜像错误，而且是我的操作失误，不是工具的。经过：`11296d42` 为履行 §6.4 义务做变异 M1（把 Edit 2 还原成旧形式、跑测试、再从 `$TEMP` 备份逐字节复原）。**我的 `git add` 恰好落在那几秒的窗口里**，于是 `a415482` 提交的是**被变异的那一行**：`missing = [key for key in GATE_KEYS if key not in incoming or key not in model]`（错误：拒绝两边都缺的键，正是这次工作要消除的 ANT 失败模式），而不是约定的 `if (key in incoming) != (key in model)`。**该提交是红的**：那 3 个测试在变异形式下必然失败（agent 的 M1 从另一方向证明了同一件事）。已由 `8f8fe28` 修正（一行），并核实 blob 内容与 agent 声称已验证的磁盘哈希 `2429F7EC…AC1F` **归一化后完全一致**（盘上 CRLF 334 / blob 里 LF 334，纯行尾差异） | 根因不是「读得不够仔细」——我在 09:58:30 **确实读到的是正确版本**，是**读完到提交之间**那个 agent 把它改掉了。**只要还有别的进程在写同一个文件，提交前的单次读取就不构成检查。**而且错误方向最坏：它把「被篡改的代码」推到了远端，Mac 一拉就撞 3 个失败 | ① 相信「我刚才看过」② 只依赖 `git status`（当时的 `git diff` 里那一行看起来就是 agent 的正常改动，**看不出它是变异**）③ 等所有 agent 停手再提交（会无限期阻塞交付） | **硬要求（§6.4 再追加一条）**：提交**之后**必须核验**已提交的 blob**（`git show HEAD:<path>`），而不是提交**之前**看工作区文件。本次正是这样发现的：工作区与 HEAD 差一行，`git show HEAD:…` 打出了错的那一行。更强的形式是把「读到的内容」与「提交的内容」放在**同一条命令**里比较。另记一条相关事实：`decoder.py` 在工作树里是 CRLF（334 个）、入库 blob 是 LF，所以**行尾归一化后的哈希**才是可比对象，直接比原始 sha256 会误判为「内容不一致」 | `8f8fe28` |
| D-58 | 09-14 | **Mac 上跑 demo 时绝不能加 `--no-browser`——它会静默地把页面变成抢输的一方**。链：`scripts/auditory_ui/serve.sh` 在**未**给 `--no-browser` 时会追加 `--open-browser`（L237），而 `demo.py:172-173` 在 `--open-browser` 时把媒体归属解析为 **`page`**（页面独占、不赌）；反之解析为 **`standby`**（`demo.py:174`），页面只拿到 **10 秒**（`DEFAULT_STANDBY_SECONDS`）窗口，超时后替身拿走槽位，**页面再也播不出声音**。而「等 URL 打印 → 启动浏览器 → 点 Play」很容易超过 10 秒 | 用户的最终交付就是「Mac 上跑出 demo」；而 `--no-browser` 看起来是个无害的省事参数，**实际把唯一能保证页面拥有槽位的路径关掉了**。这个陷阱之所以危险，是因为它不会报错、不会警告，只表现为「没声音」，与最初那个竞态症状**一模一样**，会让人以为是修复失败  | ① 在两个启动器里加 `--media-owner` 透传（本次不做，见下条）② 把 `auto` 默认改成 `page`（会让无人值守/无头验收路径失去衰减，且 `--browser` 并不等于「有人在看」——该 agent 已实测此点）③ 在 Mac 指引里保留 `--no-browser` 并靠加大 `--standby-seconds` 弥补（`serve.sh` 也不转发这个参数）  | `documents/mac_setup_prompt.md` 的运行命令改为**不带** `--no-browser`，并写明这个参数为什么不能加；同时给出一条**绕过启动器**的直接命令（带 `--media-owner page`），因为 `serve.sh` 不转发该参数。**同时修掉 `serve.sh` 的一句假注释**：它原称「Anything after the flags is passed straight through to the demo」并给出 `./serve.sh --seconds 30 --margin 0.05 --eeg-display-channel Cz` 作示例，**而那条示例从来跑不通**——参数循环以 `*) fail "unknown option: $1"` 收尾，且 `set --` 会从零重建命令行、丢弃调用者的参数。该承诺被**删除而非实现**：一个静默转发自己看不懂的参数的启动器，就不再是启动器了 | 待提交 |
| D-59 | 09-14 | **收到一份外部独立复核（`NOVA2026_integration_Independent_Verification.pdf`），判决：本分支在被复核的那个问题上成立，但有两处实质问题。** 它把仓库在云端干净克隆、自己重跑、自己核对数字。**核心结论（背书）**：在 KU Leuven 上 16 人、留出故事、14 528 个决策窗口，平衡准确率 **0.6195**；**关键在少数类**——B 类召回 0.6272 > A 类 0.6117（占 20% 的少数类反而更准），靠先验作弊的模型不可能做到这点；16 折标准差 0.0295，高出随机约 15 个标准差。**弱，但真实。** 对比 `audio_flo`（自测 100% / 换数据 38.46%，低于随机），这是**不同性质的结果**。它还独立确认了审计文档预言的那个泄漏陷阱**真的被躲开了**（`TrialRef.group` 取自 `metadata['group_key']`，`rep_` 已折叠，`assert_group_disjoint` 正面强制）。**问题一（我已修）**：`a415482` 是回归——它宣称「两个修正都做了」，实际发出去的那一行仍是旧的 `key not in incoming or key not in model`，3 个原本通过的测试因此红。**我已在 `8f8fe28` 修好**（外部复核的快照正好拍在修复前 5 分钟）。**问题二（未解决）**：契约那半确实修好了（它单独验证过：宣告 Cz 的模型现在接受宣告 CPz 的窗口，差异记为 provenance 而非拒绝），**但录音上仍是 0 决策**——失败从「116/116 被拒」变成「121 窗打了分、0 个下结论」。**这是另一个失败，不是解决了。** | 问题一暴露了一个流程缺陷，值得单独记：**新增的 14 个契约测试里，没有任何一个是「两边都缺同一个键」的情形**。外部复核原话：「The one correction that was not implemented is the one the new tests do not cover.」——**测试覆盖的缺口恰好就是漏掉的那个修正**。问题二则是把「契约修好了」误当成「录音能用了」；报告写在那次运行之前，所以不能怪它，但读 §5.1 就以为障碍只剩参考电极的人是错的 | ① 把复核当作「项目整体通过验收」（它自己声明**没碰任何音频硬件、放大器、被试**，物理时序这个最重要的未测项它一点没验证，浏览器竞态也没跑——它验收的只是**代码与数字那一半**）② 继续纠结合约（已经修好了，该追的是那 0 个决策）③ 把 0.875 和 0.620 并排引用（见下） | **这次要落地的四件事**：① 给 `test_contract_split` 补「两边都缺」的用例（**最重要**——它正是漏掉的那个修正的守卫）② 把依赖缺失产物的测试改成 **skip 而非 fail**（干净克隆常态就有 11 个红，真回归会被淹没——外部复核认为这就是问题一溜过去的原因）③ 修正 3.2 节的人群口径 ④ 登记 `a415482` 那句不成立的验证声明。**关于 ③**：窗口曲线 0.637→0.875 **只测了 S1–S4 四个被试**（源文件原话「a documented quarter of the corpus」），而 0.620 用全部 16 人；同一个 5 秒配置是 0.637 对 0.6195。源文件是诚实的，**是我在摘要里丢了这句前提**，两个数不能并排读。**关于 ④**：`a415482` 的提交信息写着「61 tests … OK」，在干净检出不成立；历史不改写，改为在此登记该声明无效。**外部复核给出的下一步**：修那一行 + 补用例；纠正提交记录；**追 0 决策而不是契约**（建议先在这段录音自己的分数分布上重新标定 margin）；训 60 秒模型；把 3.2 的人群写出来；缺产物的测试改为 skip；**F8/F3 拿去查硬件接触**（两套独立方法一致认定它们坏了，占 20 通道里的 2 个） | 待提交 |

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
6. **主 agent 可以读子 agent 的报告与产物**，并据此判断是否可信；**认为不对时，派一个新的子 agent 去处理**，
   而不是自己动手修正（§9 第 1 条）。
5. **同一问题最多 3 个子 agent**（用户 2026-09-14 要求，防止无限套娃）：
   第 1 个失败 → 派第 2 个（附诊断）；第 2 个失败 → 派第 3 个（附两次诊断）；
   **第 3 个仍失败 → 停下向用户报告**，并写明三次尝试各自的做法与失败点，由用户决定改方向还是放弃该项。
   计数规则：以「同一交付物」为单位，正常完成后追加的小修不算新问题。
6. **并行派发时必须检查文件集不相交**（D-18 的教训）：两个子 agent 不得同时改同一个文件；
   `pyproject.toml`、`scripts/run_tests.py`、`.gitignore` 等共享文件一次只派一个。

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
| 2026-09-14 | v1.8：步骤 4 **DONE**（commit `48c412d`）；新增 §3.13 扰动用例矩阵（1 项正常 + 15 项注入故障）与 §3.14 传入 EEG 的对抗性用例，并写入验收 V9；新增 §8 决策记录（D-01…D-15，含主 agent 自己的错误判断与被否方案）与 §9 主 agent 纪律（不写代码/不写测试、可读报告但纠错须派子 agent、同一问题最多 3 个子 agent）；**修正主 agent 关于 `source_unit_exponent` 的错误判断**（µV 的正确值就是上游默认的 `-6`，不是 `0`） | 主 agent |
| 2026-09-14 | v1.9：步骤 5、6、7 **DONE**（`1a6bc3c` / `7f2ab43` / `61ca57c`）；新增 §3.15 呈现方式目标变更（两耳同混合 + 回退选项 + D-21 两种模式分文件夹）、§3.16 扰动矩阵（原 3.13 改号）、**§3.17 步骤 5 暴露的三条跨步硬约束**；新增 §6.4「测试必须自造前提且能被变异证伪」；决策记录扩到 **D-22**（含 D-18 提交污染、D-19 空洞断言、D-20 呈现路线、D-21 模式分目录、D-22 全仓整理）；antio 测试修复（`fb0e9ed`）；ANT 测试集报告经独立核验后修正（`d212c03`），**证伪主 agent 三处写法**（`REF:CPz` 已写入头部、0.0833 V 是贴轨通道、误触标记算术错） | 主 agent |
| 2026-09-14 | v2.0：**D-22 全仓大整理 DONE**（`4f4a9e5`/`32ee369`/`c95d075`）：删除死路线与两处 legacy（26 文件 / −4384 行，判据"已跟踪 + 无 live 引用"），文档统一为 `<topic>_<kind>[_<date>].md` 并重写根 README（七类脚本用法 + 明确"不声称什么"），派生物 119 路径重命名（映射表入库）；删后 **641 + 53 全绿**；主 agent 独立复跑 streaming+auditory 477 tests 全绿。目标轮次上限由 40 提升至 300 并把 V1–V9 写进目标描述 | 主 agent |
