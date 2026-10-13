# LiveSoundFix

面向 **A7C2 等相机在中小型 Livehouse 单点录音**的本地后处理工具：修复削波/底噪/交流声、重建中低频与高频、抑制混响、扩展声场、母带响度达标。

全部算法用**纯 numpy + 标准库**实现，零云端、零外网依赖、无 scipy / torch。

| | |
|---|---|
| 运行形态 | 本地 HTTP 服务 + 单页 UI（`http://127.0.0.1:8765/`） |
| 依赖 | `numpy`（唯一硬依赖）；可选 `ffmpeg`（读 mp3/m4a/mov/mp4 时用） |
| 实测规模 | 8.4 分钟 / 48k / 立体声素材整链约 **149 s**（≈295 ms/秒音频） |
| 处理链 | 9 段固定顺序，逐段可开关、可调干湿比、可调音台推子 |
| 许可证 | [MIT](LICENSE) |

> 工程名 `LiveFix`，发布名 `LiveSoundFix`，两者指同一个项目。

---

## 快速开始

```powershell
git clone https://github.com/Naiyalatuotip/LiveSoundFix.git
cd LiveSoundFix
python run.py                 # 依赖 numpy；浏览器自动打开 http://127.0.0.1:8765/
```

本机开发用的解释器（无 scipy / 无 pytest 的精简运行时）：

```powershell
& 'C:\Users\24379\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe' run.py
```

流程：**选文件 → 诊断 → 调处理链 → 预览 A/B → 导出**。

- 非 WAV（mp3/m4a/flac/mov/mp4）需要本机 `ffmpeg` 在 PATH 中，否则请先转成 WAV。
- `run.py --port 8765 --no-browser` 可改端口 / 不自动开浏览器。

---

## 三区 UI

界面按用户草图分为三区（`grid: minmax(0,1.5fr) | minmax(0,1fr)`）：

| 区 | 位置 | 内容 | 延迟 |
|---|---|---|---|
| 区1 主要功能 | 左列（滚动） | 文件诊断、进度条 + ETA、9 段处理链、干湿比、预览 A/B、导出、AI 接口 | 数秒 ~ 数分钟（走 `/api/job` 异步作业） |
| 区2 动态曲线 | 右上 | Web Audio `AnalyserNode` 实时频谱 + 波形带（rAF 刷新） | 0（跟随播放） |
| 区3 实时处理 | 右下 | 拖拽 EQ（8× `BiquadFilterNode`，即时发声）+ 调音台推子 | EQ 0 ms；推子混音 ≈30 ms |

---

## 即时 A/B：两轨同播，切换只调音量

**A/B 对比不用「换整轨 `src` + seek」。** 原因很实际：145 MB 的整轨换 `src` 后浏览器只缓冲开头很小一段，此时给 `currentTime` 赋值会被**静默夹回 0**，表现就是「一切换就从头播放」。

现在的做法（两区音轨并行）：

```
auMain（处理后）──► gMain ─┐
                          ├─► busIn ─► 实时 EQ ─► Analyser ─► 输出
auOrig（原片）  ──► gOrig ─┘
```

- **两条完整音轨同时播放**，切换 = 两路增益的 6 ms 交叉 → **零延迟、无缝、无换源、无 seek**。
- `auMain` 仍是真正的 transport：原生播放/暂停/拖进度/音量全部照常，原片轨每帧与它对齐（偏差 > 35 ms 就拉回）。
- 路由进同一个 `busIn`，所以**实时 EQ 与频谱分析看到的就是你在听的那一条**。
- 原片轨只在进入对比时才开始解码，退出即暂停 —— 常听状态不会白白多解一份音频。

服务端配套：`POST /api/clips` 可一次切出「当前时刻起 N 秒」**双侧**小片段（约 0.8 MB，30 ms 内返回，两端各 5 ms 淡入淡出，两个文件用常驻 `ProcessPoolExecutor` 并行落盘）。当前 A/B 主路径已不需要它，保留给「改参数即时试听 3 秒」这一后续用途。

---

## 处理链（固定顺序，逐段可旁通）

`sep → ai → m5 → m1 → m2 → m3 → m4 → m6 → dyneq`

| 模块 | 作用 | 当前算法 |
|---|---|---|
| **sep** 源分离 | 置于所有后处理之前，分级处理人声/低频/打击 | HPSS 滑动中值（时间核 25 帧 / 频率核 31 bin）；`nmf` 为秩 4 固定先验字典投影；四路掩码逐 bin 归一使**和恒为 1**（全增益恒等） |
| **ai** 重建 | 低频/人声 AI 重建 | **接口预留**，`process` 透传并在 stats 标记 `interface_only` |
| **m5** 修复层 | 去削波 / 交流声 / 底噪 / 泵动 | Hermite 削波重建；市电陷波 Q=30 且**按陷波半带宽自适应换算 bin 数**（只做 1、2 次谐波）；最小值统计谱减（Martin 2001 思路）；反向 AGC 泵动修正（±9 dB） |
| **m1** 中低频重建 | 提密度、不提浑 | **静态 + 自参考双层频段增益**：`weight` 静态 +9 dB@30-150Hz 叠自适应动态；`density` 静态 +6 dB@40-250Hz 叠最多 +12 dB 向上压缩；`punch` 用**自适应通量阈值**（本轨 p55）最多 +9 dB；`tame` 驻波下压最多 −12 dB；`sub` 用**相位减半（θ/2）生成真次谐波** + <70Hz 低架 |
| **m2** 中高频补充 | 补信息、不抬噪 | `air` 用 **dB 目标制**倒谱包络外推（拉满 = ext_hz 以上带内能量 +18 dB，交叉淡化在 `ext_hz` **以下** 1/3 八度，扩展上界收在 18 kHz）；`detail` 用**相位加倍（2θ）生成真二次谐波** + 4-8 kHz 高架；`dehiss` 压制**贴近噪声底**的 bin（最多 −24 dB） |
| **m3** 人声优化 | 去掩蔽 / 去混响 / 去刺 / 齿音 | 四个子处理器全部改成**自参考分位激活**（clarity p55→p92、harsh p75、ess p85，保证任何素材上参数都有权威度）；room 为谱域晚期混响统计抑制（前视窗最小值）；ess 用「一帧到位 + hold + release」专门咬短脉冲 |
| **m4** 声场扩展 | 宽度 / 纵深 / 旋转 | 分频段 M/S 宽度曲线（幅度域、保相位）；纵深用 **M 的全通去相关分量注入 S**（一级全通级联，幅度平坦不染色）+ **(M,S) 联合功率归一**（总电平不变、S/M 比真实上升）；M/S 旋转矩阵 |
| **m6** 母带响度 | 响度达标、真峰限制 | BS.1770-4 K 加权与门控；LRA glue 沿分位数扫参考点取最接近目标的曲线；4× 过采样前瞻真峰限制器（5 ms 前瞻 / 1 ms 网格 / attack 1 ms / release 80 ms） |
| **dyneq** 动态 EQ | 最后一步精修 | RBJ peaking 静态曲线（精确）+ 软边频段掩码近似动态曲线；`static` / `up` / `down` 三模式 |

分级路由（`sep.on` 且 `routing=graded`）：**m1 只作用于 backing，m3 只作用于 vocal，m2 作用于两路后合回**，m5 按轨执行。

---

## 冻结默认参数（用户签字版 · 2026-10-14）

这是**每次打开应用时的初始参数**。它不是「自动校准出来的最优」，而是用户在 UI 里实听调好后冻结的一版：

```text
sep   : on  hpss  graded  vocal/low/perc=1.0  other_gain=0.6
ai    : off
m5    : on  denoise=0.00  floor_db=-60  hum=0.20   mains=50  declip=0.00  pump=0.00
m1    : on  weight=0.20  density=0.25  sub=0.15   punch=0.50  tame=0
m2    : on  air=0.45     detail=0.50   dehiss=0.10 ext_hz=15100
m3    : on  clarity=0.35 room=0.50     harsh=0.00  ess=0.00
m4    : on  width_low=0.56 width_mid=1.75 width_high=1.75 depth=0.40 rotation=0
m6    : OFF target_lufs=-16 lra=6 ceiling=-1.0
dyneq : off bands[0].thr_db=-30
export: channels=stereo  bit_depth=24  loudnorm=true
所有模块 mix = 100
```

改动入口：`livefix/engine.py` 的 `DEFAULT_CONFIG`。改它 = 改所有用户每次打开看到的初始参数。

> ⚠ `tests/test_engine.py` 里有一张 `FROZEN` 表逐项断言（47 项）+ 校验 `mix`/`export`，**误改会立刻测试失败**。要改默认值请连同那张表一起改，并同步 `spec/API.md §4.3`。

**关于 `m6 = OFF`**：处理链内不做母带响度归一（实测 60 s livehouse 素材输出 −20.5 LUFS，仅比输入高 0.8 dB）。两点要注意：

1. **导出仍会归一** —— `exporter` 按 `export.loudnorm=true` 独立做响度归一与真峰保护，「关掉 m6」≠「导出不归一」。
2. **链上就没有限制器了** —— m6 之前还兼着真峰限制器。峰值很热的素材可能过 0 dBFS（见下方「已知问题」里的削顶说明）。

---

## 参数权威度：每个效果器 ≥ ±3 dB

要求：**任何效果器参数拉满，在它的目标频带至少产生 3 dB 的实际变化。** 真实 60 s livehouse 素材实测（`_range_audit.py`）：

| 参数（拉满） | 目标带 | 总能量Δ | 峰值Δ |
|---|---|---|---|
| m1.weight / density / sub / punch | 30-150 / 40-250 / 20-90 / 100-700 Hz | **+11.3 / +7.7 / +5.4** / +6.2 | +15.5 / +20.1 / +12.8 / **+9.0** |
| m1.tame | 40-100 Hz | −0.6 | **−4.4** |
| m2.air / detail / dehiss | 11-18k / 1.4-7k / 6-16k | **+7.0** / **+3.5** / −1.1 | +23.1 / +8.2 / **−18.4** |
| m3.clarity / room / harsh / ess | 2.2-5k / 200-16k / 2.2-5.5k / 5.8-13k | **+5.1** / **−5.6** / −0.2 / −0.5 | +8.8 / −5.9 / **−6.6** / **−8.7** |
| m4.width_low/mid/high · depth | 分段侧信号 / 侧信号 | **+11~12** · **+7.1** | +13~17 · +25.3 |
| m5.denoise / hum / pump / declip | 3-8k / 45-55 / 200-8k / 重建误差 | **−15.4** / **−20.7** / +0.2 / **+11.6 dB 改善** | −17.2 / −24.8 / **+9.0** / — |
| m6.target_lufs / ceiling | 全带 | **+9.5** / **+11.2** | +13.0 / +17.8 |
| sep.other_gain / low_gain | 5-16k / 30-150 Hz | **+8.2** / **+19.8** | +15.0 / +38.9 |
| dyneq.gain_db / range_db(down) | 2-4.5 kHz | **+17.3** / +1.4 | +20.9 / **+4.0** |

- 「总能量Δ」= 目标带总能量比（**静态/准静态**参数看这一列）；「峰值Δ」= 逐帧最大带能量差（**动态**参数看这一列 —— 齿音/刺耳/泵动只在部分帧动作）。
- 唯一例外 **`m6.lra`**：它是**目标型参数（单位 LU，不是 dB 增益）**，不适用 ±3 dB 判据。实测它精确追踪目标（合成 10.93 LU 素材上 `lra=1` → 1.54 LU，Δ−9.39 LU）；低动态素材上可达变化受素材自身 LRA 限制（真实 livehouse 素材 LRA 3.98 LU → 最多压到 1.49 LU）。

---

## 调音台与干湿比

**调音台**：引擎渲染时捕获每段贡献量 `delta = out − in`（float16），满足望远镜恒等

```
原片 + Σ 贡献量_k == 处理后        实测最大偏差 ≤ 3e-4（float16 量化级）
```

推子只缩放贡献量：`混音 = 原片 + Σ 增益_k × 贡献量_k`（`POST /api/remix`，约 30 ms，**不重跑处理链**）。100% = 正常渲染；0% = 该模块效果移除（精确回到原片）；>100% = 放大该模块贡献。捕获上限 150 s 渲染时长（内存控制），超长素材用「快速窗口」渲染即可启用。

**干湿比 `mix`**（0..100，每模块独立，默认 100）与模块自身参数**分离**：引擎在段边界做 `out = (1-mix)·in + mix·proc(in)`；`sep` 特例是混合四个 stem 增益（`g_eff = 1-mix+mix·g`），`mix=0` 时 stems 与输出严格一致且完全跳过计算。改动即触发重算，支持「完整 / 快速 90 秒」两种范围，快速结果标记 `windowed=true`。

---

## 反复处理同一文件不再重读

调参数意味着同一文件要跑很多次，所以有三层缓存：

| 层 | 键 | 内容 | 生效范围 |
|---|---|---|---|
| **解码缓存** | `(绝对路径, mtime_ns, size)` | `(float32 (n,ch), sr)` | 进程内 LRU，上限 1.5 GB |
| **输入统计缓存** | 解码键 + 窗口 | `lufs_in / corr_in / spec_before` | 进程内，8 条 |
| **转码缓存** | `sha1(绝对路径\|mtime\|size)[:10]` | ffmpeg 转出的 PCM float wav | **落盘，跨重启有效** |

- 键里带 `mtime/size`，**源文件被替换会自然失效**。
- 实测（60 s 素材）：第 1 次读盘 213 ms + 输入统计约 3.9 s；第 2 次起 `source_cached / stats_cached` 均为 `true`，**这两项开销归零**（总耗时约 −13%）。
- 非 WAV 的价值更大：转码缓存名曾是 `abs(hash(path))`，而 Python 的 str hash 默认随机化 —— **每次重启都会把同一个视频重新转码一遍**；现在名字稳定且已存在就直接复用。
- 处理结果里会带 `load: {source_cached, stats_cached, last_read_ms}`，可直接确认本次有没有重读。

### 缓存清理（UI「7 缓存」区 + `/api/cache`）

`cache/` 里全是**可再生**中间产物，按类统计与清理（实测这套机制把 10.08 GB 回收到 1.66 GB）：

| 类别 | 命名 | 默认可删 |
|---|---|---|
| `preview` | `p_<hex>.wav`（处理后预览 + 即时 A/B 片段） | ✅ |
| `original` | `o_<hex>.wav`（原片转成的 wav） | ✅ |
| `stems` | `stems_job_<hex>.npz`（贡献量） | ✅ |
| `transcode` | `<源文件名>_<10位hex>.wav`（ffmpeg 转码结果） | ✅ |
| `other` | 测试素材等非本工具产物 | ❌ 默认不动 |

- **保护集永不删除**：内存中驻留的音频、在跑/最近作业引用的文件、以及**每类最新的 8 个**；删除前还会二次校验「解析后的父目录必须就是 `cache/`」，不越界。
- 清理参数：`max_mb`（默认 2048）、`max_age_days`（可选）、`keep_recent`（默认 8）、`include_other`（默认 false）。**`dry_run` 默认 true** —— 不显式传 `false` 就只报告不删。UI 上「扫描」= dry-run（并弹确认），「清理」才真删。
- **自动回收**：启动时按默认上限回收一次；每次作业完成后若占用超过 3 GB 再回收一次。清理失败不影响作业。
- **内存音频缓存也有界**：LRU，上限 1 GB 且至少保留最近 6 条；淘汰只丢内存引用（磁盘 wav 还在），后续读取自动回退到读盘。

---

## HTTP 接口

以下是**关键接口摘要**，完整契约（JSON 形状、配置对象、参数语义）见 [`spec/API.md`](spec/API.md)。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 单页 UI |
| GET | `/api/state` | 版本、**默认配置**、可选模型、链路顺序、解码缓存状态 |
| GET | `/api/models` · `/api/cache` | 模型列表 · cache/ 分类占用（含「保护集外可清理量」） |
| POST | `/api/list` | 目录浏览（仅音频扩展名） |
| POST | `/api/analyze` | 诊断：LUFS/LRA/峰值/削波占比/相关度/噪底/高频截止/市电/倍频程 + 问题列表 + 参数建议 |
| POST | `/api/process` | 同步处理（兼容旧客户端） |
| POST | `/api/job` → GET `/api/job?id=` | 异步作业：`frac` / `stage` / `stage_frac` / `eta_s`，每 300-500 ms 轮询 |
| GET | `/api/audio?id=` | WAV 字节流。**支持 HTTP Range（206 / 416 / `Accept-Ranges`）与 HEAD**，按块流式发送 |
| POST | `/api/remix` | 调音台快速混音（不重跑处理链） |
| POST | `/api/clips` | 即时 A/B 双侧片段（3 s，约 0.8 MB，多核并行落盘） |
| POST | `/api/cache` | `{"action":"info"\|"clear"\|"clean", ...}` —— `clean` 清理磁盘产物（默认 `dry_run=true` 只看不删） |
| POST | `/api/export` | 导出：声道布局 / 位深 / 响度归一 / EQ 烘焙 / 调音台混音 |

`/api/audio` 的 **Range 支持是硬要求**：没有它，浏览器换 `<audio>.src` 后无法定位，拖进度条与 A/B 都会失效（旧版遗漏这一点，正是「切换后从头播放」的根因）。

**进度与 ETA**：成本模型 = 每段 ms/秒音频先验 + 完成后 EWMA 自更新，再按实际速率缩放；分块阶段细到每 60 s 块，统计阶段细到 5 个子步骤。

---

## 导出

- 声道：**stereo / mono / 5.1**（BS.775 被动上混，分频在 STFT 幅度域完成）。
- 位深：**16 / 24 / 32float**。
- 可选 LUFS 归一（默认目标 −16）；真峰保证回环（目标 ≤ −1.05 dBTP）。
- 导出 = 渲染结果 + **实时 EQ 烘焙**（RBJ 复频响级联 + overlap-save 精确滤波）+ **调音台混音**，一次完成。

---

## 实测性能与内存

**整链耗时**（60 s / 48 k / 立体声，冻结默认参数）：

| 段 | sep | m5 | m1 | m2 | m3 | m4 | m6 | 合计 |
|---|---|---|---|---|---|---|---|---|
| ms | 3610 | 1266 | 1176 | 4342 | 1988 | 3431 | 关 | **≈15.9 s**（265 ms/秒音频） |

**整轨实测**（503.5 s 即 8.4 分钟素材）：

| 指标 | 值 |
|---|---|
| 处理耗时 | **148.7 s**（295 ms/秒音频） |
| 进程峰值工作集 | **6031 MB ≈ 5.89 GB** |
| 60 s 素材对照 | sep 单独 748 MB / 整链 1098 MB |

内存随时长近似线性增长，**30 分钟立体声 48 k 需要约 20 GB**（详见「已知问题」）。

---

## 关于「用真实基准校准」（这些素材不在仓库里）

校准用的三份真实录音属于**训练/评判数据，不随仓库分发**：

| 文件 | 采样率 | 时长 | 含义 |
|---|---|---|---|
| `源1.wav` | 48 k | 503.5 s | A7C2 直出（待修复） |
| `调整1.wav` | 96 k | 502.2 s | 人工修好 **+ 外接麦叠加** |
| `音源前.wav` | 192 k | 264.1 s | 源曲（保真参照） |

评判器本体在 [`livefix/bench.py`](livefix/bench.py)（`spec_db` / `rms_dist` / `lufs_match` /
`profile` / `score`）。原先 `origin/` 下还有一层 `judge_chain.py` 适配器（提供
`load` / `spec96` / `rmsdist` / `to48`）供 `scan_*.py` 调用，**该文件已丢失**，
因此那批扫描脚本目前不可运行；按 `bench.py` 的接口重写一个约 20 行的适配器即可恢复。

历史结论（当时脚本可用）：不处理 4.536 → 旧默认 8.172（**比不处理更糟**）→ 校准后 3.912。
⚠ 该判据把每条谱各自归一化，**对响度与峰值完全不可见**，且目标文件混入了外接麦（内容不同），
所以任何真实修复都倾向于被判为「偏离」—— 这正是早期默认值一路偏向保守、大量参数被关成 0 的原因，
也是后来改用「用户实听冻结版 + 参数权威度 ≥ ±3 dB」这套双重标准的由来。

---

## 测试

```powershell
# 逐个直接跑（17 个脚本，全部自带 PASS/FAIL 输出与退出码）
Get-ChildItem tests\test_*.py | ForEach-Object { python -X utf8 $_.FullName }
```

| 脚本 | 覆盖 |
|---|---|
| `test_kernel` | core / measure / wavio 冒烟、转码缓存命名稳定性 |
| `test_engine` | 全链集成 + **冻结默认参数 47 项断言** |
| `test_m1`…`test_m6` · `test_sep` · `test_dyneq` | 单模块：恒等、无 NaN、长度/声道不变、分段安全、性能 |
| `test_mix_progress` | 干湿比语义 + 进度回调单调性 |
| `test_mixer_eq` | 望远镜恒等、remix 恒等式、EQ 烘焙、导出路径 |
| `test_cache` | 缓存分类统计 / 容量与天数上限 / 保护集 / 不越界 / 内存 LRU（全部在临时目录上做，不碰真实 cache） |
| `test_exporter` | 多声道 / 位深 / 响度归一 |
| `test_true_peak` | 真峰数值 |
| `test_e2e` · `test_server_e2e` | 真实 server 端到端（40+ 项，全走 HTTP） |

共 **17 个脚本**，全部 exit=0。

> 这些是**独立脚本**而非 unittest/pytest 用例（模块级 `print` + `sys.exit`），所以 `python -m unittest discover` 会报 ImportError，这是预期行为；环境里也没有 pytest。

---

## 目录结构

```text
run.py              入口（启动服务器 + 打开浏览器）
server.py           HTTP 服务（仅标准库；Range/HEAD、/api/clips、/api/cache）
web/index.html      单页 UI（零外网依赖，全部内联）
livefix/            内核
  core.py           STFT/IFFT、RBJ 双二阶、零相位滤波、分段交叉淡化
  measure.py        BS.1770-4 响度/LRA、真峰、频谱、相关度
  wavio.py          WAV 读写（PCM 8/16/24/32 + float32/64）、ffmpeg 转码缓存
  engine.py         处理链装配、分级路由、贡献量捕获、**DEFAULT_CONFIG**
  api.py            server 的唯一业务依赖：作业/进度/缓存/remix/clips/导出
  analysis.py       诊断与参数建议
  m1…m6 sep dyneq   9 个功能模块（各自唯一公开接口 process / info / KEY）
  exporter.py       多声道 / 位深 / 响度归一导出
spec/API.md         HTTP/JSON 与配置契约（含参数权威度、冻结默认值、缓存契约与清理规则）
spec/CORE.md        内核与模块契约
tests/              17 个自测脚本
origin/             真实基准素材 + 校准脚本（judge_chain.py 缺失，见上）
AUDIT.md            完整算法与工程审计报告（含实测证据）
cache/              运行时产物：预览/原片 WAV、贡献量 npz、转码与片段（可自动回收，不要提交）
```

---

## 已知问题 / 待办

按严重度排列，完整版见 [`AUDIT.md`](AUDIT.md)：

1. **真峰（True Peak）实现有误** —— `measure._true_peak_env` / `m6._env_oversampled` 用「频谱镜像 + 基带 ×2」而非零填充插值。最坏情形信号（采样点全在 ±0.7071、真实真峰 0.00 dBTP）实测报 **−3.01 dBTP**，低估 3 dB。因此「真峰安全导出」在极端素材上不可靠（实测用 ceiling −1.0 导出后真实真峰可到 +0.83 dBTP）。**未修**。修法：改标准零填充或 4× 多相 FIR，并取 `max(TP, sample peak)`。
2. **长素材内存** —— 8.4 分钟实测峰值 **5.89 GB**，30 分钟约 20 GB。根因：`engine._BLOCKED = {"m6","sep"}` 让这两个最重的模块整轨单次执行（sep 全片 STFT + 4 声道一次 istft；m6 4× 过采样包络 + 整段 K 加权 FFT）。
3. ~~**`cache/` 无回收**~~ —— **已修**：新增分类统计 + 按容量/天数清理 + 保护集（内存驻留 / 在跑作业 / 每类最新 8 个），启动与作业结束后自动回收；内存字典 `_audio` 也从无上限改为 1 GB LRU。实测把 10.08 GB 回收到 1.66 GB。UI「7 缓存」区可查看与手动清理。
4. ~~**无限制器时的削顶**~~ —— **已修**：预览/中转/片段缓存改为「样本超 ±1 时自动用 32 bit float」，不再被 24 位定点 `np.clip` 削掉（那会同时破坏调音台恒等式）。导出位深仍由用户选择决定。
5. **诊断层不可靠** —— 市电检测阈值过低（实测 50 Hz 族 +7.0 dB、60 Hz 族 +15.3 dB **双双超阈**，易被音乐内容误判）；高频截止判据在门限附近抖动（明明 8k→16k 掉 16.6 dB 却不建议做 BWE）。
6. **契约与实现漂移** —— `m1.tame` / m3 的 8 个扩展参数未写进 `spec/API.md §4`；`spec/CORE.md` 说 m5/m1/m2/m3 由 `block_apply` 包裹，实际 `sep`/`m6` 是例外。
7. **未实现** —— AI 重建模型与训练程序（接口已留）；WPE 全复杂度去混响（现为统计近似）；m2 的空气感合成仍用随机相位（输出依赖分段粒度）。
8. **m4 加宽的单声道代价** —— 宽度拉满时相关性 0.92 → 0.68，折叠成单声道会有相位抵消损失。`stats.m4.mono_fold_db` 已上报该损失，但 UI 尚未显示。

---

## 算法参考

HPSS（Fitzgerald 2010）、NMF（Lee-Seung 2000）、最小值统计噪声估计（Martin 2001）、谱减法（Boll 1979）、WPE 思路的统计简化（Nakatani 2010）、缺失基频心理声学、RBJ 双二阶滤波器、Linkwitz-Riley 分频思路、ITU-R BS.1770-4 响度、EBU R128 / True Peak、BS.775 5.1 上混矩阵、M/S 声场处理（Fairlight 思路）。

---

## 发布到 GitHub 前须知

`cache/`（9.6 GB）、`origin/*.wav`（约 690 MB）与 `__pycache__/` **都不应提交**。仓库根目录至少需要：

```gitignore
# 运行时产物（预览/原片 WAV、贡献量 npz、转码与 A/B 片段）
cache/
__pycache__/
*.pyc

# 真实素材与派生产物（体积大且可能涉版权/隐私）
origin/*.wav
origin/*.json

# 本机审计探针（可选保留）
_audit_probe*.py
_*_probe.py
```

当前 `LiveSoundFix/` 仓库只有一次 Initial commit（含 `.gitattributes`），工作区尚无代码 —— 需要把 `run.py` / `server.py` / `web/` / `livefix/` / `spec/` / `tests/` / `README.md` / `.gitignore` 复制进去后提交。

---

## 许可证

[MIT](LICENSE) © 2026 Naiyalatuotip（见 `LICENSE`）。

第三方说明：唯一运行时依赖是 **NumPy**（BSD-3-Clause）。可选的外部程序是 **ffmpeg**（LGPL/GPL，仅在你读取 mp3/m4a/mov/mp4 时被调用，未链接进本项目）。
算法均为按公开论文自行实现，**未包含任何第三方项目源码**；审计中提到的备选方案
`rajmic/declipping2020_codes`（GPL-3.0）与 `facebookresearch/denoiser`（CC-BY-NC-4.0）
仅作参考，代码未被采用。
