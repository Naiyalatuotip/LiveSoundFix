# LiveFix 接口契约 v1（server.py / web/index.html 必须严格遵守）

## 0. 顶层目录

```
E:\作品集\音频修复v1\
  run.py                # 入口：启动本地服务器并打开浏览器（主代理写）
  server.py             # HTTP 服务（子代理 A 写）
  web/index.html        # 单页 UI（子代理 A 写；全部内联，禁止外网 CDN/字体/库）
  livefix/              # DSP 内核包
    core.py wavio.py measure.py engine.py api.py analysis.py ai.py   # 主代理写，只读
    m1.py m2.py m3.py m4.py m5.py m6.py sep.py dyneq.py exporter.py  # 各功能子代理写
  livefix_models/       # AI 模型 *.npz 存放处（本期不实现模型，仅留接口）
  cache/                # 处理缓存 wav（server 创建）
  spec/API.md  spec/CORE.md
```

## 1. server.py 对 livefix.api 的 Python 调用（唯一业务依赖）

```python
from livefix import api     # 必须在请求处理函数内部延迟 import（内核可能仍在开发中）

api.state()      # -> {"version","config","models":["none"],"chain":[9个key，固定顺序]}
api.list_dir(p)  # -> {"dir","parent","dirs":[{"name","path"}],"files":[{"name","path"}]}
api.analyze(p)   # -> 见 §3；失败抛异常
api.process(path, cfg, window=None)   # 同步处理（测试/旧客户端）；见 §3
api.start_job(path, cfg, window=None) # -> {"job": "job_xxx"}，后台线程执行
api.job_status(job_id)                # -> {"job","state","frac","stage","stage_frac","eta_s","error","result"}
                                      #    state: "running"|"done"|"error"；done 时 result = §3 process 形状
                                      #    frac 0..1 单调；eta_s 剩余秒数（估算，可能为 null）
                                      #    window: [start_s, dur_s] 快速试听窗口，None=完整文件
api.remix(job_id, gains)              # 调音台快速混音 -> {"id","url"}；gains={"sep":1.2,...}（缺省 1.0）
                                      #   语义 out = 原片 + Σ g_k·贡献量_k；不重跑引擎，约 0.3-0.8s
                                      #   前提：该作业渲染时长 ≤150s（已捕获贡献量），否则抛错
api.export_wav(audio_id, channels, bit_depth, loudnorm, out_path,
               eq_bands=None, gains=None, job_id=None)
                 # -> {"ok":True,"path","channels","duration"}
                 #   gains+job_id 给定时先按调音台混音再导出（替代 audio_id 路径）
                 #   eq_bands=[{"f","q","gain_db"}]：导出时烘焙实时 EQ（RBJ peaking 级联）
api.models()     # -> {"models":["none", ...livefix_models/*.npz 文件名]}
```

- **进度与 ETA**：`start_job` 内部以 `engine.process(..., progress_cb=...)` 获取总进度；
  ETA 基于成本模型（每段 ms/秒音频的先验 + 完成后 EWMA 自更新）与实际耗时速率缩放。
- **快速试听窗口**：`window=[start_s, dur_s]` 时只处理该片段（result.windowed=true），
  用于干湿比实时调整的快速迭代；导出前 UI 应确认是否需要完整渲染。

- 业务异常一律包成 HTTP 200 + `{"ok":false,"error":"..."}`（服务器自身崩溃除外）。
- 用 `http.server.ThreadingHTTPServer`；每个请求一个线程。
- 启动时创建 `cache/`（路径用 `Path(__file__).parent` 定位，与进程 cwd 无关）。
- **开发期钩子**：若环境变量 `LIVEFIX_FAKE_API=<某个.py的绝对路径>` 存在，则用 importlib 加载该文件代替 `livefix.api`（仅用于内核未就绪时联调 UI 管线；生产不设置）。

## 2. HTTP 端点

| 方法 | 路径 | 请求 | 响应 |
|---|---|---|---|
| GET | `/` | - | web/index.html |
| GET | `/api/state` | - | api.state()（含 `cache` 解码缓存状态） |
| GET | `/api/models` | - | api.models() |
| GET | `/api/cache` | - | cache/ 分类占用 + `reclaimable`（实际可清理量）+ 内存/解码缓存状态 |
| POST | `/api/cache` | `{"action":"info"\|"clear"\|"clean", ...}` | `info` 同 GET；`clear` 清内存解码/统计缓存；`clean` 清理磁盘产物（见 §1.2） |
| POST | `/api/list` | `{"dir":string}` | api.list_dir(dir) |
| POST | `/api/analyze` | `{"path":string}` | `{"ok":true, ...api.analyze(path)}` |
| POST | `/api/process` | `{"path":string,"config":object,"window":[s,dur]\|null}` | `{"ok":true, ...api.process(path,config,window)}`（同步，兼容旧客户端） |
| POST | `/api/job` | `{"path":string,"config":object,"window":[s,dur]\|null}` | `{"ok":true,"job":"job_xxx"}`（立即返回，异步） |
| GET | `/api/job?id=job_xxx` | - | `{"ok":true,...api.job_status(id)}`；UI 每 300-500ms 轮询直到 state≠running |
| GET | `/api/audio?id=xxx` | - | audio/wav 字节；id 必须匹配 `^[po]_[a-z0-9]{4,16}$`，否则 400 |
| POST | `/api/remix` | `{"job":"job_xxx","gains":{"sep":1.0,...}}` | `{"ok":true,"id","url"}`；无贡献量时 `{"ok":false,"error":...}` |
| POST | `/api/clips` | `{"id":p_xxx,"orig_id":o_xxx,"start":69.0,"dur":3.0,"job"?,"gains"?}` | `{"ok":true,"start","dur","sr","samples","parallel","a":{id,url},"b":{id,url}}`（即时 A/B 双侧片段） |
| POST | `/api/export` | `{"id","channels","bit_depth","loudnorm","out_path","eq_bands"?, "gains"?, "job"?}` | api.export_wav(...)；带 gains+job 时按调音台混音后导出 |

- POST 体 UTF-8 JSON；GET 参数用 urllib.parse 解析。
- `/api/audio` 只从 `cache/` 目录取文件（按 id 白名单，防目录穿越）。
- **`/api/audio` 必须支持 HTTP Range**：解析 `Range: bytes=s-e|s-|-N`，命中返回 `206` +
  `Content-Range: bytes s-e/size`，不可满足返回 `416` + `Content-Range: bytes */size`，
  所有响应带 `Accept-Ranges: bytes`；未带 Range 时返回 `200` 全文件。**必须按块流式发送**
  （不要 `read_bytes()` 把整文件读进内存）。必须实现 `HEAD`（只回头、不回体），否则拿到 501。
  理由：浏览器换 `<audio>.src` 时只缓冲很小一段，服务端忽略 Range 会让
  `audio.currentTime = t` 被静默夹回 0 —— 即「A/B 切换后从头播放」。
  这条是 §5「A/B 切换与调音台换 src 时都保持播放位置」的硬前提。

### `/api/clips` · 即时 A/B（零延迟切换）

`{"id":p_xxx, "orig_id":o_xxx, "start":秒, "dur":秒≤10, "job"?:job_xxx, "gains"?:{...}}`
→ `{"ok":true, "start", "dur", "sr", "samples", "parallel":bool, "a":{id,url}, "b":{id,url}}`

- `a` = 处理后（给了 `gains`+`job` 时按调音台混音结果切）、`b` = 原片；两端各 5 ms 淡入淡出。
- **两侧写盘用常驻 `ProcessPoolExecutor` 并行（多核）**，池不可用时自动串行回退并在
  `parallel` 上报 false。片段保留最近 24 个（LRU 自动裁剪），避免 cache 膨胀。
- 为什么需要它：整轨 145 MB，浏览器换 src 只能缓冲很小一段，seek 后 `currentTime`
  会被夹回 0。切成「当前时刻起 3 秒」（约 0.8 MB）后，前端可以**两路同时连续播放**
  并把切换做成增益交叉 —— 零延迟、零 seek、样本级对齐，还能一路连续播下去
  （前端每窗口剩 1.2 s 时预取下一段并按 AudioContext 绝对时刻排程衔接）。
- 片段来自**已渲染缓存**，不重跑 DSP；因此这里的多核只加速「切+写盘」，
  单次约 30 ms（首次含进程池启动约 0.6 s）。真正需要多核的是「改参数后即时重渲染
  3 秒片段」，属于后续可扩展项。

## 3. JSON 形状

**analyze（`{"ok":true, ...以下...}`）：**
```json
{"path":"...","sr":48000,"channels":2,"duration":532.1,
 "diagnosis":{"lufs":-19.4,"lra":9.2,"peak_db":-0.2,"clipped_pct":0.31,
   "corr":0.92,"noise_floor_db":-58.0,"cutoff_hz":11200,"hum_hz":50,
   "bands":[{"f":"63","db":-18.2}],
   "issues":[{"level":"info|warn|bad","text":"..."}],
   "suggest":{"m1":{"weight":0.7},"m4":{"width_mid":1.3}}}}
```
`bands` 为 11 个倍频程 63Hz..16kHz；`suggest` 是可直接合并进 config.modules 的增量。

**process：**
```json
{"ok":true,"id":"p_ab12cd","orig_id":"o_ab12cd","url":"/api/audio?id=p_ab12cd","orig_url":"/api/audio?id=o_ab12cd",
 "sr":48000,"channels":2,"duration":532.1,
 "stats":{"lufs_in":-19.4,"lufs_out":-16.0,"peak_db":-1.1,"true_peak_db":-0.8,
   "corr_in":0.92,"corr_out":0.41,
   "stages":[{"key":"sep","on":false,"ms":0},{"key":"ai","on":false,"ms":0}, ...9项，链顺序]},
 "spec_before":[96个数字],"spec_after":[96个数字],
 "mixer":{"available":true,"keys":["sep","m5","m1","m2","m3","m4","m6"],"job":"job_xxx"}}
```
- `stats[]` 每项含 `mix`（干湿比 0..100）。`mixer` 为调音台可用性：`available=true` 时
  `keys` 是有贡献量推子的模块；渲染时长 >150s 时 `available=false` 且带 `reason`，
  UI 应禁用调音台并提示改用快速窗口。
- **`load`**（新增）：`{"source_cached","stats_cached","last_read_ms"}` —— 本次处理是否
  复用了「解码缓存 / 输入统计缓存」。同一文件重复处理时二者都会是 `true`，
  此时**不再读磁盘、也不重算输入侧统计**（详见 §1.1）。
- `spec_*`：96 个对数等距频点（30Hz..16kHz）整段平均幅度 dB，各自归一化到自身最大值 = 0。
- `id`/`orig_id` 每次 process 重新生成（8 位小写字母数字）；处理后 wav 与「原片转成的 wav」都写入 cache/，供 `/api/audio` 播放。
- server 把处理结果 `(audio_ndarray, sr)` 缓存在内存字典 `{id: (audio, sr)}` 供 export 使用（同时写 cache 才能播放）。

### §1.1 反复处理同一文件不再重读

调参数意味着同一个文件要跑很多次，所以有两层缓存：

| 层 | 键 | 内容 | 生效范围 |
|---|---|---|---|
| **解码缓存** | `(绝对路径, mtime_ns, size)` | `(float32 (n,ch), sr)` | 进程内，LRU，上限 1.5 GB（约 4 份 8 分钟立体声 48k） |
| **输入统计缓存** | 解码键 + `(window_start, window_len)` | `lufs_in / corr_in / spec_before` | 进程内，最多 8 条 |
| **转码缓存** | `sha1(绝对路径\|mtime\|size)[:10]` | ffmpeg 转出的 PCM float wav | **落盘、跨重启有效** |

- 键里带 `mtime/size`，**源文件被替换会自然失效**，不会读到旧内容。
- 输入数组全程只读（各模块都是 `X = stft(x)` / `x.copy()` 的用法），多个作业可安全共享同一份数组。
- 实测（60s 素材，默认链）：第 1 次读盘 213 ms、输入统计约 3.9 s；第 2 次起
  `source_cached=true / stats_cached=true`，**这两项开销归零**（总耗时约 −13%）。
- 非 WAV（相机 `.mp4/.mov`、`.mp3`）的价值更大：旧实现用 `abs(hash(path))` 命名，
  而 Python 的 str hash 默认随机化，**每次重启服务都会把同一个文件重新转码一遍**；
  现在名字稳定且已存在就直接复用。
- 注意：**m6 的响度归一与速度无关的部分不缓存**（`lufs_out / true_peak / spec_after`
  依赖输出，每次都必须重算）。

### §1.2 cache/ 产物清理

`cache/` 里全是**可再生**中间产物，按类统计与清理：

| 类别 | 命名 | 默认可删 |
|---|---|---|
| `preview` | `p_<hex>.wav`（处理后预览 + 即时 A/B 片段） | ✅ |
| `original` | `o_<hex>.wav`（原片转成的 wav） | ✅ |
| `stems` | `stems_job_<hex>.npz`（贡献量） | ✅ |
| `transcode` | `<源文件名>_<10位hex>.wav`（ffmpeg 转码结果） | ✅ |
| `other` | 其它（测试素材等非本工具产物） | ❌ 默认不动，需显式 `include_other=true` |

`POST /api/cache {"action":"clean", ...}` 参数：

| 参数 | 默认 | 含义 |
|---|---|---|
| `dry_run` | **`true`** | 只报告不删除（`deleted=0`）。**必须显式传 false 才真删** |
| `max_mb` | `2048` | 目标容量上限；从最旧的开始删到达标 |
| `max_age_days` | `null` | 只删比它更旧的（与 `max_mb` 可叠加，先按天数后按容量） |
| `keep_recent` | `8` | **每个类别**至少保留最新的 N 个 |
| `include_other` | `false` | 是否连「其它」类别一起清 |

**保护集（永不删除）**：内存中驻留的音频 id（`_audio`）、在跑与最近作业引用的
`_orig_id` / `_stems_file` / `result.id`、以及每类最新 `keep_recent` 个。
删除时还会二次校验「解析后的父目录必须就是 `cache/`」，不会越界。

返回：`{ok, dry_run, candidates, deleted, freed_bytes, per_group, protected, remain_bytes, params}`。

`GET /api/cache` 的 `reclaimable` 字段给出**保护集之外**的真实可清理量与文件数
（`cache_info` 与 `cache_clean` 共用同一套保护规则，避免「可清理量」虚高）。

**自动回收**：启动时按默认上限（2 GB）回收一次；每次作业完成后，若占用超过
`auto_trigger_bytes`（3 GB）再回收一次。清理失败不会影响作业。

**内存音频缓存同样有界**：`_audio` 是 LRU，上限 1 GB 且至少保留最近 6 条 —— 淘汰
只丢内存引用（磁盘 wav 还在），后续读取自动回退到读盘，不会失败。

## 4. 配置对象（client 内存持有，process 时整体上送）

链顺序固定：`sep → ai → m5 → m1 → m2 → m3 → m4 → m6 → dyneq`（dyneq 是最后一步；导出只做峰值安全保护）。

```json
{"modules":{
 "sep":{"on":true,"mix":100,"mode":"hpss","routing":"graded","vocal_gain":1.0,"low_gain":1.0,"perc_gain":1.0,"other_gain":0.6},
 "ai":{"on":false,"mix":100,"model":"none","focus":"low","strength":0.7},
 "m5":{"on":true,"mix":100,"denoise":0.0,"floor_db":-60,"hum":0.2,"mains":50,"declip":0.0,"pump":0.0},
 "m1":{"on":true,"mix":100,"weight":0.2,"density":0.25,"sub":0.15,"punch":0.5,"tame":0.0,"tame_f":63.0,"tame_q":2.0},
 "m2":{"on":true,"mix":100,"air":0.45,"detail":0.5,"dehiss":0.1,"ext_hz":15100},
 "m3":{"on":true,"mix":100,"clarity":0.35,"room":0.5,"harsh":0.0,"ess":0.0},
 "m4":{"on":true,"mix":100,"width_low":0.56,"width_mid":1.75,"width_high":1.75,"depth":0.4,"rotation":0.0},
 "m6":{"on":false,"mix":100,"target_lufs":-16,"lra":6,"ceiling":-1.0},
 "dyneq":{"on":false,"mix":100,"bands":[{"f":3000,"q":1.0,"gain_db":0,"thr_db":-30,"ratio":3.0,"range_db":6,"mode":"down"}]}}},
 "export":{"channels":"stereo","bit_depth":24,"loudnorm":true}}
```
> 上面就是**当前冻结的默认值**（用户签字版，见 §4.3）。`GET /api/state` 返回的
> `config` 即此对象；客户端每次打开都应从它取值，**不要缓存上次改过的配置**。

- **每个模块都带 `"mix"` 干湿比（0..100，默认 100）**：独立于模块自身参数，由引擎在段边界
  线性混合 `out=(1-mix)·in + mix·proc(in)`；sep 特例（混合四个 stem 增益，g_eff=1-mix+mix·g）；
  mix=0 等效旁通（引擎跳过计算）。`mix` 不传给模块（引擎剥掉 "on" 和 "mix" 再传参）。
- `dyneq.bands[]`：`mode` ∈ `static` | `up`（低于阈值时向上补）| `down`（高于阈值时向下压）；最多 8 点；`gain_db` 静态基础增益。
- `sep.mode`：`hpss`（快）| `nmf`（准）。`routing`：`graded`（分级）| `mix`（分离后立即合回）。分级语义见 spec/CORE.md §6。
- `ai`：**本期为接口预留**——`model` 只有 `none` 可选，process 时透传音频并在 `stats.stages` 里标记 `"note":"interface_only"`。UI 卡片需显示「接口已预留 · 模型待实现」，focus/strength 控件禁用。
- `m5.mains`：50 或 60（市电频率）。

### §4.1 参数权威度（2026-10-14 实测校准 · 每个效果器 ≥ ±3 dB）

**要求：任何效果器的参数拉满，在它的目标频带至少产生 3 dB 的实际变化。**
下表是真实 60s livehouse 素材上的实测值（`_range_audit.py`）：
- 「总能量Δ」= 目标带总能量比（静态/准静态参数看它）
- 「峰值Δ」= 逐帧最大带能量差（动态参数看它 —— 齿音/刺耳/泵动只在部分帧动作）

| 模块.参数 | 目标带 | 总能量Δ | 峰值Δ |
|---|---|---|---|
| m1.weight | 30-150Hz | **+11.3** | +15.5 |
| m1.density | 40-250Hz | **+7.7** | +20.1 |
| m1.sub | 20-90Hz | **+5.4** | +12.8 |
| m1.punch | 100-700Hz | +6.2 | **+9.0** |
| m1.tame | 40-100Hz | −0.6 | **−4.4** |
| m2.air | 11-18kHz | **+7.0** | +23.1 |
| m2.detail | 1.4-7kHz | **+3.5** | +8.2 |
| m2.dehiss | 6-16kHz | −1.1 | **−18.4** |
| m3.clarity | 2.2-5kHz | **+5.1** | +8.8 |
| m3.room | 200-16kHz | **−5.6** | −5.9 |
| m3.harsh | 2.2-5.5kHz | −0.2 | **−6.6** |
| m3.ess | 5.8-13kHz | −0.5 | **−8.7** |
| m4.width_low/mid/high | 分段侧信号 | **+11~12** | +13~17 |
| m4.depth | 侧信号 | **+7.1** | +25.3 |
| m5.denoise | 3-8kHz | **−15.4** | −17.2 |
| m5.hum | 45-55Hz | **−20.7** | −24.8 |
| m5.pump | 200-8kHz | +0.2 | **+9.0** |
| m5.declip | 重建误差 | **+11.6 dB 改善** | — |
| m6.target_lufs | 全带 | **+9.5** | +13.0 |
| m6.ceiling | 全带 | **+11.2** | +17.8 |
| sep.other_gain | 5-16kHz | **+8.2** | +15.0 |
| sep.low_gain | 30-150Hz | **+19.8** | +38.9 |
| dyneq.gain_db | 2-4.5kHz | **+17.3** | +20.9 |
| dyneq.range_db(down) | 2-4.5kHz | +1.4 | **+4.0** |

**唯一例外 `m6.lra`**：它是**目标型参数（单位 LU，不是 dB 增益）**，不适用 ±3 dB 判据。
实测它精确追踪目标（合成 10.93 LU 素材：`lra=1` → 1.54 LU，Δ−9.39 LU；
`lra=6` → 6.45 LU）。在低动态素材上可达变化受素材自身 LRA 限制
（真实 livehouse 素材 LRA 3.98 LU → 最多降到 1.49 LU）。

### §4.2 为此修改的算法（与旧版行为不同，UI 文案需相应更新）

| 模块 | 旧行为 | 新行为 |
|---|---|---|
| m1 | 阈值是绝对 dBFS（REF_W_DB=−34 / REF_D_DB=−30）；本素材带内中位 −30.2 dBFS 故只有 22% 帧触发，`weight` 拉满只动 0.79dB | 改成「**静态增益 + 相对本轨分位数的动态分量**」两段式：`weight` 拉满 = 30-150Hz **静态 +9dB** 再叠加自适应动态；`density` 静态 +6dB + 最多 +12dB 向上压缩；`punch` 阈值改自适应分位；`sub` 低架 +9dB 且**相位减半**生成真次谐波（旧版复制相位=假谐波）；`tame` 上限 12dB 且默认改 0（实测净损失） |
| m2 | `air` 合成 4-8k 的 8% 能量，与素材高频存量耦合，拉满只动 0.48dB；斜边放在 ext_hz **以上**导致实际从 13.9k 才满量 | `air` 改「**dB 目标制**」：拉满 = ext_hz 以上带内能量 **+18dB**（相对该带自身，带 4-8k 的 −40dB 下限）；交叉淡化移到 ext_hz **以下** 1/3 八度，`ext_hz` 才真的是「从哪开始补」；扩展上界收在 18kHz（旧版一路铺到 24k，能量大半听不见）；`detail` 改**相位加倍**（真二次谐波）+ 4-8k 高架 6dB；`dehiss` 改为压制**贴近噪声底**的 bin（旧版压的是超底噪 +18dB 的内容，实测几乎不触发） |
| m3 | `clarity`/`harsh`/`ess` 用绝对阈值（r>1.2 / 超参考+4dB / r2>2），真实混音几乎不触发（拉满 0.17 / 0.11 / 0.01 dB） | 三个都改「**对本轨自身分布的高分位帧生效**」（clarity p55→p92、harsh p75、ess p85），保证任何素材上都有权威度；上限提到 9 / 9 / 18 dB |
| m4.depth | 对 S 加低通梳状反射后把 S 的 RMS **归一回原值** —— 归一让 S/M 比恒不变，实测 depth 0→1 只动 **0.03dB**（参数完全无效） | 改为把 **M 的全通去相关分量**（3 级一级全通级联，幅度平坦不染色）注入 S，再做 **(M,S) 联合功率归一**（只按总功率缩放两者）→ S/M 比真实上升（实测 ×2.65），而总电平不变（RMS 差 0.000dB）。stats 新增 `mono_fold_db`（单声道折叠损失）与 `norm_db` |
| m5.pump | 单帧修正上限 ±6dB | ±9dB |
| m6.lra | 参考点固定 `p10 + lra_t`，目标远低于素材 LRA 时压不动 | 沿分位数扫参考点取最接近目标的曲线。**注意**：不要加「向上抬升」来压 LRA —— 抬安静段会让它们穿过 EBU 相对门进入门控集合，实测把高动态素材 LRA 从 ~7 推到 15.8 |
| dyneq | `thr_db` 默认 −24，但实测 3kHz 带检测电平中位约 **−32dB** → down 模式几乎不触发 | 默认 `thr_db` 改 **−30**；`stats.dyneq.env_db_median` 上报该带检测电平，供 UI 提示用户把阈值设在什么量级 |

### §4.3 引擎默认值（`engine.DEFAULT_CONFIG`）· **用户冻结版 2026-10-14**

```text
sep   : on  hpss  graded  vocal/low/perc=1.0  other_gain=0.6      （截图未含，维持旧校准值）
ai    : off                                                       （接口预留，截图未含）
m5    : on  denoise=0.00  floor_db=-60  hum=0.20  mains=50  declip=0.00  pump=0.00
m1    : on  weight=0.20  density=0.25  sub=0.15  punch=0.50  tame=0
m2    : on  air=0.45  detail=0.50  dehiss=0.10  ext_hz=15100
m3    : on  clarity=0.35  room=0.50  harsh=0.00  ess=0.00
m4    : on  width_low=0.56  width_mid=1.75  width_high=1.75  depth=0.40  rotation=0
m6    : OFF target_lufs=-16  lra=6  ceiling=-1.0
dyneq : off  bands[0].thr_db=-30                                  （截图未含，维持关闭）
export: channels=stereo  bit_depth=24  loudnorm=true
所有模块 mix = 100
```

- 这一版是**用户在 UI 里调好并签字冻结**的参数：改它 = 改「每次打开应用时的初始参数」。
- `tests/test_engine.py` 里有一张 `FROZEN` 表逐项断言（47 项），**误改会立刻测试失败**。
- `m6 = OFF` 的含义：处理链内不做母带响度归一（实测 20s 测试素材输出 −10.8 LUFS）；
  但 **导出时 `exporter` 仍按 `export.loudnorm=true` 做响度归一** —— 关掉 m6 ≠ 导出不归一。
- `m5.floor_db=-60` 在 `denoise=0` 时不参与计算（它只是谱减的增益下限）。

### §4.2 为此修改的算法（与旧版行为不同，UI 文案需相应更新）

## 5. UI 要求（web/index.html）—— 三区布局（2026-10-13 用户草图定稿）

总体：`display:grid; grid-template-columns: minmax(0,1.5fr) minmax(0,1fr); gap`。
左列 = 区1（可滚动，长渲染）；右列上下分两块 = 区2（实时曲线）+ 区3（实时处理）。
青瓷蓝数据风：纸底 `#F7F2EB`、墨蓝 `#081F5C`、主蓝 `#334EAC`、中蓝 `#7096D1`、浅蓝
`#BAD6EB`/`#D0E3FF`；系统字体栈；无 emoji，图标内联 SVG；纯原生单文件、零外网；错误横幅。

### 区1 · 主要功能（长渲染，左列，自上而下）

所有这里的操作都**可能需要数秒到数分钟**，统一走进度条。
1. **文件与诊断**：目录浏览（/api/list）+ 当前文件 + 「诊断」→ 指标卡 + 问题列表 + 「应用建议」。
2. **进度条**（吸顶或紧贴「开始处理」）：细横条 + 百分比 + 阶段中文名 + 阶段内小进度 +
   **预计剩余**（eta_s →「x 分 y 秒」，null →「估算中…」）。POST /api/job → 400ms 轮询
   GET /api/job → done 时取 result 渲染预览；error 显示横幅。stage 映射：
   sep=源分离 ai=AI重建 m5=修复层 m1=中低频 m2=中高频 m3=人声 m4=声场 m6=母带
   dyneq=动态EQ load=读取文件 stats=统计 done=完成。
3. **处理链**：9 张模块卡（链顺序），每卡独立开关 + **渲染参数**控件（改动需重算）+ 算法小标签；
   ai 卡显示「接口已预留 · 模型待实现」。
4. **干湿比专区**（与模块参数物理分离的独立分组）：9 行滑块（模块名+数值%+滑杆），
   写 `config.modules[key].mix`；change（松手）即触发 /api/job 重算（复用进度条）；
   旁置「重算范围」单选 `完整` / `快速90秒`（window=[0,90]）。ai 行禁用。
   （这是**渲染期干湿比**，与区3的调音台推子不同：改它要重算，改调音台不用。）
5. **预览**：「开始处理」→ 进度 → 单 `<audio>`（配 Web Audio graph，见区2/3）+ A/B 切换
   （切换 src 保持播放位置）+ before/after 频谱 SVG + 指标 + 9 阶段耗时表（含 mix 列）。
6. **导出**：声道 stereo/mono/5.1、位深 16/24/32float、响度归一、输出路径 → POST /api/export，
   body 带 `eq_bands`（区3 实时 EQ 的 bands）与 `gains`+`job`（区3 调音台推子，来自 result.mixer.job）
   —— 导出=渲染结果 + 烘焙EQ + 调音台混音。预览 windowed 时先弹确认完整渲染再导出。
   若 `result.mixer.available=false`，导出时省略 gains 并提示调音台不可用原因。
7. **AI 模型**：下拉（/api/models）+ 刷新按钮 + 说明文案（禁用态控件）。

### 区2 · 实时音频动态曲线（右上，固定高度约 300px）

- 数据源：Web Audio。`<audio>` → `MediaElementAudioSourceNode` →（区3的 EQ 链）→
  `AnalyserNode(fftSize=4096, smoothingTimeConstant=0.8)` → `destination`。
- 两块画布（同一 canvas 上下分栏或两个 canvas）：
  a) **频谱曲线**：`getFloatFrequencyData`（log 频率轴 30Hz-16kHz 映射，纵轴 -90..0dB），
     主蓝 `#334EAC` 1.5px 描边 + 半透明填充；`requestAnimationFrame` 持续刷新。
  b) **波形**：`getByteTimeDomainData` 细横条带（高约 48px），中蓝细线。
- 曲线旁标注当前播放的源（处理后/原片）与播放时间。
- 播放暂停时曲线自然落到静音（不需特殊处理）。**禁止**在无用户播放动作前自动播放。

### 区3 · 实时处理（右下，与区2并列，可滚动）

这里的一切**即时生效**（浏览器 Web Audio 完成，不发请求、不重算）：
1. **拖拽 EQ**（从处理链的 dyneq 卡移到这里，改为**实时静态 EQ**）：
   - canvas：对数频率 20Hz-20kHz、±10dB、0dB 基线、频率网格；每 band 一个可拖圆点
     （左右=频率、上下=gain_db）；选中显示 Q / gain_db 控件 + 删除；双击加点（上限 8）。
   - 实现：预建 8 个 `BiquadFilterNode(type="peaking")` 串在 audio graph 里（未用的 gain=0、Q 固定），
     拖动时**实时**改 `frequency/gain/Q` → 立刻听感变化；同时把 bands 同步到内存对象 `liveEqBands`。
   - 静态叠加曲线画在 canvas 上（RBJ peaking 幅度公式，与导出烘焙一致）。
   - 注意：区1 处理链里保留 dyneq 卡（动态 EQ，渲染期）作为另一可选功能，但默认 off；
     实时EQ 是这里的静态版本，导出时由服务端烘焙。两者独立。
2. **调音台**（各功能部分的音量推子）：
   - 仅当 `result.mixer.available` 时可用，推子列表 = `result.mixer.keys`（sep/m5/m1/m2/m3/m4/m6…），
     每行：模块中文名 + 推子（0..200%，步长 1，默认 100%）+ 数值。
   - change（松手）→ 防抖 350ms → `POST /api/remix {job, gains:{...全部key...}}` →
     返回 `{id,url}` → 更新 `<audio>.src`（保持播放位置与状态）。语义：
     `混音 = 原片 + Σ 推子增益 × 该模块贡献量`（服务端 f16 贡献量累加，约 0.3-0.8s，
     **不重跑处理链**）。请求中显示「混音中…」，完成后曲线下方闪一行「已更新」。
   - `available=false`：推子区禁用并显示 result.mixer.reason（提示改用快速窗口渲染）。
3. 区3 顶部放一行小字：「本区全部实时生效；区1 的改动才需要渲染」。

### 通用

- 纯原生 HTML/CSS/JS 单文件；只请求本机 API；控件写回 config 的 key 与 §4 完全一致（含 mix）。
- 断连/超时/ok:false → 顶部红色横幅（可关闭）。A/B 切换与调音台换 src 时都保持播放位置。

## 6. 约束

- server.py 只 import 标准库 +（延迟）`livefix.api`，不得 import numpy。
- 不得修改 livefix/ 下任何文件；冲突以本文件为准并在结果消息里说明。
