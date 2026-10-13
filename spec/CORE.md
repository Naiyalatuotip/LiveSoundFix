# livefix 内核契约 v1（功能子代理必读）

## 1. 文件归属（越界即失败）

- **你只能写一个文件**：`livefix/m1.py` / `m2.py` / `m3.py` / `m4.py` / `m5.py` / `m6.py` / `sep.py` / `dyneq.py` / `exporter.py` 之一（server 子代理写 `server.py` + `web/index.html`）。
- **只读，禁止修改**：`livefix/core.py`、`wavio.py`、`measure.py`、`engine.py`、`api.py`、`analysis.py`、`ai.py`、`spec/*`。
- 你的自测脚本放 `tests/test_<key>.py`（允许创建）。

## 2. 数组约定

- 音频：`np.float32`，shape `(n_samples, n_channels)`，ch = 1 或 2（也可能更多，按原样处理）；数值可能略超 `[-1,1]`，**不要**擅自硬裁。
- STFT `X`：`complex64`，shape `(ch, frames, bins)`，`N_FFT=2048, HOP=512, bins=1025`。
- 所有以 dB 表示的参数语义：0 dB = 不处理。

## 3. `livefix/core.py` 提供（签名固定，直接 import 使用）

```python
from . import core

N_FFT = 2048; HOP = 512

core.stft(x, sr)                    # float32 (n,ch)|(n,) -> complex64 (ch,T,F)（内部按块计算，控制内存）
core.istft(X, n, sr)                # (ch,T,F) -> float32 (n,ch)，与输入等长、sqrt-hann 精确重构
core.rfftfreqs(sr)                  # -> float64 (F,)
core.db(x)                          # 20log10，带下限 -100dB
core.lin(dbv)                        # 10**(dbv/20)
core.mag_response(kind, sr, f, q=0.707, gain_db=0.0)   # -> float64 (F,) 线性幅度
                                    # kind ∈ "hp"|"lp"|"peak"|"notch"|"lowshelf"|"highshelf"|"bandpass"（RBJ 双二阶幅度，零相位应用）
core.smooth_curve_db(sr, pts)       # pts=[(Hz,dB)...] -> float64 (F,)，log-f 插值 + 斜率平滑的频段增益曲线
core.band_mask(sr, f_lo, f_hi, edge_oct=0.3)  # -> float64 (F,) 平顶软边频段掩码(0..1)
core.env_updown(v, attack_s, release_s, sr)   # 帧域非对称一阶平滑，沿最后一维；v (...,T) -> 同形 float64
core.frame_rms(X)                   # -> float32 (ch,T) 线性帧 RMS
core.band_energy_db(X, sr, f_lo, f_hi)  # -> float64 (ch,T) 频段能量 dB（功率口径，见下）
# ⚠ dB 有两种口径，混用会差 3dB 或 60dB，写代码前必须确认：
#   · 幅度口径（20log10）：core.db() / core.lin() 互为逆运算。
#   · 功率口径（10log10）：core.band_energy_db() / core.noise_floor_db() / measure 里的能量。
#   band_energy_db 返回 raw 频带功率 10log10(Σ|X|²·mask)，**不是 dBFS**：
#   cal_db = 10*log10(N_FFT * hann.sum() / 2) ；e_dBFS = band_energy_db - cal_db。
#   噪声底与频带能量互比时，两者同为功率口径可直接相减；若要与 core.db() 的幅度 dB
#   比较，必须先换算：p_dB(功率) = 20log10(幅度)。
core.noise_floor_db(X, pct=15)      # -> float64 (ch,F) 时间维百分位噪声底 dB（功率口径，同样非 dBFS）
core.smooth1d(v, win, axis=-1)      # 滑动平均（频域平滑用）
core.block_apply(x, sr, fn, block_s=60.0, xfade_s=1.0)
                                    # fn(seg float32 (n_seg,ch), sr) -> 同形 seg'；分段处理 + 线性交叉淡化，
                                    # 用于长文件控制内存。引擎会用它包裹你的 process。
```

`livefix/measure.py` 提供（导出/诊断用）：
```python
measure.lufs_integrated(x, sr)      # BS.1770-4 综合响度 LKFS
measure.lufs_short(x, sr)           # -> float64 (T,) 短时(3s)响度，用于 LRA/限制器
measure.true_peak_db(x, sr)         # 4x 过采样真峰 dBTP
measure.peak_db(x) / measure.rms_db(x)
measure.spectrum_db(x, sr, n=96)    # -> list[float] 对数频点(30..16k)幅度 dB，自身归一化到 0
measure.corr(x)                     # 左右相关系数（单声道返回 1.0）
```

`livefix/wavio.py`：
```python
wavio.read_wav(path)                # -> (float32 (n,ch), sr)；支持 PCM16/24/32 与 IEEE float32/64
wavio.write_wav(path, x, sr, bit_depth=24)   # bit_depth ∈ 16|24|32；32 = IEEE float
wavio.ensure_readable(path, cache_dir)  # 非 wav 时尝试 ffmpeg 转码；失败抛清晰错误
```

## 4. 模块契约（每个功能文件的唯一公开接口）

```python
KEY = "m1"                      # 必须等于 config 中的 key

def process(x, sr, p, ctx):
    """x: float32 (n,ch) -> float32 (n,ch) 等长
    p  : 本模块的参数 dict（来自 spec/API.md §4，字段保证齐全）
    ctx : 引擎共享 dict，见 §5
    """
    ...

def info():
    return {"key": KEY, "name": "中低频重建", "algo": "LR4 分频 · 次谐波合成 · 向上密度压缩"}
```

### 模块红线
1. 纯 numpy + 标准库；**禁止** scipy / torch / 网络 / 读写文件（exporter 除外，它必须写文件）。
2. 输出无 NaN/Inf；float32；长度、声道数不变。
3. **分段安全**：引擎可能以约 60s 段 + 1s 交叉淡化调用你的 process（`core.block_apply`），不得假设拿到的是文件开头；跨帧状态须在 ≤0.5s 内收敛；不要依赖绝对时间。
4. 参数取最小值（0 / 空列表 / 1.0）时，输出与输入 RMS 差 < 0.05 dB（m6 例外：它以响度达标为准）。
5. 性能：60s@48k 立体声单模块 ≤ 3s（本机）；STFT 每次 process ≤ 2 进 2 出。
6. 随机数只用 `ctx.get("rng") or np.random.default_rng(0)`，保证可复现。

## 5. ctx 约定

```python
ctx = {
  "rng": np.random.default_rng(seed),
  "stats": {},     # 模块可写自己的统计，如 m4 写 ctx["stats"]["m4"]={"corr":..}
  "log": [],       # 可 append 一句话给人看
  "stems": None,   # 仅 sep 写
  "note": {}       # 如 ai 写 ctx["note"]["ai"]="interface_only"
}
```

## 6. 引擎链与分级路由（引擎实现，你只需理解）

```
x = 输入
if sep.on:   x = sep.process(...)   # routing=graded 时同时写 ctx["stems"]={"vocal":V,"backing":B}，且 V+B == 返回值
x = ai.process(...)                 # 本期透传 + ctx["note"]["ai"]
--- 分级路由（sep 激活且 routing=graded）时，以下四段按轨执行 ---
m5 作用于 vocal 与 backing 两条；m1 只作用于 backing；
m3 只作用于 vocal；m2 作用于两条；随后 y = vocal + backing 合回
--- 非分级时：m5、m1、m2、m3 依次作用于全轨 ---
x = m4.process(x)                   # 声场
x = m6.process(x)                   # 母带（引擎不对其分段，需全局响度）
x = dyneq.process(x)                # 最后一步
输出
```

**干湿比（mix，0..100，每模块独立，默认 100）**：引擎在段边界线性混合
`out = (1-mix)·in + mix·proc(in)`（各模块均保相位，lerp 语义正确）；
sep 特例：mix 混合的是四个 stem 增益 `g_eff = 1-mix+mix·g`（dry=全 1=不分离），
保证 mix=0 时 stems 与输出严格一致；mix=0 时该段被完全跳过（不计算）。
`mix` 与 `on` 由引擎剥掉，不会传进 `mod.process` 的 params。

**进度回调**：`engine.process(..., progress_cb=fn)`，fn(frac, stage_key, stage_frac)。
frac 单调不减、最终为 1.0；分级路由下 m5/m2 各跑两趟（weight=2）。
`engine.plan(mods)` 返回预计计算计划 [(key, weight)]，与实际执行路径严格一致，
供 api 做 ETA 估算。跨段状态仍须 ≤0.5s 收敛（见红线 3）。

**已知修复（勿回退）**：旧版分级路由里 m5 在合轨上跑完即被 stems 覆盖（结果被静默丢弃），
现改为按轨执行。

`sep.process` 的分级语义（必须满足）：
- 返回值 = `vocal*vocal_gain + low*low_gain + perc*perc_gain + other*other_gain`（各 stem 同相位、掩码和为 1）。
- `routing=="graded"` 时额外写 `ctx["stems"] = {"vocal": vocal*vocal_gain, "backing": low*low_gain+perc*perc_gain+other*other_gain}`（与返回值相加一致）。
- `routing=="mix"` 时 `ctx["stems"] = None`。

## 7. 交付流程

1. 读 `spec/API.md`（本模块参数语义）+ 本文件 + `livefix/core.py` 源码。
2. 写模块 → 写 `tests/test_<key>.py` 自测（合成信号即可）：无 NaN、长度一致、最小参数≈恒等、立体声+单声道、性能计时。
3. 跑通后结束；报告里给出：文件路径、算法说明、自测结果、参数语义一句话表。
- Python：`C:\Users\24379\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe`（numpy 2.3.5，无 scipy）。
- 工作目录：`E:\作品集\音频修复v1`。
