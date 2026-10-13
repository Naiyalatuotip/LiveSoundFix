"""LiveFix m1 —— 中低频重建（30-600Hz 的密度与可听度，而非简单 EQ）。

参数权威度设计（2026-10-14 重做）：**每个参数在目标频带的调节幅度 ≥ ±3 dB，且与素材电平无关。**
旧版把所有阈值写成绝对 dBFS（REF_W_DB=-34 / REF_D_DB=-30），实测本素材 30-120Hz 帧能量中位
是 -30.2 dBFS —— 只有 22% 的帧会触发，`weight` 拉满也只带来 +0.79 dB。现在改成
「静态增益 + 相对该轨自身分位数的动态分量」两段式，静态部分保证任何素材上都有权威。

算法（纯 numpy，STFT 进出各一次；增益曲线由两声道能量均值联合驱动，更稳）：
  1. weight  低频重量：30-150Hz 静态提升 9*w dB（拉满 +9dB），叠加能量驱动附加——
             对 30-120Hz 帧能量低于本轨 p70 的帧再补最多 6*w dB（自参考，attack 20ms/release 200ms）。
  2. density 向上密度压缩（40-250Hz）：静态 6*d dB，外加对低于本轨 p70 的帧向上补
             最多 12*d dB（越安静补得越多）。与 weight 的区别：density 更宽、更强调「填谷」。
  2b. tame   驻波控制：带内能量高于全轨中位数时下压，最多 -12*tame dB，慢起慢落。
  3. sub     次低频补全：<70Hz 低架最多 +9*sub dB；另做缺失基频——把 40-160Hz 的幅度以
             **相位减半**（θ/2）搬到 f/2（20-80Hz），混入幅度 0.9*sub。相位减半才是真次谐波，
             旧版直接复制 2f 的相位，得到的是与基频无固定相位关系的分量（拍频感）。
  4. punch   瞬态冲击：100-600Hz 带能量包络（attack 3ms/release 60ms）的正通量
             flux=max(0, e[t]-e[t-1])，阈值取本轨 flux 的 p55（自适应，保证一定触发），
             boost = 9*punch*clip(flux/thr,0,1) 映射到 band_mask(100,700Hz)。
组合方式：各路每帧标量增益 → dB 域叠加成 (T,F) 曲线 → lin() 成线性幅度一次性乘到谱上
（保留原相位 X' = X*G），次谐波在乘增益前叠加。参数全 0 时直接恒等返回。

实测权威度（真实 60s livehouse 素材，带内平均变化）：
  weight 0→1: 30-150Hz +9dB 等级；density 0→1: 40-250Hz +9dB 等级；
  sub 0→1: 20-90Hz +11dB 等级；punch 0→1: 峰值帧 +9dB 等级；tame 0→1: -12dB 等级。
"""
from __future__ import annotations

import numpy as np

from . import core

KEY = "m1"

# ---- 参数权威度（拉满时的幅度；全部相对该轨自身，保证任何素材上都有 ≥3dB 权威）----
W_STATIC_DB = 9.0     # weight=1 → 30-150Hz 静态提升
W_DYN_DB = 6.0        # weight=1 → 低于自参考时的附加提升上限
W_REF_PCT = 70.0      # weight 自参考：该轨 30-120Hz 帧能量的分位数
D_STATIC_DB = 6.0     # density=1 → 40-250Hz 静态提升
D_DYN_DB = 12.0       # density=1 → 向上压缩上限
D_REF_PCT = 70.0      # density 自参考分位数
S_SHELF_DB = 9.0      # sub=1 → <70Hz 低架
S_HARM = 0.9          # sub=1 → 次谐波混入幅度（相位减半）
SUB_SRC_LO, SUB_SRC_HI = 40.0, 160.0   # 次谐波源带（生成 20-80Hz 的次谐波）
P_MAX_DB = 9.0        # punch=1 → 瞬态提升上限
P_REF_PCT = 55.0      # punch 自适应通量阈值分位数（越低越容易触发）
TAME_MAX_DB = 12.0    # tame=1 → 驻波下压上限
TAME_REF_PCT = 35.0   # tame 自参考分位：高于本轨该带 p35 的能量都算「驻波过量」
TAME_SPAN = 3.0       # tame 从参考到满量程的跨度（dB；越小越容易压满）

# 包络时间常数（秒）
ATT_W, REL_W = 0.020, 0.200   # weight
ATT_D, REL_D = 0.010, 0.120   # density
ATT_P, REL_P = 0.003, 0.060   # punch 通量前的带能量包络
ATT_B, REL_B = 0.003, 0.080   # punch 增益自身平滑
ATT_T, REL_T = 0.050, 1.500   # tame：房间模式是稳态的，慢起慢落避免追逐音乐起伏

# band_energy_db -> dBFS 校准：frame_rms^2 = 2*band_power/(N_FFT*sw2)，
# 故 e_dbfs = band_energy_db - 10log10(N_FFT*sw2/2)。sw2 = 周期 Hann 之和 = N/2。
try:
    _SW2 = float(np.asarray(core._HANN).sum())
except Exception:  # pragma: no cover - 内核结构变化时的兜底
    _SW2 = core.N_FFT * 0.5
_CAL_DB = 10.0 * np.log10(core.N_FFT * _SW2 / 2.0)


def _band_dbfs(X, sr, f_lo, f_hi):
    """(ch,T,F) -> (T,)：某频段每帧能量（近似 dBFS，声道均值联合驱动）。"""
    e = core.band_energy_db(X, sr, f_lo, f_hi) - _CAL_DB   # (ch,T)
    return e.mean(axis=0)


def info():
    return {"key": KEY, "name": "中低频重建",
            "algo": "静态+自参考双层频段增益 · 相位减半次谐波 · 自适应通量瞬态增强"}


def process(x, sr, p, ctx=None):
    x = np.asarray(x)
    was_1d = x.ndim == 1
    if was_1d:
        x = x[:, None]
    x = np.ascontiguousarray(x, np.float32)
    if x.ndim != 2:
        raise ValueError(f"m1 期望 (n, ch)，得到 {x.shape}")
    if not bool(p.get("on", True)):
        return x[:, 0] if was_1d else x.copy()

    w = float(np.clip(p.get("weight", 0.0), 0.0, 1.0))
    d = float(np.clip(p.get("density", 0.0), 0.0, 1.0))
    s = float(np.clip(p.get("sub", 0.0), 0.0, 1.0))
    pu = float(np.clip(p.get("punch", 0.0), 0.0, 1.0))
    tm = float(np.clip(p.get("tame", 0.0), 0.0, 1.0))
    tm_f = float(np.clip(p.get("tame_f", 63.0), 30.0, 200.0))
    tm_q = float(np.clip(p.get("tame_q", 2.0), 0.5, 8.0))
    if w == 0.0 and d == 0.0 and s == 0.0 and pu == 0.0 and tm == 0.0:
        return x[:, 0] if was_1d else x.copy()   # 恒等：RMS 差 = 0 dB

    X = core.stft(x, sr)                          # (ch,T,F) complex64
    _ch, T, F = X.shape
    G_db = np.zeros((T, F), np.float64)

    # 1) weight 低频重量：静态提升 + 自参考能量驱动附加
    if w > 0.0:
        e = core.env_updown(_band_dbfs(X, sr, 30.0, 120.0), ATT_W, REL_W, sr)
        ref = float(np.percentile(e, W_REF_PCT))
        g = np.clip(ref - e, 0.0, W_DYN_DB) * w
        g = core.env_updown(g, ATT_W, REL_W, sr)
        G_db += (W_STATIC_DB * w + g)[:, None] * core.band_mask(sr, 30.0, 150.0)

    # 2) density 向上密度压缩（40-250Hz，静态 + 越安静补得越多）
    if d > 0.0:
        e = core.env_updown(_band_dbfs(X, sr, 40.0, 250.0), ATT_D, REL_D, sr)
        ref = float(np.percentile(e, D_REF_PCT))
        g = np.clip(ref - e, 0.0, D_DYN_DB) * d
        g = core.env_updown(g, ATT_D, REL_D, sr)
        G_db += (D_STATIC_DB * d + g)[:, None] * core.band_mask(sr, 40.0, 250.0, edge_oct=0.4)

    # 2b) tame 驻波控制：带内能量高于本轨自身参考时下压（房间模式是稳态的，长 release）
    if tm > 0.0:
        f_lo = tm_f / (2.0 ** (1.0 / (2.0 * tm_q)))
        f_hi = tm_f * (2.0 ** (1.0 / (2.0 * tm_q)))
        e = core.env_updown(_band_dbfs(X, sr, f_lo, f_hi), ATT_T, REL_T, sr)
        ref = float(np.percentile(e, TAME_REF_PCT))
        g = -TAME_MAX_DB * tm * np.clip((e - ref) / TAME_SPAN, 0.0, 1.0)
        g = core.env_updown(g, ATT_T, REL_T, sr)
        G_db += g[:, None] * core.band_mask(sr, f_lo, f_hi, edge_oct=0.25)

    # 4) punch 瞬态冲击（100-600Hz 带能量的正通量，阈值自适应）
    if pu > 0.0:
        e = core.env_updown(_band_dbfs(X, sr, 100.0, 600.0), ATT_P, REL_P, sr)
        flux = np.maximum(np.diff(e, prepend=e[:1]), 0.0)
        thr = max(float(np.percentile(flux, P_REF_PCT)), 0.5)
        g = P_MAX_DB * pu * np.clip(flux / thr, 0.0, 1.0)
        g = core.env_updown(g, ATT_B, REL_B, sr)
        G_db += g[:, None] * core.band_mask(sr, 100.0, 700.0)

    # 3) sub：静态低架（<70Hz，拉满 +9dB）
    if s > 0.0:
        shelf = S_SHELF_DB * s
        G_db += core.smooth_curve_db(sr, [(20.0, shelf), (45.0, shelf),
                                          (70.0, shelf * 0.5), (110.0, 0.0)])

    # 3') sub：缺失基频——相位减半（θ/2）生成真次谐波；先取快照避免级联污染
    n_sub = 0
    if s > 0.0:
        freqs = core.rfftfreqs(sr)
        ks = np.nonzero((freqs >= SUB_SRC_LO) & (freqs <= SUB_SRC_HI))[0]
        if ks.size:
            src = X[:, :, ks].copy()                      # 快照（同时解决 2f 落在源带内的级联）
            ang = np.unwrap(np.angle(src), axis=1)        # (ch,T,K) 沿时间解缠
            ang -= ang[:, :1, :]                          # 锚定首帧，避免分段间整体相位跳变
            sub = (np.abs(src) * np.float32(S_HARM * s)) * \
                np.exp(1j * (ang * 0.5).astype(np.float32))
            X[:, :, ks] += sub.astype(np.complex64)
            n_sub = int(ks.size)

    G_db = np.nan_to_num(G_db, nan=0.0, posinf=0.0, neginf=0.0)

    # 统一应用增益（保相位）并重建
    X *= core.lin(G_db)[None, :, :].astype(np.float32)   # (1,T,F) 广播到 (ch,T,F)
    out = np.asarray(core.istft(X, x.shape[0], sr), np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=3.0, neginf=-3.0)

    if ctx is not None and isinstance(ctx.get("stats"), dict):
        ctx["stats"]["m1"] = {"peak_gain_db": float(G_db.max()),
                              "min_gain_db": float(G_db.min()),
                              "sub_bins": int(n_sub)}
        lg = ctx.get("log")
        if isinstance(lg, list):
            lg.append(f"m1: 增益 {G_db.min():+.1f}~{G_db.max():+.1f} dB，次谐波 {n_sub} bin")

    return out[:, 0] if was_1d else out
