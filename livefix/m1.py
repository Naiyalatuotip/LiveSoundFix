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
  3. sub     次低频补全：<70Hz 低架最多 +9*sub dB；另做**包络驱动次谐波**——20-80Hz 的每个
              目标 bin（源 = 2f ∈ 40-160Hz）生成「幅度 = 源幅度 × S_HARM、相位 = 载波推进
              2πk·hop/N + 随机初相」的一致 tile。⚠ 相干条件：ISTFT 4 帧重叠相加要求 bin k 的
              复幅度每帧推进 2πk·hop/N，否则低 k bin 被相消（跨 bin 复制/固定初相/纯随机
              游走实测被吃掉 ~93%，仅 k≡0 mod 4 幸免，2026-10-14 内核实验实证）。
  4. punch   瞬态冲击：100-600Hz 带能量包络（attack 3ms/release 60ms）的正通量
             flux=max(0, e[t]-e[t-1])，阈值取本轨 flux 的 p55（自适应，保证一定触发），
             boost = 9*punch*clip(flux/thr,0,1) 映射到 band_mask(100,700Hz)。
  5. ring    低频混响衰减（livehouse 63/80Hz 驻波拖尾）：只在 25-180Hz 的约 10 个 bin 上
             取幅度 E（声道均值），M=过去 30 帧滑动最大、F=未来 50 帧滑动最小；
             decay_span=(M−F)/max(M,eps)，span≤0.35（一窗内掉不到 ~4.8dB）判为持续音/贝斯
             延音 → 保护不压；pos=clip((M−E)/(M−F+eps),0,1)：攻击瞬间 E≈M→pos≈0 不压 ✓，
             深入衰减 pos→1 多压 ✓。supp=ring·pos^1.5，g=1−0.75·supp（ring=1 最深 −12dB）；
             对 supp 做 env_updown(attack 15ms=快速介入 / release 120ms=缓慢恢复)，
             最终按 1+(g−1)·band_mask(25,180,软边) 落回全谱 → 带外零作用、带边平滑。
  6. dive    下潜重建（相机麦/HPF 滚降缺失的 20-50Hz）：同 sub 的包络驱动法（含载波推进）——
              20-50Hz 的每个目标 bin（源 = 0.55·|X(2f)| + 0.22·|X(3f)| 的低频源包络）生成
              一致 tile，跟随底鼓/贝斯包络给出 20-50Hz 的「体重」，再叠 20-50Hz 低架
              dive*4 dB。跨 bin 复制的旧阶梯法因缺载波项被往返相消 ~93%，已废弃。
处理顺序：ring 先（检测读原始谱），dive 后（新生成的下潜不进入 ring 检测）；两者的增益
与 weight/density/tame/punch/sub 的 dB 曲线统一叠加后一次乘谱。参数全 0 时直接恒等返回。
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
S_HARM = 0.9          # sub=1 → 次谐波混入幅度（相对源 bin 幅度）
SUB_SRC_LO, SUB_SRC_HI = 40.0, 160.0   # 次谐波源带（生成 20-80Hz 的次谐波）
P_MAX_DB = 9.0        # punch=1 → 瞬态提升上限
P_REF_PCT = 55.0      # punch 自适应通量阈值分位数（越低越容易触发）
TAME_MAX_DB = 12.0    # tame=1 → 驻波下压上限
TAME_REF_PCT = 35.0   # tame 自参考分位：高于本轨该带 p35 的能量都算「驻波过量」
TAME_SPAN = 3.0       # tame 从参考到满量程的跨度（dB；越小越容易压满）

# ---- ring（低频混响衰减）----
R_LO, R_HI = 25.0, 180.0        # 检测/作用频带（48k/2048 下 bin 宽 23.4Hz，仅约 10 个 bin）
R_W_PAST, R_W_FUT = 30, 50      # 过去峰值窗 / 未来谷值窗（帧，hop=512）
R_SPAN_THR = 0.35               # 一窗内掉不到 35%（≈4.8dB）→ 持续音，保护不压
R_MAX_SUPP = 0.75               # ring=1 最深抑制 g=0.25 → −12dB
R_EPS = 1e-6
R_ATT, R_REL = 0.015, 0.120     # 抑制量平滑：attack=快速介入 15ms / release=缓慢恢复 120ms

# ---- dive（下潜重建）----
D_LO, D_HI = 20.0, 50.0         # 下潜写入 bin 范围（跳过 DC；源 = 2f/3f）
D_W2, D_W3 = 0.55, 0.22         # 源包络权重（2f / 3f）
D_GAIN = 0.9                    # dive 总幅度
D_SHELF_DB = 4.0                # dive=1 → 20-50Hz 平滑低架
D_ATT, D_REL = 0.045, 0.300     # 源包络平滑（跟随底鼓，不抖）
D_WALK = 0.35                   # 相位随机游走步长（rad/帧，慢速=一致 tile）

# ---- sub / dive 共用的包络次谐波参数 ----
S_TGT_LO, S_TGT_HI = 20.0, 80.0  # sub 次谐波目标带（源 = 2f ∈ 40-160Hz）
S_ATT, S_REL = 0.045, 0.300      # sub 源包络平滑
S_WALK = 0.0                     # sub 相位游走（必须为 0：游走相位非平稳，能量灌 DC）

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


def _past_max(A, w):
    """M[t] = max(A[t-w..t])：逐位移原地迭代，内存安全（只留一份拷贝）。"""
    M = np.array(A, np.float64, copy=True)
    Tn = M.shape[0]
    for i in range(1, min(int(w), Tn - 1) + 1):
        np.maximum(M[i:], A[:Tn - i], out=M[i:])
    return M


def _future_min(A, w):
    """F[t] = min(A[t..t+w])：未来窗滑动最小，逐位移原地迭代（参考 m3 写法，本地版）。"""
    D = np.array(A, np.float64, copy=True)
    Tn = D.shape[0]
    for i in range(1, min(int(w), Tn - 1) + 1):
        np.minimum(D[:Tn - i], A[i:], out=D[:Tn - i])
    return D


def _env_tile(env, rng, steps, walk=0.0):
    """包络 → 一致 STFT tile：幅度=env，相位=载波推进（2πf·hop/sr 每帧）+ 每 bin 随机初相。

    相位必须以目标 bin 频率的速率推进 —— 这是 istft 4 帧 sqrt-hann 重叠相加的相干条件
    （2026-10-14 实证：注入幅度 100 的 unit tile，往返后 固定初相 bin1 仅 11.8 / bin2 仅 0.8，
    只有 k≡0 mod4 幸免于 81；加载波项后所有 bin 均 ~81 保真）。固定初相/纯随机游走都与
    载波错开 → 低 k bin 在重叠相加中相消——「跨 bin 复制被往返吃掉 93%」的统一机理。
    载波 + 极慢游走 = 跟随包络的振荡器，一致 tile。
    """
    T = env.shape[0]
    ph = np.arange(T, dtype=np.float64)[:, None] * np.asarray(steps, np.float64)[None, :]
    ph = ph + rng.uniform(0.0, 2.0 * np.pi, size=(1, env.shape[1]))
    if walk and walk > 0.0:
        ph = ph + walk * np.cumsum(rng.normal(size=env.shape), axis=0)
    return (env * np.exp(1j * ph)).astype(np.complex64)


def _rng_of(ctx, seed):
    rng = ctx.get("rng") if isinstance(ctx, dict) else None
    return rng if rng is not None else np.random.default_rng(seed)


def _ring_gain_db(X, sr, ring):
    """ring 拖尾抑制 → (idx, gdb)：低带各 bin 的增益 dB（≤0），无抑制的帧为 0。"""
    mask = core.band_mask(sr, R_LO, R_HI)
    idx = np.nonzero(mask > 1e-3)[0]
    if idx.size == 0:                                   # 极低采样率兜底
        return idx, np.zeros((X.shape[1], 0), np.float64)
    E = np.abs(X[:, :, idx]).mean(axis=0).astype(np.float64)   # (T,B) 声道均值幅度
    M = _past_max(E, R_W_PAST)                          # 过去峰值
    F = _future_min(E, R_W_FUT)                         # 未来谷值
    span = (M - F) / np.maximum(M, R_EPS)               # 一窗内的衰减跨度
    pos = np.clip((M - E) / (M - F + R_EPS), 0.0, 1.0)  # 攻击≈0 → 深入衰减→1
    pos[span <= R_SPAN_THR] = 0.0                       # 持续音/贝斯延音保护
    supp = ring * np.power(pos, 1.5)
    supp = core.env_updown(supp.T, R_ATT, R_REL, sr).T  # 沿时间平滑（attack=介入）
    g = 1.0 - R_MAX_SUPP * supp
    # 软边落回全谱：1+(g−1)·mask —— 带内满量、带边平滑、带外恒 1（0dB）
    gdb = 20.0 * np.log10(np.maximum(1.0 + (g - 1.0) * mask[idx], 1e-3))
    return idx, gdb


def info():
    return {"key": KEY, "name": "中低频重建",
            "algo": "静态+自参考双层频段增益 · 包络驱动次谐波 · 拖尾抑制 · 下潜重建 · 自适应通量瞬态"}


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
    rg = float(np.clip(p.get("ring", 0.0), 0.0, 1.0))    # 低频混响衰减
    dv = float(np.clip(p.get("dive", 0.0), 0.0, 1.0))    # 下潜重建
    if w == 0.0 and d == 0.0 and s == 0.0 and pu == 0.0 and tm == 0.0 \
            and rg == 0.0 and dv == 0.0:
        return x[:, 0] if was_1d else x.copy()   # 恒等：RMS 差 = 0 dB

    X = core.stft(x, sr)                          # (ch,T,F) complex64
    _ch, T, F = X.shape
    G_db = np.zeros((T, F), np.float64)

    # 0) ring 低频混响衰减：先治拖尾嗡声（检测读原始谱，先于 dive 的下潜生成）
    n_ring = 0
    if rg > 0.0:
        r_idx, r_gdb = _ring_gain_db(X, sr, rg)
        if r_idx.size:
            G_db[:, r_idx] += r_gdb
            n_ring = int(r_idx.size)

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

    # 3') sub：包络驱动次谐波（20-80Hz，源 = 2f ∈ 40-160Hz；一致 tile 可完整通过往返）
    n_sub = 0
    if s > 0.0:
        freqs = core.rfftfreqs(sr)
        ks = np.nonzero((freqs >= S_TGT_LO) & (freqs <= S_TGT_HI))[0]
        ks = ks[(ks > 0) & (2 * ks < F)]
        if ks.size:
            env = np.abs(X[:, :, np.minimum(2 * ks, F - 1)]).mean(axis=0)  # (T,B) 源幅度
            env = core.env_updown(env.T, S_ATT, S_REL, sr).T * (S_HARM * s)
            steps = 2.0 * np.pi * freqs[ks] * core.HOP / float(sr)         # 各 bin 载波推进
            tile = _env_tile(env, _rng_of(ctx, 0), steps, S_WALK)
            X[:, :, ks] += tile[None, :, :].astype(np.complex64)
            n_sub = int(ks.size)

    # 6) dive 下潜重建：包络驱动一致 tile（20-50Hz，源 = 2f/3f 低频包络）
    n_dive = 0
    if dv > 0.0:
        G_db += core.smooth_curve_db(sr, [(18.0, D_SHELF_DB * dv),
                                          (50.0, D_SHELF_DB * dv), (90.0, 0.0)])
        freqs = core.rfftfreqs(sr)
        ks = np.nonzero((freqs >= D_LO) & (freqs <= D_HI))[0]
        ks = ks[(ks > 0) & (3 * ks < F)]
        if ks.size:
            env = (D_W2 * np.abs(X[:, :, np.minimum(2 * ks, F - 1)])
                   + D_W3 * np.abs(X[:, :, np.minimum(3 * ks, F - 1)])).mean(axis=0)
            env = core.env_updown(env.T, D_ATT, D_REL, sr).T * (D_GAIN * dv)
            steps = 2.0 * np.pi * freqs[ks] * core.HOP / float(sr)         # 各 bin 载波推进
            tile = _env_tile(env, _rng_of(ctx, 1), steps, D_WALK)
            X[:, :, ks] += tile[None, :, :].astype(np.complex64)
            n_dive = int(ks.size)

    G_db = np.nan_to_num(G_db, nan=0.0, posinf=0.0, neginf=0.0)

    # 统一应用增益（保相位）并重建
    X *= core.lin(G_db)[None, :, :].astype(np.float32)   # (1,T,F) 广播到 (ch,T,F)
    out = np.asarray(core.istft(X, x.shape[0], sr), np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=3.0, neginf=-3.0)

    if ctx is not None and isinstance(ctx.get("stats"), dict):
        ctx["stats"]["m1"] = {"peak_gain_db": float(G_db.max()),
                              "min_gain_db": float(G_db.min()),
                              "sub_bins": int(n_sub),
                              "ring_bins": int(n_ring),
                              "dive_bins": int(n_dive)}
        lg = ctx.get("log")
        if isinstance(lg, list):
            lg.append(f"m1: 增益 {G_db.min():+.1f}~{G_db.max():+.1f} dB，次谐波 {n_sub} bin，"
                      f"拖尾 {n_ring} bin，下潜 {n_dive} bin")

    return out[:, 0] if was_1d else out
