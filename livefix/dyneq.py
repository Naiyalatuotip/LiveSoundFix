"""LiveFix 动态均衡（dyneq）——处理链最后一步。

算法（静态/动态分解）：
- 对每个频点 band(f, q, gain_db, thr_db, ratio, range_db, mode)：
  1) 检测：core.mag_response("bandpass", sr, f, q) 得频段响应 |H_bp|，
     帧能量 e[t] = SCALE · Σ_ch Σ_F |X|²·|H_bp|²（多声道联合，避免声像漂移；
     SCALE 与 core.frame_rms 同尺度归一，使 e_db 直观对应音频电平 dB），
     转 dB 后用 core.env_updown(attack=5ms, release=120ms) 非对称平滑。
  2) 动态增益 dyn[t]（作用于平滑后的 e_db）：
     mode="static" -> dyn ≡ 0，只有静态 gain_db。
     mode="down"   -> over = max(0, e_db-thr_db)；dyn = -over·(1-1/ratio)，截到 [-range_db, 0]。
     mode="up"     -> under = max(0, thr_db-e_db)；dyn = +under·(1-1/ratio)，截到 [0, +range_db]。
     总增益 = gain_db + dyn[t]（gain_db 是可选静态偏置）。
  3) 应用曲线（近似说明）：
     静态部分用精确 peaking 响应 20·log10|H_peak(gain_db)|（RBJ 双二阶，一次算好）。
     动态部分【近似】为「软边频段掩码 × 每帧标量」：
       shape = core.band_mask(sr, f/2^(1/(2q)), f·2^(1/(2q)))，
       每帧标量 = dyn[t]；即不逐帧重算 peaking 双二阶幅度（8 band × ~5k 帧的
       逐帧 biquad 频响代价过高），改用一条固定掩码乘以随时间变化的标量。
     所有 band 的曲线在 dB 域相加为 G[t,F]，取 exp 后一次性乘到 |X| 再 istft。
- 增益对所有声道相同（联合能量驱动、同一条 G），保证声像不漂移。

契约：KEY="dyneq"；x float32 (n,ch) -> float32 (n,ch) 等长；纯 numpy；
bands 为空 / 全 static 且 gain_db=0 / on=False 时走恒等快路径；
引擎可能以 ~60s 段调用（block_apply），本模块无跨段状态，天然分段安全。
"""
from __future__ import annotations

import numpy as np

from . import core

KEY = "dyneq"
_MAX_BANDS = 8

# 帧能量归一系数：与 core.frame_rms 的 2/(N_FFT * sum(hann)) 同尺度
# （periodic hann 的 sum = N/2，故 2/(N·N/2) = 4/N²）。
_E_SCALE = 2.0 / (core.N_FFT * (core.N_FFT / 2.0))

# 平滑参数（秒）
_ATTACK_S = 0.005
_RELEASE_S = 0.120

# 安全裁剪
_G_DB_CLIP = 96.0  # 合成总曲线的 dB 上限（防 exp 溢出）

# thr_db 的量纲提示：e_db 与 core.frame_rms 同尺度，实测本素材 3kHz 带中位约 -32dB、
# 1kHz 约 -29dB、8kHz 约 -37dB。故 down 模式的 thr_db 建议设在 -30 附近（默认 -24 偏高，
# 会让动态段几乎不触发：实测 range_db 拉满只动 0.91dB）。引擎默认值已同步调整。


def _clean_band(b, sr):
    """把 UI 传来的 band dict 规约为安全的 (f, q, gain, thr, ratio, rng, mode)。"""
    if not isinstance(b, dict):
        b = {}
    nyq = 0.49 * float(sr)
    f = float(np.clip(float(b.get("f", 1000.0)), 20.0, max(nyq, 21.0)))
    q = float(np.clip(float(b.get("q", 1.0)), 0.1, 20.0))
    gain = float(np.clip(float(b.get("gain_db", 0.0)), -60.0, 60.0))
    thr = float(np.clip(float(b.get("thr_db", -24.0)), -120.0, 24.0))
    ratio = float(np.clip(float(b.get("ratio", 1.0)), 1.0, 100.0))
    rng = float(np.clip(float(b.get("range_db", 6.0)), 0.0, 48.0))
    mode = str(b.get("mode", "static")).lower()
    if mode not in ("static", "up", "down"):
        mode = "static"
    return f, q, gain, thr, ratio, rng, mode


def process(x, sr, p, ctx):
    x = np.asarray(x, np.float32)
    was_1d = x.ndim == 1
    if was_1d:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"dyneq 期望 (n, ch)，得到 {x.shape}")

    p = p or {}
    bands = list(p.get("bands") or [])[:_MAX_BANDS]
    bands = [b for b in bands if isinstance(b, dict)]

    def _ident():
        y = x.copy()
        return y[:, 0] if was_1d else y

    if not p.get("on", False) or not bands:
        return _ident()

    cl = [_clean_band(b, sr) for b in bands]
    # 恒等快路径：全部为 static 且 gain_db=0 -> 总曲线恒为 0 dB
    if all(m == "static" and g == 0.0 for (_f, _q, g, _t, _r, _rg, m) in cl):
        return _ident()

    n, ch = x.shape
    X = core.stft(x, sr)                       # complex64 (ch,T,F)
    T, F = X.shape[1], X.shape[2]

    P = np.abs(X)
    P *= P                                     # float32 (ch,T,F) 功率，各 band 复用

    G = np.zeros((T, F), np.float32)           # dB 域总增益曲线
    max_dyn = 0.0
    env_med = np.zeros(0)

    for (f, q, gain, thr, ratio, rng, mode) in cl:
        # ---- 1) 频段检测（联合能量驱动） ----
        hb = core.mag_response("bandpass", sr, f, q=q)          # (F,) 线性幅度
        w2 = (hb * hb).astype(np.float32)
        e_ch = np.tensordot(P, w2, axes=([-1], [0]))            # (ch,T) 归一前
        e_db = 10.0 * np.log10(_E_SCALE * e_ch.sum(axis=0) + 1e-16)  # (T,) 联合 dB
        env = core.env_updown(e_db, _ATTACK_S, _RELEASE_S, sr)  # (T,) 非对称平滑
        if env.size and env_med.size == 0:
            # 上报该带检测电平的中位，供 UI 提示 thr_db 该设在什么量级
            # （实测 3kHz 带本素材中位约 -32dB，默认 thr_db=-24 会让 down 模式几乎不触发）
            env_med = np.array([float(np.median(env))])

        # ---- 2) 动态增益 ----
        if mode == "down":
            dyn = -np.maximum(0.0, env - thr) * (1.0 - 1.0 / ratio)
            np.clip(dyn, -rng, 0.0, out=dyn)
        elif mode == "up":
            dyn = np.maximum(0.0, thr - env) * (1.0 - 1.0 / ratio)
            np.clip(dyn, 0.0, rng, out=dyn)
        else:  # static
            dyn = np.zeros(T, np.float64)
        if dyn.size:
            ad = float(np.abs(dyn).max())
            if ad > max_dyn:
                max_dyn = ad

        # ---- 3) 应用曲线：静态精确 peak + 动态掩码近似（dB 域相加） ----
        if gain != 0.0:
            hp = core.mag_response("peak", sr, f, q=q, gain_db=gain)
            G += (20.0 * np.log10(hp + 1e-9)).astype(np.float32)
        if mode != "static" and rng > 0.0:
            bw = 2.0 ** (1.0 / (2.0 * q))
            mask = core.band_mask(sr, f / bw, f * bw).astype(np.float32)  # (F,)
            G += dyn.astype(np.float32)[:, None] * mask[None, :]

    np.clip(G, -_G_DB_CLIP, _G_DB_CLIP, out=G)
    G *= (np.log(10.0) / 20.0)
    np.exp(G, out=G)                            # 线性增益 (T,F)，复用 G 的内存
    X *= G[None, :, :]                          # 一次性作用于所有声道

    y = core.istft(X, n, sr)
    y = np.nan_to_num(y, copy=False)

    if ctx is not None:
        st = ctx.get("stats")
        if isinstance(st, dict):
            st[KEY] = {"bands": len(cl), "max_dyn_db": round(max_dyn, 2),
                       "env_db_median": round(float(np.median(env_med)), 2)
                       if env_med.size else None}
        lg = ctx.get("log")
        if isinstance(lg, list) and max_dyn > 0.5:
            lg.append(f"dyneq: {len(cl)} 频点，最大动态增益 {max_dyn:.1f} dB")

    if was_1d:
        return y[:, 0]
    return y


def info():
    return {
        "key": KEY,
        "name": "动态均衡",
        "algo": "RBJ peaking · 频段能量包络 · 上/下行动态增益 · 掩码近似曲线",
    }
