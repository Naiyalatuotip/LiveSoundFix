"""LiveFix m4 · 声场扩展（参考 DaVinci Fairlight 声场调整思路，纯 numpy）。

算法（立体声走 M/S 域）：
1. M/S 解码：M=(L+R)/2，S=(L-R)/2。
2. 分频段宽度曲线（STFT 实数幅度缩放，相位保留）：
   band_mask(0,120) / band_mask(120,1200) / band_mask(1200,sr) 三条软边掩码
   按 width_low/width_mid/width_high 加权叠加后除以掩码和，得到一条连续、无阶跃
   的 (F,) 宽度曲线（过渡带为余弦斜边，无梳状风险；DC bin 归入最低段取
   width_low）；在 STFT 域对 S 的复谱乘该实数曲线（只改幅度、保留相位），
   istft 回时域。
3. depth 纵深（伪立体声 · 全通去相关）：
    对 M 做 3 级一级全通级联（系数 0.62/-0.72/0.81，幅度平坦 → 只改相位、不染色），
    得到与 M 去相关的版本，按 depth*0.80 注入 S。之后对 (M,S) 做**联合功率归一**
    （只按总功率缩放两者），保证总电平不变、而 S/M 比真实上升 = 可听的变宽/纵深。
    旧版是对 S 自身加低通梳状反射后把 S 的 RMS 归一回原值 —— 归一让 S/M 比恒不变，
    实测 depth 0→1 只改动 0.03dB，参数完全无效。
4. rotation 声像旋转（度，限制 ±45°）：a = radians(rotation)；旋转矩阵
   M' = M*cos a + S*sin a，S' = S*cos a - M*sin a。a=+45° 时
   L'=(L-R)/√2、R'=(L+R)/√2（左耳只剩差信号、右耳只剩和信号），a=0 为恒等。
5. M/S 逆变换：L' = M'+S'，R' = M'-S'。
处理顺序：宽度 -> depth -> 联合功率归一 -> rotation。
统计里额外给出 mono_fold_db（单声道折叠损失）与 norm_db（联合归一增益），
供 UI 提示「加宽会牺牲多少单声道兼容性」。

单声道：无 S 可分，仅对 M 施加 depth 反射（权重减半 = depth*0.5*Σg），
宽度/旋转跳过，并在 ctx["log"] append 提示。
"""
from __future__ import annotations

import math

import numpy as np

from . import core

KEY = "m4"

_EDGE_OCT = 0.35            # 宽度曲线过渡带宽（八度）
_F_LO, _F_MID, _F_HI = 120.0, 1200.0, 1200.0  # 分频点：120Hz / 1.2kHz
_TAP_S = (0.013, 0.017, 0.023, 0.031)         # 反射抽头延时（秒）
_TAP_G = (0.45, 0.35, 0.28, 0.22)             # 反射抽头增益
_LP_HZ = 1200.0             # 反射源低通，抑制刺耳侧向反射
_W_MAX = 5.0                # 宽度参数安全上限
# ---- depth 重做（2026-10-14）----
# 旧版：S += depth*refl(S) 后把 S 的 RMS 归一回原值 —— 功率归一让 S/M 比**恒不变**
# （实测 depth 0→1 只改动 0.03dB，参数完全无效）。
# 新版：把 **M 的全通去相关分量**注入 S（一级全通级联，幅度平坦 → 只改相位、不染色），
# 宽度与纵深之后对 (M,S) 做**联合**功率归一（只按总功率缩放两者），
# 于是 S/M 比真实上升 = 可听的「变宽」，同时总电平不变（不给下游送电平台阶）。
_DEPTH_MIX = 0.80           # depth=1 时注入 S 的 M 分量幅度（相对 M，拉满约 +7dB 侧能量）
_AP_G = (0.62, -0.72, 0.81)  # 一级全通系数级联（交替符号，去相关更好）


def _width_curve(sr, w_low, w_mid, w_high):
    """三条软边带掩码加权叠加 -> 连续宽度曲线 (F,)，值域 [min(w), max(w)]。

    DC bin 处三条 band_mask 均为 0（内核约定），按最低频段处理取 w_low——
    这也保证 width 全 1.0 时曲线逐位等于 1，恒等路径完全跳过 STFT。
    """
    m_lo = core.band_mask(sr, 0.0, _F_LO, _EDGE_OCT)
    m_md = core.band_mask(sr, _F_LO, _F_MID, _EDGE_OCT)
    m_hi = core.band_mask(sr, _F_HI, float(sr), _EDGE_OCT)  # 上边沿落在 Nyquist 之外
    num = m_lo * w_low + m_md * w_mid + m_hi * w_high
    den = m_lo + m_md + m_hi
    return np.where(den > 1e-9, num / np.maximum(den, 1e-9), w_low)


def _reflections(s, sr):
    """(n,) -> 反射和 (n,)：lp 1.2kHz（4 阶，级联两级 2 率 biquad，24dB/oct）后 4 抽头梳状延迟叠加（有限长 FIR，无反馈）。"""
    b, a = core.biquad_coeffs("lp", sr, _LP_HZ, q=0.707)
    slp = core.iir_filt(s[:, None], b, a)
    slp = core.iir_filt(slp, b, a)[:, 0].astype(np.float64)  # 级联成 4 阶，抑制 4kHz 以上刺耳成分
    n = int(s.shape[0])
    refl = np.zeros(n, np.float64)
    for ds, g in zip(_TAP_S, _TAP_G):
        d = int(round(float(ds) * float(sr)))
        if 0 < d < n:
            refl[d:] += g * slp[: n - d]
    return refl


def _shape_side(s, sr, curve):
    """STFT 域对 S 乘实数宽度曲线（幅度缩放、相位保留）。曲线全 1 时直通。"""
    if float(np.max(np.abs(curve - 1.0))) < 1e-9:
        return s
    X = core.stft(s, sr)                                   # (1,T,F) complex64
    X = X * curve.astype(np.float32)[None, None, :]
    return core.istft(X, s.shape[0], sr)[:, 0].astype(np.float64)


def _decorrelate(m, sr):
    """一级全通级联（幅度平坦、只改相位）→ 与 M 去相关的版本。

    全通不会改变频谱包络，所以注入 S 不会染色（不像梳状反射会产生固定着色），
    适合做「伪立体声 / 纵深」的侧信号源。b/a 补零到二阶以便复用 core.iir_filt。
    """
    y = np.asarray(m, np.float64)[:, None]
    for g in _AP_G:
        b = np.array([-g, 1.0, 0.0])
        a = np.array([1.0, -g, 0.0])
        y = core.iir_filt(y, b, a)
    return y[:, 0].astype(np.float64)


def _rms(v):
    return float(np.sqrt(np.mean(np.asarray(v, np.float64) ** 2)))


def _apply_depth(s, sr, depth):
    """单声道用：侧信号不存在，仅对 M 施加低通梳状早期反射 + 功率归一（电平中性）。"""
    added = (depth * 0.5) * _reflections(s, sr)
    ratio = _rms(added) / max(_rms(s), 1e-12)
    return (s + added) / np.sqrt(1.0 + ratio * ratio)


def _sanitize(y):
    bad = ~np.isfinite(y)
    if bad.any():
        y[bad] = 0.0
    return y


def process(x, sr, p, ctx):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    x = np.ascontiguousarray(x, np.float32)
    p = p or {}
    if not bool(p.get("on", True)):
        return x.copy()

    w_low = float(np.clip(float(p.get("width_low", 1.0)), 0.0, _W_MAX))
    w_mid = float(np.clip(float(p.get("width_mid", 1.0)), 0.0, _W_MAX))
    w_high = float(np.clip(float(p.get("width_high", 1.0)), 0.0, _W_MAX))
    depth = float(np.clip(float(p.get("depth", 0.0)), 0.0, 1.0))
    rot = float(np.clip(float(p.get("rotation", 0.0)), -45.0, 45.0))

    ctx = ctx if isinstance(ctx, dict) else {}
    log = ctx.setdefault("log", [])
    stats = ctx.setdefault("stats", {})
    n, ch = x.shape

    # ---- 单声道：无 S 可分，仅 depth 反射（权重减半） --------------------
    if ch < 2:
        m = x[:, 0].astype(np.float64)
        if depth > 0.0:
            m = _apply_depth(m, sr, depth)
        y = np.empty((n, 1), np.float32)
        y[:, 0] = m
        log.append("m4 声场：单声道输入，无侧信号可分——仅施加 depth 纵深反射（权重减半），宽度/旋转跳过")
        stats[KEY] = {"width_in": 0.0, "width_out": 0.0}
        return _sanitize(y)

    # ---- 立体声（多于 2 声道时仅前两声道走 M/S，其余直通） --------------
    L = x[:, 0].astype(np.float64)
    R = x[:, 1].astype(np.float64)
    M = 0.5 * (L + R)
    S = 0.5 * (L - R)
    mr = max(_rms(M), 1e-12)
    width_in = _rms(S) / mr
    p_m0 = float(np.mean(M ** 2))
    p_s0 = float(np.mean(S ** 2))

    # 1) 分频段宽度
    S = _shape_side(S, sr, _width_curve(sr, w_low, w_mid, w_high))
    # 2) depth：把 M 的全通去相关分量注入 S —— 对近单声道素材产生真正可听的宽/纵深
    if depth > 0.0:
        S = S + (depth * _DEPTH_MIX) * _decorrelate(M, sr)
    # 3) 联合功率归一：只按总功率缩放 (M,S)。**不能**各自归一到原 RMS ——
    #    那会让 S/M 比恒不变，正是旧版 depth 无效的原因。
    p_s1 = float(np.mean(S ** 2))
    g_norm = math.sqrt((p_m0 + p_s0) / max(p_m0 + p_s1, 1e-20))
    M = M * g_norm
    S = S * g_norm
    # 4) rotation：M/S 旋转矩阵
    if rot != 0.0:
        a = rot * np.pi / 180.0
        ca, sa = float(np.cos(a)), float(np.sin(a))
        M, S = M * ca + S * sa, S * ca - M * sa

    y = np.empty((n, ch), np.float32)
    y[:, 0] = M + S
    y[:, 1] = M - S
    if ch > 2:
        y[:, 2:] = x[:, 2:]
        log.append(f"m4 声场：输入 {ch} 声道，仅前两声道做 M/S 声场处理，其余声道直通")

    m2 = 0.5 * (y[:, 0].astype(np.float64) + y[:, 1])
    s2 = 0.5 * (y[:, 0].astype(np.float64) - y[:, 1])
    fold_mono = 10.0 * np.log10((np.mean(m2 ** 2) + 1e-20)
                                / (np.mean(y[:, 0].astype(np.float64) ** 2) + 1e-20))
    stats[KEY] = {
        "width_in": float(width_in),
        "width_out": float(_rms(s2) / max(_rms(m2), 1e-12)),
        "mono_fold_db": float(fold_mono),
        "norm_db": float(20.0 * np.log10(max(g_norm, 1e-9))),
    }
    return _sanitize(y)


def info():
    return {
        "key": KEY,
        "name": "声场扩展",
        "algo": "M/S 分频宽度曲线 · 全通去相关纵深 · 联合功率归一 · 声像旋转",
    }
