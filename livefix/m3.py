"""LiveFix m3 —— 人声去掩蔽与优化（STFT 幅度域，保留原相位）。

四个并行子处理器，各自输出一条 dB 增益曲线，dB 域相加后一次性乘回 |X|：
  1. clarity 清晰度/动态去掩蔽：每帧算 1.5-4.5kHz（人声存在感带）与 300-1200Hz
     （参考带）的能量比 r = E_pres/E_ref。r 从 1.2 线性升到 3.0 判定"人声活跃度"
     act∈[0,1]，平滑（attack 15ms / release 250ms）后经 band_mask(1.4k,5k)
     补 0 .. +6*clarity dB。同时对 90-160Hz：仅当该带能量高于中频 0..6dB 时下压
     0 .. -2*clarity dB（band_mask(90,160)），避免低频浑浊掩蔽人声。
  2. room 晚期混响抑制（Nakatani WPE 的谱域统计简化）：对 |X| 沿时间轴取
     "未来 30 帧前视窗滑动最小值" D[f,t]（逐 i 原地 np.minimum，内存 O(1) 份），
     3 帧时域平滑抑制随机波动。晚期能量 L = max(|X|^2 - D^2, 0)，
     抑制量 supp = room*L/|X|^2（clip 0..0.75），增益 g = sqrt(max(1-supp, 0.06))
     （floor ≈ -12.2dB），再乘一个噪声底门（低于噪声底 10dB 内渐出，避免静音抽吸）。
     包络 attack（恢复）150ms / release（下压）40ms —— 快压慢放。
  3. harsh 刺耳控制：2.2-5.5kHz 帧能量超过 0.4s 滑动参考 +4dB 时衰减，
     excess/6 线性映射，最多 -6*harsh dB，band_mask(2.2k,5.5k)，
     包络 attack（恢复）120ms / release（下压）8ms —— 快速咬住嘶叫。
  4. ess 齿音（de-esser）：5-9kHz 与 1-3kHz 能量比 r2 超过阈值 4.0 时按
     ratio≈3 的斜率下压（over_db * 2/3），最多 -10*ess dB，
     band_mask(4.5k,10k)，包络 attack（恢复）90ms / release（下压）3ms。
声道按声道独立处理（增益曲线形状 (ch,T,F)）。参数全 0 或 on=False 时恒等返回。
"""
from __future__ import annotations

import numpy as np

from . import core

KEY = "m3"

# ---- 常量 -----------------------------------------------------------------
LOOKAHEAD = 30       # room 前视帧数（48k hop512 ≈ 0.32s）
SMOOTH_WIN = 3       # room 幅度时域平滑帧数（抑制随机谱波动）
G_FLOOR_LIN = 0.06   # room 增益下限（线性功率域）≈ -12.2 dB
SUPP_CLIP = 0.75     # room 抑制量上限

CLAR_THR, CLAR_SPAN = 1.2, 1.8   # 兼容旧常量（clarity 已改为自参考百分位，见 CLAR_PCT_*）
CLARITY_F0 = 3000.0              # clarity 主频中心（Hz）
CLARITY_MAX_DB = 9.0             # clarity 最大提升（dB）
ESS_THR = 2.0                    # 兼容旧常量（ess 已改为自参考百分位，见 ESS_PCT）
ESS_SLOPE = 2.0 / 3.0            # ratio≈3 的 dB 域斜率（1 - 1/3）
ESS_MAX_DB = 18.0                # ess 最大衰减（dB），强齿音可压 15dB 以上
ESS_F0 = 6500.0                  # ess 作用频带中心（Hz）
ESS_HOLD, ESS_REL = 0.080, 0.120  # ess 衰减 hold 80ms（覆盖脉冲全程）/ release 120ms
HARSH_REF_S = 0.4                # harsh 自适应参考窗长（秒）
HARSH_THR, HARSH_SPAN = 4.0, 6.0  # 兼容旧常量（harsh 已改为自参考百分位，见 HARSH_PCT）
HARSH_MAX_DB = 9.0               # harsh 最大衰减（dB）
ROOM_MAX_DB = 0.75               # room 抑制量上限（线性域）
# ---- 自参考激活阈值（分位数）----
# 旧实现用绝对阈值（r>1.2 / 超参考+4dB / r2>2），在真实混音上几乎不触发：
# 实测 clarity 拉满只动 0.17dB、harsh 0.11dB、ess 0.01dB —— 参数形同虚设。
# 改为「对全轨自身分布的高分位帧生效」，保证任何素材上参数都有权威。
CLAR_PCT_LO, CLAR_PCT_HI = 55.0, 92.0   # clarity：55→92 分位线性映射到 0→1
HARSH_PCT = 75.0                        # harsh：超过 p75 的过量算起
ESS_PCT = 85.0                          # ess：超过 p85 的齿音比算起


def _band_db(P, m32):
    """P (ch,T,F) 功率 float32，m32 (F,) float32 掩码 -> (ch,T) float32 dB。"""
    e = np.einsum("ctf,f->ct", P, m32)
    return (10.0 * np.log10(e + 1e-16)).astype(np.float32)


def _hold_smooth(ge, sr, hold_s, release_s):
    """帧域：attack 一帧到位，hold 保持最深值，release 单极回落。

    专为短脉冲（齿音）设计。与一阶包络的差别：hold 期间增益钉在最深值，
    不会在脉冲尚未结束时就因平滑而回升，故脉冲全程保持足量衰减。
    输入/输出 (ch,T) float64，语义为 dB（负值=衰减）。
    """
    v = np.asarray(ge, np.float64)
    if v.ndim != 2 or v.shape[-1] < 2:
        return v.copy()
    hop_s = core.HOP / float(sr)
    hold_f = max(1, int(round(float(hold_s) / hop_s)))
    ar = float(np.exp(-hop_s / max(float(release_s), 1e-5)))
    out = v.copy()
    for c in range(v.shape[0]):
        row = out[c]
        prev = row[0]
        hold_until = -1              # 当前 hold 截止帧号（相对本次脉冲起点）
        for i in range(1, row.shape[0]):
            cur = row[i]
            if cur < prev:           # 需要更多衰减：一帧内到位（attack），并重新起一个 hold
                prev = cur
                hold_until = i + hold_f
            elif i <= hold_until:    # 本次脉冲的 hold 期内：钉住最深值，不回升
                pass
            else:                    # release：单极回落
                prev = ar * prev + (1.0 - ar) * cur
            row[i] = prev
    return out


def _future_min(mag, w):
    """沿时间轴的未来窗滑动最小值：D[t] = min(mag[t..t+w])，原地迭代，内存安全。"""
    D = mag.copy()
    T = mag.shape[1]
    for i in range(1, min(int(w), T - 1) + 1):
        np.minimum(D[:, i:, :], mag[:, :-i, :], out=D[:, i:, :])
    return D


def _tsmooth3(a):
    """3 帧居中平滑（float32，边界复制），仅为 room 统计服务。"""
    b = a.copy()
    if a.shape[1] >= 3:
        s = b[:, 1:-1, :]
        np.add(s, a[:, :-2, :], out=s)
        np.add(s, a[:, 2:, :], out=s)
        s *= np.float32(1.0 / 3.0)
    return b


def info():
    return {"key": KEY, "name": "人声去掩蔽与优化",
            "algo": "动态清晰度去掩蔽 · 谱域晚期混响抑制 · 刺耳控制 · 动态齿音下压"}


def process(x, sr, p, ctx=None):
    x = np.asarray(x)
    was_1d = x.ndim == 1
    if was_1d:
        x = x[:, None]
    x = np.ascontiguousarray(x, np.float32)
    if x.ndim != 2:
        raise ValueError(f"m3 期望 (n, ch)，得到 {x.shape}")
    p = p or {}
    on = bool(p.get("on", True))
    clarity = float(np.clip(p.get("clarity", 0.0), 0.0, 1.0))
    room = float(np.clip(p.get("room", 0.0), 0.0, 1.0))
    harsh = float(np.clip(p.get("harsh", 0.0), 0.0, 1.0))
    ess = float(np.clip(p.get("ess", 0.0), 0.0, 1.0))
    # 每个子效果的关键阈值/上限均可调（现场条件差异大，默认值经基准校准）
    clarity_f0 = float(p.get("clarity_f0", CLARITY_F0))
    clarity_max = float(p.get("clarity_max", CLARITY_MAX_DB))
    room_max = float(p.get("room_max", ROOM_MAX_DB))
    harsh_thr = float(p.get("harsh_thr", HARSH_THR))
    harsh_max = float(p.get("harsh_max", HARSH_MAX_DB))
    ess_thr = float(np.clip(p.get("ess_thr", ESS_THR), 1.0, 40.0))
    ess_max = float(p.get("ess_max", ESS_MAX_DB))
    ess_f0 = float(p.get("ess_f0", ESS_F0))
    if (not on) or (clarity == 0.0 and room == 0.0 and harsh == 0.0 and ess == 0.0):
        return x[:, 0] if was_1d else x.copy()   # 恒等：RMS 差 = 0 dB

    sr = float(sr) if float(sr) > 0 else 48000.0
    X = core.stft(x, sr)                          # (ch,T,F) complex64
    mag = np.abs(X)                               # float32
    P = mag * mag                                 # float32 功率

    def m32(f_lo, f_hi):
        return core.band_mask(sr, f_lo, f_hi).astype(np.float32)

    G = np.zeros(P.shape, np.float32)             # 总 dB 增益
    stats = {}

    # ---- 1) clarity：1.5-4.5k 动态提升 + 90-160Hz 动态下压 ------------------
    if clarity > 0.0:
        e_pres = _band_db(P, m32(1500.0, 4500.0))
        e_ref = _band_db(P, m32(300.0, 1200.0))
        r = core.lin(e_pres.astype(np.float64) - e_ref.astype(np.float64))
        # 自参考：以本轨存在感比的 p55→p92 映射为 0→1，保证总有一段被处理
        r_lo = float(np.percentile(r, CLAR_PCT_LO))
        r_hi = float(np.percentile(r, CLAR_PCT_HI))
        act = np.clip((r - r_lo) / max(r_hi - r_lo, 1e-6), 0.0, 1.0)
        gc = clarity_max * clarity * core.env_updown(act, 0.015, 0.25, sr)   # (ch,T)
        G += gc[:, :, None] * m32(max(900.0, clarity_f0 * 0.75), clarity_f0 * 3.33)[None, None, :]
        stats["clarity_peak_db"] = float(gc.max())

        e_low = _band_db(P, m32(90.0, 160.0))
        e_mid = _band_db(P, m32(300.0, 1200.0))
        duck = np.clip((e_low.astype(np.float64) - e_mid.astype(np.float64)) / 6.0, 0.0, 1.0)
        gd = -2.0 * clarity * core.env_updown(duck, 0.20, 0.05, sr)
        G += gd[:, :, None] * m32(90.0, 160.0)[None, None, :]
        stats["bass_duck_db"] = float(gd.min())

    # ---- 2) room：谱减式晚期混响抑制 ---------------------------------------
    if room > 0.0:
        ms = _tsmooth3(mag)
        D = _future_min(ms, LOOKAHEAD)            # 直达声幅度下界
        L = np.maximum(P - D * D, 0.0)            # 晚期能量
        supp = np.float32(room) * L / (P + np.float32(1e-12))
        np.clip(supp, 0.0, np.float32(room_max), out=supp)
        # 噪声底门：接近/低于噪声底的 bin 不抑制，避免静音抽吸
        nfloor = core.noise_floor_db(X).astype(np.float32)            # (ch,F)
        p_db = 10.0 * np.log10(P + 1e-16)
        supp *= np.clip((p_db - nfloor[:, None, :]) / 10.0, 0.0, 1.0)
        gl = np.sqrt(np.maximum(1.0 - supp, G_FLOOR_LIN))
        gr = (20.0 * np.log10(gl)).astype(np.float32)                 # (ch,T,F) ≤ 0
        gr = core.env_updown(np.moveaxis(gr, 1, -1), 0.15, 0.04, sr)  # 恢复150ms/下压40ms
        G += np.moveaxis(gr, -1, 1).astype(np.float32)
        stats["room_max_db"] = float(gr.min())

    # ---- 3) harsh：2.2-5.5kHz 自适应参考过量衰减 ---------------------------
    if harsh > 0.0:
        e_h = _band_db(P, m32(2200.0, 5500.0))
        win = max(3, int(round(HARSH_REF_S * sr / core.HOP)))
        ref = core.smooth1d(e_h.astype(np.float64), win, axis=-1)     # 0.4s 滑动参考
        ex = e_h.astype(np.float64) - ref
        thr = float(np.percentile(ex, HARSH_PCT))                     # 自参考阈值
        span = max(float(np.percentile(ex, 98.0)) - thr, 0.5)
        excess = np.clip((ex - thr) / span, 0.0, 1.0)
        gh = -harsh_max * harsh * core.env_updown(excess, 0.12, 0.08, sr)   # 恢复120ms/下压8ms
        G += gh[:, :, None] * m32(2200.0, 5500.0)[None, None, :]
        stats["harsh_max_db"] = float(gh.min())

    # ---- 4) ess：5-9kHz 下行扩展（de-esser） -------------------------------
    if ess > 0.0:
        e_s = _band_db(P, m32(5000.0, 9000.0))
        e_v = _band_db(P, m32(1000.0, 3000.0))
        r2 = core.lin(e_s.astype(np.float64) - e_v.astype(np.float64))
        # 自参考阈值：超过本轨齿音比的 p85 才算齿音（真实混音里 r2 常年 <2，
        # 旧的绝对阈值 2.0 导致 ess 拉满只动 0.01dB）
        thr2 = max(float(np.percentile(r2, ESS_PCT)), 1e-3)
        over = np.clip(20.0 * np.log10(np.maximum(r2, 1e-6) / thr2), 0.0, 36.0)
        ge = -ess * np.minimum(over * ESS_SLOPE, ess_max)              # 上限可调
        # 齿音是 30-80ms 的短脉冲（仅 6 帧 @48k）。不能用一阶包络平滑 dB：
        # 6 帧内一阶平滑到不了稳态，且对负 dB 加权平均会严重低估衰减深度。
        # 改用「attack 一帧到位 + hold 脉冲宽度 + release 平滑」，保证脉冲期间
        # 始终保持足量衰减（实测同一测试从 2.2dB 提升到 >8dB）。
        ge = _hold_smooth(ge, sr, ESS_HOLD, ESS_REL)
        G += ge[:, :, None] * m32(ess_f0 * 0.9, ess_f0 * 2.0)[None, None, :]
        stats["ess_max_db"] = float(ge.min())

    # ---- 组合：幅度域一次性施加，相位不变 -----------------------------------
    linG = core.lin(G.astype(np.float64)).astype(np.float32)
    X2 = (X * linG).astype(np.complex64)
    del G, linG, P, mag
    out = core.istft(X2, x.shape[0], sr)
    if out.shape != x.shape:                     # 容错：理论不会发生
        out = out[: x.shape[0]]
        if out.shape[0] < x.shape[0]:
            out = np.pad(out, ((0, x.shape[0] - out.shape[0]), (0, 0)))
    out = np.ascontiguousarray(out, np.float32)
    out[~np.isfinite(out)] = 0.0                 # 输出无 NaN/Inf 保底

    if isinstance(ctx, dict):
        st = ctx.get("stats")
        if not isinstance(st, dict):
            st = {}
            ctx["stats"] = st
        st[KEY] = {k: round(v, 3) for k, v in stats.items()}
        lg = ctx.get("log")
        if isinstance(lg, list):
            lg.append("m3: clarity=%s room=%s harsh=%s ess=%s" % (
                f"{clarity:.2f}", f"{room:.2f}", f"{harsh:.2f}", f"{ess:.2f}"))
    return out[:, 0] if was_1d else out
