"""livefix · m5 修复层（去削波 / 除噪 / 降底噪 / 去泵动）。

处理顺序（契约）：
    declip（时域重建） → hum + denoise + pump（一次 STFT，各增益相乘 = dB 相加后统一作用于幅度域）

算法要点：
- declip：|x| >= 0.985*peak 的连续平坦段（run）用三次 Hermite（Catmull-Rom 切线）重建，
  接缝处用 C1 连续的 tanh 软限幅压回 ±peak；强度 declip 控制原值/重建值混合比。
- hum：mains 及其 2/3/4 次谐波各做一个 RBJ notch（Q=30，core.mag_response），深度 -30*hum dB；
  因 Q=30 的凹口比一个 STFT bin 还窄，深度按 bin 量化铺到最近的 1+2+2 个 bin 上，保证真实生效。
- denoise：最小值统计（noise_floor_db pct=12）+ 百分位偏差校正（指数分布 12% 分位 ≈ 0.128×均值，×3 拉回）
  + 沿频率中位数音调保护（持续音不被当成噪声）+ 谱减 alpha=1.2+1.5*denoise，
  增益下限 10^(floor_db/20)（同时覆盖「SNR<3dB 不塌陷」规则），频率 smooth1d(3)、
  时间 env_updown(10ms/200ms) 平滑抑制音乐噪声。
- pump：全带帧 RMS（声道均值）→ env_updown(300ms/800ms) 慢包络 → 相对中位数偏离
  （±6dB 截止）取反为修正增益 ×pump，作用于全谱；静音段不加增益。

无 NaN/Inf；分段调用安全（无跨段状态，包络以首帧值初始化）。
"""
from __future__ import annotations

import numpy as np

from . import core

KEY = "m5"

_CLIP_REL = 0.985     # 削波检测阈值（相对整段峰值）
_CTX_S = 0.005        # 前后文窗口 5ms
_SEAM_TH = 0.80       # tanh 软限幅起点（相对峰值；此处函数值与一阶导均连续）
_NPCT = 12.0          # 噪声底时间百分位
_NB_C = 3.0           # 最小值统计百分位偏差校正系数（dB 域 +20log10(3)≈+9.54dB）
_MED_WIN = 21         # 音调保护：沿频率中位数窗（bin）
_MED_MARGIN = 6.0     # 音调保护余量 dB
_ALPHA0, _ALPHA_K = 1.2, 1.5   # alpha = 1.2 + 1.5*denoise
_HUM_Q = 30.0
_HUM_DEPTH = -30.0    # hum=1 时每个谐波陷波深度 dB
_PUMP_MAX_DB = 9.0    # pump 的单帧最大修正量（dB；旧值 6，拉满的权威度不足）


# ------------------------------------------------------------------ 工具

def _amt(v, lo=0.0, hi=1.0):
    try:
        f = float(v)
    except (TypeError, ValueError):
        f = 0.0
    if not np.isfinite(f):
        f = 0.0
    return float(min(max(f, lo), hi))


def _medfilt1d(v, win):
    """沿最后一维的中位数滤波（edge 填充，长度不变）。"""
    v = np.asarray(v, np.float64)
    if win <= 1 or v.shape[-1] < 2:
        return v
    if win % 2 == 0:
        win += 1
    win = min(win, v.shape[-1] if v.shape[-1] % 2 == 1 else v.shape[-1] - 1)
    if win <= 1:
        return v
    pad = win // 2
    vp = np.pad(v, [(0, 0)] * (v.ndim - 1) + [(pad, pad)], mode="edge")
    sw = np.lib.stride_tricks.sliding_window_view(vp, win, axis=-1)
    return np.median(sw, axis=-1)


# ------------------------------------------------------------------ 削波修复（时域）

def _declip(x, sr, amt):
    """对每个声道找 |x|>=0.985*peak 的 run，Hermite 重建 + tanh 软限幅接缝。

    返回 (float32 (n,ch), 修复的 run 总数)。amt=1 完全替换，0 原样。
    批量向量化：run 按 16384 一组做 float32 网格运算，Python 循环只遍历批次数。
    """
    x64 = np.asarray(x, np.float64)
    n, ch = x64.shape
    out = x64.copy()
    total = 0
    th = _SEAM_TH
    for c in range(ch):
        v = x64[:, c]
        v32 = v.astype(np.float32)         # 批量重建全程 float32
        peak = float(np.max(np.abs(v32))) if n else 0.0
        if peak <= 1e-6:
            continue
        thr = _CLIP_REL * peak
        m = np.abs(v) >= thr
        if not m.any():
            continue
        dm = np.diff(m.astype(np.int8))
        s = np.flatnonzero(dm == 1) + 1
        e = np.flatnonzero(dm == -1) + 1
        if m[0]:
            s = np.concatenate((np.array([0]), s))
        if m[-1]:
            e = np.concatenate((e, np.array([n])))
        # 需要左右各 2 个真实样本做锚点/切线；贴边 run 跳过
        ok = (e > s) & (s >= 2) & (e <= n - 2)
        s = s[ok]
        e = e[ok]
        if s.size == 0:
            continue
        total += int(s.size)
        lensf = (e - s).astype(np.float32)
        maxl = int(lensf.max())
        krow = (np.arange(maxl, dtype=np.float32) + 0.5)[None, :]
        kinds = np.arange(maxl, dtype=np.int64)[None, :]
        B = 16384
        for i0 in range(0, s.size, B):
            i1 = min(i0 + B, s.size)
            ss, ee = s[i0:i1], e[i0:i1]
            ll = lensf[i0:i1]
            t = np.minimum(krow / ll[:, None], 1.0)
            inrun = kinds < ll.astype(np.int64)[:, None]
            # 锚点：run 前后各一个真实样本；切线：5ms 上下文内的相邻样本差（批量运算走 float32）
            p0 = v32[ss - 1]
            p3 = v32[ee]
            m0 = v32[ss - 1] - v32[ss - 2]
            m1 = v32[ee + 1] - v32[ee]
            # Hermite（Catmull-Rom 切线）
            t2 = t * t
            t3 = t2 * t
            val = ((2 * t3 - 3 * t2 + 1) * p0[:, None]
                   + (t3 - 2 * t2 + t) * m0[:, None])
            val += (-2 * t3 + 3 * t2) * p3[:, None]
            val += (t3 - t2) * m1[:, None]
            # tanh 软限幅（归一域）：f(v)=clip(v)+s*tanh((v-clip(v))/s)，s=1-th，th 处函数与导数连续
            val *= (1.0 / peak)
            clipv = np.clip(val, -th, th)
            np.subtract(val, clipv, out=val)
            val /= (1.0 - th)
            np.tanh(val, out=val)
            val *= (1.0 - th)
            np.add(val, clipv, out=val)
            val *= peak                    # 回到样本幅度，压回 ±peak
            # 混合原值（amt=1 跳过 gather）
            tgt = ss[:, None] + kinds
            np.minimum(tgt, n - 1, out=tgt)
            if amt < 1.0:
                orig = v32[tgt]
                orig *= (1.0 - amt)
                val *= amt
                np.add(val, orig, out=val)
            out[tgt[inrun], c] = val[inrun]
    return out.astype(np.float32), total


# ------------------------------------------------------------------ 市电陷波（幅度域静态曲线）

def _hum_curve(sr, mains, hum):
    """-> float64 (F,)：mains 及 2/3/4 次谐波 notch，深度 -30*hum dB。

    陷波宽度必须随频率自适应：Q=30 在 50Hz 处带宽仅 ~1.7Hz，比一个 STFT bin
    （23.4Hz @48k）还窄，若按固定 bin 数铺展，50Hz 的陷波会削掉 3~165Hz 的
    整片低频——实测这让真实 livehouse 素材离目标更远 4.28dB。改为按「陷波
    半带宽」换算应覆盖的 bin 数：只铺 ceil(带宽/bin宽) 个 bin，且最外侧 bin
    只给半深度，避免削到相邻的音乐内容。
    """
    F = core.N_FFT // 2 + 1
    freqs = core.rfftfreqs(sr)
    depth = _HUM_DEPTH * hum
    curve_db = np.zeros(F, np.float64)
    nyq = 0.49 * float(sr)
    bin_w = float(sr) / core.N_FFT
    for k in (1, 2):  # 只处理基波与二次谐波：3/4 次（150/200Hz）常落在音乐内容上
        f0 = float(mains) * k
        if f0 >= nyq:
            break
        rdb = core.db(core.mag_response("notch", sr, f0, q=_HUM_Q), floor=-80.0)
        rdb = np.maximum(rdb, depth)
        kk = int(np.argmin(np.abs(freqs - f0)))
        half_bw = f0 / _HUM_Q                      # 半功率半带宽
        span = max(1, int(np.ceil(half_bw / bin_w)))  # 应覆盖的 bin 数
        for off in range(-span, span + 1):         # 边缘 bin 半深度
            j = kk + off
            if 0 <= j < F:
                frac = 1.0 if off == 0 else (0.75 if abs(off) <= span - 1 else 0.35)
                rdb[j] = min(rdb[j], depth * frac)
        curve_db = np.minimum(curve_db, rdb)
    return core.lin(curve_db)


# ------------------------------------------------------------------ 谱减降噪（幅度域增益）

def _denoise_gain(X, sr, amt, floor_db):
    """-> float32 (ch,T,F) 幅度域增益（最小值统计谱减 + 音调保护 + 平滑）。"""
    P = np.abs(X)
    P *= P                                    # 功率 (ch,T,F) float32
    Ndb = core.noise_floor_db(X, _NPCT)       # (ch,F) 每 bin 噪声底 dB（注意：噪声底是每声道独立的）
    Ndb = Ndb + 20.0 * np.log10(_NB_C)        # 百分位偏差校正
    med = _medfilt1d(Ndb, _MED_WIN)           # 音调保护：持续音 ≠ 噪声
    Ndb = np.minimum(Ndb, med + _MED_MARGIN)
    # 注意：noise_floor_db 是功率 dB（10log10），须用 10^(dB/10) 转回功率（core.lin 是幅度约定，不能用）
    N = np.power(10.0, Ndb * 0.1).astype(np.float32)[:, None, :]  # (ch,1,F) 功率，沿时间维广播
    alpha = _ALPHA0 + _ALPHA_K * float(amt)
    g = P - alpha * N
    np.maximum(g, 0.0, out=g)
    np.maximum(P, 1e-20, out=P)             # 就地保护分母，省一个整块临时
    g /= P
    # 增益下限（覆盖「SNR<3dB 不低于 floor」；floor_db=0 即完全不压）
    np.maximum(g, core.lin(floor_db), out=g)
    g = core.smooth1d(g, 3, axis=-1).astype(np.float32)          # 频率平滑
    gt = core.env_updown(np.moveaxis(g, 1, 2), 0.010, 0.200, sr)  # 时间平滑（快起慢落）
    return np.moveaxis(np.asarray(gt, np.float32), 1, 2)


# ------------------------------------------------------------------ 泵动平滑（全谱慢增益）

def _pump_gain(X, sr, amt):
    """-> float64 (T,)：慢包络相对中位数的反向修正（±9dB 截止）× amt。"""
    fr = core.frame_rms(X)                                  # (ch,T)
    mono = np.asarray(fr.mean(axis=0), np.float64)          # (T,)
    env = core.env_updown(mono[None, :], 0.300, 0.800, sr)[0]
    mx = float(env.max()) if env.size else 0.0
    if mx <= 1e-7:
        return np.ones(env.shape, np.float64)
    valid = env > mx * 1e-3                                 # 静音段不加增益
    med = float(np.median(env[valid]))
    if med <= 1e-9:
        return np.ones(env.shape, np.float64)
    dev = 20.0 * np.log10(env / med + 1e-12)
    corr = -np.clip(dev, -_PUMP_MAX_DB, _PUMP_MAX_DB) * float(amt)
    corr[~valid] = 0.0
    return core.lin(corr)


# ------------------------------------------------------------------ 入口

def _write_ctx(ctx, d, note):
    try:
        if not isinstance(ctx, dict):
            return
        st = ctx.get("stats")
        if not isinstance(st, dict):
            st = ctx["stats"] = {}
        st[KEY] = d
        lg = ctx.get("log")
        if isinstance(lg, list):
            lg.append(note)
    except Exception:
        pass


def process(x, sr, p, ctx):
    """x: float32 (n,ch) -> float32 (n,ch) 等长。"""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    x = np.ascontiguousarray(x, np.float32)
    n, ch = x.shape
    p = p if isinstance(p, dict) else {}
    on = bool(p.get("on", True))
    denoise = _amt(p.get("denoise", 0.0))
    hum = _amt(p.get("hum", 0.0))
    declip = _amt(p.get("declip", 0.0))
    pump = _amt(p.get("pump", 0.0))
    mains = _amt(p.get("mains", 50.0), 45.0, 65.0) or 50.0
    floor_db = _amt(p.get("floor_db", -18.0), -60.0, 0.0)

    stats = {"runs": 0, "declip": declip, "denoise_db": 0.0,
             "hum_db": round(_HUM_DEPTH * hum, 2), "pump": pump}
    if (not on or n == 0
            or (denoise <= 0.0 and hum <= 0.0 and declip <= 0.0 and pump <= 0.0)):
        y = x.copy()                       # 全关：严格恒等
        _write_ctx(ctx, stats, "m5: 全部关闭，直通")
        return y

    if declip > 0.0:
        x, stats["runs"] = _declip(x, sr, declip)

    if denoise <= 0.0 and hum <= 0.0 and pump <= 0.0:
        _write_ctx(ctx, stats, "m5: 削波重建 %d 段" % stats["runs"])
        return x                            # 仅时域修复，免一次 STFT

    X = core.stft(x, sr)                    # 唯一一次 STFT
    if denoise > 0.0:
        g = _denoise_gain(X, sr, denoise, floor_db)
        stats["denoise_db"] = round(float(20.0 * np.log10(np.mean(g) + 1e-9)), 2)
        X *= g                              # complex64 *= float32，幅度域统一乘
    if hum > 0.0:
        X *= _hum_curve(sr, mains, hum)[None, None, :]
    if pump > 0.0:
        X *= _pump_gain(X, sr, pump)[None, :, None]

    y = core.istft(X, n, sr)
    y = np.asarray(y, np.float32)
    if not np.isfinite(y).all():            # 兜底：任何路径都不允许 NaN/Inf 出口
        y = np.nan_to_num(y, nan=0.0, posinf=2.0, neginf=-2.0)
    _write_ctx(ctx, stats, "m5: 削波重建 %d 段 · 降噪 %.1f dB · %.0fHz 陷波 · 泵动 %.2f"
               % (stats["runs"], stats["denoise_db"], mains, pump))
    return y


def info():
    return {"key": KEY, "name": "修复层",
            "algo": "Hermite 削波重建 · 谐波陷波 Q30 · 最小值统计谱减 · 慢 AGC 平滑"}
