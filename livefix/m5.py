"""livefix · m5 修复层（去削波 / 抑交流声 / 场地去混响抑共振 / 去泵动）。

处理顺序（契约）：
    declip（时域重建） → dereverb + hum + pump（一次 STFT，各增益相乘 = dB 相加后统一作用于幅度域）

算法要点：
- declip：|x| >= 0.985*peak 的连续平坦段（run）用三次 Hermite（Catmull-Rom 切线）重建，
  接缝处用 C1 连续的 tanh 软限幅压回 ±peak；强度 declip 控制原值/重建值混合比。
- hum：mains 及其 2/3/4 次谐波各做一个 RBJ notch（Q=30，core.mag_response），深度 -30*hum dB；
  因 Q=30 的凹口比一个 STFT bin 还窄，深度按 bin 量化铺到最近的 1+2+2 个 bin 上，保证真实生效。
- dereverb（2026-10-14 起，**取代原 denoise + floor_db**）：针对场地造成的中低频共振峰与
  混响拖尾。**只在「正在衰减」的帧上动作**（双门：一窗跌幅 > 35%，且仍在过去峰值 18dB 以内），
  因此攻击瞬间、稳态乐音（持续贝斯 / 长音）、静音段一律不动 —— 这是它不去削「持续音基频」
  的关键（静态共振陷波最大的风险就是把乐音基频当成房间模态）。
  带内每帧下压量 = 拖尾深度 × (0.55 + 0.45 × 共振权重)：
    · 拖尾深度 = 12dB × amt × pos^1.5，pos=(M−E)/(M−F) 定位衰减深度；
    · 共振权重 = 该帧每个 bin 相对「跨频中位数基线（9 bin）」的突起（>3dB 才计，
      1.2× 突起映射到 0..1，对应换算上限 15dB）——窄带模态突起高、宽带混响突起小。
  频带 70-600Hz（band_mask 软边 0.3 八度，带外恒 1），时间轴 env_updown(30ms/40ms) 慢入快出。
  ⚠ 两条如实记录的限制：(1) 48k 下 STFT bin 间隔 23.4Hz，房间模态带宽常只有几到几十 Hz，
    所以它压的是「模态所在的窄频区」，不是 Q>30 的单点陷波；(2) 对密集无间隙的素材，
    衰减段判定难以分离，效果会自然变弱 —— 这是统计式去混响的固有限制。
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
_MED_WIN = 21         # （保留：跨频中位数窗的旧值，现由 _RES_MED 使用）
_HUM_Q = 30.0
_HUM_DEPTH = -30.0    # hum=1 时每个谐波陷波深度 dB
_PUMP_MAX_DB = 9.0    # pump 的单帧最大修正量（dB；旧值 6，拉满的权威度不足）

# ---- dereverb：场地中低频共振峰抑制 + 混响拖尾抑制 ----
# 只在「正在衰减」的帧上动作（span + 范围双门）→ 稳态乐音（持续贝斯/长音）一律不动，
# 避免把「持续音的基频」当成「房间模态」削掉 —— 这是静态共振陷波最大的风险。
_DR_LO, _DR_HI = 70.0, 600.0      # 作用频带（房间模态与混响拖尾集中区）
_DR_MAX_DB = 12.0                 # 拖尾段整体下压上限（共振 bin 可打满，普通 bin 打 55%）
_DR_RANGE_DB = 18.0               # 只处理仍在峰值 18dB 以内的帧（更低的算静音，不压）
_DR_PAST, _DR_FUT = 60, 80        # 滑窗帧数（HOP=512@48k → 10.67ms/帧：0.64s / 0.85s）
_DR_SPAN_THR = 0.35               # 一窗内跌幅低于此判为持续音/延音 → 保护
_DR_ATT, _DR_REL = 0.030, 0.040   # 慢介入 / 快退出（退出快 → 下一个瞬态不被前一段拖累）
_DR_ENV_W = 7                     # 逐 bin 包络的时间平滑窗（帧，≈75ms）——单 bin 幅度抖得厉害，
                                  # 不平滑的话衰减判定会被噪声帧拉偏（实测权威度掉一半）
_RES_MED = 9                      # 共振突起：跨频中位数基线窗（bin）
_RES_MIN_DB = 3.0                 # 突起低于此不算共振峰
_RES_MAX_DB = 15.0                # 共振峰额外下压的换算上限
_RES_GAIN = 1.2                   # 突起量 → 权重斜率
_RES_BASE = 0.55                  # 非共振 bin 享受的拖尾下压比例


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


# ------------------------------------------------------------------ 场地去混响 / 抑共振

def _past_max(A, w):
    """M[t] = max(A[t-w..t])：逐位移原地迭代，内存安全（只留一份拷贝）。"""
    M = np.array(A, np.float64, copy=True)
    Tn = M.shape[0]
    for i in range(1, min(int(w), Tn - 1) + 1):
        np.maximum(M[i:], A[:Tn - i], out=M[i:])
    return M


def _future_min(A, w):
    """F[t] = min(A[t..t+w])：未来窗滑动最小，逐位移原地迭代（本地版，与 m1 同写法）。"""
    D = np.array(A, np.float64, copy=True)
    Tn = D.shape[0]
    for i in range(1, min(int(w), Tn - 1) + 1):
        np.minimum(D[:Tn - i], A[i:], out=D[:Tn - i])
    return D


def _dereverb_gain(X, sr, amt):
    """-> (gain (T,F) float32, stats dict)

    带内 = 「拖尾整体下压 × (0.55 + 0.45×共振权重)」，带外恒 1（借 band_mask 软边落回）。
    只在衰减帧上动作，所以攻击、稳态乐音、静音段都不受影响。
    """
    mask = core.band_mask(sr, _DR_LO, _DR_HI)
    idx = np.nonzero(mask > 1e-3)[0]
    Tn, F = X.shape[1], X.shape[2]
    st = {"dereverb_db": 0.0, "res_peaks": 0, "res_max_db": 0.0}
    if idx.size == 0 or Tn < 4:                       # 极低采样率 / 太短兜底
        return np.ones((Tn, F), np.float32), st

    amp0 = np.abs(X).mean(axis=0).astype(np.float64)[:, idx]     # (T,B) 声道均值幅度
    amp = core.smooth1d(amp0, _DR_ENV_W, axis=0)                # 平滑版：只用于逐 bin 衰减门

    # 压多少：用**整带**衰减深度（包络稳、抗单 bin 抖动）
    E = amp0.mean(axis=1)                                       # (T,) 带内包络
    Mb = _past_max(E, _DR_PAST)
    Fb = _future_min(E, _DR_FUT)
    span_b = (Mb - Fb) / np.maximum(Mb, 1e-9)
    pos = np.clip((Mb - E) / (Mb - Fb + 1e-9), 0.0, 1.0)        # 攻击≈0 → 深入衰减→1
    pos[(span_b <= _DR_SPAN_THR)
        | (E <= Mb * (10.0 ** (-_DR_RANGE_DB / 20.0)))] = 0.0   # 持续音 / 静音段不动

    # 压谁：**逐 bin** 衰减门。持续贝斯的基频 bin 自身平坦 → span≈0 → 被保护；
    # 鼓激起的宽带衰减与慢衰减的模态 bin 各自开门。整带门在密集素材上会被鼓打开，
    # 从而连带压到同时响着的持续音，所以保护必须按 bin 判。
    M = _past_max(amp, _DR_PAST)                                # (T,B) 过去峰值
    Fu = _future_min(amp, _DR_FUT)                              # (T,B) 未来谷值
    span = (M - Fu) / np.maximum(M, 1e-9)
    gate = (span > _DR_SPAN_THR) & (amp > M * (10.0 ** (-_DR_RANGE_DB / 20.0)))

    # 共振权重：该帧内每个 bin 相对「跨频中位数基线」的突起（窄带模态突出，宽带混响不突出）
    base = _medfilt1d(amp0, _RES_MED)                           # (T,B)
    prom = 20.0 * np.log10(np.maximum(amp0, 1e-9)) - 20.0 * np.log10(np.maximum(base, 1e-9))
    prom = np.maximum(prom - _RES_MIN_DB, 0.0)
    res_w = np.minimum(prom * _RES_GAIN, _RES_MAX_DB) / _RES_MAX_DB    # 0..1

    supp = (_DR_MAX_DB * float(amt) * np.power(pos, 1.5))[:, None] \
        * (_RES_BASE + (1.0 - _RES_BASE) * res_w) * gate        # (T,B) dB
    supp = core.env_updown(supp.T, _DR_ATT, _DR_REL, sr).T      # 沿时间平滑（末维须为时间）
    g = np.power(10.0, -supp / 20.0)

    out = np.ones((Tn, F), np.float32)
    out[:, idx] = (1.0 + (g - 1.0) * mask[idx][None, :]).astype(np.float32)   # 软边落回全谱
    st["dereverb_db"] = round(float(20.0 * np.log10(np.mean(out) + 1e-9)), 2)
    if gate.any():
        st["res_peaks"] = int(np.count_nonzero((res_w * gate) > 0.05))
        st["res_max_db"] = round(float(-supp.max()), 2)
    return out, st


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
    dereverb = _amt(p.get("dereverb", 0.0))
    hum = _amt(p.get("hum", 0.0))
    declip = _amt(p.get("declip", 0.0))
    pump = _amt(p.get("pump", 0.0))
    mains = _amt(p.get("mains", 50.0), 45.0, 65.0) or 50.0

    stats = {"runs": 0, "declip": declip, "dereverb_db": 0.0, "res_peaks": 0,
             "res_max_db": 0.0, "hum_db": round(_HUM_DEPTH * hum, 2), "pump": pump}
    if (not on or n == 0
            or (dereverb <= 0.0 and hum <= 0.0 and declip <= 0.0 and pump <= 0.0)):
        y = x.copy()                       # 全关：严格恒等
        _write_ctx(ctx, stats, "m5: 全部关闭，直通")
        return y

    if declip > 0.0:
        x, stats["runs"] = _declip(x, sr, declip)

    if dereverb <= 0.0 and hum <= 0.0 and pump <= 0.0:
        _write_ctx(ctx, stats, "m5: 削波重建 %d 段" % stats["runs"])
        return x                            # 仅时域修复，免一次 STFT

    X = core.stft(x, sr)                    # 唯一一次 STFT
    if dereverb > 0.0:
        g, dstat = _dereverb_gain(X, sr, dereverb)
        stats.update(dstat)
        X *= g                              # (T,F) 广播到 (ch,T,F)，幅度域统一乘
    if hum > 0.0:
        X *= _hum_curve(sr, mains, hum)[None, None, :]
    if pump > 0.0:
        X *= _pump_gain(X, sr, pump)[None, :, None]

    y = core.istft(X, n, sr)
    y = np.asarray(y, np.float32)
    if not np.isfinite(y).all():            # 兜底：任何路径都不允许 NaN/Inf 出口
        y = np.nan_to_num(y, nan=0.0, posinf=2.0, neginf=-2.0)
    _write_ctx(ctx, stats, "m5: 削波重建 %d 段 · 去混响 %.1f dB（共振峰 %d 个，最深 %.1f dB）"
               " · %.0fHz 陷波 · 泵动 %.2f"
               % (stats["runs"], stats["dereverb_db"], stats["res_peaks"],
                  stats["res_max_db"], mains, pump))
    return y


def info():
    return {"key": KEY, "name": "修复层",
            "algo": "Hermite 削波重建 · 谐波陷波 Q30 · 场地去混响 / 中低频抑共振 · 慢 AGC 平滑"}
