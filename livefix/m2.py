"""LiveFix m2 —— 中高频补充（带宽扩展 / BWE）。

三个互相联动的谱域处理（全部 STFT 幅度域，保留原相位）：

1. air（空气感 · 带宽扩展）
   - 倒谱包络：对每帧 log 幅度沿频率做 rfft，只保留前 40 个倒谱系数再 irfft，
     得到平滑谱包络 E[ch,T,F]（dB）；
   - 外推：在 4k-8k 段对时间平均包络做 log2(f) 线性拟合得斜率 s（dB/oct），
     外推斜率 s_ext = 0.6*s（衰减避免过亮），锚点取每帧 4k-8k 包络均值，
     ext_hz 以上 E_ex = anchor + s_ext*(log2 f - log2 f_anchor)；
   - 合成：ext_hz 以上 M_syn = lin(E_ex) * 瑞利幅度 * 随机相位（白噪声底），
     整体按 sqrt(0.08*air*E(4-8k)/E(syn)) 缩放（逐声道）；
   - 交叉淡化：ext_hz 起 1/3 八度余弦斜边，与原谱复数域交叉淡化；
   - dehiss 联动：6-16k 噪声底高于 -60 dBFS 时，合成量按 dehiss 缩减
     （severity 归一，最多 -80%），避免往嘶声上"补噪声"。
2. detail（细节 · 谐波激励）
   - 700-3500Hz 的 bin 以 0.12*detail 幅度复制到 2f（bin 域 k→2k 精确翻倍，
     幅度相位同拷贝，产生二次谐波"振动感"）；
   - 另加 4-8k 静态高架 detail*2.5 dB（smooth_curve_db）。
3. dehiss（嘶声控制）
   - noise_floor_db(X, 15) 作参考底，6-16k 内超出底噪 +18dB 的部分软压缩：
     gain_dB = -min(over*0.4*dehiss, 8)，帧域 env_updown(5ms/120ms) 平滑。

参数全 0 时恒等（early-return，位级不变）；ext_hz 仅 air>0 时生效。
纯 numpy；随机数取 ctx["rng"]（缺省 default_rng(0)）保证可复现。
"""
from __future__ import annotations

import numpy as np

from . import core

KEY = "m2"

_KEPT_CEPS = 40        # 倒谱提升保留的倒谱系数个数
_SLOPE_DAMP = 0.6      # 外推斜率衰减（防过亮）
_SYNTH_SHARE = 0.08    # 兼容旧常量（不再用于 air；air 改为 dB 目标制，见 _AIR_MAX_DB）
# ---- 参数权威度（拉满时的幅度，均可自参考，保证任何素材上 ≥3dB）----
_AIR_MAX_DB = 18.0     # air=1 → ext_hz 以上带内能量抬升 18 dB（相对该带自身）
_AIR_FLOOR = 1e-4      # ext 带过弱时的下限（相对 4-8k 能量，-40dB），避免补了个听不见的量
_AIR_HI_HZ = 18000.0   # 扩展上界：旧实现一路铺到 Nyquist(24k)，能量大半落在听不见的 18-24k，
                       # 既浪费余量又让 11-16k 的可听变化只有 0.94dB。现在收在 18k 以内。
_DETAIL_MIX = 0.5      # detail=1 → 2f 混入幅度（相对源 bin 幅度）
_DETAIL_SHELF_DB = 6.0  # detail=1 → 4-8k 静态高架（dB）
_DEHISS_MAX_DB = 24.0  # dehiss=1 → 6-16k 最大衰减（dB）
_DEHISS_MARGIN_DB = 18.0  # 距噪声底 18dB 以内开始衰减（真正的嘶声/底噪压制）


def _f(v, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float(default)
    return v if np.isfinite(v) else float(default)


def _clip01(v, default=0.5):
    return float(min(max(_f(v, default), 0.0), 1.0))


def _cep_env_db(Ldb):
    """(T,F) log 幅度 dB -> 平滑谱包络 dB：倒谱提升（保留前 _KEPT_CEPS 个系数）。"""
    F = Ldb.shape[-1]
    C = np.fft.rfft(Ldb, axis=-1)
    K = min(_KEPT_CEPS, C.shape[-1])
    C[..., K:] = 0.0
    return np.fft.irfft(C, n=F, axis=-1)


def _dehiss(X, sr, nf, dehiss):
    """6-16k 嘶声/底噪压制：把接近或低于噪声底的 bin 按距离线性下压，最多 _DEHISS_MAX_DB*dehiss。

    旧实现压缩的是「超出底噪 +18dB 的部分」（即响的高频内容），这在真实素材上几乎不触发
    （实测拉满只动 0.12dB）；改为压制「贴近底噪的部分」——这才是嘶声控制的物理含义，
    并且任何有底噪的素材都会触发。level 是 20log10|X|，nf 是 10log10|X|²，两者同刻度可直接比。
    """
    hi = min(16000.0, 0.47 * float(sr))
    if hi <= 6500.0:
        return X, 0.0
    mask = core.band_mask(sr, 6000.0, hi, edge_oct=0.25)
    bins = np.nonzero(mask > 1e-6)[0]                     # 只处理 6-16k 的 bin
    if bins.size == 0:
        return X, 0.0
    Xb = X[:, :, bins]
    level = 20.0 * np.log10(np.abs(Xb) + 1e-12)           # (ch,T,B)
    near = np.clip((nf[:, None, bins] + _DEHISS_MARGIN_DB - level) / _DEHISS_MARGIN_DB,
                   0.0, 1.0)                              # 越贴近底噪越接近 1
    g = -(_DEHISS_MAX_DB * dehiss) * near * mask[bins][None, None, :]
    g = core.env_updown(np.moveaxis(g, 1, -1), 0.005, 0.120, sr)
    g = np.moveaxis(g, -1, 1)
    cut = float(-g.min()) if g.size else 0.0
    X = X.copy()
    X[:, :, bins] = Xb * core.lin(g)
    return X, cut


def _detail(X, sr, detail):
    """二次谐波加成（k→2k，**相位加倍** 2θ 生成真谐波）+ 4-8k 静态高架。

    旧实现把 X[k] 整体搬到 2k（复制相位），得到的是与基频无固定相位关系的分量
    （拍频/不稳定感），不是二次谐波。真谐波要求 θ→2θ。另外 2k 可能同时落在源带内，
    故先取快照再写，避免级联污染。
    """
    F = X.shape[-1]
    freqs = core.rfftfreqs(sr)
    k_lo = int(np.searchsorted(freqs, 700.0))
    k_hi = int(np.searchsorted(freqs, 3500.0))
    ks = np.arange(k_lo, max(k_hi, k_lo + 1))
    ks = ks[2 * ks < F]
    if ks.size:
        X = X.copy()
        src = X[:, :, ks].copy()                     # 快照
        ang = np.angle(src)
        har = (np.abs(src) * np.float32(_DETAIL_MIX * detail)) * \
            np.exp(1j * (2.0 * ang).astype(np.float32))
        X[:, :, 2 * ks] += har.astype(np.complex64)
    shelf_db = _DETAIL_SHELF_DB * detail
    curve = core.smooth_curve_db(sr, [(1500.0, 0.0), (4000.0, shelf_db)])
    return (X * core.lin(curve)[None, None, :]).astype(np.complex64)


def _band_ext(X, sr, air, dehiss, nf, rng, ext_hz):
    """带宽扩展主体（见模块 docstring 第 1 条）。"""
    freqs = core.rfftfreqs(sr)
    F = freqs.shape[0]

    ext = min(max(float(ext_hz), 2000.0), 0.45 * float(sr))
    # 交叉淡化放在 ext **以下**的 1/3 八度内：这样 ext_hz 才是「从哪个频率以上开始合成」。
    # 旧实现把斜边放在 ext 以上（11k→13.9k），于是声明的 11kHz 实际到 13.9kHz 才满量，
    # 11-14k 全被原有内容占着 —— 实测 air 拉满在 11-16k 只动 0.94dB，参数形同虚设。
    ext_lo = ext / (2.0 ** (1.0 / 3.0))
    fl_fit, fh_fit = 4000.0, min(8000.0, 0.44 * float(sr))
    ext_top = min(0.47 * float(sr), _AIR_HI_HZ)
    if ext_top <= ext or fh_fit <= fl_fit * 1.15:
        return X, None                            # 频带不够，放弃扩展

    # dehiss 联动：嘶声明显时按 dehiss 缩减合成量（最多 -80%）
    hi616 = min(16000.0, 0.47 * float(sr))
    m616 = core.band_mask(sr, 6000.0, hi616, edge_oct=0.25)
    nf_band = (nf * m616[None, :]).sum(axis=-1) / max(float(m616.sum()), 1e-9)  # (ch,)
    sev = np.clip((nf_band + 60.0) / 24.0, 0.0, 1.0)                            # (ch,)
    air_eff = air * (1.0 - 0.8 * dehiss * sev)                                   # (ch,)

    fit_bins = np.nonzero((freqs >= fl_fit) & (freqs <= fh_fit))[0]
    x_fit = np.log2(freqs[fit_bins])
    f_anchor = float(2.0 ** x_fit.mean())

    m_cross = ((freqs >= ext_lo) & (freqs < ext)).astype(np.float32)
    if m_cross.any():
        t = (freqs[m_cross > 0] - ext_lo) / max(ext - ext_lo, 1e-9)
        m_cross[m_cross > 0] = 0.5 - 0.5 * np.cos(np.pi * t)
    w = np.zeros(F, np.float32)
    w[freqs >= ext] = 1.0
    w[freqs >= ext_lo] = np.maximum(w[freqs >= ext_lo], m_cross[freqs >= ext_lo])
    # 上界收敛到听感有效的 18kHz（软边），避免把能量灌进 18-24kHz 的听不见区
    if ext_top > ext:
        w *= core.band_mask(sr, ext, ext_top, edge_oct=0.15).astype(np.float32)
    ext_bins = np.nonzero(w > 1e-6)[0]
    if ext_bins.size < 4:
        return X, None
    w_e = w[ext_bins]                       # (nb,) 交叉淡化权重
    w2 = (w_e ** 2).astype(np.float32)
    log2_ext_bins = np.log2(freqs[ext_bins]).astype(np.float32) - np.float32(np.log2(f_anchor))
    k_exp = np.float32(np.log(10.0) / 20.0)

    out = X.copy()
    scale_db = np.zeros(X.shape[0], np.float64)
    T = X.shape[1]
    nb = ext_bins.size
    for c in range(X.shape[0]):
        A = np.abs(X[c])                                        # (T,F) float32
        Ldb = (20.0 * np.log10(A + 1e-10)).astype(np.float32)
        E = _cep_env_db(Ldb)                                    # (T,F) float32 包络 dB
        Ef = E[:, fit_bins]
        s = float(np.polyfit(x_fit, Ef.mean(axis=0), 1)[0])     # dB/oct
        s = float(np.clip(s, -40.0, 6.0)) * _SLOPE_DAMP
        s = np.float32(np.clip(s, -40.0, 3.0))
        E_ex = Ef.mean(axis=1)[:, None] + s * log2_ext_bins[None, :]   # (T,nb) dB

        # 复高斯噪声：幅度天然服从瑞利分布、相位均匀，一次生成替代 uniform+log+exp
        Z = rng.standard_normal((T, nb), dtype=np.float32)
        Z = Z + 1j * rng.standard_normal((T, nb), dtype=np.float32)
        M_syn = np.exp(E_ex * k_exp)                            # (T,nb) 线性幅度

        pref = float(np.sum(A[:, fit_bins] ** 2, dtype=np.float64))
        # air 改为「dB 目标」制：把 ext_hz 以上的带内能量抬高 _AIR_MAX_DB*air dB（相对该带自身）。
        # 旧实现是「合成 8% 的 4-8k 能量」，与素材高频存量耦合 —— 实测拉满只动 0.48dB，
        # 参数形同虚设。自参考后 air 与「补多少 dB」一一对应，且任何素材都能达到目标。
        pext = float(np.sum((A[:, ext_bins] ** 2) * w2[None, :], dtype=np.float64))
        pbase = max(pext, pref * _AIR_FLOOR)          # 扩展带过弱时退化到 4-8k 的 -40dB
        psyn = float(np.sum((M_syn ** 2) * w2[None, :], dtype=np.float64))
        g = np.sqrt((10.0 ** (_AIR_MAX_DB * float(air_eff[c]) / 10.0) * pbase)
                    / (psyn + 1e-30))
        scale_db[c] = 20.0 * np.log10(max(g, 1e-9))

        syn = (g * M_syn) * Z                                   # complex64
        reg = out[c][:, ext_bins]
        # 只补能量、不替换原谱：ext_hz 以上常已有真实内容（乐器泛音、房间反射），
        # 用 (1-w)*原谱 + w*合成 会把它们抹掉——实测 ext_hz=7000 时让真实素材
        # 离目标更远 1.80dB。改为幅度域按 w 加合成（非相干叠加，避免相位抵消），
        # 原谱完整保留，合成只是补上缺失的高频能量。
        add = (w_e[None, :] * np.abs(syn)).astype(np.complex64)
        out[c][:, ext_bins] = (reg + add).astype(np.complex64)
    return out, {"ext_hz": round(ext, 1), "nf_band_db": [round(float(v), 1) for v in nf_band],
                 "synth_gain_db": [round(float(v), 1) for v in scale_db],
                 "air_eff": [round(float(v), 3) for v in air_eff]}


def process(x, sr, p, ctx):
    """x: float32 (n,ch) -> float32 (n,ch) 等长。"""
    x = np.asarray(x)
    squeeze = False
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    x = x.astype(np.float32, copy=False)
    if x.ndim != 2:
        raise ValueError(f"音频应为 (n, ch)，得到 {x.shape}")
    n = x.shape[0]
    p = p or {}
    on = bool(p.get("on", True))
    air = _clip01(p.get("air"), 0.5)
    detail = _clip01(p.get("detail"), 0.5)
    dehiss = _clip01(p.get("dehiss"), 0.4)
    ext_hz = _f(p.get("ext_hz"), 11000.0)

    if (not on) or n == 0 or (air <= 0.0 and detail <= 0.0 and dehiss <= 0.0):
        return x.squeeze(-1) if squeeze else x          # 恒等（位级不变）

    if isinstance(ctx, dict):
        rng = ctx.get("rng") or np.random.default_rng(0)
        stats = ctx.setdefault("stats", {}) if isinstance(ctx.get("stats"), dict) else {}
    else:
        rng = np.random.default_rng(0)
        stats = {}
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(0)

    X = core.stft(x, sr)
    nf = core.noise_floor_db(X, pct=15.0)               # (ch,F) 参考噪声底 dB

    cut = 0.0
    if dehiss > 0.0:
        X, cut = _dehiss(X, sr, nf, dehiss)
    if detail > 0.0:
        X = _detail(X, sr, detail)
    air_info = None
    if air > 0.0:
        X, air_info = _band_ext(X, sr, air, dehiss, nf, rng, ext_hz)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = core.istft(X, n, sr).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    if isinstance(ctx, dict):
        stats["m2"] = {"air_on": air > 0.0, "dehiss_cut_db": round(cut, 2),
                       "detail": detail, "air": air}
        if air_info:
            stats["m2"].update(air_info)
        log = ctx.get("log")
        if isinstance(log, list):
            log.append(f"m2: air={air} detail={detail} dehiss={dehiss} ext={ext_hz}Hz"
                       + (f" 外推至 {air_info['ext_hz']}Hz" if air_info else " 外推跳过"))
    return y.squeeze(-1) if squeeze else y


def info():
    return {"key": KEY, "name": "中高频补充", "algo": "倒谱包络外推 · 谐波激励 · 嘶声软压"}
