"""LiveFix 源分离分级处理（置于所有后处理之前）。

算法（纯 numpy，无 scipy）：
  共享相位掩码：|X| 取各声道均值 (T,F) -> 四个掩码 vocal/low/perc/other，
  逐 bin 归一化使掩码和恒为 1（静音 bin 回退为 0.25x4），各 stem 保留共同相位 X*mask。

  mode="hpss"（快）：Fitzgerald 2010 经典 HPSS——|X| 沿时间轴滑动中值（核 25 帧）
    得谐波分量 H，沿频率轴滑动中值（核 31 bin）得打击分量 P；谐波性 h = H/(H+P)。
    再细分：150-4500Hz 且谐波性强 -> vocal；<150Hz 且谐波性强 -> low；
    打击分量且 >2kHz -> perc；其余（含宽带噪声底）-> other。四掩码归一化。
  mode="nmf"（准）：秩 4 NMF（Lee-Seung 欧氏距离乘性更新，30 次迭代），
    在隔 2 帧取样的 |X| 上运行；字典 W 固定为四个先验模板（vocal 谐波峰形 / low<250Hz 平坦 /
    perc 全带+高频倾斜 / other 平缓斜坡），仅更新激活 U，再按各分量重构幅度做 Wiener 式掩码归一化。
    掩码由双声道 |X| 均值算出并套用到所有声道（省算力、相位一致）。

分级路由（spec/CORE.md §6，引擎依赖）：
  返回值 = vocal*vocal_gain + low*low_gain + perc*perc_gain + other*other_gain；
  routing=="graded" 时写 ctx["stems"] = {"vocal", "backing"}，且 vocal+backing == 返回值；
  routing=="mix" 时 ctx["stems"] = None。

分段安全：引擎以 core.block_apply（约 60s 段 + 1s 线性交叉淡化）分段调用本模块时，
  本模块用输入段的原始采样精确检测相邻段重叠长度，并复刻 block_apply 的交叉淡化权重，
  把各段 stems 累加成整轨 stems，保证 ctx["stems"]["vocal"]+["backing"] 与引擎最终输出一致。
"""
from __future__ import annotations

import numpy as np

from . import core

KEY = "sep"

_EPS = 1e-12
_MED_CHUNK = 1024      # 中值滤波分块帧数（控制内存）
_NMF_ITERS = 30        # NMF 迭代次数
_NMF_STEP = 2          # NMF 帧取样步长


def info():
    return {"key": KEY, "name": "源分离分级",
            "algo": "HPSS 滑动中值 / 秩4 NMF · Wiener 掩码和恒为1 · vocal/low/perc/other 分级路由"}


# ---------------------------------------------------------------- 滑动中值（分块）

def _med1d(a2d: np.ndarray, k: int, axis: int) -> np.ndarray:
    """a2d (T,F) float32 -> 同形 float32：沿 axis 的滑动中值（核 k，自动取奇数），分块控制内存。

    实现 = sliding_window_view + np.partition（取中间序统计量）。k 为奇数时与 np.median
    逐样本一致（np.median 内部同样走 partition，但额外做均值/拷贝，实测慢约 2.3 倍）。
    """
    k = int(k) | 1
    pad = k // 2
    T = a2d.shape[0]
    out = np.empty_like(a2d)
    if axis == 0:
        # 转置后沿末轴开窗：窗口在内存中连续，partition 快约 25%（实测 1.38s -> 1.04s @60s）
        p = np.pad(a2d, ((pad, pad), (0, 0)), mode="edge")
        p = np.ascontiguousarray(p.T)                                    # (F, T+2p)
        w = np.lib.stride_tricks.sliding_window_view(p, k, axis=1)       # (F, T, k)
        tmp = np.empty((p.shape[0], T), np.float32)
        step = _MED_CHUNK * 2
        for a in range(0, T, step):
            b = min(a + step, T)
            tmp[:, a:b] = np.partition(w[:, a:b], k // 2, axis=-1)[..., k // 2]
        out[:] = tmp.T
    else:
        p = np.pad(a2d, ((0, 0), (pad, pad)), mode="edge")
        for a in range(0, T, _MED_CHUNK):
            b = min(a + _MED_CHUNK, T)
            w = np.lib.stride_tricks.sliding_window_view(p[a:b], k, axis=1)  # (m,F,k)
            out[a:b] = np.partition(w, k // 2, axis=-1)[..., k // 2]
    return out


# ---------------------------------------------------------------- 掩码归一化

def _normalize(raws):
    """4 个原始掩码 -> (float32 掩码列表, 最大和偏差)；逐 bin 除以总和使和恒为 1，
    静音 bin（总和≈0）回退为各 0.25，保证全增益恒等。内部断言和为 1。"""
    r0, r1, r2, r3 = (np.asarray(r, np.float32) for r in raws)
    s = r0 + r1 + r2 + r3
    nz = s > np.float32(_EPS)
    one = np.float32(1.0)
    q = np.float32(0.25)
    inv = np.where(nz, one / np.where(nz, s, one), q)
    out = [np.where(nz, r * inv, q) for r in (r0, r1, r2, r3)]
    total = out[0] + out[1] + out[2] + out[3]
    dev = float(np.max(np.abs(total - 1.0))) if total.size else 0.0
    if dev > 1e-6:
        raise AssertionError(f"sep: 掩码和偏离 1 过大：{dev:.3e}")
    return out, dev


# ---------------------------------------------------------------- hpss

def _masks_hpss(mag: np.ndarray, sr):
    """mag (T,F) float32 -> (4 掩码, 和偏差)。经典 HPSS + 频带/谐波性细分（全程 float32）。"""
    H = _med1d(mag, 25, 0)          # 时间轴中值 -> 谐波分量
    P = _med1d(mag, 31, 1)          # 频率轴中值 -> 打击分量
    H2 = H * H
    P2 = P * P
    den = H2 + P2 + np.float32(_EPS)
    Hm = H2 / den                   # 谐波 Wiener 掩码
    Pm = np.float32(1.0) - Hm       # 打击 Wiener 掩码（与 Hm 严格互补）
    h = (H + np.float32(1e-6)) / (H + P + np.float32(1e-6))   # 谐波性度量 0..1

    wv = core.band_mask(sr, 150.0, 4500.0).astype(np.float32)   # 人声带
    wl = core.band_mask(sr, 10.0, 150.0).astype(np.float32)     # 低频带
    wp = core.band_mask(sr, 2000.0, float(sr)).astype(np.float32)  # 高频打击带（上边不裁）
    wh = np.maximum(wv, wl)

    raw_v = (Hm * h) * wv
    raw_l = (Hm * h) * wl
    raw_p = Pm * wp
    raw_o = Hm * (np.float32(1.0) - wh) + Pm * (np.float32(1.0) - wp)
    return _normalize([raw_v, raw_l, raw_p, raw_o])


# ---------------------------------------------------------------- nmf

def _nmf_templates(sr) -> np.ndarray:
    """四个先验模板 -> (F,4) float64，列序 = vocal/low/perc/other，列单位范数。"""
    f = core.rfftfreqs(sr)
    fnz = np.maximum(f, 1.0)
    nyq = max(float(sr) * 0.5, 20.0)
    lg = np.log2(fnz)

    vocal = np.exp(-0.5 * ((lg - np.log2(800.0)) / 1.4) ** 2)           # 150-4500Hz 谐波峰形
    vocal *= 0.65 + 0.35 * np.cos(2.0 * np.pi * np.log2(fnz / 200.0) * 3.0)
    vocal[f < 80.0] = 0.0
    low = 1.0 / (1.0 + (fnz / 220.0) ** 4)                              # <250Hz 平坦
    low[f < 1.0] = 1.0
    perc = 0.15 + 0.85 * np.clip(f / nyq, 0.0, 1.0) ** 1.5              # 全带 + 高频倾斜
    other = 1.0 / (1.0 + (fnz / 550.0) ** 2.5)                          # 平缓斜坡

    W = np.stack([vocal, low, perc, other], axis=1)
    W = np.maximum(W, 1e-4)
    W /= np.sqrt((W ** 2).sum(axis=0, keepdims=True))
    return W


def _masks_nmf(mag: np.ndarray, sr):
    """mag (T,F) float32 -> (4 掩码, 和偏差)。

    秩 4 NMF（Lee-Seung 欧氏距离乘性更新，30 次迭代）在隔 2 帧取样的 |X| 上运行。
    字典 W 固定为四个先验模板（仅更新激活 U）：30 次迭代的预算内让 W 自由漂移
    会迅速破坏模板与 vocal/low/perc/other 的对应关系，实测冻结字典的分级质量更稳。
    之后按各分量重构幅度做 Wiener 式掩码归一化。
    """
    T = mag.shape[0]
    V = np.ascontiguousarray(mag[::_NMF_STEP], np.float64)     # (M,F) 隔帧取样
    W = _nmf_templates(sr)                                     # (F,4) 固定字典
    wn = np.maximum((W ** 2).sum(axis=0), 1e-12)
    U = np.maximum(V @ W / wn[None, :], 1e-6)                  # (M,4) 初始化
    WtW = W.T @ W
    eps = 1e-10
    for _ in range(_NMF_ITERS):
        U *= (V @ W) / (U @ WtW + eps)                         # 欧氏距离乘性更新
    Uf = np.repeat(U, _NMF_STEP, axis=0)
    if Uf.shape[0] > T:
        Uf = Uf[:T]
    elif Uf.shape[0] < T:
        Uf = np.pad(Uf, ((0, T - Uf.shape[0]), (0, 0)), mode="edge")
    Uf32 = Uf.astype(np.float32)
    Wf32 = W.T.astype(np.float32)                               # (4,F)
    comps = [np.ascontiguousarray(Uf32[:, k:k + 1] * Wf32[k:k + 1]) for k in range(4)]
    return _normalize(comps)


# ---------------------------------------------------------------- stems 跨段拼接

def _overlap_len(prev: np.ndarray, cur: np.ndarray, max_cand: int = 64) -> int:
    """精确检测 cur 的开头与 prev 结尾的重叠样本数（原始采样逐位相等，O(候选) 校验）。"""
    P = prev.shape[0]
    N = cur.shape[0]
    kmax = min(P, N)
    if kmax <= 0:
        return 0
    cand = np.flatnonzero(prev[:, 0] == cur[0, 0])
    cand = cand[cand >= P - kmax]
    for idx in cand[:max_cand]:
        i = int(idx)
        k = P - i
        if np.array_equal(cur[:k], prev[i:]):
            return k
    return 0


def _stems_push(ctx, V: np.ndarray, B: np.ndarray, xseg: np.ndarray):
    """把本段 (vocal, backing) 累加进整轨 stems，并复刻 core.block_apply 的重叠交叉淡化。

    相邻段重叠区在引擎侧的合成是 prev*(1-t)+cur*t（wsum 恒为 1）；本函数保持同样权重，
    使 ctx["stems"]["vocal"]+["backing"] 与 block_apply 的最终输出逐样本一致（仅浮点舍入差）。
    """
    V = np.asarray(V, np.float32)
    B = np.asarray(B, np.float32)
    st = ctx.get("_sep_stems")
    if st is None:                      # 首段（或不分段的整轨）
        v = V.copy()
        b = B.copy()
        ctx["_sep_stems"] = {"v": v, "b": b, "x": np.array(xseg, np.float32, copy=True)}
        ctx["stems"] = {"vocal": v, "backing": b}
        return
    Lp = st["x"].shape[0]
    ov = min(_overlap_len(st["x"], xseg), Lp, V.shape[0])
    v = st["v"]
    b = st["b"]
    if ov > 0:
        Ev = v.shape[0]                       # 累加器当前覆盖长度（上一段的结尾）
        t = ((np.arange(ov, dtype=np.float64) + 0.5) / ov)[:, None]
        sl = slice(Ev - ov, Ev)               # 重叠区此前只有上一段以权重 1 贡献
        v[sl] = (v[sl].astype(np.float64) * (1.0 - t) + V[:ov] * t).astype(np.float32)
        b[sl] = (b[sl].astype(np.float64) * (1.0 - t) + B[:ov] * t).astype(np.float32)
        vt = V[ov:]
        bt = B[ov:]
    else:
        vt = V
        bt = B
    if vt.shape[0] > 0:
        v = np.concatenate([v, vt])
        b = np.concatenate([b, bt])
    ctx["_sep_stems"] = {"v": v, "b": b, "x": np.array(xseg, np.float32, copy=True)}
    ctx["stems"] = {"vocal": v, "backing": b}


# ---------------------------------------------------------------- 主入口

def process(x, sr, p, ctx):
    """x: float32 (n,ch) -> float32 (n,ch) 等长；分级路由时写 ctx["stems"]。"""
    p = dict(p or {})
    ctx.setdefault("stats", {})
    ctx.setdefault("log", [])
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    x = np.ascontiguousarray(x, np.float32)
    if not bool(p.get("on", True)):                 # 引擎不传 on；显式 False 时透传
        ctx["stems"] = None
        ctx["stats"]["sep"] = {"on": False}
        return x
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    n = x.shape[0]
    mode = str(p.get("mode", "hpss")).lower()
    routing = str(p.get("routing", "graded")).lower()
    vg = float(p.get("vocal_gain", 1.0))
    lg = float(p.get("low_gain", 1.0))
    pg = float(p.get("perc_gain", 1.0))
    og = float(p.get("other_gain", 1.0))

    X = core.stft(x, sr)                            # (ch,T,F) complex64
    mag = np.abs(X).mean(axis=0).astype(np.float32)  # 双声道共享幅度 -> 掩码（相位一致）
    if mode == "nmf":
        (mv, ml, mp, mo), dev = _masks_nmf(mag, sr)
    else:
        (mv, ml, mp, mo), dev = _masks_hpss(mag, sr)

    g_mix = ((vg * mv + lg * ml) + pg * mp) + og * mo     # 合轨掩码（全 1 增益时恒为 1）
    if routing == "graded":
        g_v = (vg * mv).astype(np.float32)
        # vocal/backing 堆成 4 声道一次 istft：Python 帧循环开销减半（istft 对声道线性）
        Xs = np.concatenate([X * g_v[None], X * (g_mix - g_v)[None]], axis=0)
        y4 = core.istft(Xs, n, sr)
        V = np.ascontiguousarray(y4[:, :x.shape[1]])
        B = np.ascontiguousarray(y4[:, x.shape[1]:])
        del Xs, y4
        out = (V + B).astype(np.float32)             # 保证 stems 之和与返回值逐样本一致
        _stems_push(ctx, V, B, x)
    else:
        ctx["stems"] = None
        ctx.pop("_sep_stems", None)
        out = core.istft(X * g_mix[None], n, sr).astype(np.float32)

    st = ctx["stats"].get("sep") or {}
    st.update({"on": True, "mode": mode, "routing": routing,
               "mask_sum_dev": dev, "calls": int(st.get("calls", 0)) + 1,
               "stems_len": int(ctx["stems"]["vocal"].shape[0]) if ctx.get("stems") else None})
    ctx["stats"]["sep"] = st
    ctx["log"].append(f"sep: {mode}/{routing} 掩码和偏差 {dev:.1e}")
    return out
