"""LiveFix DSP 内核公共实现（纯 numpy，无 scipy）。

契约见 spec/CORE.md。本文件由主代理维护，功能模块只读。
STFT 采用 sqrt-Hann 分析/综合窗、hop=N/4，保证精确重构（COLA 归一化）。
"""
from __future__ import annotations

import numpy as np

N_FFT = 2048
HOP = 512
_PAD = N_FFT


def _periodic_hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / n)


_HANN = _periodic_hann(N_FFT)
_WIN = np.sqrt(_HANN).astype(np.float32)  # 分析窗 = 综合窗 = sqrt(hann)
_WIN2 = _HANN.astype(np.float32)  # win*win


# ---------------------------------------------------------------- 结构 / STFT

def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"音频应为 (n, ch)，得到 {x.shape}")
    return x


def stft(x, sr):
    """float32 (n,ch)|(n,) -> complex64 (ch,T,F)。内部按块计算以控制内存。"""
    del sr  # 采样率不影响变换本身
    x = _as_2d(x).astype(np.float32, copy=False)
    n, ch = x.shape
    tail = (HOP - ((n + 2 * _PAD - N_FFT) % HOP)) % HOP
    xp = np.pad(x, ((_PAD, _PAD + tail), (0, 0)))
    nf = (xp.shape[0] - N_FFT) // HOP + 1
    F = N_FFT // 2 + 1
    X = np.empty((ch, nf, F), np.complex64)
    frames = np.lib.stride_tricks.sliding_window_view(xp, N_FFT, axis=0)[::HOP]
    per_chunk = max(1, int(4_000_000 // (ch * N_FFT)))
    for a in range(0, nf, per_chunk):
        b = min(a + per_chunk, nf)
        seg = np.ascontiguousarray(frames[a:b])  # (m, ch, N)
        seg *= _WIN
        spec = np.fft.rfft(seg, axis=-1)
        X[:, a:b, :] = np.transpose(spec, (1, 0, 2))
    return X


def istft(X, n, sr):
    """(ch,T,F) -> float32 (n,ch)，与输入等长。"""
    del sr
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"X 应为 (ch,T,F)，得到 {X.shape}")
    ch, nf, _F = X.shape
    total = (nf - 1) * HOP + N_FFT
    acc = np.zeros((total, ch), np.float64)
    den = np.zeros(total, np.float64)
    for i in range(nf):
        seg = np.fft.irfft(X[:, i], n=N_FFT, axis=-1)  # (ch, N)
        seg = np.transpose(seg, (1, 0)) * _WIN[:, None]  # 综合窗与分析窗一致：win*win = hann
        s = i * HOP
        acc[s:s + N_FFT] += seg
        den[s:s + N_FFT] += _WIN2
    lo, hi = _PAD, _PAD + n
    if hi > total:  # 容错：外部传入长度不一致时补零
        acc = np.pad(acc, ((0, hi - total), (0, 0)))
        den = np.pad(den, (0, hi - total))
    den[den < 1e-9] = 1.0
    out = acc[lo:hi] / den[lo:hi, None]
    return out.astype(np.float32)


def rfftfreqs(sr, n_fft=N_FFT):
    """频率轴。默认 N_FFT；传 n_fft 可获得任意长度 STFT 的频率轴。"""
    return np.fft.rfftfreq(n_fft, 1.0 / float(sr))


# ---------------------------------------------------------------- 数值工具

def db(x, floor=-100.0):
    return np.maximum(20.0 * np.log10(np.abs(x) + 1e-12), floor)


def lin(dbv):
    return np.power(10.0, np.asarray(dbv, np.float64) / 20.0)


def smooth1d(v, win, axis=-1):
    """滑动平均（边缘 edge 填充，长度不变；win 自动取奇数）。"""
    v = np.asarray(v, np.float64)
    if win <= 1:
        return v
    if win % 2 == 0:
        win += 1
    pad = win // 2
    vp = np.moveaxis(v, axis, -1)
    vp = np.pad(vp, [(0, 0)] * (vp.ndim - 1) + [(pad, pad)], mode="edge")
    c = np.cumsum(vp, axis=-1)
    c = np.concatenate([np.zeros(c.shape[:-1] + (1,)), c], axis=-1)
    out = (c[..., win:] - c[..., :-win]) / win
    return np.moveaxis(out, -1, axis)


def smooth_curve_db(sr, pts):
    """pts=[(Hz,dB)...] -> float64 (F,)：log-f 插值 + 平滑的频段增益曲线。

    范围外取端点值（常数延拓）；重复频率保留最后一个点。
    """
    if len(pts) == 0:
        return np.zeros(N_FFT // 2 + 1)
    pairs = {}
    for f, d in pts:
        pairs[float(f)] = float(d)  # 后写覆盖先写
    pf = np.array(sorted(pairs), np.float64)
    pd = np.array([pairs[f] for f in pf], np.float64)
    nyq = float(sr) * 0.49
    grid = np.geomspace(20.0, max(min(20000.0, nyq), 40.0), 161)
    g = np.interp(np.log2(grid), np.log2(pf), pd)
    g = smooth1d(smooth1d(g, 7), 7)
    freqs = rfftfreqs(sr)
    return np.interp(np.log2(np.maximum(freqs, 1.0)), np.log2(grid), g)


def band_mask(sr, f_lo, f_hi, edge_oct=0.3, n_fft=N_FFT):
    """平顶软边频段掩码 (F,)，值域 0..1（余弦斜边，宽度 edge_oct 个八度）。

    n_fft 默认取内核 N_FFT；用于非内核长度的 STFT 时显式传入，返回长度随之改变。
    """
    freqs = rfftfreqs(sr, n_fft)
    fl = np.log2(np.maximum(freqs, 1e-3))
    lf = np.log2(max(float(f_lo), 1e-3))
    lh = np.log2(max(float(f_hi), 1e-3))
    eo = max(float(edge_oct), 1e-6)
    t_lo = np.clip((fl - (lf - eo)) / eo, 0.0, 1.0)
    t_hi = np.clip(((lh + eo) - fl) / eo, 0.0, 1.0)
    m = np.minimum(0.5 - 0.5 * np.cos(np.pi * t_lo), 0.5 - 0.5 * np.cos(np.pi * t_hi))
    m[freqs < 1e-3] = 0.0
    return m


def env_updown(v, attack_s, release_s, sr):
    """帧域非对称一阶平滑，沿最后一维（必须是时间维）。"""
    v = np.asarray(v, np.float64)
    if v.ndim == 0 or v.shape[-1] < 2:
        return v.copy()
    hop_s = HOP / float(sr)
    aa = float(np.exp(-hop_s / max(float(attack_s), 1e-5)))
    ar = float(np.exp(-hop_s / max(float(release_s), 1e-5)))
    out = v.copy()
    flat = out.reshape(-1, out.shape[-1])
    prev = flat[:, 0].copy()
    for i in range(1, flat.shape[1]):
        cur = flat[:, i]
        a = np.where(cur > prev, aa, ar)
        prev = a * prev + (1.0 - a) * cur
        flat[:, i] = prev
    return out


def frame_rms(X):
    """STFT -> 线性帧 RMS (ch,T)（窗功率归一）。"""
    p = np.abs(np.asarray(X)) ** 2
    weighted = 2.0 * p.sum(axis=-1) - p[..., 0] - p[..., -1]
    sw2 = float(_HANN.sum())
    return np.sqrt(np.maximum(weighted, 0.0) / (N_FFT * sw2))


def band_energy_db(X, sr, f_lo, f_hi):
    m = band_mask(sr, f_lo, f_hi)
    e = (np.abs(np.asarray(X)) ** 2 * m).sum(axis=-1)
    return 10.0 * np.log10(e + 1e-16)


def noise_floor_db(X, pct=15.0):
    """时间维百分位噪声底 (ch,F)，dB。"""
    p = np.abs(np.asarray(X)) ** 2
    v = np.percentile(p, float(pct), axis=-2)
    return 10.0 * np.log10(v + 1e-16)


# ---------------------------------------------------------------- 滤波器

def biquad_coeffs(kind, sr, f, q=0.707, gain_db=0.0):
    """RBJ 双二阶 -> (b, a)，a0 已归一化。kind: hp/lp/peak/notch/lowshelf/highshelf/bandpass。"""
    kind = str(kind).lower()
    f = float(np.clip(f, 5.0, 0.49 * sr))
    w0 = 2.0 * np.pi * f / float(sr)
    cw, sw = float(np.cos(w0)), float(np.sin(w0))
    q = max(float(q), 1e-3)
    alpha = sw / (2.0 * q)
    if kind == "lp":
        b0, b1, b2 = (1 - cw) / 2, 1 - cw, (1 - cw) / 2
        a0, a1, a2 = 1 + alpha, -2 * cw, 1 - alpha
    elif kind == "hp":
        b0, b1, b2 = (1 + cw) / 2, -(1 + cw), (1 + cw) / 2
        a0, a1, a2 = 1 + alpha, -2 * cw, 1 - alpha
    elif kind == "notch":
        b0, b1, b2 = 1.0, -2 * cw, 1.0
        a0, a1, a2 = 1 + alpha, -2 * cw, 1 - alpha
    elif kind == "bandpass":
        b0, b1, b2 = alpha, 0.0, -alpha
        a0, a1, a2 = 1 + alpha, -2 * cw, 1 - alpha
    elif kind == "peak":
        A = 10.0 ** (gain_db / 40.0)
        aa = alpha / A
        b0, b1, b2 = 1 + alpha * A, -2 * cw, 1 - alpha * A
        a0, a1, a2 = 1 + aa, -2 * cw, 1 - aa
    elif kind in ("lowshelf", "highshelf"):
        A = 10.0 ** (gain_db / 40.0)
        al = sw / 2.0 * np.sqrt(2.0)  # S=1
        sq = np.sqrt(A)
        if kind == "lowshelf":
            b0 = A * ((A + 1) - (A - 1) * cw + 2 * sq * al)
            b1 = 2 * A * ((A - 1) - (A + 1) * cw)
            b2 = A * ((A + 1) - (A - 1) * cw - 2 * sq * al)
            a0 = (A + 1) + (A - 1) * cw + 2 * sq * al
            a1 = -2 * ((A - 1) + (A + 1) * cw)
            a2 = (A + 1) + (A - 1) * cw - 2 * sq * al
        else:
            b0 = A * ((A + 1) + (A - 1) * cw + 2 * sq * al)
            b1 = -2 * A * ((A - 1) + (A + 1) * cw)
            b2 = A * ((A + 1) + (A - 1) * cw - 2 * sq * al)
            a0 = (A + 1) - (A - 1) * cw + 2 * sq * al
            a1 = 2 * ((A - 1) - (A + 1) * cw)
            a2 = (A + 1) - (A - 1) * cw - 2 * sq * al
    else:
        raise ValueError(f"未知滤波器类型：{kind}")
    return np.array([b0, b1, b2], np.float64) / a0, np.array([a0, a1, a2], np.float64) / a0


def mag_response(kind, sr, f, q=0.707, gain_db=0.0):
    """-> float64 (F,) 线性幅度响应（零相位应用用它）。"""
    b, a = biquad_coeffs(kind, sr, f, q, gain_db)
    w = 2.0 * np.pi * rfftfreqs(sr) / float(sr)
    z1 = np.exp(-1j * w)
    z2 = z1 * z1
    return np.abs((b[0] + b[1] * z1 + b[2] * z2) / (a[0] + a[1] * z1 + a[2] * z2))


def iir_filt(x, b, a, h_len=8192, fft_n=65536):
    """零相位无关的时域 IIR 滤波（截断脉冲响应 + overlap-save FFT），精确线性卷积。"""
    x = _as_2d(x)
    b = np.asarray(b, np.float64) / a[0]
    a = np.asarray(a, np.float64) / a[0]
    L = min(int(h_len), int(fft_n) // 8)
    h = np.zeros(L, np.float64)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(L):
        xn = 1.0 if i == 0 else 0.0
        yn = b[0] * xn + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        x2, x1 = x1, xn
        y2, y1 = y1, yn
        h[i] = yn
    N = int(fft_n)
    if L > N:
        raise ValueError("fft_n 太小")
    H = np.fft.rfft(h, n=N)
    block = N // 8
    n, ch = x.shape
    pad = N - block
    xp = np.pad(x.astype(np.float64), ((pad, 0), (0, 0)))
    out = np.empty((n, ch), np.float64)
    pos = 0
    k = 0
    while pos < n:
        s = k * block
        chunk = xp[s:s + N]
        if chunk.shape[0] < N:
            chunk = np.pad(chunk, ((0, N - chunk.shape[0]), (0, 0)))
        yv = np.fft.irfft(np.fft.rfft(chunk, axis=0) * H[:, None], n=N, axis=0)
        seg = yv[N - block:]
        m = min(block, n - pos)
        out[pos:pos + m] = seg[:m]
        pos += m
        k += 1
    return out.astype(np.float32)


def iir_impulse(b, a, length=8192):
    """双二阶的（截断）脉冲响应 (length,) float64，用于级联 EQ 烘焙。"""
    b = np.asarray(b, np.float64) / a[0]
    a = np.asarray(a, np.float64) / a[0]
    L = int(length)
    h = np.zeros(L, np.float64)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(L):
        xn = 1.0 if i == 0 else 0.0
        yn = b[0] * xn + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        x2, x1 = x1, xn
        y2, y1 = y1, yn
        h[i] = yn
    return h


def freq_filt(x, H, fft_n=65536):
    """按复频响 H 做 overlap-save 精确滤波（导出烘焙 EQ 用）。

    H 为 rfft 格点长度 fft_n//2+1 的**复**频响——把多个双二阶的频响连乘后传入，
    一次变换完成整条 EQ 级联（比逐个 iir_filt 快 N 倍，且含各级相位，完全等价
    于时域级联）。等效脉冲 = irfft(H) 截断到 fft_n//4，拖尾误差可忽略
    （0.34s ≫ 典型 peaking 衰减时间常数）。
    """
    x = _as_2d(x)
    N = int(fft_n)
    H = np.asarray(H)
    if H.shape[0] != N // 2 + 1:
        raise ValueError(f"H 长度应为 {N // 2 + 1}，得到 {H.shape}")
    L = N // 4                       # 脉冲保留长度；要求 block <= N - L + 1
    h = np.fft.irfft(H, n=N)[:L]
    Hh = np.fft.rfft(h, n=N)
    block = N - L                    # 49152 @ N=65536
    n, ch = x.shape
    pad = L - 1
    xp = np.pad(x.astype(np.float64), ((pad, 0), (0, 0)))
    out = np.empty((n, ch), np.float64)
    pos = 0
    k = 0
    while pos < n:
        s = k * block
        chunk = xp[s:s + N]
        if chunk.shape[0] < N:
            chunk = np.pad(chunk, ((0, N - chunk.shape[0]), (0, 0)))
        yv = np.fft.irfft(np.fft.rfft(chunk, axis=0) * Hh[:, None], n=N, axis=0)
        seg = yv[L - 1: L - 1 + block]
        m = min(block, n - pos)
        out[pos:pos + m] = seg[:m]
        pos += m
        k += 1
    return out.astype(np.float32)

def block_apply(x, sr, fn, block_s=60.0, xfade_s=1.0, cb=None):
    """fn(seg float32 (n_seg,ch), sr) -> 同形 seg'；分段处理 + 线性交叉淡化（权重互补和=1）。

    cb(done_blocks, total_blocks) 可选：每完成一块回调一次，供引擎上报进度。
    """
    x = _as_2d(x).astype(np.float32, copy=False)
    n, ch = x.shape
    L = int(block_s * float(sr))
    O = max(1, int(xfade_s * float(sr)))
    O = min(O, max(1, L // 4))
    if n <= L:
        y = np.asarray(fn(x.copy(), sr), np.float32)
        if y.shape != x.shape:
            raise ValueError(f"分段输出形状不符：{y.shape} != {x.shape}")
        if cb is not None:
            cb(1, 1)
        return y
    starts = [0]
    while True:
        s = starts[-1] + (L - O)
        if s + L >= n:
            break
        starts.append(s)
    if starts[-1] + L < n:
        starts.append(n - L)
    out = np.zeros((n, ch), np.float64)
    wsum = np.zeros((n, 1), np.float64)
    total = len(starts)
    for idx, s in enumerate(starts):
        e = min(s + L, n)
        seg = x[s:e].copy()
        y = np.asarray(fn(seg, sr), np.float32)
        if y.shape != seg.shape:
            raise ValueError(f"分段输出形状不符：{y.shape} != {seg.shape}")
        w = np.ones((e - s, 1), np.float64)
        if idx > 0:
            ov_in = (starts[idx - 1] + L) - s
            if ov_in > 0:
                t = (np.arange(ov_in) + 0.5) / ov_in
                w[:ov_in, 0] = t
        if idx < len(starts) - 1:
            ov_out = e - starts[idx + 1]
            if ov_out > 0:
                t = (np.arange(ov_out) + 0.5) / ov_out
                w[len(w) - ov_out:, 0] = 1.0 - t
        out[s:e] += y * w
        wsum[s:e] += w
        if cb is not None:
            cb(idx + 1, total)
    wsum[wsum < 1e-9] = 1.0
    return (out / wsum).astype(np.float32)
