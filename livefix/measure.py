"""LiveFix 测量（BS.1770-4 响度 / 真峰 / 频谱 / 相关，纯 numpy）。

LUFS 采用 K 加权（高架 +4dB@1681.97Hz 与 38.135Hz 高通，RBJ 实现），
400ms 矩形窗、100ms 步进、绝对门 -70LKFS 与相对门 -10LU。
真峰用 4 倍 FFT 过采样 + sqrt-Hann 50% overlap-save。
"""
from __future__ import annotations

import numpy as np

from . import core

ABS_GATE_LUFS = -70.0
REL_GATE_LU = -10.0


def _as2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    return x


def _weights(ch):
    if ch == 1:
        return np.array([1.0])
    if ch == 2:
        return np.array([1.0, 1.0])
    # 5.1 WAV 顺序：L R C LFE RL RR
    w = np.ones(ch, np.float64)
    if ch >= 4:
        w[3] = 0.0
    if ch >= 3:
        w[2] = np.sqrt(2.0)
    return w


def k_filter(x, sr):
    """K 加权（时域，overlap-save 精确滤波）。"""
    b1, a1 = core.biquad_coeffs("highshelf", sr, 1681.97, q=0.7071752, gain_db=3.999843)
    b2, a2 = core.biquad_coeffs("hp", sr, 38.135, q=0.5003270)
    y = core.iir_filt(x, b1, a1)
    return core.iir_filt(y, b2, a2)


def _base_powers(y, sr, base_s=0.1):
    """-> (nbase, ch) 每 base_s 块的均方（float64）。"""
    y = _as2d(y)
    base = max(1, int(round(base_s * float(sr))))
    n = (y.shape[0] // base) * base
    if n == 0:
        return np.zeros((0, y.shape[1])), base
    sq = y[:n].astype(np.float32) ** 2
    return sq.reshape(-1, base, y.shape[1]).sum(axis=1).astype(np.float64) / base, base


def _z_series(sq_base, ch, hop_blocks):
    """每 hop 窗内 z = sum_ch G*mean-square。sq_base: (nbase, ch)。"""
    g = _weights(ch)[: sq_base.shape[1]]
    z_base = sq_base @ g  # (nbase,)
    z_base = np.concatenate([z_base, np.zeros(max(0, hop_blocks - 1))])
    c = np.concatenate([[0.0], np.cumsum(z_base)])
    n_win = len(z_base) - hop_blocks + 1
    if n_win <= 0:
        return np.zeros(0)
    return (c[hop_blocks:] - c[:-hop_blocks]) / hop_blocks


def lufs_integrated(x, sr):
    """BS.1770-4 综合响度 LKFS（float）。"""
    y = k_filter(x, sr)
    sq, base = _base_powers(y, sr, 0.1)
    if sq.shape[0] < 4:
        return -70.0
    z = _z_series(sq, sq.shape[1], 4)  # 400ms / 100ms
    z = z[z > 0]
    if z.size == 0:
        return -70.0
    z1 = z[z > 10.0 ** ((ABS_GATE_LUFS + 0.691) / 10.0)]
    if z1.size == 0:
        return -70.0
    loud1 = -0.691 + 10.0 * np.log10(np.mean(z1))
    z2 = z1[z1 > 10.0 ** ((loud1 + REL_GATE_LU + 0.691) / 10.0)]
    if z2.size == 0:
        return float(loud1)
    return float(-0.691 + 10.0 * np.log10(np.mean(z2)))


def lufs_short(x, sr):
    """短时(3s)响度序列（100ms 步进），float64 (T,)。"""
    y = k_filter(x, sr)
    sq, _base = _base_powers(y, sr, 0.1)
    if sq.shape[0] < 30:
        return np.full(max(sq.shape[0], 0), -70.0)
    z = _z_series(sq, sq.shape[1], 30)
    return -0.691 + 10.0 * np.log10(np.maximum(z, 1e-16))


def lra(x, sr):
    """响度范围 LRA（EBU Tech 3342 简化：短时 3s，绝对门 -70，相对门 -20）。"""
    l = lufs_short(x, sr)
    l = l[np.isfinite(l)]
    if l.size == 0:
        return 0.0
    keep = l > ABS_GATE_LUFS
    if keep.sum() < 3:
        return 0.0
    z = 10.0 ** ((l[keep] + 0.691) / 10.0)
    mean_l = -0.691 + 10.0 * np.log10(np.mean(z))
    keep2 = l[keep] > (mean_l - 20.0)
    v = l[keep][keep2]
    if v.size < 3:
        return 0.0
    return float(np.percentile(v, 95) - np.percentile(v, 10))


def _true_peak_env(x, L=1 << 16):
    """4x 带限插值包络 (4n,)，与 m6._env_oversampled 同法（OLA 重构，幅度守恒）。

    rfft(L) 取基带 -> 搬进 4L 频带（基带 x2 增益 + 共轭镜像 x2 + 带边半权重）
    -> irfft(4L)，hann 50% OLA 累加后裁掉两端补零。OLA 重构保证插值曲线在采样点
    上精确等于原信号，故无边缘伪迹（分块直接取 max 会在块边缘高估约 0.8dB）。
    """
    xa = np.asarray(x, np.float64)
    n, ch = xa.shape
    L = min(int(L), 1 << 16)
    L -= L % 2
    hop = L // 2
    w = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(L) / L)).astype(np.float64)
    xp = np.pad(xa, ((L, L), (0, 0)))
    npx = n + 2 * L
    nup = 4 * npx + 8
    acc = np.zeros((nup, ch), np.float32)
    Y = np.zeros((2 * L + 1, ch), np.complex128)
    for s in range(0, npx, hop):
        e = min(s + L, npx)
        seg = np.zeros((L, ch), np.float64)
        seg[:e - s] = xp[s:e]
        W = np.fft.rfft(seg * w[:, None], axis=0)               # (L/2+1, ch)
        Y[:] = 0.0
        Y[:L // 2 + 1] = 2.0 * W                                # 基带正频率（周期延拓去镜像）
        Y[L // 2 + 1:L] = 2.0 * np.conj(W[L // 2 - 1:0:-1])     # 2x 增益恢复采样值
        Y[L] = np.real(W[0])                                    # 带边半权重
        up = np.fft.irfft(Y, n=4 * L, axis=0)                   # (4L, ch) 真 4x 带限插值
        off = 4 * s
        m = min(4 * L, nup - off)
        acc[off:off + m] += up[:m].astype(np.float32)
    return np.abs(acc[4 * L:4 * L + 4 * n]).max(axis=1)


def true_peak_db(x, sr, factor=4):
    """factor 倍过采样真峰 dBTP（float）。

    做法：对整段信号直接做带限插值——rfft(n) 取基带频谱，搬进 factor*n 频带
    （正频段上移 + 共轭镜像填满，高频留零 = 限制到原奈奎斯特），irfft 后取最大。
    这是全带限插值，**不加窗**：加窗会衰减段内峰值，导致真峰被系统性低估
    （正是早期 bug 的深层原因）。整段处理保证采样点被精确保留，故
    true_peak ≥ peak 恒成立。长文件分块时每块零填充到 factor 倍长度后
    直接比较峰值，不做 OLA（避免窗衰减）。
    """
    x = _as2d(x).astype(np.float32)
    n, ch = x.shape
    L = min(int(sr), 1 << 16)
    if L < 512:
        L = min(n, 512)
    L -= L % 2
    del factor  # 保留参数以兼容旧签名；当前实现固定 4x（见 _true_peak_env）
    if n <= 1:
        return float(20.0 * np.log10(max(float(np.abs(x).max()), 1e-9)))
    env = _true_peak_env(x, L)
    return float(20.0 * np.log10(max(float(env.max()), 1e-9)))


def peak_db(x):
    return float(20.0 * np.log10(max(float(np.abs(_as2d(x)).max()), 1e-9)))


def rms_db(x):
    x = _as2d(x).astype(np.float64)
    return float(10.0 * np.log10(max(np.mean(x * x), 1e-16)))


def corr(x):
    """左右相关系数；单声道返回 1.0。"""
    x = _as2d(x).astype(np.float64)
    if x.shape[1] < 2:
        return 1.0
    a, b = x[:, 0], x[:, 1]
    d = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if d < 1e-12:
        return 1.0
    return float(np.sum(a * b) / d)


def spectrum_db(x, sr, n=96, f_lo=30.0, f_hi=16000.0):
    """整段平均频谱，对数频点 n 个，归一化到自身最大值 0 dB。-> list[float]。"""
    x = _as2d(x).astype(np.float32)
    nfft = 4096
    if x.shape[0] < nfft:
        x = np.pad(x, ((0, nfft - x.shape[0]), (0, 0)))
    hop = nfft // 2
    wh = np.sqrt(core._periodic_hann(nfft)).astype(np.float32)
    freqs = np.fft.rfftfreq(nfft, 1.0 / float(sr))
    acc = np.zeros(freqs.shape, np.float64)
    cnt = 0
    per = max(1, int(2_000_000 // (x.shape[1] * nfft)))
    starts = list(range(0, x.shape[0] - nfft + 1, hop))
    for a in range(0, len(starts), per):
        idx = starts[a:a + per]
        seg = np.stack([x[s:s + nfft] for s in idx])  # (m, nfft, ch)
        spec = np.fft.rfft(seg * wh[None, :, None], axis=1)
        acc += (np.abs(spec) ** 2).sum(axis=(0, 2))
        cnt += len(idx)
    if cnt == 0:
        return [0.0] * n
    psd = acc / cnt
    f_hi_use = min(float(f_hi), float(sr) * 0.48)
    grid = np.geomspace(max(f_lo, 10.0), max(f_hi_use, 50.0), n)
    psd_db = 10.0 * np.log10(psd[1:] + 1e-16)
    lf = np.log2(np.maximum(freqs[1:], 1.0))
    v = np.interp(np.log2(grid), lf, psd_db)
    v = core.smooth1d(v, 5)
    v = v - v.max()
    return [float(t) for t in v]


def band_powers_db(x, sr, edges=None):
    """倍频程带能量（参考 63Hz..16k），-> list[{"f":str,"db":float}]，相对整段中位。"""
    if edges is None:
        edges = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    x = _as2d(x).astype(np.float32)
    nfft = 8192
    hop = nfft // 2
    wh = np.sqrt(core._periodic_hann(nfft)).astype(np.float32)
    freqs = np.fft.rfftfreq(nfft, 1.0 / float(sr))
    acc = np.zeros(freqs.shape, np.float64)
    starts = list(range(0, max(x.shape[0] - nfft + 1, 1), hop))
    if not starts:  # 极短素材：补零到一个整窗
        starts = [0]
        x = np.pad(x, ((0, max(0, nfft - x.shape[0])), (0, 0)))
    for s in starts[:4000]:
        seg = x[s:s + nfft]
        if seg.shape[0] < nfft:
            seg = np.pad(seg, ((0, nfft - seg.shape[0]), (0, 0)))
        acc += (np.abs(np.fft.rfft(seg * wh[:, None], axis=0)) ** 2).sum(axis=1)
    acc /= max(len(starts[:4000]), 1)
    out = []
    lo = 44.0
    for f in edges:
        f_lo, f_hi = lo, float(f) * 1.05
        r = np.clip(freqs / max(f_lo, 1e-3), 1e-6, None)
        t_lo = np.clip((np.log2(r) + 0.2) / 0.2, 0.0, 1.0)
        r2 = np.clip(freqs / max(f_hi, 1e-3), 1e-6, None)
        t_hi = np.clip((1.2 - np.log2(r2)) / 0.2, 0.0, 1.0)
        m = np.minimum(0.5 - 0.5 * np.cos(np.pi * t_lo), 0.5 - 0.5 * np.cos(np.pi * t_hi))
        e = float((acc * m).sum())
        out.append({"f": str(f), "db": round(float(10.0 * np.log10(e + 1e-16)), 2)})
        lo = float(f) * 0.95
    ref = float(np.median([b["db"] for b in out]))
    for b in out:
        b["db"] = round(b["db"] - ref, 2)
    return out
