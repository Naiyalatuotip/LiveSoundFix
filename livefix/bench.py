"""基准评判器：用三份真实文件（音源前/源1/调整1）给处理链与各模块打分。

口径约定：所有频带/频谱量统一用功率 dB（10log10），噪声底与频带能量可直接相减。
所有曲线先做 LUFS 对齐，隔离响度差异、只比音色与声场。
"""
from __future__ import annotations

import numpy as np

from . import core, measure

EDGES = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
SPEC_N = 96
CAL_CACHE: dict = {}


def _cal_db(nfft: int) -> float:
    """band/谱能量 -> dBFS 的校准量（功率口径）。"""
    key = nfft
    if key not in CAL_CACHE:
        w = np.sqrt(core._periodic_hann(nfft)).astype(np.float64)
        CAL_CACHE[key] = 10.0 * np.log10(nfft * (w ** 2).sum() / 2.0)
    return CAL_CACHE[key]


def _as2d(x):
    x = np.asarray(x, np.float32)
    return x[:, None] if x.ndim == 1 else x


def lufs_match(x, sr, target):
    """响度对齐（不裁剪，返回 float32）。"""
    x = _as2d(x)
    cur = measure.lufs_integrated(x, sr)
    g = 10.0 ** ((target - cur) / 20.0)
    return (x * np.float32(g)).astype(np.float32)


def band_db(x, sr, nfft=2048):
    """倍频程能量（功率 dB，已校准到 dBFS）。返回 list[float]，长度 len(EDGES)。"""
    x = _as2d(x).astype(np.float32)
    hop = nfft // 2
    w = np.sqrt(core._periodic_hann(nfft)).astype(np.float32)
    freqs = np.fft.rfftfreq(nfft, 1.0 / float(sr))
    acc = np.zeros(freqs.shape, np.float64)
    starts = list(range(0, max(x.shape[0] - nfft + 1, 1), hop))
    if not starts:
        starts = [0]
    per = max(1, int(1_500_000 // (x.shape[1] * nfft)))
    cnt = 0
    for a in range(0, len(starts), per):
        seg = np.stack([x[s:s + nfft] for s in starts[a:a + per]])
        seg = np.where(seg.shape[1] == nfft, seg, np.pad(
            seg, ((0, 0), (0, max(0, nfft - seg.shape[1])), (0, 0))))[:, :nfft]
        acc += (np.abs(np.fft.rfft(seg * w[None, :, None], axis=1)) ** 2).sum(axis=(0, 2))
        cnt += seg.shape[0]
    acc /= max(cnt, 1)
    cal = _cal_db(nfft)
    out, lo = [], 44.0
    for f in EDGES:
        m = core.band_mask(sr, lo, float(f) * 1.05, edge_oct=0.2, n_fft=nfft)
        e = float((acc * m).sum())
        out.append(round(10.0 * np.log10(e + 1e-16) - cal, 2))
        lo = float(f) * 0.95
    return out


def spec_db(x, sr, n=SPEC_N, nfft=2048):
    """对数频点频谱（功率 dB，已校准到 dBFS，长度 n）。"""
    x = _as2d(x).astype(np.float32)
    hop = nfft // 2
    w = np.sqrt(core._periodic_hann(nfft)).astype(np.float32)
    freqs = np.fft.rfftfreq(nfft, 1.0 / float(sr))
    acc = np.zeros(freqs.shape, np.float64)
    starts = list(range(0, max(x.shape[0] - nfft + 1, 1), hop))
    if not starts:
        starts = [0]
    per = max(1, int(1_500_000 // (x.shape[1] * nfft)))
    cnt = 0
    for a in range(0, len(starts), per):
        idx = starts[a:a + per]
        seg = np.stack([x[s:s + nfft] for s in idx])
        acc += (np.abs(np.fft.rfft(seg * w[None, :, None], axis=1)) ** 2).sum(axis=(0, 2))
        cnt += len(idx)
    acc /= max(cnt, 1)
    grid = np.geomspace(30.0, min(16000.0, sr * 0.48), n)
    psd = 10.0 * np.log10(acc[1:] + 1e-16) - _cal_db(nfft)
    v = np.interp(np.log2(grid), np.log2(np.maximum(freqs[1:], 1.0)), psd)
    return [round(float(t), 2) for t in core.smooth1d(v, 3)]


def noise_floor_db(x, sr, nfft=2048):
    """全带噪声底（功率 dB，已校准到 dBFS）：先算谱，再取时间维低百分位的中位。"""
    x = _as2d(x).astype(np.float32)
    hop = nfft // 4
    w = np.sqrt(core._periodic_hann(nfft)).astype(np.float32)
    starts = list(range(0, max(x.shape[0] - nfft + 1, 1), hop))[:4000]
    if not starts:
        return -120.0
    per = max(1, int(1_200_000 // (x.shape[1] * nfft)))
    cols = []
    for a in range(0, len(starts), per):
        seg = np.stack([x[s:s + nfft] for s in starts[a:a + per]])
        p = (np.abs(np.fft.rfft(seg * w[None, :, None], axis=1)) ** 2).mean(axis=(0, 2))
        cols.append(p)
    p = np.concatenate(cols, axis=0)
    floor = np.percentile(p, 10.0)
    return float(10.0 * np.log10(floor + 1e-16) - _cal_db(nfft))


def profile(x, sr, target=-16.0):
    """一份音频的完整画像（响度对齐后测量）。"""
    x = _as2d(x)
    xm = lufs_match(x, sr, target)
    return {
        "lufs": round(measure.lufs_integrated(x, sr), 2),
        "lra": round(measure.lra(x, sr), 2),
        "peak_db": round(measure.peak_db(x), 2),
        "true_peak_db": round(measure.true_peak_db(x, sr), 2),
        "corr": round(measure.corr(x), 3),
        "noise_floor_db": round(noise_floor_db(xm, sr), 1),
        "bands": band_db(xm, sr),
        "spec": spec_db(xm, sr),
        "lufs_aligned": target,
    }


def rms_dist(a, b):
    """两条曲线的 RMS 距离（dB）。"""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.sqrt(((a - b) ** 2).mean()))


def score(processed, sr, ref_fixed, ref_source, target=-16.0):
    """给一份处理结果打分。越小越好。

    - dist_to_fixed : 离「人工修好的版本」有多近（主目标）
    - dist_to_source: 离源曲有多近（信息保真，过大=过度处理）
    - improvement   : 相对相机直出的改善量（正数=有帮助）
    """
    p = profile(processed, sr, target)
    d_fixed = rms_dist(p["spec"], ref_fixed["spec"])
    d_src = rms_dist(p["spec"], ref_source["spec"])
    d_base = rms_dist(ref_source["spec"], ref_fixed["spec"])
    return {
        "dist_to_fixed": round(d_fixed, 3),
        "dist_to_source": round(d_src, 3),
        "improvement": round(d_base - d_fixed, 3),
        "corr": p["corr"],
        "noise_floor_db": p["noise_floor_db"],
        "lufs": p["lufs"],
        "true_peak_db": p["true_peak_db"],
        "bands": p["bands"],
        "spec": p["spec"],
    }


__all__ = [
    "EDGES", "lufs_match", "band_db", "spec_db", "noise_floor_db",
    "profile", "score", "rms_dist",
]
