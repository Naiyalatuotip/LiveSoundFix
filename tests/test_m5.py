"""m5 修复层自测：削波重建 / 市电陷波 / 谱减降噪 / 泵动平滑 / 恒等 / 单声道 / 性能。

合成信号（spec 配方）：1kHz 正弦 + 白噪声（SNR 10dB）+ 50Hz hum；削波测试额外
做「clip ±0.2 再 ×5」。降噪/陷波测量用未削波版本，避免削波谐波污染 3-8k 频带计量。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from livefix import core
from livefix import m5

sr = 48000
rng = np.random.default_rng(7)


def synth(dur=6.0, clip=False):
    t = np.arange(int(dur * sr)) / sr
    s = np.sin(2 * np.pi * 1000.0 * t)                 # 1kHz 正弦，功率 0.5
    noise = rng.standard_normal(t.shape)
    noise *= np.sqrt(0.5 / 10.0) / np.std(noise)       # SNR 10dB
    hum50 = 0.05 * np.sin(2 * np.pi * 50.0 * t)
    x = s + noise + hum50
    if clip:
        x = np.clip(x, -0.2, 0.2) * 5.0                # 人为削波
    return np.stack([x, x * 0.97], axis=1).astype(np.float32)


def params(**kw):
    base = {"on": True, "denoise": 0.0, "floor_db": -18.0,
            "hum": 0.0, "mains": 50, "declip": 0.0, "pump": 0.0}
    base.update(kw)
    return base


def run(x, p, ctx=None):
    t0 = time.perf_counter()
    y = m5.process(x, sr, p, ctx if ctx is not None else {"stats": {}, "log": []})
    ms = (time.perf_counter() - t0) * 1000.0
    assert y.shape == x.shape and y.dtype == np.float32
    assert np.isfinite(y).all(), "输出含 NaN/Inf"
    return y, ms


def band_diff_db(x_in, x_out, f_lo, f_hi):
    X0 = core.stft(x_in, sr)
    X1 = core.stft(x_out, sr)
    e0 = core.band_energy_db(X0, sr, f_lo, f_hi)
    e1 = core.band_energy_db(X1, sr, f_lo, f_hi)
    return float(np.mean(e1 - e0))


def rms_db(v):
    v = np.asarray(v, np.float64)
    return 10.0 * np.log10(np.mean(v * v) + 1e-20)


# ---------------------------------------------------------------- 1) 全 0 恒等
x_nc = synth(6.0, clip=False)
x_cl = synth(6.0, clip=True)
y0, ms = run(x_nc, params())
r = rms_db(y0) - rms_db(x_nc)
print("[identity] rms diff = %.6f dB (%.1f ms), max diff = %.2e"
      % (r, ms, float(np.abs(y0 - x_nc).max())))
assert abs(r) < 0.05 and float(np.abs(y0 - x_nc).max()) == 0.0

# ---------------------------------------------------------------- 2) denoise
y, ms = run(x_nc, params(denoise=1.0))
d_noise = band_diff_db(x_nc, y, 3000, 8000)
d_sig = band_diff_db(x_nc, y, 800, 1200)
print("[denoise] 3-8k noise band = %+.2f dB | 1kHz signal band = %+.2f dB (%.1f ms)"
      % (d_noise, d_sig, ms))
assert d_noise < -6.0, d_noise
assert d_sig > -1.5, d_sig

# ---------------------------------------------------------------- 3) hum
y, ms = run(x_nc, params(hum=1.0))
freqs = core.rfftfreqs(sr)
k50 = int(np.argmin(np.abs(freqs - 50.0)))
X0 = core.stft(x_nc, sr)
X1 = core.stft(y, sr)
e0 = float(np.mean((np.abs(X0) ** 2)[:, :, k50]))
e1 = float(np.mean((np.abs(X1) ** 2)[:, :, k50]))
d50 = 10.0 * np.log10(e1 / e0 + 1e-20)
print("[hum] bin %d (%.1f Hz) energy = %+.2f dB (%.1f ms)"
      % (k50, freqs[k50], d50, ms))
assert d50 < -15.0, d50

# ---------------------------------------------------------------- 4) declip
y, ms = run(x_cl, params(declip=1.0))
peak_in = float(np.abs(x_cl).max())
n_over_in = int(np.sum(np.abs(x_cl) >= 0.985 * peak_in))
n_over_out = int(np.sum(np.abs(y) >= 0.985 * peak_in))
print("[declip] samples >0.985*peak: %d -> %d (%.1f ms)"
      % (n_over_in, n_over_out, ms))
assert n_over_in > 10000
assert n_over_out < 0.2 * n_over_in

# ---------------------------------------------------------------- 5) pump
# 慢泵动（2s 一档，14dB 起伏）：AGC 包络 attack 300ms / release 800ms，需在档位后半段计量
t = np.arange(int(8.0 * sr)) / sr
mod = np.where((t % 4.0) < 2.0, 0.5, 0.125)
pn = rng.standard_normal(t.shape) * 0.002
xp = (0.2 * np.sin(2 * np.pi * 200.0 * t) * mod + pn).astype(np.float32)
xp = np.stack([xp, xp], axis=1)
loud = (t % 4.0 >= 1.0) & (t % 4.0 <= 1.9)
quiet = (t % 4.0 >= 3.0) & (t % 4.0 <= 3.9)
y, ms = run(xp, params(pump=1.0))
depth_before = rms_db(xp[loud, 0]) - rms_db(xp[quiet, 0])
depth_after = rms_db(y[loud, 0]) - rms_db(y[quiet, 0])
print("[pump] loud-quiet depth %.2f -> %.2f dB (reduced %.2f, %.1f ms)"
      % (depth_before, depth_after, depth_before - depth_after, ms))
assert depth_after < depth_before - 2.0, (depth_before, depth_after)

# ---------------------------------------------------------------- 6) 单声道 + 全链 + ctx
y, ms = run(x_cl[:, :1], params(denoise=0.5, hum=0.6, declip=0.6, pump=0.4))
ctx = {"stats": {}, "log": []}
y, ms = run(x_cl, params(denoise=0.5, hum=0.6, declip=0.6, pump=0.4), ctx=ctx)
print("[chain/stereo] mono+stereo ok, ctx stats =", ctx["stats"]["m5"],
      "log =", ctx["log"][-1], "(%.1f ms)" % ms)
y, ms = run(x_nc, params(denoise=0.5, hum=0.6, declip=0.6, pump=0.4), ctx=None)
print("[chain/ctx=None] ok (%.1f ms)" % ms)

# ---------------------------------------------------------------- 7) 性能 60s@48k 立体声
x60 = synth(60.0, clip=True)
t0 = time.perf_counter()
y60 = m5.process(x60, sr, params(denoise=0.5, hum=0.6, declip=0.6, pump=0.4),
                 {"stats": {}, "log": []})
elapsed = time.perf_counter() - t0
assert y60.shape == x60.shape and np.isfinite(y60).all()
print("[perf] 60s@48k stereo full-chain = %.2f s" % elapsed)
assert elapsed < 5.0, elapsed

print("M5 OK")
