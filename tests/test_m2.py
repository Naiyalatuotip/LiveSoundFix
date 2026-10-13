"""m2（中高频补充 / 带宽扩展）自测。

合成信号 → 断言：air 带宽扩展提升 10-16k 带能量；全 0 恒等；无 NaN；
单声道不崩溃；60s 立体声耗时 < 5s。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from livefix import core, m2

sr = 48000


def mk_ctx(seed=0):
    return {"rng": np.random.default_rng(seed), "stats": {}, "log": []}


def band_db(x, lo, hi):
    """整段平均频段能量 dB（对全帧取均值）。"""
    X = core.stft(x, sr)
    return float(core.band_energy_db(X, sr, lo, hi).mean())


def rms_db(x):
    return float(10.0 * np.log10(np.mean(np.asarray(x, np.float64) ** 2) + 1e-20))


# ---------------------------------------------------------------- 1) 带宽扩展
rng = np.random.default_rng(1)
n = 6 * sr
t = np.arange(n) / sr
sines = sum(0.12 * np.sin(2 * np.pi * f * t) for f in (1100.0, 1750.0, 2400.0, 3300.0))
noise = 0.02 * rng.standard_normal(n)
x = (sines + noise).astype(np.float32)
x = np.stack([x, np.roll(x, 128)], axis=1)          # 模拟立体声

# 高频缺失：谱域级联 lp@8k（6 次，>10k 残留 < -30dB，模拟老录音硬带限）
X = core.stft(x, sr)
lp = core.mag_response("lp", sr, 8000.0, q=0.707)
X = X * lp[None, None, :] ** 6
xb = core.istft(X, n, sr)

before = band_db(xb, 10000, 16000)
p = {"on": True, "air": 1.0, "detail": 0.0, "dehiss": 0.0, "ext_hz": 9000}
y = m2.process(xb, sr, p, mk_ctx())
after = band_db(y, 10000, 16000)
print(f"[1] 10-16k 带能量：before={before:.2f} dB  after={after:.2f} dB  rise={after - before:+.2f} dB")
assert after - before > 6.0, f"带宽扩展提升不足：{after - before:.2f} dB"
assert y.shape == xb.shape and y.dtype == np.float32

# ---------------------------------------------------------------- 2) 全 0 恒等
p0 = {"on": True, "air": 0.0, "detail": 0.0, "dehiss": 0.0, "ext_hz": 11000}
y0 = m2.process(xb, sr, p0, mk_ctx())
d = abs(rms_db(y0) - rms_db(xb))
print(f"[2] 全 0 恒等：RMS 差 = {d:.6f} dB（要求 < 0.05）")
assert d < 0.05
assert np.array_equal(y0, xb), "全 0 时应位级等于输入"

# air=0 时 ext_hz 不生效
p0b = {"on": True, "air": 0.0, "detail": 0.0, "dehiss": 0.4, "ext_hz": 4000}
y0b = m2.process(xb, sr, p0b, mk_ctx())
p0c = {"on": True, "air": 0.0, "detail": 0.0, "dehiss": 0.4, "ext_hz": 20000}
y0c = m2.process(xb, sr, p0c, mk_ctx())
assert np.array_equal(y0b, y0c), "air=0 时 ext_hz 不应影响输出"

# ---------------------------------------------------------------- 3) 全开无 NaN
pf = {"on": True, "air": 0.7, "detail": 0.6, "dehiss": 0.5, "ext_hz": 11000}
ctxf = mk_ctx()
yf = m2.process(xb, sr, pf, ctxf)
assert np.isfinite(yf).all(), "全开输出含 NaN/Inf"
assert yf.shape == xb.shape and yf.dtype == np.float32
print("[3] 全开：无 NaN，形状/类型一致；stats =", ctxf["stats"]["m2"])

# 静音不崩溃、不爆音
xz = np.zeros((sr, 2), np.float32)
yz = m2.process(xz, sr, pf, mk_ctx())
assert np.isfinite(yz).all() and float(np.abs(yz).max()) < 1e-3
print(f"[4] 静音：max = {float(np.abs(yz).max()):.2e}")

# ---------------------------------------------------------------- 5) 单声道
xm = xb[:, 0:1].copy()
ym = m2.process(xm, sr, pf, mk_ctx())
assert ym.ndim == 2 and ym.shape == (n, 1) and np.isfinite(ym).all()
y1 = m2.process(xb[:, 0].copy(), sr, pf, mk_ctx())          # 1D 输入也不崩
assert y1.shape == (n,) and np.isfinite(y1).all()
print(f"[5] 单声道：OK mono_shape={ym.shape} 1d_shape={y1.shape}")

# ---------------------------------------------------------------- 6) 性能 60s 立体声
n60 = 60 * sr
x60 = (0.1 * rng.standard_normal((n60, 2))).astype(np.float32)
p60 = {"on": True, "air": 0.5, "detail": 0.5, "dehiss": 0.4, "ext_hz": 11000}
t0 = time.perf_counter()
y60 = m2.process(x60, sr, p60, mk_ctx())
ms = (time.perf_counter() - t0) * 1000.0
assert np.isfinite(y60).all() and y60.shape == x60.shape
print(f"[6] 60s@48k 立体声耗时：{ms:.0f} ms（要求 < 5000）")
assert ms < 5000.0

print("M2 SMOKE OK")
