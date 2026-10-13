"""m1（中低频重建）自测。

覆盖：60s 立体声形状/有限性/时长、全 0 恒等（RMS 差 < 0.05 dB）、weight=1 时
30-120Hz 带能量提升 > 3 dB、单声道不崩溃、60s 立体声处理耗时 < 5s。
运行：python tests/test_m1.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from livefix import core, m1  # noqa: E402

SR = 48000
DUR = 60.0


def _synth_stereo(sr=SR, dur=DUR):
    """低频正弦扫频 + 噪声 + 鼓脉冲（60s 立体声）。"""
    n = int(sr * dur)
    t = np.arange(n) / sr
    rng = np.random.default_rng(7)
    f0, f1 = 35.0, 220.0
    fc = f0 * (f1 / f0) ** (t / dur)
    phase = 2.0 * np.pi * np.cumsum(fc) / sr
    sweep = 0.06 * np.sin(phase)
    noise = 0.02 * rng.standard_normal(n)
    klen = int(0.25 * sr)
    tt = np.arange(klen) / sr
    kick = 0.22 * np.sin(2.0 * np.pi * 58.0 * tt) * np.exp(-tt / 0.06)
    pulse = np.zeros(n)
    for st in range(0, n - klen, int(0.5 * sr)):
        pulse[st:st + klen] += kick
    L = sweep + noise + pulse
    R = 0.95 * sweep + 0.90 * noise + 0.98 * pulse
    return np.stack([L, R], axis=1).astype(np.float32)


def _thin_bass(sr=SR, dur=6.0):
    """低频很瘦的信号（30-120Hz 帧能量约 -44 dBFS），给 weight 留补正空间。"""
    n = int(sr * dur)
    rng = np.random.default_rng(3)
    t = np.arange(n) / sr
    x = 0.012 * np.sin(2.0 * np.pi * 62.0 * t) + 0.0006 * rng.standard_normal(n)
    return x.astype(np.float32)[:, None]


def _rms_db(v):
    v = np.asarray(v, np.float64)
    return 10.0 * np.log10(float(np.mean(v * v)) + 1e-30)


def _band_energy_mean_db(x, sr, lo, hi):
    X = core.stft(x, sr)
    e = core.band_energy_db(X, sr, lo, hi)      # (ch,T)
    v = e[e > -80.0]                             # 只看有内容的帧
    return float(v.mean()) if v.size else -120.0


def test_stereo_60s():
    x = _synth_stereo()
    p = {"on": True, "weight": 0.6, "density": 0.5, "sub": 0.4, "punch": 0.5}
    t0 = time.perf_counter()
    y = m1.process(x, SR, p, {})
    dt = time.perf_counter() - t0
    assert y.shape == x.shape, f"形状不符：{y.shape} != {x.shape}"
    assert y.dtype == np.float32
    assert np.isfinite(y).all(), "输出含 NaN/Inf"
    assert y.shape[0] == x.shape[0], "时长不一致"
    # 中低频确实被抬起来了（整段平均）
    b0 = _band_energy_mean_db(x, SR, 30, 600)
    b1 = _band_energy_mean_db(y, SR, 30, 600)
    print(f"[60s 立体声] 耗时 {dt:.2f}s | 30-600Hz 带能量 {b0:.1f} -> {b1:.1f} dB")
    assert dt < 5.0, f"性能超限：{dt:.2f}s >= 5s"
    assert b1 > b0, "默认参数下中低频未提升"


def test_identity_all_zero():
    x = _synth_stereo(dur=4.0)
    p = {"on": True, "weight": 0.0, "density": 0.0, "sub": 0.0, "punch": 0.0}
    y = m1.process(x, SR, p, {})
    diff = abs(_rms_db(y) - _rms_db(x))
    print(f"[全 0 恒等] RMS 差 {diff:.4f} dB")
    assert diff < 0.05, f"恒等失败：RMS 差 {diff:.4f} dB"
    assert np.array_equal(y, x), "全 0 时应为逐样本恒等"


def test_weight_boost():
    x = _thin_bass()
    before = _band_energy_mean_db(x, SR, 30, 120)
    p = {"on": True, "weight": 1.0, "density": 0.0, "sub": 0.0, "punch": 0.0}
    y = m1.process(x, SR, p, {})
    after = _band_energy_mean_db(y, SR, 30, 120)
    print(f"[weight=1] 30-120Hz 带能量 {before:.1f} -> {after:.1f} dB（+{after - before:.1f}）")
    assert after - before > 3.0, f"weight=1 提升不足 3dB：+{after - before:.1f}"
    assert np.isfinite(y).all()


def test_mono():
    x = _synth_stereo(dur=3.0)[:, :1]            # (n,1)
    p = {"on": True, "weight": 0.6, "density": 0.5, "sub": 0.4, "punch": 0.5}
    y = m1.process(x, SR, p, {})
    assert y.shape == x.shape and np.isfinite(y).all()
    y1 = m1.process(x[:, 0], SR, p, {})          # 1D 输入也不崩
    assert y1.ndim == 1 and y1.shape[0] == x.shape[0] and np.isfinite(y1).all()
    # 各子效果单独开关也不崩
    for key in ("weight", "density", "sub", "punch"):
        q = {"on": True, "weight": 0.0, "density": 0.0, "sub": 0.0, "punch": 0.0}
        q[key] = 1.0
        z = m1.process(x, SR, q, {})
        assert np.isfinite(z).all(), f"{key}=1 时输出非有限"
    print("[单声道] (n,1) 与 1D 均通过；四个子效果单开均有限")


def main():
    t0 = time.perf_counter()
    test_identity_all_zero()
    test_weight_boost()
    test_mono()
    test_stereo_60s()
    print(f"ALL TESTS PASSED（总耗时 {time.perf_counter() - t0:.1f}s）")


if __name__ == "__main__":
    main()
