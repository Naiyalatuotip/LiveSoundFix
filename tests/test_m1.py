"""m1（中低频重建）自测：恒等 / weight / ring 拖尾 / dive 下潜 / 同开安全 / 单声道 / 性能 / 分段冒烟。

1) 全参数 0（含 ring/dive）→ 逐样本恒等
2) ring=1：冲击 ±30ms 带能量下降 <1.5dB（攻击保住）；冲击后 150-400ms 下降 >6dB（砍尾巴）；
   持续 63Hz 正弦段变化 <1.5dB（valid 保护）
3) dive=1：HP 切除后的「缺失」信号 20-40Hz 带能量相对 dive=0 提升 >6dB；源带 90-160Hz 完好
   （|Δ|<0.5dB）；dive=0 恒等
4) ring=1 & dive=1 同开：无 NaN、形状 dtype 不变、|y|<3
5) 单声道 (n,1) 与 1D 输入不崩
6) 60s@48k 立体声 ring=1 & dive=1 耗时 ≤3s（打印）
7) 15s 信号经 core.block_apply(5s/1s) 分段跑 ring=1：无 NaN、长度一致
运行：python -X utf8 tests/test_m1.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from livefix import core, m1  # noqa: E402

SR = 48000
DUR = 60.0
P0 = {"on": True, "weight": 0.0, "density": 0.0, "sub": 0.0, "punch": 0.0,
      "tame": 0.0, "ring": 0.0, "dive": 0.0}


# ----------------------------------------------------------------- 合成信号
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
    """低频很瘦的信号，给 weight 留补正空间。"""
    n = int(sr * dur)
    rng = np.random.default_rng(3)
    t = np.arange(n) / sr
    x = 0.012 * np.sin(2.0 * np.pi * 62.0 * t) + 0.0006 * rng.standard_normal(n)
    return x.astype(np.float32)[:, None]


def _ring_signal(sr=SR, dur=8.0):
    """63Hz 冲击（30ms）→ τ≈0.35s 模态拖尾；4.0-7.0s 另有持续 63Hz 正弦。"""
    n = int(sr * dur)
    t = np.arange(n) / sr
    y = np.zeros(n)
    f = 63.0
    t0, ti = 1.0, 0.03
    m = (t >= t0) & (t < t0 + ti)                      # 冲击：sin² 包络，峰值 1.0
    loc = (t[m] - t0) / ti
    y[m] += np.sin(np.pi * loc) ** 2 * np.sin(2.0 * np.pi * f * t[m])
    m2 = t >= t0 + ti                                  # 拖尾：0.25 起，τ=0.35s 指数衰减
    y[m2] += 0.25 * np.exp(-(t[m2] - (t0 + ti)) / 0.35) * np.sin(2.0 * np.pi * f * t[m2])
    m3 = (t >= 4.0) & (t < 7.0)                        # 持续段（valid 保护测试）
    w = np.ones(int(m3.sum()))
    r = int(0.03 * sr)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(r) / r)
    w[:r] = ramp
    w[-r:] = ramp[::-1]
    y[m3] += 0.30 * w * np.sin(2.0 * np.pi * f * t[m3])
    x = y[:, None]
    return np.ascontiguousarray(np.concatenate([x, x * 0.97], axis=1), np.float32)


def _dive_missing(sr=SR, dur=6.0):
    """50-150Hz 带限噪声（低频内容的真实形态：宽频、相位演化）、<50Hz 已切除。

    注意：不用对齐 bin 的纯音——纯音的复数复制会与残存泄漏同频相干抵消，
    是阶梯合成的最差情形；真实低频内容按功率非相干叠加（见 m1.py D_W2 注释）。
    """
    n = int(sr * dur)
    rng = np.random.default_rng(11)
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(n, 1.0 / sr)
    X[(fr < 50.0) | (fr > 150.0)] = 0.0
    y = np.fft.irfft(X, n)
    return (0.4 * y / (np.abs(y).max() + 1e-9)).astype(np.float32)[:, None]


# ----------------------------------------------------------------- 度量
def _rms_db(v):
    v = np.asarray(v, np.float64)
    return 10.0 * np.log10(float(np.mean(v * v)) + 1e-30)


def _band_energy_mean_db(x, sr, lo, hi):
    X = core.stft(x, sr)
    e = core.band_energy_db(X, sr, lo, hi)      # (ch,T)
    v = e[e > -80.0]
    return float(v.mean()) if v.size else -120.0


def _win_band_db(x, sr, lo, hi, t0, t1):
    """窗口 [t0,t1) 内频段能量：帧线性能量平均后转 dB（对输入/输出可比）。"""
    X = core.stft(np.asarray(x, np.float32), sr)
    e = core.band_energy_db(X, sr, lo, hi)                       # (ch,T) dB
    tt = np.arange(e.shape[1]) * core.HOP / sr - (core.N_FFT / 2) / sr
    sel = (tt >= t0) & (tt < t1)
    if not sel.any():                                            # 帧稀疏兜底
        sel = np.zeros(e.shape[1], bool)
        sel[e.shape[1] // 2] = True
    p = np.power(10.0, e[:, sel] / 10.0)
    return 10.0 * np.log10(float(p.mean()) + 1e-30)


# ----------------------------------------------------------------- 测试
def test_identity_all_zero():
    x = _synth_stereo(dur=4.0)
    y = m1.process(x, SR, dict(P0), {})
    diff = abs(_rms_db(y) - _rms_db(x))
    print(f"[全 0 恒等] RMS 差 {diff:.4f} dB（含 ring/dive/tame）")
    assert diff < 0.05, f"恒等失败：RMS 差 {diff:.4f} dB"
    assert np.array_equal(y, x), "全 0 时应为逐样本恒等"


def test_weight_boost():
    x = _thin_bass()
    before = _band_energy_mean_db(x, SR, 30, 120)
    y = m1.process(x, SR, dict(P0, weight=1.0), {})
    after = _band_energy_mean_db(y, SR, 30, 120)
    print(f"[weight=1] 30-120Hz 带能量 {before:.1f} -> {after:.1f} dB（+{after - before:.1f}）")
    assert after - before > 3.0, f"weight=1 提升不足 3dB：+{after - before:.1f}"
    assert np.isfinite(y).all()


def test_ring():
    x = _ring_signal()
    y = m1.process(x, SR, dict(P0, ring=1.0), {})
    a0, a1 = (_win_band_db(v, SR, 40, 110, 0.97, 1.03) for v in (x, y))   # 攻击 ±30ms
    t0, t1 = (_win_band_db(v, SR, 40, 110, 1.15, 1.40) for v in (x, y))   # 尾巴 150-400ms
    s0, s1 = (_win_band_db(v, SR, 40, 110, 4.5, 6.5) for v in (x, y))     # 持续正弦
    d_atk, d_tail, d_sus = a0 - a1, t0 - t1, s0 - s1
    print(f"[ring=1] 攻击 −{d_atk:.2f}dB(<1.5) | 尾巴 −{d_tail:.2f}dB(>6) | "
          f"持续 −{d_sus:.2f}dB(<1.5)")
    assert d_atk < 1.5, f"攻击被压掉 {d_atk:.2f}dB"
    assert d_tail > 6.0, f"尾巴只压了 {d_tail:.2f}dB"
    assert abs(d_sus) < 1.5, f"持续段被误压 {d_sus:.2f}dB（valid 保护失效）"
    assert np.isfinite(y).all()


def test_dive():
    x = _dive_missing()
    y0 = m1.process(x, SR, dict(P0, dive=0.0), {})
    assert np.array_equal(y0, x), "dive=0 应逐样本恒等"
    y1 = m1.process(x, SR, dict(P0, dive=1.0), {})
    b0 = _win_band_db(x, SR, 20, 40, 1.0, 5.0)   # 20-40Hz：真正缺失的区域（2048-STFT 仅 bin1）
    b1 = _win_band_db(y1, SR, 20, 40, 1.0, 5.0)
    c0 = _win_band_db(x, SR, 90, 160, 1.0, 5.0)  # 源带：dive bin/低架尾均不涉及，应完好
    c1 = _win_band_db(y1, SR, 90, 160, 1.0, 5.0)
    w0 = _win_band_db(x, SR, 20, 50, 1.0, 5.0)
    w1 = _win_band_db(y1, SR, 20, 50, 1.0, 5.0)
    print(f"[dive=1] 20-40Hz {b0:.1f} -> {b1:.1f} dB（+{b1 - b0:.1f}，>6）；"
          f"20-50Hz +{w1 - w0:.1f}；源带 90-160Hz {c0:.1f} -> {c1:.1f}（|Δ|<0.5）")
    assert b1 - b0 > 6.0, f"下潜提升不足（20-40Hz）：+{b1 - b0:.1f}"
    assert abs(c1 - c0) < 0.5, f"源带被扰动: {c1 - c0:+.1f} dB"
    assert np.isfinite(y1).all() and y1.shape == x.shape and y1.dtype == np.float32


def test_ring_dive_together():
    x = _ring_signal()
    y = m1.process(x, SR, {"on": True, "ring": 1.0, "dive": 1.0}, {})
    peak = float(np.abs(y).max())
    print(f"[ring=1 & dive=1 同开] shape={y.shape} dtype={y.dtype} 峰值 {peak:.3f}（<3）")
    assert y.shape == x.shape and y.dtype == np.float32
    assert np.isfinite(y).all(), "同开输出含 NaN/Inf"
    assert peak < 3.0, f"峰值爆了：{peak:.3f}"


def test_mono():
    x = _synth_stereo(dur=3.0)[:, :1]            # (n,1)
    y = m1.process(x, SR, dict(P0, ring=1.0, dive=1.0), {})
    assert y.shape == x.shape and np.isfinite(y).all()
    y1 = m1.process(x[:, 0], SR, dict(P0, ring=1.0, dive=1.0), {})   # 1D 输入
    assert y1.ndim == 1 and y1.shape[0] == x.shape[0] and np.isfinite(y1).all()
    for key in ("weight", "density", "sub", "punch", "tame", "ring", "dive"):
        q = dict(P0)
        q[key] = 1.0
        z = m1.process(x, SR, q, {})
        assert np.isfinite(z).all(), f"{key}=1 时输出非有限"
    print("[单声道] (n,1) 与 1D 均通过；七个参数单开均有限")


def test_block_apply():
    x = _ring_signal(dur=15.0)
    p = dict(P0, ring=1.0, dive=1.0)
    fn = lambda seg, sr: m1.process(seg, sr, p, {})   # noqa: E731
    y = core.block_apply(x, SR, fn, block_s=5.0, xfade_s=1.0)
    assert y.shape == x.shape and y.dtype == np.float32
    assert np.isfinite(y).all(), "分段处理输出含 NaN"
    print(f"[分段冒烟] block_apply 5s/1s → {y.shape}，长度一致、无 NaN")


def test_stereo_60s():
    x = _synth_stereo()
    p = dict(P0, weight=0.6, density=0.5, sub=0.4, punch=0.5)
    t0 = time.perf_counter()
    y = m1.process(x, SR, p, {})
    dt = time.perf_counter() - t0
    assert y.shape == x.shape and y.dtype == np.float32
    assert np.isfinite(y).all() and y.shape[0] == x.shape[0]
    b0 = _band_energy_mean_db(x, SR, 30, 600)
    b1 = _band_energy_mean_db(y, SR, 30, 600)
    print(f"[60s 立体声 默认参数] 耗时 {dt:.2f}s | 30-600Hz {b0:.1f} -> {b1:.1f} dB")
    assert dt < 5.0, f"性能超限：{dt:.2f}s >= 5s"
    assert b1 > b0, "默认参数下中低频未提升"


def test_perf_ring_dive():
    x = _synth_stereo()
    p = dict(P0, weight=0.6, density=0.5, sub=0.4, punch=0.5, ring=1.0, dive=1.0)
    t0 = time.perf_counter()
    y = m1.process(x, SR, p, {})
    dt = time.perf_counter() - t0
    print(f"[60s 立体声 ring=1 dive=1] 耗时 {dt:.2f}s（≤3s）")
    assert dt <= 3.0, f"性能超限：{dt:.2f}s > 3s"
    assert y.shape == x.shape and np.isfinite(y).all()


def main():
    t0 = time.perf_counter()
    test_identity_all_zero()
    test_weight_boost()
    test_ring()
    test_dive()
    test_ring_dive_together()
    test_mono()
    test_block_apply()
    test_stereo_60s()
    test_perf_ring_dive()
    print(f"ALL TESTS PASSED（总耗时 {time.perf_counter() - t0:.1f}s）")


if __name__ == "__main__":
    main()
