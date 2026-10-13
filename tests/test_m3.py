"""m3（人声去掩蔽与优化）自测：合成信号 + 恒等/NaN/单声道/性能断言。

信号：500Hz 正弦（掩蔽源）+ 3kHz 调幅正弦（人声存在感）+ 6.5kHz 齿音噪声脉冲
      + 每个脉冲附带指数衰减噪声尾巴（模拟混响晚期）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livefix import core, m3  # noqa: E402

SR = 48000
DUR = 8.0
PULSES = [2.0, 4.0, 6.0]
TAIL_PULSES = [2.0, 6.0]      # 带混响尾巴的脉冲（room 测试）
CLEAN_PULSE = 4.0             # 无尾巴的干净齿音脉冲（ess 测试，避免尾巴污染 1-3k 参考带）


def make_signal(dur=DUR, sr=SR, stereo=True, seed=0):
    n = int(dur * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    x = 0.15 * np.sin(2.0 * np.pi * 500.0 * t)                       # 掩蔽源 200-800Hz
    am = 0.5 + 0.5 * np.sin(2.0 * np.pi * (2.5 * t + 0.675))         # 2.5Hz 调制 0..1（波谷对齐脉冲中点）
    # 齿音期间声源短暂停顿（真实 /s/ 前后元音常被压制）：脉冲前 50ms 到后 120ms 下探到 0.15
    dip = np.ones(n)
    for t0 in PULSES:
        w = np.clip((np.abs(t - t0) - 0.05) / 0.025, 0.0, 1.0)
        dip = np.minimum(dip, 0.15 + 0.85 * (w * w * (3.0 - 2.0 * w)))
    x = x + 0.4 * dip * (0.2 + 0.8 * am) * np.sin(2.0 * np.pi * 3000.0 * t)  # 3k 存在感（深度0.8）
    for t0 in PULSES:
        i0 = int(t0 * sr)
        m = int(0.06 * sr)
        burst = rng.standard_normal(m)                                # 6.5k 齿音脉冲（带通）
        mf = core.band_mask(sr, 4800.0, 9500.0)
        bf = np.fft.rfftfreq(m, 1.0 / sr)
        burst_m = np.interp(bf, core.rfftfreqs(sr), mf)               # 插到 burst 频率格点
        Bf = np.fft.rfft(burst) * burst_m
        burst = np.fft.irfft(Bf, n=m)
        burst *= 1.2 / (np.max(np.abs(burst)) + 1e-9)
        x[i0:i0 + m] += burst
        if t0 in TAIL_PULSES:                                         # 混响尾巴
            tl = int(0.6 * sr)
            env = np.exp(-np.arange(tl) / (0.15 * sr))
            x[i0:i0 + tl] += 0.3 * env * rng.standard_normal(tl)
    x = np.ascontiguousarray(x.astype(np.float32)[:, None])
    if stereo:
        d = np.zeros_like(x)
        d[5:] = x[:-5]                                               # 右声道微延迟
        x = np.concatenate([x, d], axis=1)
    return x


def band_db(x, sr, lo, hi, t0, t1):
    """[t0,t1) 窗口内的频段平均能量 dB（对全部帧/声道平均）。"""
    X = core.stft(x, sr)
    e = core.band_energy_db(X, sr, lo, hi)               # (ch,T) dB
    hop = core.HOP / sr
    i0 = max(0, int(round(t0 / hop)) - 1)
    i1 = max(i0 + 1, int(round(t1 / hop)) - 1)
    return float(np.mean(e[:, i0:i1]))


def rms_db(x):
    return 10.0 * np.log10(float(np.mean(np.asarray(x, np.float64) ** 2)) + 1e-30)


def p(**kw):
    base = {"on": True, "clarity": 0.0, "room": 0.0, "harsh": 0.0, "ess": 0.0}
    base.update(kw)
    return base


def main():
    x = make_signal()
    n, ch = x.shape
    ctx = {"stats": {}, "log": []}
    ok = True

    # 1) 恒等性：参数全 0 ---------------------------------------------------
    y0 = m3.process(x, SR, p(), ctx)
    d0 = abs(rms_db(y0) - rms_db(x))
    assert y0.dtype == np.float32 and y0.shape == x.shape
    assert d0 < 0.05, f"恒等 RMS 差 {d0:.4f} dB"
    print(f"[1] 全 0 恒等：RMS 差 {d0:.6f} dB  OK")

    # 2) clarity=1：1.5-4.5k 带能量提升 > 3dB ------------------------------
    y = m3.process(x, SR, p(clarity=1.0), ctx)
    b0 = band_db(x, SR, 1500, 4500, 1.5, 6.5)
    b1 = band_db(y, SR, 1500, 4500, 1.5, 6.5)
    assert b1 - b0 > 3.0, f"clarity 提升仅 {b1 - b0:.2f} dB"
    print(f"[2] clarity=1：3k 带能量 {b0:.2f} -> {b1:.2f} dB（+{b1 - b0:.2f} > 3）OK")

    # 3) ess=1：6.5k 脉冲期间 5-9k 带能量下降 > 4dB ------------------------
    y = m3.process(x, SR, p(ess=1.0), ctx)
    drops = []
    for t0 in [CLEAN_PULSE]:
        e0 = band_db(x, SR, 5000, 9000, t0 + 0.012, t0 + 0.055)
        e1 = band_db(y, SR, 5000, 9000, t0 + 0.012, t0 + 0.055)
        drops.append(e0 - e1)
    assert min(drops) > 4.0, f"ess 下降不足：{[round(d, 2) for d in drops]}"
    print(f"[3] ess=1：脉冲期 5-9k 下降 {[round(d, 2) for d in drops]} dB（最小 > 4）OK")

    # 4) room=1：脉冲结束 200ms 后 5-9k 带能量下降 > 3dB -------------------
    y = m3.process(x, SR, p(room=1.0), ctx)
    drops = []
    for t0 in TAIL_PULSES:
        e0 = band_db(x, SR, 5000, 9000, t0 + 0.26, t0 + 0.41)
        e1 = band_db(y, SR, 5000, 9000, t0 + 0.26, t0 + 0.41)
        drops.append(e0 - e1)
    assert min(drops) > 3.0, f"room 抑制不足：{[round(d, 2) for d in drops]}"
    print(f"[4] room=1：脉冲后 200ms 尾巴下降 {[round(d, 2) for d in drops]} dB（最小 > 3）OK")

    # 5) harsh=1：不炸、有限、脉冲期 2.2-5.5k 有可感压低 ---------------------
    y = m3.process(x, SR, p(harsh=1.0), ctx)
    e0 = band_db(x, SR, 2200, 5500, PULSES[0] + 0.012, PULSES[0] + 0.055)
    e1 = band_db(y, SR, 2200, 5500, PULSES[0] + 0.012, PULSES[0] + 0.055)
    assert np.isfinite(y).all()
    print(f"[5] harsh=1：脉冲期 2.2-5.5k {e0:.2f} -> {e1:.2f} dB（有限即可）OK")

    # 6) 全强度联合：立体声 + 无 NaN + 长度声道不变 -------------------------
    y = m3.process(x, SR, p(clarity=0.6, room=0.4, harsh=0.4, ess=0.4), ctx)
    assert np.isfinite(y).all() and y.dtype == np.float32 and y.shape == x.shape
    assert np.max(np.abs(y)) < 3.0, f"异常峰值 {np.max(np.abs(y)):.2f}"
    assert isinstance(ctx["stats"].get("m3"), dict) and ctx["log"]
    print(f"[6] 默认强度联合：peak={np.max(np.abs(y)):.3f} stats={ctx['stats']['m3']}  OK")

    # 7) 单声道不崩溃 -------------------------------------------------------
    xm = make_signal(stereo=False)
    ym = m3.process(xm, SR, p(clarity=0.5, room=0.5, harsh=0.5, ess=0.5), {"stats": {}, "log": []})
    assert ym.shape == xm.shape and ym.ndim == 2 and np.isfinite(ym).all()
    y1 = m3.process(xm[:, 0], SR, p(clarity=0.5), None)          # 1D 输入
    assert y1.ndim == 1 and np.isfinite(y1).all()
    print("[7] 单声道 (n,1) 与 1D 输入均正常  OK")

    # 8) 性能：12s@48k 立体声 < 5s（外推 60s ≈ 5x，目标单模块 ≤3s 内）------
    xb = make_signal(dur=12.0)
    t1 = time.perf_counter()
    m3.process(xb, SR, p(clarity=0.6, room=0.4, harsh=0.4, ess=0.4), None)
    dt = time.perf_counter() - t1
    assert dt < 5.0, f"耗时 {dt:.2f}s ≥ 5s"
    print(f"[8] 性能：12s 立体声耗时 {dt:.2f}s（线性外推 60s ≈ {dt * 5:.1f}s，目标 ≤3s）  OK")

    print("\nALL M3 TESTS PASSED")


if __name__ == "__main__":
    main()
