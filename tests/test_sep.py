"""sep（源分离分级）模块自测：合成混合信号，跑 hpss / nmf 两模式。

运行：cd E:\\作品集\\音频修复v1 && <python> tests\\test_sep.py
"""
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from livefix import core, sep  # noqa: E402

SR = 48000
EPS = 1e-12


def make_mix(sr=SR, dur=12.0, stereo=True, seed=7):
    """低频正弦 100Hz + 3kHz 调幅正弦（人声感）+ 宽带噪声脉冲串（打击）+ 底噪。"""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    t = np.arange(n) / sr
    low = 0.22 * np.sin(2 * np.pi * 100.0 * t)
    voice = 0.28 * np.sin(2 * np.pi * 3000.0 * t) * (0.55 + 0.45 * np.sin(2 * np.pi * 1.7 * t))
    noise = rng.standard_normal(n) * 0.004
    perc = np.zeros(n)
    ln = int(0.035 * sr)
    env = np.exp(-np.linspace(0.0, 5.0, ln))
    pos = int(0.30 * sr)
    step = int(0.50 * sr)
    while pos + ln < n:
        perc[pos:pos + ln] += rng.standard_normal(ln) * env * 0.55
        pos += step
    mono = low + voice + perc + noise
    if stereo:
        x = np.stack([mono, mono * 0.92 + 0.01 * np.sin(2 * np.pi * 3020.0 * t)], axis=1)
    else:
        x = mono[:, None]
    return np.ascontiguousarray(x, np.float32)


def rms_db(x):
    return 20.0 * np.log10(np.sqrt(float(np.mean(np.asarray(x, np.float64) ** 2))) + EPS)


def band_db(x, sr, f_lo, f_hi):
    X = core.stft(x, sr)
    m = core.band_mask(sr, f_lo, f_hi)[None, None, :]
    e = float(((np.abs(X) ** 2) * m).sum())
    return 10.0 * np.log10(e + 1e-20)


def params(**kw):
    """按引擎方式构造参数（engine 会把 on 从参数里剥掉）。"""
    p = {"mode": "hpss", "routing": "graded",
         "vocal_gain": 1.0, "low_gain": 1.0, "perc_gain": 1.0, "other_gain": 1.0}
    p.update(kw)
    return p


def new_ctx():
    return {"rng": np.random.default_rng(0), "stats": {}, "log": [], "stems": None, "note": {}}


def run_mode(mode, dur=12.0):
    print(f"--- mode={mode} ({dur}s stereo) ---")
    x = make_mix(dur=dur)
    n, ch = x.shape
    ctx = new_ctx()
    t0 = time.perf_counter()
    y = sep.process(x, SR, params(mode=mode), ctx)
    ms = (time.perf_counter() - t0) * 1000.0

    assert y.shape == (n, ch) and y.dtype == np.float32, (y.shape, y.dtype)
    assert np.isfinite(y).all(), "输出含 NaN/Inf"
    dev = ctx["stats"]["sep"]["mask_sum_dev"]
    assert dev < 1e-6, f"掩码和偏差过大：{dev}"
    print(f"  运行 {ms:.0f} ms · 掩码和最大偏差 {dev:.2e}")

    # 增益全 1：恒等
    d = abs(rms_db(y) - rms_db(x))
    assert d < 0.05, f"全 1 增益 RMS 差 {d:.3f} dB"
    assert np.max(np.abs(y - x)) < 1e-3
    print(f"  全 1 增益恒等：RMS 差 {d:.4f} dB · 最大样本差 {np.max(np.abs(y - x)):.2e}")

    # graded：ctx["stems"] 存在且 vocal+backing == 返回值
    stems = ctx["stems"]
    assert stems is not None and set(stems) == {"vocal", "backing"}
    assert stems["vocal"].shape == (n, ch) and stems["backing"].shape == (n, ch)
    err = float(np.max(np.abs(stems["vocal"] + stems["backing"] - y)))
    assert err < 1e-3, f"vocal+backing 与返回值不符：{err}"
    print(f"  graded stems 一致：max|V+B-y| = {err:.2e}")

    # mix：stems 为 None
    ctx2 = new_ctx()
    y2 = sep.process(x, SR, params(mode=mode, routing="mix"), ctx2)
    assert ctx2["stems"] is None
    d2 = abs(rms_db(y2) - rms_db(x))
    assert d2 < 0.05 and np.max(np.abs(y2 - x)) < 1e-3
    print(f"  mix 路由 stems=None · 恒等 RMS 差 {d2:.4f} dB")

    # vocal_gain=0：人声带（3kHz）能量显著下降
    ctx3 = new_ctx()
    y3 = sep.process(x, SR, params(mode=mode, vocal_gain=0.0), ctx3)
    drop = band_db(x, SR, 2600, 3400) - band_db(y3, SR, 2600, 3400)
    assert drop > 6.0, f"vocal_gain=0 时 3kHz 带仅下降 {drop:.1f} dB"
    # 低频带应基本保留
    dl = abs(band_db(x, SR, 80, 120) - band_db(y3, SR, 80, 120))
    assert dl < 1.0, f"vocal_gain=0 不应影响低频带：{dl:.2f} dB"
    print(f"  vocal_gain=0：2.6-3.4kHz 带下降 {drop:.1f} dB · 80-120Hz 带变化 {dl:.3f} dB")

    # low_gain=0：低频带显著下降，人声带基本保留
    ctx4 = new_ctx()
    y4 = sep.process(x, SR, params(mode=mode, low_gain=0.0), ctx4)
    drop_lo = band_db(x, SR, 80, 120) - band_db(y4, SR, 80, 120)
    dv = abs(band_db(x, SR, 2600, 3400) - band_db(y4, SR, 2600, 3400))
    assert drop_lo > 6.0, f"low_gain=0 时 100Hz 带仅下降 {drop_lo:.1f} dB"
    assert dv < 1.0, f"low_gain=0 不应影响人声带：{dv:.2f} dB"
    print(f"  low_gain=0：80-120Hz 带下降 {drop_lo:.1f} dB · 2.6-3.4kHz 带变化 {dv:.3f} dB")
    return ms


def run_mono():
    print("--- mono + 极短信号健壮性 ---")
    x = make_mix(dur=3.0, stereo=False)
    ctx = new_ctx()
    y = sep.process(x, SR, params(mode="hpss"), ctx)
    assert y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all()
    assert ctx["stems"]["vocal"].shape == x.shape
    tiny = make_mix(dur=0.05, stereo=False)
    ctx2 = new_ctx()
    y2 = sep.process(tiny, SR, params(mode="hpss"), ctx2)
    assert y2.shape == tiny.shape and np.isfinite(y2).all()
    d = abs(rms_db(y2) - rms_db(tiny))
    assert d < 0.05
    ctx3 = new_ctx()
    y3 = sep.process(tiny, SR, params(mode="nmf"), ctx3)
    assert y3.shape == tiny.shape and np.isfinite(y3).all()
    # 显式 on=False：透传且 stems 为 None
    ctx4 = new_ctx()
    y4 = sep.process(x, SR, params(on=False), ctx4)
    assert np.array_equal(y4, x) and ctx4["stems"] is None
    print("  mono 3s OK · 极短 0.05s hpss/nmf OK · on=False 透传 OK")


def run_block_apply():
    """模拟引擎分段调用（core.block_apply），验证跨段 stems 累加与引擎输出一致。"""
    print("--- core.block_apply 分段调用（block_s=5, xfade=0.5, 13s）---")
    x = make_mix(dur=13.0)
    ctx = new_ctx()
    t0 = time.perf_counter()
    y = core.block_apply(x, SR, lambda seg, _sr: sep.process(seg, _sr, params(mode="hpss"), ctx),
                         block_s=5.0, xfade_s=0.5)
    ms = (time.perf_counter() - t0) * 1000.0
    stems = ctx["stems"]
    assert stems is not None
    assert stems["vocal"].shape == x.shape == y.shape
    err = float(np.max(np.abs(stems["vocal"] + stems["backing"] - y)))
    assert err < 1e-3, f"分段 stems 与引擎输出不一致：{err}"
    d = abs(rms_db(y) - rms_db(x))
    assert d < 0.05
    print(f"  {ms:.0f} ms · max|V+B-y| = {err:.2e} · 全 1 增益恒等 RMS 差 {d:.4f} dB")


if __name__ == "__main__":
    t_all = time.perf_counter()
    # 中值一致性
    rng = np.random.default_rng(3)
    a = rng.standard_normal((300, 64)).astype(np.float32)
    for k, axis in ((25, 0), (31, 1)):
        pad = k // 2
        ref = np.empty_like(a)
        if axis == 0:
            p = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
            for i in range(a.shape[0]):
                ref[i] = np.median(p[i:i + k], axis=0)
        else:
            p = np.pad(a, ((0, 0), (pad, pad)), mode="edge")
            for j in range(a.shape[1]):
                ref[:, j] = np.median(p[:, j:j + k], axis=1)
        got = sep._med1d(a, k, axis)
        assert np.array_equal(got, ref), f"_med1d 不一致 k={k} axis={axis}"
    print("--- _med1d 与 np.median 逐样本一致（k=25/axis0, k=31/axis1） ---")

    ms_hpss = run_mode("hpss")
    ms_nmf = run_mode("nmf")
    run_mono()
    run_block_apply()

    print("--- 60s@48k 立体声性能 ---")
    big = make_mix(dur=60.0)
    for mode in ("hpss", "nmf"):
        ctx = new_ctx()
        t0 = time.perf_counter()
        y = sep.process(big, SR, params(mode=mode), ctx)
        ms = (time.perf_counter() - t0) * 1000.0
        assert np.isfinite(y).all() and y.shape == big.shape
        d = abs(rms_db(y) - rms_db(big))
        assert d < 0.05
        lim = 4000.0 if mode == "hpss" else 8000.0
        flag = "OK" if ms <= lim else "超预算!"
        print(f"  {mode}: {ms:.0f} ms（预算 {lim:.0f} ms）{flag} · 恒等 RMS 差 {d:.4f} dB")
        assert ms <= lim
    print(f"全部通过，总耗时 {time.perf_counter() - t_all:.1f}s（12s 段：hpss {ms_hpss:.0f} ms / nmf {ms_nmf:.0f} ms）")
