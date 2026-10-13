"""m6 母带响度 自测：合成大动态音乐感信号，验证响度/天花板/LRA/单声道/数值安全与耗时。

运行：python tests/test_m6.py
"""
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from livefix import m6, measure  # noqa: E402

SR = 48000


def synth_music(dur=60.0, stereo=True, seed=7):
    """安静段/响段交替的音乐感信号：和声簇 + 节拍脉冲 + 噪声 + 单样本瞬态。"""
    n = int(SR * dur)
    rng = np.random.default_rng(seed)
    ch = 2 if stereo else 1
    x = np.zeros((n, ch), np.float64)
    seg = int(5.0 * SR)                      # 段长 > 3s 短时窗，保证真实 LRA
    nseg = (n + seg - 1) // seg
    for i, s in enumerate(range(0, n, seg)):
        m = min(seg, n - s)
        tt = np.arange(m) / SR
        loud = (i % 2 == 1) and (i < nseg - 1)   # 末段强制安静（真实母带惯例，也避开 measure.lra 尾部零稀释窗的假性拉偏）
        amp = 0.45 if loud else 0.018   # 安静段比响段低约 28dB（大动态）
        f0 = 110.0 * (2.0 ** ((i % 5) / 12.0))
        sig = np.zeros(m)
        for mul, ga in ((1, 1.0), (2, 0.55), (3, 0.35), (4, 0.22), (5, 0.12), (7, 0.08)):
            sig += ga * np.sin(2 * np.pi * f0 * mul * tt + rng.uniform(0, 6.28))
        if loud:
            ph = np.mod(tt, 0.625)
            sig += 2.2 * (ph < 0.02) * np.exp(-ph * 120.0)
        sig *= amp * (0.85 + 0.15 * np.sin(2 * np.pi * 0.4 * tt))
        noise = rng.standard_normal(m) * amp * 0.05
        for c in range(ch):
            x[s:s + m, c] = sig + noise + (0.15 * sig if c else 0.0) * rng.uniform(0.8, 1.2)
    for pos, v in ((12.3, 0.99), (31.7, 0.95)):
        k = int(pos * SR)
        if k < n:
            x[k, 0] += v  # 单样本冲击，考验限制器
    return x.astype(np.float32)


def ctx_new():
    return {"rng": np.random.default_rng(0), "stats": {}, "log": [], "stems": None, "note": {}}


def test_fast_path_agreement():
    """快速 K 滤波路径与 measure 对拍。"""
    x = synth_music(dur=20.0, seed=11)
    p = {"on": True, "target_lufs": -16, "lra": 6, "ceiling": -1.0}
    ctx = ctx_new()
    out = m6.process(x, SR, p, ctx)
    ref_lufs = measure.lufs_integrated(x, SR)
    fast_lufs = ctx["stats"]["m6"]["lufs_before"]
    d = abs(ref_lufs - fast_lufs)
    print(f"[fast] lufs measure={ref_lufs:.3f} fast={fast_lufs:.3f} diff={d:.3f}")
    assert d < 0.2, f"快速响度路径偏差过大: {d:.3f} LU"
    ref_lra = measure.lra(x, SR)
    fast_lra = ctx["stats"]["m6"]["lra_before"]
    print(f"[fast] lra  measure={ref_lra:.3f} fast={fast_lra:.3f} diff={abs(ref_lra - fast_lra):.3f}")
    assert abs(ref_lra - fast_lra) < 0.3
    _ = out


def test_rolling_min():
    rng = np.random.default_rng(3)
    v = rng.standard_normal(1000)
    for w in (1, 5, 37, 240):
        got = m6._rolling_min(v, w)
        want = np.array([v[i:i + w].min() for i in range(v.shape[0])])
        assert np.allclose(got, want, atol=1e-6), f"rolling_min w={w} 不一致"
    print("[unit] _rolling_min OK")


def test_env_oversampled():
    n = SR * 3
    x = synth_music(dur=3.0, seed=5)
    x[SR, 0] = 0.97
    env = m6._env_oversampled(x, SR)
    assert env.shape == (4 * n,)
    xa = np.abs(x).max(axis=1)
    assert env[::4].max() >= xa.max() - 1e-3, "4x 包络应覆盖采样峰"
    dev = float(np.abs(env[::4] - xa).max())
    print(f"[unit] env 对齐采样点最大偏差 {dev:.2e}，env_max={env.max():.4f} x_max={np.abs(x).max():.4f}")
    assert dev < 1e-3


def test_silence():
    x = np.zeros((SR, 1), np.float32)
    ctx = ctx_new()
    out = m6.process(x, SR, {"on": True, "target_lufs": -16, "lra": 6, "ceiling": -1.0}, ctx)
    assert out.shape == (SR, 1) and np.isfinite(out).all() and np.abs(out).max() == 0.0
    print("[edge] 静音 OK")


def test_stereo_main():
    p = {"on": True, "target_lufs": -16.0, "lra": 6.0, "ceiling": -1.0}
    x = synth_music(dur=60.0, seed=7)
    lufs_in = measure.lufs_integrated(x, SR)
    lra_in = measure.lra(x, SR)
    tp_in = measure.true_peak_db(x, SR)
    ctx = ctx_new()
    t0 = time.perf_counter()
    out = m6.process(x, SR, p, ctx)
    ms = (time.perf_counter() - t0) * 1000.0

    assert out.dtype == np.float32 and out.shape == x.shape, "形状/dtype 不一致"
    assert np.isfinite(out).all(), "存在 NaN/Inf"
    lufs_out = measure.lufs_integrated(out, SR)
    lra_out = measure.lra(out, SR)
    tp_out = measure.true_peak_db(out, SR)
    print(f"[main] 输入 LUFS={lufs_in:.2f} LRA={lra_in:.2f} TP={tp_in:.2f}")
    print(f"[main] 输出 LUFS={lufs_out:.2f} LRA={lra_out:.2f} TP={tp_out:.2f}")
    print(f"[main] stats={ctx['stats']['m6']}")
    print(f"[main] 耗时 {ms:.0f} ms (60s 立体声)")
    assert abs(lufs_out - (-16.0)) < 0.5, f"LUFS 未达标: {lufs_out:.2f}"
    assert tp_out <= -1.0 + 0.1, f"真峰超天花板: {tp_out:.3f}"
    assert lra_out <= 6.0 + 1.5, f"LRA 未收敛: {lra_out:.2f}"
    st = ctx["stats"]["m6"]
    for k in ("lufs_before", "lufs_after", "lra_before", "lra_after", "gain_db"):
        assert k in st
    assert ms < 6000, f"耗时过长: {ms:.0f} ms"
    return ms


def test_mono():
    p = {"on": True, "target_lufs": -16.0, "lra": 6.0, "ceiling": -1.0}
    x = synth_music(dur=12.0, stereo=False, seed=21)
    ctx = ctx_new()
    out = m6.process(x, SR, p, ctx)
    assert out.shape == x.shape and np.isfinite(out).all()
    lo = measure.lufs_integrated(out, SR)
    tp = measure.true_peak_db(out, SR)
    print(f"[mono] LUFS={lo:.2f} TP={tp:.2f} stats={ctx['stats']['m6']}")
    assert abs(lo - (-16.0)) < 0.5
    assert tp <= -0.9


def test_at_target_quiet():
    """已在目标响度、动态小的信号：归一增益应接近 0，输出与输入接近。"""
    n = SR * 20
    t = np.arange(n) / SR
    rng = np.random.default_rng(9)
    x = (0.1 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.sin(2 * np.pi * 330 * t)
         + 0.005 * rng.standard_normal(n)).astype(np.float32)[:, None]
    g = 10 ** ((-16.0 - measure.lufs_integrated(x, SR)) / 20.0)
    x = (x * g).astype(np.float32)
    before = measure.lufs_integrated(x, SR)
    ctx = ctx_new()
    out = m6.process(x, SR, {"on": True, "target_lufs": -16.0, "lra": 6.0, "ceiling": -1.0}, ctx)
    after = measure.lufs_integrated(out, SR)
    rms_in = measure.rms_db(x)
    rms_out = measure.rms_db(out)
    print(f"[quiet] LUFS {before:.2f}->{after:.2f}, RMS {rms_in:.2f}->{rms_out:.2f}")
    assert abs(after - (-16.0)) < 0.5
    assert abs(rms_out - rms_in) < 1.0


def main():
    for fn in (test_rolling_min, test_env_oversampled, test_fast_path_agreement,
               test_silence, test_at_target_quiet, test_mono):
        fn()
    ms = test_stereo_main()
    print(f"\n全部通过。主用例（60s@48k 立体声）耗时 {ms/1000:.2f} s")


if __name__ == "__main__":
    main()
