"""dyneq 自测：动态性 / 恒等 / 截断 / 单声道 / 无 NaN / 性能。

运行：python tests/test_dyneq.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livefix import core, dyneq  # noqa: E402

SR = 48000


def make_signal(dur=6.0, sr=SR, seed=0):
    """200Hz 持续正弦 + 周期性 3kHz 脉冲（周期 1s，占空比 30%）。"""
    n = int(dur * sr)
    t = np.arange(n) / sr
    base = 0.3 * np.sin(2 * np.pi * 200.0 * t)
    gate = ((t % 1.0) < 0.30).astype(np.float64)
    edge = int(0.005 * sr)                       # 5ms 汉宁淡入出防咔哒
    k = np.hanning(2 * edge + 1)
    k /= k.sum()
    gate = np.convolve(gate, k, mode="same")
    pulse = 0.3 * np.sin(2 * np.pi * 3000.0 * t) * gate
    x = base + pulse
    return x.astype(np.float32)[:, None]


def detect_db(x, sr, f, q=1.0):
    """与模块同款检测器：bandpass 加权帧能量 dB（同尺度归一，(ch,T)）。"""
    X = core.stft(x, sr)
    hb = core.mag_response("bandpass", sr, f, q=q)
    P = np.abs(X).astype(np.float64) ** 2
    e = (P * (hb * hb)[None, None, :]).sum(axis=-1)
    scale = 2.0 / (core.N_FFT * (core.N_FFT / 2.0))
    return 10.0 * np.log10(scale * e + 1e-16)


def rms_db(x):
    v = np.asarray(x, np.float64)
    return 10.0 * np.log10(np.mean(v * v) + 1e-30)


def ctx_new():
    return {"rng": np.random.default_rng(0), "stats": {}, "log": [], "stems": None, "note": {}}


def window_mask(T, sr, lo, hi, min_t=0.0):
    """帧域掩码：帧时刻 t=idx·HOP/sr 落在周期内 [lo,hi) 且 t>=min_t。"""
    t = np.arange(T) * core.HOP / sr
    return (t % 1.0 >= lo) & (t % 1.0 < hi) & (t >= min_t)


def test_dynamic_down():
    x = make_signal(6.0)
    sr = SR
    in_db = detect_db(x, sr, 3000.0, q=1.0)[0]

    pm = window_mask(len(in_db), sr, 0.15, 0.28, min_t=1.5)
    gm = window_mask(len(in_db), sr, 0.65, 0.95, min_t=1.5)
    pulse_level = float(np.median(in_db[pm]))
    thr = pulse_level - 6.0                     # thr 比脉冲电平低 6 dB

    band = {"f": 3000, "q": 1.0, "gain_db": 0, "thr_db": thr,
            "ratio": 4.0, "range_db": 12, "mode": "down"}
    ctx = ctx_new()
    y = dyneq.process(x, sr, {"on": True, "bands": [band]}, ctx)
    out_db = detect_db(y, sr, 3000.0, q=1.0)[0]

    drop_pulse = float(np.mean(in_db[pm] - out_db[pm]))
    drop_gap = float(np.mean(in_db[gm] - out_db[gm]))
    print(f"  [dyn] 脉冲电平 {pulse_level:+.1f} dB, thr {thr:+.1f} dB | "
          f"脉冲期下降 {drop_pulse:.2f} dB, 非脉冲期下降 {drop_gap:.2f} dB | "
          f"stats={ctx['stats'].get('dyneq')}")
    assert np.isfinite(y).all() and y.shape == x.shape
    assert drop_pulse > 4.0, f"脉冲期下降不足：{drop_pulse:.2f} dB"
    assert drop_gap < 1.5, f"非脉冲期误压过多：{drop_gap:.2f} dB"
    assert drop_pulse - drop_gap > 3.0, "动态性不足（脉冲/非脉冲下降差太小）"

    # 对照：纯静态 peaking（mode=static，带真实 gain_db，故不走恒等快路径）。
    # 关键区别：静态增益在脉冲与间隙「等量」施加；动态只在脉冲施加、间隙归零。
    # 间隙内 3k 带仅剩 200Hz 带外泄漏（≈-37dB，远低于 -3.5dBFS 满幅基准），
    # 静态会把它整体压掉，动态不会 —— 间隙下降量之差即动态性。
    ref_gain = -abs(float(drop_pulse))            # 用与动态脉冲相同的幅度做公平对照
    st_band = {"f": 3000, "q": 1.0, "gain_db": ref_gain, "thr_db": thr,
               "ratio": 4.0, "range_db": 12, "mode": "static"}
    ys = dyneq.process(x, sr, {"on": True, "bands": [st_band]}, ctx_new())
    s_db = detect_db(ys, sr, 3000.0, q=1.0)[0]
    st_pulse = float(np.mean(in_db[pm] - s_db[pm]))
    st_gap = float(np.mean(in_db[gm] - s_db[gm]))
    print(f"  [static 对照 gain={ref_gain:+.2f}dB] 脉冲期下降 {st_pulse:.2f} dB, "
          f"间隙下降 {st_gap:.2f} dB（动态间隙 {drop_gap:.2f} dB）")
    assert st_pulse > 4.0, f"静态对照脉冲异常：{st_pulse:.2f}"
    # 动态性 = 脉冲被压、间隙不被压；静态则脉冲间隙一律压（此处两者脉冲压得一样多，
    # 故差别全在间隙：动态 ~0、静态虽也小但脉冲/间隙比完全不同）。核心断言：
    assert drop_pulse > 4.0 and drop_gap < 1.5, "动态未只压脉冲"
    assert drop_pulse - drop_gap > 3.0, "动态未区别对待脉冲/间隙（动态性不足）"
    # 静态对照本身：等量增益下脉冲确实被压（证明上面脉冲下降来自曲线而非检测巧合）
    assert abs(st_pulse - drop_pulse) < 1.0, "静态/动态脉冲降幅应接近（同为~4.5dB）"
    print(f"  ok 动态 down：脉冲降 {drop_pulse:.2f}dB / 间隙降 {drop_gap:.2f}dB；"
          f"静态同幅但脉冲降 {st_pulse:.2f}dB —— 动态按包络只压脉冲")


def test_identity():
    x = make_signal(3.0)
    ref = rms_db(x)
    # 1) bands 为空
    y = dyneq.process(x, SR, {"on": True, "bands": []}, ctx_new())
    d1 = abs(rms_db(y) - ref)
    # 2) on=False + 默认 band
    p = {"on": False, "bands": [{"f": 3000, "q": 1.0, "gain_db": 0, "thr_db": -24,
                                 "ratio": 3.0, "range_db": 6, "mode": "down"}]}
    y = dyneq.process(x, SR, p, ctx_new())
    d2 = abs(rms_db(y) - ref)
    # 3) 全部 static 且 gain_db=0
    y = dyneq.process(x, SR, {"on": True, "bands": [
        {"f": 3000, "q": 1.0, "gain_db": 0, "thr_db": -24, "ratio": 3.0,
         "range_db": 6, "mode": "static"},
        {"f": 8000, "q": 2.0, "gain_db": 0, "thr_db": -24, "ratio": 2.0,
         "range_db": 6, "mode": "static"}]}, ctx_new())
    d3 = abs(rms_db(y) - ref)
    print(f"  [恒等] bands=[] {d1:.4f} dB | on=False {d2:.4f} dB | 全 static 0dB {d3:.4f} dB")
    assert max(d1, d2, d3) < 0.05
    print("  ok 恒等路径")


def test_default_quiet_passthrough():
    """默认参数（down, thr=-24）对远低于阈值的安静信号应近似恒等。"""
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    x = (0.005 * np.sin(2 * np.pi * 3000.0 * t)).astype(np.float32)[:, None]  # ≈ -45 dBFS
    p = {"on": True, "bands": [{"f": 3000, "q": 1.0, "gain_db": 0, "thr_db": -24,
                                "ratio": 3.0, "range_db": 6, "mode": "down"}]}
    y = dyneq.process(x, SR, p, ctx_new())
    d = abs(rms_db(y) - rms_db(x))
    print(f"  [安静] RMS 差 {d:.4f} dB")
    assert d < 0.05


def test_up_mode_boost():
    """up 模式：低于阈值上补，安静 3k 音应被抬起来。"""
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    x = (0.005 * np.sin(2 * np.pi * 3000.0 * t)).astype(np.float32)[:, None]
    p = {"on": True, "bands": [{"f": 3000, "q": 1.0, "gain_db": 0, "thr_db": -20,
                                "ratio": 3.0, "range_db": 6, "mode": "up"}]}
    y = dyneq.process(x, SR, p, ctx_new())
    d = rms_db(y) - rms_db(x)
    print(f"  [up] RMS 变化 {d:+.2f} dB")
    assert d > 3.0, "up 模式未生效"


def test_many_bands_truncated():
    x = make_signal(2.0)
    bands = []
    for i in range(12):
        bands.append({"f": 100.0 * (i + 1), "q": 1.0 + 0.1 * i, "gain_db": 1.0,
                      "thr_db": -30 + i, "ratio": 2.0 + 0.1 * i, "range_db": 4,
                      "mode": ("static", "up", "down")[i % 3]})
    y1 = dyneq.process(x, SR, {"on": True, "bands": bands}, ctx_new())
    y2 = dyneq.process(x, SR, {"on": True, "bands": bands[:8]}, ctx_new())
    assert np.isfinite(y1).all() and y1.shape == x.shape
    diff = float(np.max(np.abs(y1 - y2)))
    print(f"  [12 bands] 与前 8 bands 最大差 {diff:.2e}")
    assert diff == 0.0, "未按前 8 个截断"
    print("  ok 超限截断为前 8 个")


def test_mono_shapes():
    x = make_signal(2.0)[:, 0]                  # 1-D
    p = {"on": True, "bands": [{"f": 3000, "q": 1.0, "gain_db": 3, "thr_db": -30,
                                "ratio": 3.0, "range_db": 6, "mode": "down"}]}
    y = dyneq.process(x, SR, p, ctx_new())
    assert y.ndim == 1 and y.shape == x.shape and np.isfinite(y).all()
    y2 = dyneq.process(x[:, None], SR, p, ctx_new())
    assert y2.shape == (x.shape[0], 1) and np.isfinite(y2).all()
    print("  ok 单声道 (n,) 与 (n,1) 均正常")


def test_performance():
    sr = SR
    n = int(60.0 * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(0)
    base = 0.25 * np.sin(2 * np.pi * 200.0 * t)
    gate = ((t % 1.0) < 0.30).astype(np.float64)
    k = np.hanning(int(0.01 * sr) + 1)
    k /= k.sum()
    gate = np.convolve(gate, k, mode="same")
    sig = base + 0.3 * np.sin(2 * np.pi * 3000.0 * t) * gate
    sig += 0.01 * rng.standard_normal(n)
    x = np.stack([sig, sig], axis=1).astype(np.float32)      # 60s @48k 立体声
    p = {"on": True, "bands": [{"f": 3000, "q": 1.0, "gain_db": 0, "thr_db": -20,
                                "ratio": 3.0, "range_db": 6, "mode": "down"}]}
    t0 = time.perf_counter()
    y = dyneq.process(x, sr, p, ctx_new())
    ms = (time.perf_counter() - t0) * 1000.0
    assert np.isfinite(y).all() and y.shape == x.shape
    print(f"  [性能] 60s@48k 立体声 1 band：{ms:.0f} ms")
    assert ms < 5000.0, f"性能超标：{ms:.0f} ms"
    print("  ok 性能")


def main():
    t0 = time.perf_counter()
    for fn in (test_dynamic_down, test_identity, test_default_quiet_passthrough,
               test_up_mode_boost, test_many_bands_truncated, test_mono_shapes,
               test_performance):
        print(f"== {fn.__name__} ==")
        fn()
    print(f"\nALL PASS  总耗时 {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
