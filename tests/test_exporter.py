"""exporter 自测：声道映射 / 5.1 上混结构 / 位深往返 / 响度归一 / 峰值安全 / 计时。

信号配方：20s 立体声 = 1kHz 正弦 + 白噪声 + 80Hz 低频 + 高频（L 7k / R 5.5k）。
所有断言以「读回文件后重新测量」为准。
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from livefix import exporter
from livefix import measure
from livefix import wavio

sr = 48000
rng = np.random.default_rng(11)
TMP = tempfile.mkdtemp(prefix="livefix_export_")
timings = {}


def timed(tag, fn):
    t0 = time.perf_counter()
    out = fn()
    timings[tag] = time.perf_counter() - t0
    return out


def synth(dur=20.0):
    t = np.arange(int(dur * sr)) / sr
    l = (0.45 * np.sin(2 * np.pi * 1000.0 * t) + 0.10 * np.sin(2 * np.pi * 80.0 * t)
         + 0.06 * np.sin(2 * np.pi * 7000.0 * t) + 0.03 * rng.standard_normal(t.shape))
    r = (0.42 * np.sin(2 * np.pi * 1000.0 * t + 0.3) + 0.10 * np.sin(2 * np.pi * 80.0 * t)
         + 0.05 * np.sin(2 * np.pi * 5500.0 * t) + 0.03 * rng.standard_normal(t.shape))
    return np.stack([l, r], axis=1).astype(np.float32)


x = synth()
print(f"信号：{x.shape[0]/sr:.1f}s @ {sr}Hz  shape={x.shape}  dtype={x.dtype}")

# ------------------------------------------------ 1) 声道数：mono / stereo / 5.1
print("\n[1] 声道映射导出（读回验证）")
for tgt, chn in (("mono", 1), ("stereo", 2), ("5.1", 6)):
    out = timed("export_" + tgt, lambda tgt=tgt: exporter.export(
        x, sr, os.path.join(TMP, f"ch_{tgt.replace('.', '')}.wav"),
        channels=tgt, bit_depth=24))
    y, sr2 = wavio.read_wav(out["path"])
    assert y.shape[1] == chn, (tgt, y.shape)
    assert abs(y.shape[0] / sr2 - 20.0) < 0.01
    assert np.all(np.isfinite(y))
    assert out["channels"] == chn
    print(f"  {tgt:6s} -> {y.shape[1]}ch  {y.shape[0]} 采样  "
          f"lufs={out['lufs']:.2f}  tp={out['true_peak_db']:.2f}")

# ------------------------------------------------ 2) to_channels 语义
print("\n[2] to_channels 语义")
xm = exporter.to_channels(x, sr, "mono")
assert np.abs(xm[:, 0] - x.astype(np.float64).mean(axis=1)).max() < 1e-6
xs = exporter.to_channels(x, sr, "stereo")
assert np.abs(xs - x).max() == 0.0
x1 = exporter.to_channels(x[:, :1], sr, "stereo")
assert x1.shape[1] == 2 and np.abs(x1[:, 0] - x1[:, 1]).max() == 0.0
print("  mono=均值 ok / stereo=直通 ok / 单声道复制 ok")

# ------------------------------------------------ 3) 5.1 上混结构
print("\n[3] 5.1 上混结构")
x5 = exporter.to_channels(x, sr, "5.1")
assert x5.shape == (x.shape[0], 6) and np.all(np.isfinite(x5))
mid = 0.5 * (x[:, 0].astype(np.float64) + x[:, 1].astype(np.float64))
assert np.abs(x5[:, 2] - 0.707 * mid).max() < 1e-4                      # C = 0.707*mid
assert np.abs(x5[:, 0] - (x[:, 0].astype(np.float64) - 0.5 * mid)).max() < 1e-4
assert np.abs(x5[:, 1] - (x[:, 1].astype(np.float64) - 0.5 * mid)).max() < 1e-4
rms = lambda v: float(np.sqrt(np.mean(np.asarray(v, np.float64) ** 2)))


def bin_amp(v, f):
    seg = np.asarray(v, np.float64)[: int(2 * sr)]
    sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    return float(sp[int(round(f * len(seg) / sr))])


r1k = bin_amp(x5[:, 3], 1000.0) / max(bin_amp(mid, 1000.0), 1e-12)
r80 = bin_amp(x5[:, 3], 80.0) / max(bin_amp(mid, 80.0), 1e-12)
assert r1k < 0.05, r1k                                                 # 1kHz 被 LP120 压 >26dB
assert 0.3 < r80 < 0.75, r80                                           # 80Hz 保留（0.5 增益）
assert rms(x5[:, 4]) > 1e-4 and rms(x5[:, 5]) > 1e-4                   # 环绕非空
corr_ls_rs = float(np.corrcoef(x5[:, 4], x5[:, 5])[0, 1])
raw_side = 0.5 * (x[:, 0].astype(np.float64) - x[:, 1].astype(np.float64))
corr_raw = float(np.corrcoef(raw_side[:-1], raw_side[1:].astype(np.float64))[0, 1])
assert corr_ls_rs < corr_raw, (corr_ls_rs, corr_raw)                   # 微延迟去相关生效
xm1 = exporter.to_channels(x[:, :1], sr, "5.1")
mono = x[:, 0].astype(np.float64)
assert np.abs(xm1[:, 2] - mono).max() < 1e-6                           # C = mid 主导
assert np.abs(xm1[:, 0] - mono).max() < 1e-6 and np.abs(xm1[:, 1] - mono).max() < 1e-6
assert np.abs(xm1[:, 4]).max() < 1e-9 and np.abs(xm1[:, 5]).max() < 1e-9  # 无 side
print(f"  LFE: 1kHz 衰减 x{r1k:.4f} / 80Hz 保留 x{r80:.3f}  "
      f"corr(Ls,Rs)={corr_ls_rs:.3f} < corr(raw side)={corr_raw:.3f}")
print("  单声道源：C=mid 主导 / L=R 复制 / Ls=Rs=0  ok")

# ------------------------------------------------ 4) 响度归一（读回文件测量）
print("\n[4] loudnorm=True -> -16 LUFS / 真峰 <= -0.9")
for tgt, bd in (("stereo", 24), ("5.1", 24), ("mono", 32)):
    out = timed("loudnorm_" + tgt, lambda tgt=tgt, bd=bd: exporter.export(
        x, sr, os.path.join(TMP, f"ln_{tgt.replace('.', '')}.wav"),
        channels=tgt, bit_depth=bd, loudnorm=True))
    y, _ = wavio.read_wav(out["path"])
    lu = measure.lufs_integrated(y, sr)
    tp = measure.true_peak_db(y, sr)
    assert np.all(np.isfinite(y))
    assert abs(lu - (-16.0)) < 0.6, (tgt, lu)
    assert tp <= -0.9, (tgt, tp)
    print(f"  {tgt:6s} bd{bd}: lufs={lu:.3f} (Δ={lu+16.0:+.3f} LU)  tp={tp:.3f} dBTP")

# ------------------------------------------------ 5) 峰值安全（非归一）
print("\n[5] loudnorm=False + 输入峰值 +3dB -> 输出 tp <= -0.4")
hot = (x * float(10 ** (3.0 / 20.0))).astype(np.float32)
print(f"  输入 tp={measure.true_peak_db(hot, sr):+.2f} dBTP")
out = timed("peak_safe", lambda: exporter.export(
    hot, sr, os.path.join(TMP, "hot.wav"), channels="stereo", bit_depth=24))
y, _ = wavio.read_wav(out["path"])
tp = measure.true_peak_db(y, sr)
assert tp <= -0.4, tp
print(f"  输出 tp={tp:.3f} dBTP  (纯等比缩放，无频谱改变)")

# ------------------------------------------------ 6) 位深往返
print("\n[6] 位深往返误差")
seg = (x[: int(3 * sr)] * 0.2).astype(np.float32)  # 峰值约 -18dB，不触发任何缩放
for bd, tol in ((16, 1e-3), (24, 2e-6), (32, 1e-6)):
    out = timed(f"bd{bd}", lambda bd=bd: exporter.export(
        seg, sr, os.path.join(TMP, f"bd{bd}.wav"), bit_depth=bd, loudnorm=False))
    y, _ = wavio.read_wav(out["path"])
    err = float(np.abs(y.astype(np.float64) - seg.astype(np.float64)).max())
    assert err < tol, (bd, err)
    print(f"  {bd:2d}bit: max_err={err:.3e}  (< {tol:g})")

# ------------------------------------------------ 7) 杂项：补后缀 / 参数校验 / NaN
print("\n[7] 补 .wav 后缀 / 参数校验")
out = timed("suffix", lambda: exporter.export(
    x[:1000], sr, os.path.join(TMP, "no_suffix"), bit_depth=32))
assert out["path"].endswith("no_suffix.wav") and os.path.exists(out["path"])
for bad in (lambda: exporter.export(x, sr, os.path.join(TMP, "b1.wav"), channels="7.4"),
            lambda: exporter.export(x, sr, os.path.join(TMP, "b2.wav"), bit_depth=8)):
    try:
        bad()
        raise AssertionError("应当抛出 ValueError")
    except ValueError:
        pass
print("  无后缀自动补 .wav ok；非法 channels/bit_depth 抛 ValueError ok")

print("\n耗时：")
for k, v in timings.items():
    print(f"  {k:16s} {v*1000:8.1f} ms")
print(f"  合计             {sum(timings.values())*1000:8.1f} ms")
print("\nALL EXPORTER TESTS PASSED")
