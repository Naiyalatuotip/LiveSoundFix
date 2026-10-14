"""m5 修复层自测：削波重建 / 市电陷波 / 场地去混响抑共振 / 泵动平滑 / 恒等 / 单声道 / 性能。

合成信号（spec 配方）：1kHz 正弦 + 白噪声（SNR 10dB）+ 50Hz hum；削波测试额外
做「clip ±0.2 再 ×5」。dereverb 另用专用信号：每 1s 一个宽带瞬态 + 指数衰减混响尾巴
+ 180Hz 持续共振峰，用来分别量「共振峰被削」与「尾巴被压、干声基本不动、带外无副作用」。
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


def synth_reverb(dur=8.0, tail_s=0.45, res_hz=180.0, res_tau_s=0.25):
    """场地混响素材：每 1s 一个宽带瞬态 + 指数衰减尾巴 + **被瞬态激起后振铃**的共振峰。

    共振峰必须「振铃」而不是持续存在 —— 房间模态是被激励后才响的，这既是物理事实，
    也是 dereverb 区分「模态」与「持续乐音」的依据（持续音走 held 用例验证不被削）。
    """
    n = int(dur * sr)
    x = np.zeros(n)
    onsets = []
    for k in np.arange(0.3, dur - tail_s - 0.1, 1.0):
        i0 = int(k * sr)
        onsets.append(i0)
        L = int(0.012 * sr)                             # 干声：12ms 宽带点击
        x[i0:i0 + L] += rng.standard_normal(L) * 0.80
        Lt = int(tail_s * sr)                           # 尾巴：指数衰减噪声（τ=100ms）
        tt = np.arange(Lt) / sr
        x[i0:i0 + Lt] += rng.standard_normal(Lt) * np.exp(-tt / 0.10) * 0.45
        # 房间模态振铃：180Hz 正弦，τ=250ms（比尾巴长 → 拖尾段里它最突出）
        x[i0:i0 + Lt] += 0.34 * np.sin(2 * np.pi * res_hz * tt) * np.exp(-tt / res_tau_s)
    return np.stack([x, x * 0.98], axis=1).astype(np.float32), onsets


def synth_held(dur=8.0, res_hz=180.0):
    """稳态素材：持续 180Hz 乐音（长音/贝斯）+ 稳定宽带底，用来验证「不削持续音」。"""
    t = np.arange(int(dur * sr)) / sr
    x = (0.34 * np.sin(2 * np.pi * res_hz * t) + 0.05 * rng.standard_normal(t.shape)
         + 0.30 * np.sin(2 * np.pi * 1000.0 * t))
    return np.stack([x, x * 0.98], axis=1).astype(np.float32)


def params(**kw):
    base = {"on": True, "dereverb": 0.0,
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

# ---------------------------------------------------------------- 2) dereverb：压拖尾 + 抑共振
def band_power_db(sig, f_lo, f_hi):
    """整段带内**聚合**能量 dB（先功率求和再取 dB）。

    不能用逐帧 dB 求平均 —— 帧间静音处的 dB 是数值噪声，会把结果带偏。
    """
    Xs = core.stft(sig, sr)
    sel = (core.rfftfreqs(sr) >= f_lo) & (core.rfftfreqs(sr) <= f_hi)
    return 10.0 * np.log10(float(np.mean((np.abs(Xs) ** 2)[:, :, sel])) + 1e-30)


x_rv, onsets = synth_reverb()
y_rv0, _ = run(x_rv, params())                          # dereverb=0 必须逐样本恒等
assert np.array_equal(y_rv0, x_rv), "dereverb=0 应严格恒等"
y_rv, ms = run(x_rv, params(dereverb=1.0))

freqs = core.rfftfreqs(sr)
kres = int(np.argmin(np.abs(freqs - 180.0)))


def seg_bin_db(sig, t0_s, t1_s, k):
    """瞬态后某时间窗内、某条 bin 的聚合能量 dB（跨所有瞬态平均）。"""
    tot = []
    for i0 in onsets:
        a, b = i0 + int(t0_s * sr), i0 + int(t1_s * sr)
        if b > sig.shape[0]:
            continue
        Xs = core.stft(sig[a:b], sr)
        tot.append(float(np.mean((np.abs(Xs) ** 2)[:, :, k])))
    return 10.0 * np.log10(float(np.mean(tot)) + 1e-30)


def seg_band_db(sig, t0_s, t1_s):
    tot = []
    sel = (freqs >= 70.0) & (freqs <= 600.0)
    for i0 in onsets:
        a, b = i0 + int(t0_s * sr), i0 + int(t1_s * sr)
        if b > sig.shape[0]:
            continue
        Xs = core.stft(sig[a:b], sr)
        tot.append(float(np.mean((np.abs(Xs) ** 2)[:, :, sel])))
    return 10.0 * np.log10(float(np.mean(tot)) + 1e-30)


# 2a) 共振峰：拖尾段里 180Hz 那条 bin 的下降（权威度要求 ≥3dB）
res_b = seg_bin_db(x_rv, 0.05, 0.40, kres)
res_a = seg_bin_db(y_rv, 0.05, 0.40, kres)
d_res = res_a - res_b
# 2b) 拖尾整体：120-400ms 段 vs 0-12ms 干声段
tail_b = seg_band_db(x_rv, 0.12, 0.40)
tail_a = seg_band_db(y_rv, 0.12, 0.40)
dry_b = seg_band_db(x_rv, 0.0, 0.012)
dry_a = seg_band_db(y_rv, 0.0, 0.012)
# 2c) 带外（3-8kHz）不应有副作用
d_out = band_power_db(y_rv, 3000, 8000) - band_power_db(x_rv, 3000, 8000)
print("[dereverb] 180Hz 共振峰 %.2f dB | 拖尾 %.2f dB | 干声 %.2f dB"
      " | 尾-干 %.2f -> %.2f dB | 3-8k %.2f dB (%.1f ms)"
      % (d_res, tail_a - tail_b, dry_a - dry_b,
         tail_b - dry_b, tail_a - dry_a, d_out, ms))
assert d_res < -3.0, f"共振峰衰减不足 3dB：{d_res:.2f}"
assert tail_a - tail_b < -3.0, f"拖尾衰减不足 3dB：{tail_a - tail_b:.2f}"
assert (tail_a - dry_a) < (tail_b - dry_b) - 2.0, "尾/干比未改善"
assert abs(dry_a - dry_b) < 1.5, f"干声（攻击）被动到：{dry_a - dry_b:.2f}"
assert abs(d_out) < 0.3, f"带外不应有副作用：{d_out:.2f}"

# 2d) 稳态乐音保护：持续 180Hz 长音 + 稳定底噪 → 基本不动（防止削到乐音基频）
x_held = synth_held()
y_held, _ = run(x_held, params(dereverb=1.0))
d_held = band_power_db(y_held, 70, 600) - band_power_db(x_held, 70, 600)
d_held1k = band_power_db(y_held, 800, 1200) - band_power_db(x_held, 800, 1200)
print("[dereverb/held] 持续长音带内 %.2f dB | 1kHz %.2f dB（应≈0，证明不削持续音）"
      % (d_held, d_held1k))
assert abs(d_held) < 1.0, f"持续乐音被削了 {d_held:.2f} dB"
assert abs(d_held1k) < 0.3, d_held1k

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
y, ms = run(x_cl[:, :1], params(dereverb=0.5, hum=0.6, declip=0.6, pump=0.4))
ctx = {"stats": {}, "log": []}
y, ms = run(x_cl, params(dereverb=0.5, hum=0.6, declip=0.6, pump=0.4), ctx=ctx)
print("[chain/stereo] mono+stereo ok, ctx stats =", ctx["stats"]["m5"],
      "log =", ctx["log"][-1], "(%.1f ms)" % ms)
y, ms = run(x_nc, params(dereverb=0.5, hum=0.6, declip=0.6, pump=0.4), ctx=None)
print("[chain/ctx=None] ok (%.1f ms)" % ms)

# ---------------------------------------------------------------- 7) 性能 60s@48k 立体声
x60 = synth(60.0, clip=True)
t0 = time.perf_counter()
y60 = m5.process(x60, sr, params(dereverb=0.5, hum=0.6, declip=0.6, pump=0.4),
                 {"stats": {}, "log": []})
elapsed = time.perf_counter() - t0
assert y60.shape == x60.shape and np.isfinite(y60).all()
print("[perf] 60s@48k stereo full-chain = %.2f s" % elapsed)
assert elapsed < 5.0, elapsed

print("M5 OK")
