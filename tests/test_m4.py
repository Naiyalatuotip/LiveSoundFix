"""m4 声场扩展自测：相关性下降 / 恒等 / 单声道 / 旋转 / 分频语义 / 性能。

运行：python tests/test_m4.py（工作目录 E:\\作品集\\音频修复v1）
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from livefix import core, measure, m4

sr = 48000
rng = np.random.default_rng(7)
FAIL = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def new_ctx():
    return {"rng": np.random.default_rng(0), "stats": {}, "log": [], "stems": None, "note": {}}


def rms_db(v):
    v = np.asarray(v, np.float64)
    return 10.0 * np.log10(max(np.mean(v * v), 1e-16))


# ---------------------------------------------------------------- 1) 相关性下降
# 模拟相机单点录音：L = R + 麦克风自噪声（宽带，方差比 r≈0.45）
n = 20 * sr
common = rng.standard_normal(n) * 0.2
# diff 用中频带通噪声（模拟麦克风中频自噪声），使 width_mid=1.6 主导相关性下降；
# 幅度取到 width_in≈0.28 量级——加宽系数 1.6 时 corr 才能显著下降（corr_in 仍 >0.85 强相关）
_bb, _aa = core.biquad_coeffs("bandpass", sr, 450.0, q=0.7)
diff = core.iir_filt((rng.standard_normal(n) * 0.58).astype(np.float32)[:, None], _bb, _aa)[:, 0]
stereo = np.stack([common + diff, common], 1).astype(np.float32)

p_wide = {"on": True, "width_low": 0.10, "width_mid": 1.6, "width_high": 1.50, "depth": 0.4, "rotation": 0.0}
ctx = new_ctx()
t0 = time.perf_counter()
y = m4.process(stereo, sr, p_wide, ctx)
dt = time.perf_counter() - t0
c_in, c_out = measure.corr(stereo), measure.corr(y)
check("20s stereo corr 下降 >0.15", (c_in - c_out) > 0.15,
      f"corr {c_in:.3f} -> {c_out:.3f} (drop {c_in - c_out:.3f}), {dt:.2f}s")
check("corr_in 仍强相关 >0.75", c_in > 0.75, f"{c_in:.3f}")
check("形状/类型不变", y.shape == stereo.shape and y.dtype == np.float32, f"{y.shape} {y.dtype}")
check("无 NaN/Inf", bool(np.all(np.isfinite(y))))
st = ctx["stats"]["m4"]
check("stats m4 宽度上升", st["width_out"] > st["width_in"],
      f"in={st['width_in']:.4f} out={st['width_out']:.4f}")

# ---------------------------------------------------------------- 2) 恒等：全 1.0 / 0
p_id = {"on": True, "width_low": 1.0, "width_mid": 1.0, "width_high": 1.0, "depth": 0.0, "rotation": 0.0}
y_id = m4.process(stereo, sr, p_id, new_ctx())
dr = abs(rms_db(y_id) - rms_db(stereo))
check("恒等 RMS 差 <0.05 dB", dr < 0.05, f"{dr:.6f} dB")
check("恒等 corr 几乎不变", abs(measure.corr(y_id) - c_in) < 0.01,
      f"{c_in:.4f} -> {measure.corr(y_id):.4f}")
check("恒等样本级误差 <1e-4", float(np.abs(y_id - stereo).max()) < 1e-4,
      f"max err {float(np.abs(y_id - stereo).max()):.2e}")

# on=False 透传
y_off = m4.process(stereo, sr, {"on": False}, new_ctx())
check("on=False 原样透传", float(np.abs(y_off - stereo).max()) == 0.0)

# ---------------------------------------------------------------- 3) 单声道
mono = (rng.standard_normal(n) * 0.2).astype(np.float32)
ctxm = new_ctx()
ym = m4.process(mono, sr, {"on": True, "width_low": 0.1, "width_mid": 1.6, "width_high": 1.5, "depth": 0.5, "rotation": 20.0}, ctxm)
check("mono 返回 mono 不崩溃", ym.shape == (n, 1) and ym.dtype == np.float32, f"{ym.shape}")
check("mono 无 NaN", bool(np.all(np.isfinite(ym))))
check("mono log 有提示", any("单声道" in s for s in ctxm["log"]))
check("mono depth>0 有变化", float(np.abs(ym[:, 0] - mono).max()) > 1e-6)
ym0 = m4.process(mono, sr, {"on": True, "width_low": 1.0, "width_mid": 1.0, "width_high": 1.0, "depth": 0.0, "rotation": 0.0}, new_ctx())
check("mono depth=0 恒等", float(np.abs(ym0[:, 0] - mono).max()) < 1e-4)

# ---------------------------------------------------------------- 4) rotation 定义
p_rot45 = {"on": True, "width_low": 1.0, "width_mid": 1.0, "width_high": 1.0, "depth": 0.0, "rotation": 45.0}
yr = m4.process(stereo, sr, p_rot45, new_ctx())
L, R = stereo[:, 0].astype(np.float64), stereo[:, 1].astype(np.float64)
check("rot=+45: L'=(L-R)/sqrt2", float(np.abs(yr[:, 0] - (L - R) / np.sqrt(2)).max()) < 1e-3)
check("rot=+45: R'=(L+R)/sqrt2", float(np.abs(yr[:, 1] - (L + R) / np.sqrt(2)).max()) < 1e-3)
yrn = m4.process(stereo, sr, {**p_rot45, "rotation": -45.0}, new_ctx())
check("rot=-45: L'=(L+R)/sqrt2", float(np.abs(yrn[:, 0] - (L + R) / np.sqrt(2)).max()) < 1e-3)

# ---------------------------------------------------------------- 5) 分频语义
# 注意：depth/宽度之后有「(M,S) 联合功率归一」，所以侧信号的**绝对**电平不再等于
# 宽度系数（电平中性是有意设计）。可验证的量是 侧/中 比 —— 它才代表「宽了多少」。
def band_sm(sig, out, f_lo, f_hi):
    """输出在 [f_lo,f_hi] 的 侧/中 RMS 比。"""
    s_out = (0.5 * (out[:, 0].astype(np.float64) - out[:, 1])).astype(np.float32)
    m_out = (0.5 * (out[:, 0].astype(np.float64) + out[:, 1])).astype(np.float32)
    m = core.band_mask(sr, f_lo, f_hi)

    def band_rms(v):
        V = core.stft(v, sr)
        return float(np.sqrt(np.mean((np.abs(V) ** 2 * m).sum(axis=-1))))

    return band_rms(s_out) / max(band_rms(m_out), 1e-12)


t = np.arange(8 * sr) / sr
car = (0.4 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)   # 中置载体（M）
side_lo = (0.3 * np.sin(2 * np.pi * 60 * t)).astype(np.float32)  # 低频侧信号
st_lo = np.stack([car + side_lo, car - side_lo], 1).astype(np.float32)
p_lo = {"on": True, "width_low": 0.10, "width_mid": 1.2, "width_high": 1.5,
        "depth": 0.0, "rotation": 0.0}
p_id = {"on": True, "width_low": 1.0, "width_mid": 1.0, "width_high": 1.0,
        "depth": 0.0, "rotation": 0.0}
sm_lo = band_sm(st_lo, m4.process(st_lo, sr, p_lo, new_ctx()), 40, 90)
sm_lo_id = band_sm(st_lo, m4.process(st_lo, sr, p_id, new_ctx()), 40, 90)
check("width_low=0.10: <120Hz 侧/中比 ≈0.1x", 0.07 < sm_lo / sm_lo_id < 0.13,
      f"ratio {sm_lo / sm_lo_id:.3f}")

side_hi = (0.3 * np.sin(2 * np.pi * 8000 * t)).astype(np.float32)
st_hi = np.stack([car + side_hi, car - side_hi], 1).astype(np.float32)
sm_hi = band_sm(st_hi, m4.process(st_hi, sr, p_lo, new_ctx()), 6000, 12000)
sm_hi_id = band_sm(st_hi, m4.process(st_hi, sr, p_id, new_ctx()), 6000, 12000)
check("width_high=1.5: >1.2kHz 侧/中比 ≈1.5x", 1.40 < sm_hi / sm_hi_id < 1.60,
      f"ratio {sm_hi / sm_hi_id:.3f}")

# ---------------------------------------------------------------- 6) depth 纵深
# 设计（2026-10-14 重做）：把 M 的全通去相关分量注入 S，再做 (M,S) 联合功率归一。
# 旧版是对 S 自身加反射后把 S 的 RMS 归一回原值 —— 归一让 S/M 比恒不变，
# 实测 depth 0→1 只改动 0.03dB，参数完全无效。故旧断言「S/M 比近似不变」已废弃。
# (a) 单声道路径仍走低通梳状反射：冲击响应应在 13/17/23/31ms 出现
imp = np.zeros(2 * sr, np.float32)
imp[1000] = 1.0
ym_imp = m4.process(imp[:, None], sr,
                    {"on": True, "width_low": 1.0, "width_mid": 1.0, "width_high": 1.0,
                     "depth": 1.0, "rotation": 0.0}, new_ctx())
s_m = ym_imp[:, 0].astype(np.float64)
refl_m = s_m.copy()
refl_m[960:1041] = 0.0
rmax_m = float(np.abs(refl_m).max())
tap_ok_m = []
for ds in (0.013, 0.017, 0.023, 0.031):
    k = 1000 + int(round(ds * sr))
    tap_ok_m.append(float(np.abs(s_m[k - 40: k + 40]).max()) > 0.2 * rmax_m)
check("mono depth 抽头 13/17/23/31ms 均出现反射", all(tap_ok_m), f"rmax={rmax_m:.4f}")

# (b) 全通去相关器：幅度应平坦（只改相位、不染色），100Hz-10kHz 内起伏 < 3dB
_noise = (rng.standard_normal(4 * sr) * 0.3).astype(np.float32)
_dec = m4._decorrelate(_noise, sr)
_h1 = np.abs(np.fft.rfft(_noise * np.hanning(len(_noise))))
_h2 = np.abs(np.fft.rfft(_dec * np.hanning(len(_dec))))
_fr = np.fft.rfftfreq(len(_noise), 1.0 / sr)
_sel = (_fr > 100) & (_fr < 10000)
_ratio_db = 20 * np.log10((_h2[_sel] + 1e-12) / (_h1[_sel] + 1e-12))
check("depth 全通去相关器幅度平坦（100Hz-10kHz 起伏 <3dB）",
      float(np.percentile(_ratio_db, 95) - np.percentile(_ratio_db, 5)) < 3.0,
      f"起伏 {float(np.percentile(_ratio_db,95)-np.percentile(_ratio_db,5)):.2f} dB")

# (c) depth 权威度：拉满必须真的把侧信号抬起来（用户要求每个效果器 ≥±3dB）
s_lo2 = rng.standard_normal(n // 4).astype(np.float32)
b, a = core.biquad_coeffs("lp", sr, 1500.0, q=0.707)
s_lo2 = core.iir_filt(s_lo2[:, None], b, a)[:, 0] * 0.3
c2 = (rng.standard_normal(n // 4) * 0.3).astype(np.float32)
st_d = np.stack([c2 + s_lo2, c2 - s_lo2], 1).astype(np.float32)
p_dw = {"on": True, "width_low": 1.0, "width_mid": 1.0, "width_high": 1.0,
        "depth": 0.8, "rotation": 0.0}
ctxd = new_ctx()
yd = m4.process(st_d, sr, p_dw, ctxd)
yd0 = m4.process(st_d, sr, {**p_dw, "depth": 0.0}, new_ctx())
std_ = ctxd["stats"]["m4"]
_sw = std_["width_out"] / max(std_["width_in"], 1e-9)
check("depth 反射确实改变信号", float(np.abs(yd - yd0).max()) > 1e-4)
check("depth 权威度：S/M 比提升 ≥30%（旧版恒不变）", _sw >= 1.30,
      f"in={std_['width_in']:.3f} out={std_['width_out']:.3f} x{_sw:.2f}")
check("depth 电平中性（联合功率归一，RMS 差 <1dB）", abs(rms_db(yd) - rms_db(yd0)) < 1.0,
      f"{rms_db(yd) - rms_db(yd0):.3f} dB")
check("depth 上报 mono_fold_db", "mono_fold_db" in std_,
      f"mono_fold={std_.get('mono_fold_db'):.2f} dB")
check("depth 输出无 NaN 且长度不变", bool(np.all(np.isfinite(yd))) and yd.shape == st_d.shape)

# ---------------------------------------------------------------- 7) 边界
ys = m4.process(stereo[:512], sr, p_wide, new_ctx())
check("超短输入(512采样)不崩溃", ys.shape == (512, 2) and bool(np.all(np.isfinite(ys))))
y3 = m4.process(np.concatenate([stereo, np.zeros((n, 1), np.float32)], 1), sr, p_wide, new_ctx())
check("3 声道：前两声道处理、第3声道直通",
      y3.shape == (n, 3) and float(np.abs(y3[:, 2]).max()) == 0.0 and bool(np.all(np.isfinite(y3))))
check("info() 形状", m4.info()["key"] == "m4" and "M/S" in m4.info()["algo"], str(m4.info()))

# ---------------------------------------------------------------- 8) 性能：60s@48k 立体声
n60 = 60 * sr
c60 = (rng.standard_normal(n60) * 0.2).astype(np.float32)
d60 = (rng.standard_normal(n60) * 0.13).astype(np.float32)
st60 = np.stack([c60 + d60, c60], 1).astype(np.float32)
p_def = {"on": True, "width_low": 0.10, "width_mid": 1.20, "width_high": 1.50, "depth": 0.4, "rotation": 0.0}
t0 = time.perf_counter()
y60 = m4.process(st60, sr, p_def, new_ctx())
dt60 = time.perf_counter() - t0
check("60s@48k 立体声耗时 <5s", dt60 < 5.0, f"{dt60:.2f}s")
check("60s 输出无 NaN", bool(np.all(np.isfinite(y60))))

print("-" * 60)
if FAIL:
    print("M4 TEST FAILED:", ", ".join(FAIL))
    sys.exit(1)
print("M4 TEST OK")
