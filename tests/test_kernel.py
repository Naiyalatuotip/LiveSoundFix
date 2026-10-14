"""内核冒烟测试：STFT 重构 / 滤波 / LUFS / 真峰 / 分段调用。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from livefix import core, measure, wavio

sr = 48000
rng = np.random.default_rng(0)

# 1) STFT 精确重构
x = (rng.standard_normal((123456, 2)) * 0.1).astype(np.float32)
X = core.stft(x, sr)
y = core.istft(X, x.shape[0], sr)
err = float(np.abs(x - y).max())
print("stft roundtrip max err:", err)
assert err < 1e-4, err

# 2) mag_response：峰值滤波器在 f0 处有增益
resp = core.mag_response("peak", sr, 1000.0, q=1.0, gain_db=6.0)
k = int(np.argmax(resp))
print("peak@f0:", core.rfftfreqs(sr)[k], "gain dB:", 20 * np.log10(resp[k]))
assert abs(core.rfftfreqs(sr)[k] - 1000) < 50

# 3) block_apply 恒等
xb = x[: 60 * sr + 777]
yb = core.block_apply(xb, sr, lambda s, srr: s * 1.0, block_s=8.0, xfade_s=0.5)
print("block_apply err:", float(np.abs(xb - yb).max()))
assert float(np.abs(xb - yb).max()) < 1e-5

# 4) LUFS：正弦 -20dBFS 立体声 ≈ -20-3.01+~2.0(K加权对1k约+0.7?)，只要求可测且稳定
t = np.arange(6 * sr) / sr
s = (0.1 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
stereo = np.stack([s, s], axis=1)
lufs = measure.lufs_integrated(stereo, sr)
print("lufs 1kHz -20dB stereo:", round(lufs, 2))
assert -24 < lufs < -14

# 5) 真峰：限幅方波
sq = np.sign(np.sin(2 * np.pi * 200 * t)).astype(np.float32) * 0.999
print("true_peak square:", round(measure.true_peak_db(np.stack([sq, sq], 1), sr), 2))
print("peak sample:", round(measure.peak_db(np.stack([sq, sq], 1)), 2))

# 6) 频谱 / 相关 / LRA
print("spectrum len:", len(measure.spectrum_db(stereo, sr)), "corr:", measure.corr(stereo))
print("lra:", round(measure.lra(stereo, sr), 2))

# 7) wavio 往返
os.makedirs("cache", exist_ok=True)
wavio.write_wav("cache/_smoke.wav", stereo, sr, 24)
back, sr2 = wavio.read_wav("cache/_smoke.wav")
print("wav roundtrip:", back.shape, sr2, "err:", float(np.abs(back - stereo).max()))
assert sr2 == sr and float(np.abs(back - stereo).max()) < 1e-4

wavio.write_wav("cache/_smoke32.wav", stereo, sr, 32)
back32, _ = wavio.read_wav("cache/_smoke32.wav")
assert float(np.abs(back32 - stereo).max()) < 1e-6

wavio.write_wav("cache/_smoke16.wav", stereo, sr, 16)
back16, _ = wavio.read_wav("cache/_smoke16.wav")
print("16bit err:", float(np.abs(back16 - stereo).max()))

# 7b) 转码缓存名稳定性：同一文件必须固定名字（旧版 hash() 每次重启都会变）
_src = "cache/_smoke.wav"
_t1 = wavio.transcode_tag(_src)
_t2 = wavio.transcode_tag(_src)
assert _t1 == _t2, f"转码缓存名不稳定：{_t1} != {_t2}"
import time as _time
_time.sleep(0.01)
_st = os.stat(_src)
os.utime(_src, (_st.st_atime, _st.st_mtime - 3600.0))   # 显式回拨 1h：Windows 时间戳粒度
                                                        # 会让 utime(now) 落回原刻度（15.6ms 计时器）
_t3 = wavio.transcode_tag(_src)
print(f"transcode tag: {_t1} (稳定) -> {_t3} (改 mtime 后)")
assert _t3 != _t1, "mtime 变化后转码名应失效"
# 同一个 WAV 走 ensure_readable 不应触发转码，且必须解析到自身
assert os.path.samefile(wavio.ensure_readable(_src, cache_dir="cache"), _src)
print("转码缓存命名 OK")

# 8) iir_filt 频响正确性：低通 1k 后 5k 成分应显著衰减
t2 = np.arange(2 * sr) / sr
mixed = (0.3 * np.sin(2 * np.pi * 100 * t2) + 0.3 * np.sin(2 * np.pi * 5000 * t2)).astype(np.float32)
b, a = core.biquad_coeffs("lp", sr, 1000.0, q=0.707)
filtered = core.iir_filt(np.stack([mixed, mixed], 1), b, a)
m_low = core.band_mask(sr, 60, 200)
m_high = core.band_mask(sr, 4000, 8000)
def band_energy(sig, m):
    S = core.stft(sig, sr)
    e = (np.abs(S) ** 2 * m).sum()
    return 10 * np.log10(e)
print("lp 1k: low before/after:", round(band_energy(np.stack([mixed, mixed], 1), m_low), 1),
      round(band_energy(filtered, m_low), 1))
print("lp 1k: high before/after:", round(band_energy(np.stack([mixed, mixed], 1), m_high), 1),
      round(band_energy(filtered, m_high), 1))
assert band_energy(filtered, m_high) < band_energy(np.stack([mixed, mixed], 1), m_high) - 25

print("KERNEL SMOKE OK")
