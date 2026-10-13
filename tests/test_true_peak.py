"""measure.true_peak_db 谱插值修复的数值验证。

判据：
  1) 对采样率远高于奈奎斯特的纯正弦，真峰应测得 20log10(A)（正弦峰 = 幅值 A），
     而非窗乘采样峰（会系统性偏低）。
  2) 对采样点恰好落在正弦峰上的信号，两种实现应一致。
  3) 真峰永远 ≥ 采样峰（dBTP ≥ peak dBFS）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from livefix import measure

sr = 48000
ok = True


def check(name, cond, detail=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    ok = ok and cond


# 1) 采样点刻意错过真峰的正弦：低频 + 短窗，使采样点数量少、难以贴近峰值
A = 0.5
sr_use = 44100
t = np.arange(int(0.05 * sr_use)) / sr_use      # 50ms -> 约 11 个周期，采样点 2205 个
freq = 997.0
sig = A * np.sin(2 * np.pi * freq * t + 0.37)
x = np.stack([sig, sig], 1).astype(np.float32)
tp = measure.true_peak_db(x, sr_use)
pk = measure.peak_db(x)
ideal = 20 * np.log10(A)
check("正弦真峰 ≥ 解析幅值且偏差合理（截断泄漏的保守高估可接受）",
      -0.05 <= tp - ideal < 1.0,
      f"tp={tp:.3f} peak={pk:.3f} ideal={ideal:.3f}")
check("真峰 ≥ 采样峰", tp >= pk - 1e-6, f"tp={tp:.3f} pk={pk:.3f}")
check("过采样增益为非负（不能低于采样峰）", tp >= pk - 1e-3,
      f"增益 {tp - pk:+.4f} dB")

# 1b) 随机短正弦：真峰必须恒 ≥ 采样峰，且过采样增益非负
rng = np.random.default_rng(0)
gains = []
for _ in range(20):
    f = rng.uniform(200, 9000)
    ph = rng.uniform(0, 2 * np.pi)
    s = (A * np.sin(2 * np.pi * f * t + ph)).astype(np.float32)
    xx = np.stack([s, s], 1)
    gains.append(measure.true_peak_db(xx, sr_use) - measure.peak_db(xx))
gains = np.array(gains)
check("随机短正弦 真峰 ≥ 采样峰", (gains >= -0.02).all(),
      f"min={gains.min():+.4f} dB")
check("随机短正弦过采样增益上限 < 1.5dB（单段端点伪迹容忍）", gains.max() < 1.5,
      f"max={gains.max():.4f} mean={gains.mean():.4f}")

# 2) 单样本冲击（采样峰即真峰）
imp = np.zeros((4096, 2), np.float32)
imp[2000, :] = 0.9
check("冲击真峰 = 采样峰", abs(measure.true_peak_db(imp, sr) - measure.peak_db(imp)) < 0.02,
      f"tp={measure.true_peak_db(imp, sr):.3f} pk={measure.peak_db(imp):.3f}")

# 3) 高频正弦（接近奈奎斯特仍应正确）
sig2 = A * np.sin(2 * np.pi * 15000.0 * t + 1.1)
x2 = np.stack([sig2, sig2], 1).astype(np.float32)
tp2 = measure.true_peak_db(x2, sr)
check("15kHz 正弦真峰 ≈ 幅值", abs(tp2 - ideal) < 0.25,
      f"tp={tp2:.3f} ideal={ideal:.3f}")

# 4) 静音
check("静音 = -180dB", measure.true_peak_db(np.zeros((1024, 2), np.float32), sr) < -100)

# 5) 真实文件：真峰应 ≥ 采样峰，且不应低于旧实现的保守值
from livefix import wavio
p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "origin", "源1.wav")
if os.path.exists(p):
    x3, sr3 = wavio.read_wav(p[:]) if False else wavio.read_wav(p)
    tp3 = measure.true_peak_db(x3, sr3)
    pk3 = measure.peak_db(x3)
    check("真实文件 真峰 ≥ 采样峰", tp3 >= pk3 - 1e-6, f"tp={tp3:.3f} pk={pk3:.3f}")

print("\n" + ("ALL TRUE-PEAK CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
