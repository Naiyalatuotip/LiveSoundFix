"""引擎集成测试：全链处理（各模块缺失时自动跳过对应断言）。"""
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from livefix import engine

sr = 48000
t = np.arange(20 * sr) / sr
rng = np.random.default_rng(1)
# livehouse 模拟：低频正弦 + 3k 调幅“人声” + 噪声 + 窄立体声
L = (0.3 * np.sin(2 * np.pi * 80 * t) + 0.15 * np.sin(2 * np.pi * 3000 * t)
     * (0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t)) + 0.02 * rng.standard_normal(t.shape))
R = L + 0.01 * rng.standard_normal(t.shape)
x = np.stack([L, R], 1).astype(np.float32)

missing = []
for k in engine.CHAIN:
    try:
        importlib.import_module(f"livefix.{k}")
    except Exception as e:  # noqa: BLE001
        missing.append(f"{k}({type(e).__name__})")
print("缺失模块:", missing or "无")

cfg = engine.load_config(None)
t0 = time.perf_counter()
y, stages, ctx = engine.process(x, sr, cfg)
dt = time.perf_counter() - t0
print(f"全链处理 20s 立体声耗时 {dt:.2f}s")
print("stages:", [(s["key"], s["on"], s["ms"]) for s in engine.order_stages(stages)])

assert y.shape == x.shape, y.shape
assert np.all(np.isfinite(y)), "输出含 NaN/Inf"

# ---------------------------------------------------------------- 冻结默认参数
# 用户在 UI 里调好并确认「每次打开都用这一版」的参数（2026-10-14 冻结）。
# 这不是「当前最优」，而是**用户签字版**：任何改动都会让他们每次打开的参数变掉。
# 若确实要改默认，请连同本表一起改，并同步 spec/API.md §4.3。
FROZEN = {
    "sep": {"on": True, "mode": "hpss", "routing": "graded", "vocal_gain": 1.0,
            "low_gain": 1.0, "perc_gain": 1.0, "other_gain": 0.6},
    "ai": {"on": False, "model": "none", "focus": "low", "strength": 0.7},
    "m5": {"on": True, "dereverb": 0.0, "hum": 0.20, "mains": 50,
           "declip": 0.0, "pump": 0.0},
    "m1": {"on": True, "weight": 0.20, "density": 0.25, "sub": 0.15, "punch": 0.50,
           "tame": 0.0, "tame_f": 63.0, "tame_q": 2.0, "ring": 0.0, "dive": 0.0},
    "m2": {"on": True, "air": 0.45, "detail": 0.50, "dehiss": 0.10, "ext_hz": 15100},
    "m3": {"on": True, "clarity": 0.35, "room": 0.50, "harsh": 0.0, "ess": 0.0},
    "m4": {"on": True, "width_low": 0.56, "width_mid": 1.75, "width_high": 1.75,
           "depth": 0.40, "rotation": 0.0},
    "m6": {"on": False, "target_lufs": -16, "lra": 6, "ceiling": -1.0},
    "dyneq": {"on": False},
}
d = engine.default_config()
bad = []
for mk, params in FROZEN.items():
    for k, want in params.items():
        got = d["modules"][mk].get(k)
        if got != want:
            bad.append(f"{mk}.{k}: {got!r} != 冻结值 {want!r}")
assert not bad, "默认参数被改动（应与用户冻结版一致）：\n  " + "\n  ".join(bad)
# 已删除的参数不得复活（2026-10-14：m5 的 denoise / floor_db 被 dereverb 取代）
for gone in ("denoise", "floor_db"):
    assert gone not in d["modules"]["m5"], f"m5.{gone} 已被删除，不应再出现在默认配置里"
assert all(d["modules"][mk]["mix"] == 100 for mk in engine.CHAIN), "所有模块 mix 应为 100"
assert d["export"] == {"channels": "stereo", "bit_depth": 24, "loudnorm": True}
print(f"默认参数冻结校验 OK（{sum(len(v) for v in FROZEN.values())} 项逐项一致）")

# ---------------------------------------------------------------- m6 母带链路
# m6 在用户冻结版里是 OFF，所以「响度归一到 -16」必须显式开启 m6 来验证，
# 不能依赖默认值（否则默认一关，这条断言就跟着失效）。
if "m6" not in missing:
    from livefix import measure
    cfg_m6 = engine.load_config({"modules": {"m6": {"on": True}}})
    y6, _, _ = engine.process(x, sr, cfg_m6)
    l = measure.lufs_integrated(y6, sr)
    print("m6 显式开启后 lufs_out:", round(l, 2))
    assert abs(l - (-16)) < 1.0, l
    # 而默认配置（m6 关闭）不应被归一
    l_def = measure.lufs_integrated(y, sr)
    print("默认配置（m6 关）lufs_out:", round(l_def, 2))
    assert abs(l_def - (-16)) > 0.5, "m6 默认关闭时不应发生响应归一"

# 分级路由跑一遍
cfg2 = engine.load_config({"modules": {"sep": {"on": True, "routing": "graded"}}})
y2, _, ctx2 = engine.process(x, sr, cfg2)
assert y2.shape == x.shape and np.all(np.isfinite(y2))
if "sep" not in missing:
    assert ctx2.get("stems") is not None
print("ENGINE INTEGRATION OK" if not missing else f"ENGINE INTEGRATION PARTIAL (缺 {missing})")
