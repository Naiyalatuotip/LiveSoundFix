"""干湿比（mix）与进度回调（progress_cb）的引擎级测试。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from livefix import engine

sr = 48000
rng = np.random.default_rng(3)
t = np.arange(15 * sr) / sr
L = (0.3 * np.sin(2 * np.pi * 70 * t) + 0.2 * np.sin(2 * np.pi * 1000 * t)
     + 0.05 * rng.standard_normal(t.shape))
R = L + 0.02 * rng.standard_normal(t.shape)
x = np.stack([L, R], 1).astype(np.float32)


def only(key, **over):
    cfg = engine.default_config()
    for k in engine.CHAIN:
        cfg["modules"][k]["on"] = (k == key)
    cfg["modules"][key].update(over)
    return cfg


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, np.float64) ** 2)))


# 1) 单模块 mix 扫描：0=恒等，100=全处理，50 介于两者之间
for key in ("m1", "m4", "m5"):
    y0, _, _ = engine.process(x, sr, only(key, mix=0))
    y50, _, _ = engine.process(x, sr, only(key, mix=50))
    y100, _, _ = engine.process(x, sr, only(key, mix=100))
    d0 = np.abs(y0 - x).max()
    d50 = np.abs(y50 - x).max()
    d100 = np.abs(y100 - x).max()
    print(f"[{key}] mix0 maxdev={d0:.2e}  mix50={d50:.4f}  mix100={d100:.4f}")
    assert d0 < 1e-6, f"{key} mix=0 不是恒等"
    assert 1e-4 < d50 < d100 * 1.01, f"{key} mix=50 应介于 0 与 100 之间"
    assert np.all(np.isfinite(y50))

# 2) mix=50 是否精确等于 lerp(输入, mix100 结果) —— 单模块由定义成立
y50, _, _ = engine.process(x, sr, only("m4", mix=50))
y100, _, _ = engine.process(x, sr, only("m4", mix=100))
lerp = x * 0.5 + y100 * 0.5
err = float(np.abs(y50 - lerp).max())
print(f"[m4] mix50 vs lerp(mix100) max err = {err:.2e}")
assert err < 1e-5

def sep_only(mix, routing="graded", other_gain=0.7):
    """只开 sep 的配置（其余全关，隔离测 sep 的干湿比语义）。"""
    cfg = engine.default_config()
    for k in engine.CHAIN:
        cfg["modules"][k]["on"] = (k == "sep")
    cfg["modules"]["sep"].update(on=True, mix=mix, routing=routing, other_gain=other_gain)
    return cfg


# 3) sep 分级路由 mix=0：输出恒等，且 stems 与输出一致（V+B==y）
cfg = sep_only(0, "graded")
y0, st, ctx = engine.process(x, sr, cfg)
print(f"[sep] mix=0 graded: max|y-x|={np.abs(y0 - x).max():.2e}  stems={ctx.get('stems') is not None}")
assert np.abs(y0 - x).max() < 1e-5
if ctx.get("stems"):
    vb = ctx["stems"]["vocal"] + ctx["stems"]["backing"]
    assert np.abs(vb - y0).max() < 1e-4, "stems 之和应等于输出"

# 4) sep mix=50：stem 增益按 g_eff=1-mix+mix*g 混合（对 other_gain 验证）
y0s, _, _ = engine.process(x, sr, sep_only(0, "mix"))
y50s, _, _ = engine.process(x, sr, sep_only(50, "mix"))
y100s, _, _ = engine.process(x, sr, sep_only(100, "mix"))
print(f"[sep] mix0 dev={np.abs(y0s - x).max():.2e}  mix50 dev={np.abs(y50s - x).max():.4f}"
      f"  mix100 dev={np.abs(y100s - x).max():.4f}")
assert np.abs(y0s - x).max() < 1e-5
assert np.abs(y50s - x).max() < np.abs(y100s - x).max()

# 5) 进度回调：单调不减、以 1.0 收尾、stage 合法
events = []


def cb(frac, stage, sfrac):
    events.append((frac, stage, sfrac))


cfg = engine.default_config()
plan = engine.plan(cfg["modules"])
y, stages, _ = engine.process(x, sr, cfg, progress_cb=cb)
fracs = [e[0] for e in events]
print(f"[progress] 回调 {len(events)} 次, plan={plan}")
print(f"           frac 单调: {all(b >= a - 1e-9 for a, b in zip(fracs, fracs[1:]))}"
      f"  首={fracs[0]:.3f} 末={fracs[-1]:.3f}")
assert events and events[-1][0] == 1.0
assert all(b >= a - 1e-9 for a, b in zip(fracs, fracs[1:])), "frac 必须单调不减"
valid_keys = {"sep", "ai", "m5", "m1", "m2", "m3", "m4", "m6", "dyneq", "done"}
assert all(e[1] in valid_keys for e in events)
active = {k for k, _ in plan}
seen = {e[1] for e in events if e[1] != "done"}
assert active <= seen, f"plan 里的段未全部上报: {active - seen}"

# 6) 阶段表：9 行 + mix 字段
ordered = engine.order_stages(stages)
print(f"[stages] {[(s['key'], s['on'], s.get('mix')) for s in ordered]}")
assert [s["key"] for s in ordered] == engine.CHAIN
assert all("mix" in s for s in ordered)

# 7) 全关时进度直接到 1、输出恒等
cfg_off = engine.default_config()
for k in engine.CHAIN:
    cfg_off["modules"][k]["on"] = False
ev = []
y_off, _, _ = engine.process(x, sr, cfg_off, progress_cb=lambda f, s, g: ev.append((f, s)))
print(f"[all-off] maxdev={np.abs(y_off - x).max():.2e}  last_frac={ev[-1][0] if ev else None}")
assert np.abs(y_off - x).max() < 1e-6
assert ev and ev[-1][0] == 1.0

print("MIX & PROGRESS TESTS PASSED")
