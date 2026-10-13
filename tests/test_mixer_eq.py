"""调音台贡献量（chain_stems）与快速混音（remix）、EQ 烘焙的测试。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from livefix import api, core, engine, wavio

sr = 48000
rng = np.random.default_rng(11)
t = np.arange(15 * sr) / sr
L = (0.3 * np.sin(2 * np.pi * 66 * t) + 0.25 * np.sin(2 * np.pi * 997 * t)
     + 0.06 * rng.standard_normal(t.shape))
R = L + 0.03 * rng.standard_normal(t.shape)
x = np.stack([L, R], 1).astype(np.float32)

# ------------------------------------------------- 1) 望远镜恒等（引擎层）
cfg = engine.default_config()
y, stages, ctx = engine.process(x, sr, cfg, capture_stems=True)
stems = ctx.get("chain_stems") or {}
print(f"[capture] keys={sorted(stems.keys())}")
assert stems, "应捕获到贡献量"
acc = x.astype(np.float32).copy()
for k in stems:
    assert stems[k].dtype == np.float16, f"{k} 应为 f16"
    acc = acc + stems[k].astype(np.float32)
err = float(np.abs(acc - y).max())
print(f"[telescope] max|输入+Σdelta - 输出| = {err:.5f}（f16 量化级）")
assert err < 0.02, err
active = {k for k, _ in engine.plan(cfg["modules"])}
assert active <= set(stems.keys()), f"激活段缺贡献量: {active - set(stems.keys())}"

# 单模块关闭时其贡献量应缺失（省内存）
cfg2 = engine.default_config()
cfg2["modules"]["m4"]["on"] = False
_, _, ctx2 = engine.process(x, sr, cfg2, capture_stems=True)
assert "m4" not in (ctx2.get("chain_stems") or {}), "关闭的模块不应有贡献量"

# ------------------------------------------------- 2) api 作业 + remix（服务层）
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
src = os.path.join(CACHE, "_t_mixer.wav")
wavio.write_wav(src, x, sr, 24)

job = api.start_job(src, engine.default_config())["job"]
t0 = time.time()
while True:
    s = api.job_status(job)
    if s["state"] != "running":
        break
    assert time.time() - t0 < 120, "作业超时"
    time.sleep(0.3)
assert s["state"] == "done", s["error"]
r = s["result"]
print(f"[job] mixer={r['mixer']}")
assert r["mixer"]["available"], "15s 素材应捕获贡献量"
assert r["mixer"]["job"] == job

# 2a) 全部推子 = 100%：remix 应 ≈ 处理后音频
m1 = api.remix(job, {})
x1, sr1 = wavio.read_wav(api.audio_bytes(m1["id"]))
proc, _ = wavio.read_wav(api.audio_bytes(r["id"]))
d_full = float(np.abs(x1 - proc).max())
print(f"[remix all=1] vs 处理后 max diff = {d_full:.5f}")
assert d_full < 0.02, d_full

# 2b) sep 推子 = 0：remix 的语义是 out = 原片 + Σ g_k·δ_k，
#     所以 sep=0 时必须**逐样本等于 proc − δ_sep**（这是由定义保证的恒等式）。
#
# 旧断言写的是「去掉 sep 贡献应更接近原片」，用 max|·| 度量 —— 这个前提**不成立**：
# 峰值是单个最坏样本，去掉一项贡献完全可能抬高它。实测（用户冻结默认参数）：
#   sep 贡献 RMS 0.0126，其余贡献之和 RMS 0.1085，两者相关系数 −0.123（轻微反相）
#   -> 去掉 sep 后离原片 RMS 0.1077 → 0.1085（反而略远）
# 所以这里改成验证 mixer 的代数恒等式，并顺带打印各模块贡献量做诊断。
m0 = api.remix(job, {"sep": 0.0})
x0, _ = wavio.read_wav(api.audio_bytes(m0["id"]))
d_sep = float(np.abs(x0 - proc).max())
sf = os.path.join(CACHE, f"stems_{job}.npz")
assert os.path.exists(sf), "应落盘贡献量 npz"
with np.load(sf) as data:
    keys_npz = set(data.files)
    delta_sep = data["sep"].astype(np.float32)
expect = proc - delta_sep                      # = 原片 + Σ_{k≠sep} δ_k
err_mix = float(np.abs(x0 - expect).max())
print(f"[remix sep=0] 与处理后差 {d_sep:.4f}；"
      f"与「proc − δ_sep」恒等式偏差 {err_mix:.6f}")
assert set(r["mixer"]["keys"]) <= keys_npz, "npz 应含全部推子模块"
assert err_mix < 0.005, f"mixer 恒等式不成立: {err_mix}"
assert d_sep > 0.001, "sep 推子归零应改变输出"
dev_full = float(np.abs(proc - x).max())
print(f"[remix sep=0] 离原片 {float(np.abs(x0 - x).max()):.4f} vs 全量 {dev_full:.4f}"
      f"（不比较大小：峰值口径下无必然方向）")

# 各模块贡献量诊断（哪一段在主导整体改动）
print("[contrib] 各模块贡献量 RMS（越大=对整体改动贡献越多）：")
_rms = {}
for k in keys_npz:
    with np.load(sf) as data:
        _rms[k] = float(np.sqrt((data[k].astype(np.float32) ** 2).mean()))
for k in sorted(_rms, key=lambda q: -_rms[q]):
    print(f"           {k:<6} {_rms[k]:.6f}")

# 2c) 全部推子 = 0：应精确回到原片
mz = api.remix(job, {k: 0.0 for k in r["mixer"]["keys"]})
xz, _ = wavio.read_wav(api.audio_bytes(mz["id"]))
d_zero = float(np.abs(xz - x).max())
print(f"[remix all=0] vs 原片 max diff = {d_zero:.5f}")
assert d_zero < 0.02, d_zero

# ------------------------------------------------- 3) EQ 烘焙
imp = np.zeros((4096, 1), np.float32)
imp[100, 0] = 1.0
y_eq = api._apply_eq(imp, sr, [])
assert float(np.abs(y_eq - imp).max()) < 1e-6, "空 EQ 应恒等"

tone = (0.3 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
tone = np.stack([tone, tone], 1)
up = api._apply_eq(tone, sr, [{"f": 3000, "q": 1.0, "gain_db": 6.0}])
dn = api._apply_eq(tone, sr, [{"f": 3000, "q": 1.0, "gain_db": -6.0}])


def band_e(sig, lo, hi):
    X = core.stft(sig, sr)
    m = core.band_mask(sr, lo, hi)
    return float(10 * np.log10((np.abs(X) ** 2 * m).sum() + 1e-16))


e0, e_up, e_dn = band_e(tone, 2500, 3600), band_e(up, 2500, 3600), band_e(dn, 2500, 3600)
print(f"[eq] 3k 带: 原 {e0:.2f}  +6dB烘焙 {e_up:.2f}  -6dB烘焙 {e_dn:.2f}")
assert e_up - e0 > 4.5, "正增益烘焙应提升 3k 带"
assert e_dn - e0 < -4.5, "负增益烘焙应压低 3k 带"
assert np.all(np.isfinite(up)) and np.all(np.isfinite(dn))

# ------------------------------------------------- 4) 导出走 gains+eq 路径
out = os.path.join(CACHE, "_t_mixer_export.wav")
res = api.export_wav(None, channels="stereo", bit_depth=24, loudnorm=False,
                     out_path=out, gains={"sep": 0.0}, job_id=job,
                     eq_bands=[{"f": 120, "q": 1.2, "gain_db": -3.0}])
print(f"[export] {res}")
assert os.path.exists(out)
xe, sre = wavio.read_wav(out)
assert xe.shape[1] == 2 and np.all(np.isfinite(xe))

print("MIXER / REMIX / EQ TESTS PASSED")
