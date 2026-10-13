"""端到端验证：合成 livehouse 模拟素材 -> analyze -> process -> 三声道导出。

缺失的模块自动跳过（子代理尚未交付时也能跑）。
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from livefix import api, engine, measure, wavio

SR = 48000
DUR = 25.0
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")


def make_test_wav(path, sr=SR, dur=DUR):
    """模拟中型 livehouse 相机录音：窄立体声、低频驻波、高频缺失、50Hz 哼声、底噪、轻微削波。"""
    n = int(sr * dur)
    t = np.arange(n) / sr
    rng = np.random.default_rng(7)
    # “乐队”：低频驻波 67Hz 强 + 149Hz 弱、250Hz 和弦、3k 调幅人声感
    # 避开 50/60Hz 及其谐波，保证哼声检测不受泄漏干扰
    low = 0.55 * np.sin(2 * np.pi * 67 * t) + 0.20 * np.sin(2 * np.pi * 149 * t)
    mid = 0.12 * np.sin(2 * np.pi * 250 * t + 0.4) + 0.08 * np.sin(2 * np.pi * 420 * t)
    voice = 0.22 * np.sin(2 * np.pi * 3000 * t) * (0.4 + 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * 2.2 * t)))
    voice += 0.03 * np.sin(2 * np.pi * 11500 * t) * (np.sin(2 * np.pi * 3.1 * t) > 0.8)  # 齿音，落在 9k 滚降区之上
    mono = low + mid + voice
    # 相机削波/AGC 软压缩（在滚降之前，避免 tanh 谐波重新填满高频）
    mono = np.tanh(mono * 1.15) / 1.15
    # 相机麦高频滚降：9kHz 三阶级联一阶低通（-18dB/oct），纯滤波不混直通
    a = np.exp(-2 * np.pi * 9000 / sr)
    lp = mono.copy()
    for _ in range(3):
        acc = 0.0
        out = np.empty_like(lp)
        for i in range(n):
            acc = a * acc + (1 - a) * lp[i]
            out[i] = acc
        lp = out
    mono = lp  # 纯滚降，模拟相机麦高频缺失
    hum = 0.015 * np.sin(2 * np.pi * 50 * t) + 0.008 * np.sin(2 * np.pi * 100 * t)
    noise = 0.0012 * rng.standard_normal(n)  # 低噪底
    sig = mono + hum + noise
    # 高频真空必须在加噪之后施加（否则白噪会把 10k 以上平铺成地板，掩盖滚降）
    sig_spec = np.fft.rfft(sig)
    fr = np.fft.rfftfreq(n, 1 / sr)
    sig_spec[fr > 10000] = 0.0
    sig = np.fft.irfft(sig_spec, n)
    sig = sig / np.abs(sig).max() * 0.995
    # 真实机采立体声：含 ITD 延迟与侧向反射，side 有一定内容（corr≈0.9），
    # 完全单声道的测试信号无法观测 m4 的声场扩展
    itd = int(0.0008 * sr)
    amb = np.zeros(n)
    for d, g in ((int(0.007 * sr), 0.14), (int(0.013 * sr), 0.10), (int(0.023 * sr), 0.07)):
        amb += g * np.roll(sig, d)
    r_ch = 0.88 * sig + 0.12 * np.roll(sig, itd) + 0.5 * amb
    r_ch = r_ch / np.abs(r_ch).max() * 0.995
    stereo = np.stack([sig, r_ch], 1).astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wavio.write_wav(path, stereo, sr, 24)
    return path


def has(key):
    try:
        importlib.import_module(f"livefix.{key}")
        return True
    except Exception:  # noqa: BLE001
        return False


def main():
    src = make_test_wav(os.path.join(CACHE, "_e2e_src.wav"))
    print("素材:", src)

    # 1) 诊断
    d = api.analyze(src)
    dg = d["diagnosis"]
    print(f"诊断: LUFS={dg['lufs']} corr={dg['corr']} cutoff={dg['cutoff_hz']}Hz "
          f"hum={dg['hum_hz']} noise={dg['noise_floor_db']}dB issues={len(dg['issues'])}")
    for i in dg["issues"]:
        print("   -", i["level"], i["text"])
    assert dg["hum_hz"] == 50, "应检测到 50Hz 哼声"
    assert dg["cutoff_hz"] <= 14000, "应检测到高频滚降或有效高频缺失"

    # 2) 全链处理（仅开启已交付的模块）
    cfg = engine.load_config(None)
    for k in engine.CHAIN:
        if k not in ("ai",):
            cfg["modules"][k]["on"] = has(k)
    print("开启模块:", [k for k in engine.CHAIN if cfg["modules"][k]["on"]])
    r = api.process(src, cfg)
    st = r["stats"]
    print(f"处理: LUFS {st['lufs_in']} -> {st['lufs_out']}, peak={st['peak_db']}, "
          f"corr {st['corr_in']} -> {st['corr_out']}")
    for s in st["stages"]:
        print(f"   {s['key']:6s} on={s['on']} {s['ms']:8.1f}ms")
    assert r["channels"] == 2 and r["duration"] == DUR
    assert len(r["spec_before"]) == 96 == len(r["spec_after"])
    if has("m6"):
        assert abs(st["lufs_out"] - (-16)) < 1.0, st["lufs_out"]

    # 3) 导出三声道
    if has("exporter"):
        for ch, expect in (("stereo", 2), ("mono", 1), ("5.1", 6)):
            out = os.path.join(CACHE, f"_e2e_{ch.replace('.', '')}.wav")
            res = api.export_wav(r["id"], channels=ch, bit_depth=24, loudnorm=False, out_path=out)
            x, sr = wavio.read_wav(out)
            print(f"导出 {ch}: channels={x.shape[1]} dur={x.shape[0] / sr:.1f}s")
            assert x.shape[1] == expect, (ch, x.shape)
            assert np.all(np.isfinite(x))
    else:
        print("exporter 未交付，跳过导出验证")

    print("E2E OK" if all(has(k) for k in ("m1", "m2", "m3", "m4", "m5", "m6", "sep", "dyneq", "exporter"))
          else "E2E PARTIAL（部分模块未交付）")


if __name__ == "__main__":
    main()
