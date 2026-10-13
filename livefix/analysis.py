"""LiveFix 诊断：自动分析录音问题并给出建议参数。"""
from __future__ import annotations

import numpy as np

from . import core, measure

HUM_CANDIDATES = (50, 60, 100, 120, 150, 180)


def _hum_hz(X, sr):
    """检测市电哼声：比较各候选频率（及其谐波）与邻近频段的能量差。"""
    freqs = core.rfftfreqs(sr)
    p = (np.abs(X) ** 2).mean(axis=(0, 1))
    best, best_score = 50, 0.0
    for base in (50, 60):
        score = 0.0
        for h in (1, 2, 3, 4):
            f = base * h
            k = int(np.argmin(np.abs(freqs - f)))
            lo = max(k - 8, 1)
            hi = min(k + 9, len(p))
            local = np.median(np.concatenate([p[lo:k], p[k + 1:hi]])) + 1e-16
            score += float(10 * np.log10((p[k] + 1e-16) / local))
        if score > best_score:
            best, best_score = base, score
    return (best, round(best_score, 1)) if best_score > 6 else (None, 0.0)


def _cutoff_hz(X, sr, nf_db=None):
    """高频截止：找内容与残留噪声尾巴之间的滚降拐点。

    参考电平取 5-8k 中位（有实质内容的最高频段），自顶向下找首个达到
    ref-20dB 的频率。用中位数当门限会失败：滚降后的残留噪声尾巴会抬高
    8-16k 中位，使门限低于尾巴，把尾巴误判为有效高频。返回 0 表示无有效高频。
    """
    del nf_db  # 保留签名兼容，实际不再使用全局噪底
    freqs = core.rfftfreqs(sr)
    p = 10 * np.log10((np.abs(X) ** 2).mean(axis=(0, 1)) + 1e-16)
    ref_mask = (freqs >= 5000) & (freqs <= 8000)
    if not np.any(ref_mask):
        return float(sr) / 2
    ref = float(np.median(p[ref_mask]))
    thr = ref - 20.0
    for k in range(len(freqs) - 1, 1, -1):
        if p[k] > thr:
            return float(freqs[min(k + 1, len(freqs) - 1)])
    return 0.0


def analyze(x, sr):
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = x[:, None]
    clip_pct = float((np.abs(x) >= 0.985).mean() * 100)
    X = core.stft(x[: min(len(x), sr * 20)], sr)  # 诊断最多取前 20s
    nf = float(np.median(core.noise_floor_db(X)))
    hum, hum_score = _hum_hz(X, sr)
    cutoff = _cutoff_hz(X, sr)
    lufs = measure.lufs_integrated(x, sr)
    lra = measure.lra(x, sr)
    peak = measure.peak_db(x)
    corr = measure.corr(x)
    bands = measure.band_powers_db(x, sr)

    issues = []

    def issue(level, text):
        issues.append({"level": level, "text": text})

    low = [b for b in bands if int(b["f"]) in (63, 125)]
    low_span = max(b["db"] for b in low) - min(b["db"] for b in low) if low else 0
    if low_span > 4:
        issue("warn", f"低频 60-160Hz 峰谷约 ±{low_span / 2:.0f}dB，疑似房间驻波，建议中低频动态重建")
    if 0 < cutoff < 14000:
        issue("warn", f"高频截止约 {cutoff / 1000:.1f}kHz，中高频信息缺失，建议带宽扩展补充")
    elif cutoff == 0:
        issue("bad", "未检测到有效高频内容（8k 以上低于高频段典型电平 25dB），建议带宽扩展补充")
    if clip_pct > 0.01:
        issue("bad", f"检测到削波样本占 {clip_pct:.2f}%，建议先走去削波修复")
    if nf > -50:
        issue("warn", f"噪底约 {nf:.0f}dBFS 偏高，建议适度降噪（注意保留信息）")
    if hum:
        issue("warn", f"检测到 {hum}Hz 交流哼声（显著度 {hum_score}dB），建议陷波")
    if corr > 0.8 and x.shape[1] > 1:
        issue("warn", f"左右相关系数 {corr:.2f} 过高，声场偏单点，建议声场扩展")
    if lufs < -24 or lufs > -9:
        issue("info", f"综合响度 {lufs:.1f} LUFS 偏离常规交付范围，建议母带归一")
    if x.shape[1] == 1:
        issue("info", "单声道素材：声场扩展将采用伪立体声合成，效果有限")
    if not issues:
        issue("info", "各项指标在合理范围内，可小幅修饰")

    suggest = {"m1": {}, "m2": {}, "m3": {}, "m4": {}, "m5": {}, "m6": {}}
    if low_span > 4:
        suggest["m1"].update(weight=0.75, density=0.6)
    if 0 < cutoff < 14000:
        suggest["m2"].update(air=0.7, ext_hz=min(11000, cutoff * 0.9))
    elif cutoff == 0:
        suggest["m2"].update(air=0.85, ext_hz=9000)
    if clip_pct > 0.01:
        suggest["m5"].update(declip=0.8)
    if nf > -50:
        suggest["m5"].update(denoise=min(0.75, 0.3 + (-nf - 50) / 30))
    if hum:
        suggest["m5"].update(hum=0.8, mains=hum)
    if corr > 0.8 and x.shape[1] > 1:
        suggest["m4"].update(width_mid=1.3, width_high=1.6, depth=0.5)
    if lufs < -24 or lufs > -9:
        suggest["m6"].update(target_lufs=-16)
    suggest = {k: v for k, v in suggest.items() if v}

    return {
        "sr": int(sr),
        "channels": int(x.shape[1]),
        "duration": round(float(len(np.asarray(x))) / sr, 3),
        "diagnosis": {
            "lufs": round(float(lufs), 2),
            "lra": round(float(lra), 2),
            "peak_db": round(float(peak), 2),
            "clipped_pct": round(clip_pct, 3),
            "corr": round(float(corr), 3),
            "noise_floor_db": round(float(nf), 1),
            "cutoff_hz": round(float(cutoff), 0) if cutoff else 0,
            "hum_hz": hum,
            "bands": bands,
            "issues": issues,
            "suggest": suggest,
        },
    }
