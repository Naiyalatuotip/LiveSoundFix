"""LiveFix 多声道导出（stereo / mono / 5.1 · 16/24/32bit · 响度归一）。

声道映射：
- mono   ：跨声道均值。
- stereo ：已是立体声直接返回；单声道复制到 L/R；>2 声道取前两声道（WAV 常规布局 L,R）。
- 5.1    ：BS.775 被动上混，WAV 声道顺序 L R C LFE Ls Rs（见 _upmix_51 注释）。

滤波做法：side/mid 的分频（HP 180Hz / LP 120Hz）在 STFT 幅度域完成——
side 与 mid 合成 2 通道一次 stft，用 core.mag_response("hp"/"lp") 幅度相乘
（零相位）后 istft 回到时域；Rs 对 HP(side) 施加 3.5ms 微延迟做去相关。

响度处理：
- loudnorm=True ：measure.lufs_integrated 测得响度 -> 增益到 target_lufs
  （截 ±24dB）-> 时域包络限制器（采样域天花板 -1.3dB，瞬时 attack / 指数
  release / 逐点强制）-> 真峰保证回环（线性缩放到 ≤ -1.05dBTP，满足 ≤ -0.9）。
- loudnorm=False：若 true_peak_db > -0.5dBTP，纯等比缩到 -0.5（不改频谱）。

只用 numpy + 标准库 + 本包 wavio/measure。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import core, measure, wavio

TARGETS = ("stereo", "mono", "5.1")
BIT_DEPTHS = (16, 24, 32)

LIMIT_TARGET_DB = -1.05   # loudnorm 真峰保证目标（契约要求 <= -0.9，留余量）
LIMIT_SAMPLE_DB = -1.3    # 包络限制器采样域天花板（给过采样真峰留余量）
PEAK_SAFE_DB = -0.5       # 非归一导出的峰值安全线
LIMIT_RELEASE_S = 0.06    # 限制器释放时间常数
LIMIT_BLOCK_MS = 2.0      # 限制器控制块宽度


def _as2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError(f"音频应为 (n, ch)，得到 {np.asarray(x).shape}")
    return x.astype(np.float32, copy=False)


def _filter_band(stacked, sr, kinds, freqs):
    """(n,k) -> float64 (n,k)：STFT 幅度域零相位滤波，第 i 通道用 kinds[i]@freqs[i]。"""
    n = stacked.shape[0]
    X = core.stft(np.ascontiguousarray(stacked, np.float32), sr)  # (ch, T, F)
    for i, (k, f) in enumerate(zip(kinds, freqs)):
        X[i, :, :] *= core.mag_response(k, sr, f).astype(np.float32)  # 第 i 声道（轴0=声道）
    return core.istft(X, n, sr).astype(np.float64)


def _upmix_51(x, sr):
    """BS.775 被动上混 -> float32 (n,6)，WAV 声道顺序 L R C LFE Ls Rs。

    立体声源：
        mid  = (L+R)/2,  side = (L-R)/2
        C    = 0.707 * mid
        L'   = L - 0.707*C,   R' = R - 0.707*C
        Ls   = 0.5 * HP(side, 180Hz)
        Rs   = 0.5 * HP(side 经 3.5ms 微延迟, 180Hz)   # 延迟去相关
        LFE  = 0.5 * LP(mid, 120Hz)
    分频在 STFT 幅度域完成（见模块头注释）。
    单声道源：C = mid 主导，L/R 直接复制，无 side（Ls = Rs = 0）。
    输入已 >=6 声道时直接取前 6 声道。
    """
    n, ch = x.shape
    if ch >= 6:
        return x[:, :6].copy()
    xf = x.astype(np.float64)
    if ch == 1:
        m = xf[:, 0]
        c = m  # C = mid 主导
        lp = _filter_band(np.stack([m, np.zeros_like(m)], 1), sr, ("lp",), (120.0,))[:, 0]
        z = np.zeros(n, np.float64)
        return np.stack([m, m, c, 0.5 * lp, z, z], 1).astype(np.float32)
    l0, r0 = xf[:, 0], xf[:, 1]  # >2 声道时同样取前两声道
    mid = 0.5 * (l0 + r0)
    side = 0.5 * (l0 - r0)
    c = 0.707 * mid
    lp_l = l0 - 0.707 * c
    lp_r = r0 - 0.707 * c
    fb = _filter_band(np.stack([side, mid], 1), sr, ("hp", "lp"), (180.0, 120.0))
    hp_side, lp_mid = fb[:, 0], fb[:, 1]
    d = max(1, int(round(0.0035 * float(sr))))  # 3.5ms 微延迟去相关
    hp_side_d = np.concatenate([np.zeros(d, np.float64), hp_side[:-d]])
    ls = 0.5 * hp_side
    rs = 0.5 * hp_side_d
    lfe = 0.5 * lp_mid
    return np.stack([lp_l, lp_r, c, lfe, ls, rs], 1).astype(np.float32)


def to_channels(x, sr, target):
    """-> float32 (n, ch_out)。target: "stereo"|"mono"|"5.1"。"""
    x = _as2d(x)
    t = str(target).lower()
    if t == "mono":
        if x.shape[1] == 1:
            return x.copy()
        return np.ascontiguousarray(
            x.astype(np.float64).mean(axis=1, keepdims=True), np.float32)
    if t == "stereo":
        if x.shape[1] == 2:
            return x.copy()
        if x.shape[1] == 1:
            return np.repeat(x, 2, axis=1)  # 单声道复制到 L/R
        return x[:, :2].copy()  # >2 声道：按 WAV 常规布局取前两声道（L,R），其余忽略
    if t == "5.1":
        return _upmix_51(x, sr)
    raise ValueError(f"不支持的声道布局：{target}（可用 stereo / mono / 5.1）")


def _envelope_limit(y, sr, ceil_db=LIMIT_SAMPLE_DB):
    """时域包络限制器（全声道 link）：瞬时 attack、指数 release、逐点强制天花板。

    |y| 全声道最大值 -> 2ms 块最大（保守包络）-> 期望块增益 min(1, ceil/bm)
    -> 释放平滑（attack 瞬时取小，release 一阶指数）-> 线性插值到采样率
    -> 与本块期望增益逐点取 min（保证采样域 |y*gain| <= ceil）。
    """
    yy = y.astype(np.float64)
    n = yy.shape[0]
    H = max(16, int(round(LIMIT_BLOCK_MS * 1e-3 * float(sr))))
    nb = max(1, (n + H - 1) // H)
    m = np.pad(np.abs(yy).max(axis=1), (0, nb * H - n))
    bm = m.reshape(nb, H).max(axis=1)  # 块最大 = 保守包络
    ceil_lin = float(core.lin(ceil_db))
    g = np.minimum(1.0, ceil_lin / np.maximum(bm, 1e-12))
    a = float(np.exp(-(H / float(sr)) / LIMIT_RELEASE_S))
    gs = np.empty(nb)
    cur = 1.0
    for k in range(nb):  # 瞬时 attack（取小即降）/ 指数 release（缓慢恢复）
        cur = g[k] if g[k] < cur else a * cur + (1.0 - a) * g[k]
        gs[k] = cur
    idx = (np.arange(n) + 0.5) / H - 0.5
    gain = np.interp(idx, np.arange(nb), gs)              # 线性平滑
    gain = np.minimum(gain, np.repeat(g, H)[:n])          # 逐点强制天花板
    return yy * gain[:, None]


def _loudnorm(y, sr, target_lufs):
    lufs0 = measure.lufs_integrated(y, sr)
    g = float(np.clip(core.lin(float(target_lufs) - lufs0),
                      core.lin(-24.0), core.lin(24.0)))  # 增益截 ±24dB
    z = y.astype(np.float64) * g
    z = _envelope_limit(z, sr)
    tp = measure.true_peak_db(z, sr)
    if tp > LIMIT_TARGET_DB:  # 真峰保证回环：线性缩放必然收敛
        z = z * float(core.lin(LIMIT_TARGET_DB - tp))
    return z


def export(x, sr, out_path, channels="stereo", bit_depth=24, loudnorm=False,
           target_lufs=-16.0):
    """多声道导出并写 wav。

    -> {"path": 绝对路径, "channels": int, "duration": 秒,
       "lufs": float, "true_peak_db": float}
    """
    x = _as2d(x)
    if x.shape[0] == 0:
        raise ValueError("音频为空，无法导出")
    sr = int(sr)
    bit_depth = int(bit_depth)
    if bit_depth not in BIT_DEPTHS:
        raise ValueError(f"不支持的位深：{bit_depth}（可用 16 / 24 / 32）")
    y = np.nan_to_num(to_channels(x, sr, channels), nan=0.0, posinf=0.0, neginf=0.0)
    if loudnorm:
        y = _loudnorm(y, sr, float(target_lufs))
    else:
        tp = measure.true_peak_db(y, sr)
        if tp > PEAK_SAFE_DB:  # 峰值安全：纯等比缩放，不改频谱
            y = y.astype(np.float64) * float(core.lin(PEAK_SAFE_DB - tp))
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    p = Path(out_path)
    if p.suffix.lower() != ".wav":  # 无 .wav 后缀自动补
        p = p.with_suffix(p.suffix + ".wav") if p.suffix else p.with_suffix(".wav")
    wavio.write_wav(p, y, sr, bit_depth)
    return {
        "path": str(p.resolve()),
        "channels": int(y.shape[1]),
        "duration": round(float(y.shape[0]) / float(sr), 3),
        "lufs": round(measure.lufs_integrated(y, sr), 2),
        "true_peak_db": round(measure.true_peak_db(y, sr), 2),
    }
