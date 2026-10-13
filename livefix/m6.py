"""m6 母带响度（纯 numpy）。

三步流水（引擎对本模块不分段，可依赖全文件统计）：
  1. LRA glue：短时响度（3s 窗/100ms 步进，增益按窗中心对齐）高于 ref = EBU 门控 p10 + lra_target
     的部分按 ratio=2 下压；压缩量闭式 dyn=-(l-ref)·(1-2^-k) 迭代至预测 LRA 收敛（参考轨由输入
     分布一次定出，预测在 K 加权均方域精确推演，无需重复滤波）。
  2. 响度归一：静态增益 g = target_lufs - 当前综合响度（截 ±24dB），在预测均方域上重算门控
     修正残差后与动态增益合并，一次应用。
  3. 真峰限制器：块 FFT 频域插值求 4x 过采样包络 → 目标增益 min(1, ceiling/env) → 5ms 前瞻
     min 窗（分块滑窗最小值，O(n)）→ 1ms 网格非对称平滑（attack 1ms / release 80ms）→ 与瞬时
     需求取 min 硬钳位（保证不过冲）→ 应用。

性能设计：K 加权用「整段一次 FFT 频域合成双二阶」实现（与 measure.k_filter 同系数同序，
数值一致但快 ~4x），整条链 60s@48k 立体声约 2~3s。

已发现的 measure.py 问题（未修改，见交付报告）：
  - measure.true_peak_db 的 rfft(seg*w, n=4L) 是时域补零（返回 [窗乘段, 0...]），
    并非 4x 频域插值，实际量到的是「窗乘采样峰」而非真峰；
  - measure.k_filter / lufs_* 走 core.iir_filt 截断 IR 卷积，60s 立体声单次 ~2.5s，
    本模块为满足性能预算改用等效的整段 FFT 实现（测试中与 measure 对拍 < 0.2LU）。
"""
from __future__ import annotations

import numpy as np

from . import core

KEY = "m6"

_ABS_GATE = -70.0
_REL_GATE_INT = -10.0
_REL_GATE_LRA = -20.0
# ---- LRA glue 的权威度（2026-10-14）----
# 参考点沿分位数扫一遍取最接近目标的曲线（旧版固定 ref=p10+lra_t，目标远低于素材
# LRA 时压不动）。注意：**不能**加向上抬升来压 LRA —— 抬安静段会让它们穿过 EBU
# 相对门进入门控集合，实测把高动态素材的 LRA 从 ~7 推到 15.8，适得其反。
_GLUE_DOWN_MAX_DB = 24.0   # 单帧最大下压


# ---------------------------------------------------------------- K 加权（快速等效路径）

def _k_weights(ch: int) -> np.ndarray:
    """与 measure._weights 一致：mono=1；立体声=1,1；>=3 声道按 5.1 顺序（C=√2, LFE=0）。"""
    if ch == 1:
        return np.array([1.0])
    if ch == 2:
        return np.array([1.0, 1.0])
    w = np.ones(ch, np.float64)
    if ch >= 4:
        w[3] = 0.0
    if ch >= 3:
        w[2] = np.sqrt(2.0)
    return w


def _kfilter_fft(x: np.ndarray, sr: float) -> np.ndarray:
    """整段一次 FFT 实现 K 加权（shelf + HP 合成一个频响，频域相乘）。

    与 measure.k_filter 同系数、同级联；尾部零 pad 保证循环卷积等价于线性卷积。
    """
    xa = np.asarray(x)
    n = xa.shape[0]
    N = 1 << int(np.ceil(np.log2(n + 65536)))
    b1, a1 = core.biquad_coeffs("highshelf", sr, 1681.97, q=0.7071752, gain_db=3.999843)
    b2, a2 = core.biquad_coeffs("hp", sr, 38.135, q=0.5003270)
    w = 2.0 * np.pi * np.fft.rfftfreq(N, 1.0)  # rad/sample
    z1 = np.exp(-1j * w)
    z2 = z1 * z1
    H = ((b1[0] + b1[1] * z1 + b1[2] * z2) / (a1[0] + a1[1] * z1 + a1[2] * z2)) * \
        ((b2[0] + b2[1] * z1 + b2[2] * z2) / (a2[0] + a2[1] * z1 + a2[2] * z2))
    X = np.fft.rfft(xa.astype(np.float64), n=N, axis=0)
    X *= H[:, None]
    return np.fft.irfft(X, n=N, axis=0)[:n]


# ---------------------------------------------------------------- 响度统计（100ms 均方域）

def _mean_square_blocks(y: np.ndarray, base: int) -> np.ndarray:
    """每 base 采样块的均方 (nb, ch)，float64；与 measure._base_powers 同精度路径。"""
    n, ch = y.shape
    nb = n // base
    if nb == 0:
        return np.zeros((0, ch), np.float64)
    sq = (y[:nb * base].astype(np.float32) ** 2).reshape(nb, base, ch).sum(axis=1)
    return (sq / base).astype(np.float64)


def _window_means(z: np.ndarray, win: int) -> np.ndarray:
    """滑动窗均值（hop=1 块）；尾部不足窗补零稀释，与 measure._z_series 完全一致。"""
    zp = np.concatenate([z, np.zeros(win - 1, np.float64)])
    c = np.concatenate([[0.0], np.cumsum(zp)])
    return (c[win:] - c[:-win]) / win


def _gated_integrated(z400: np.ndarray) -> float:
    """BS.1770-4 门控综合响度（400ms 块均方序列 -> LKFS），与 measure.lufs_integrated 同逻辑。"""
    z = z400[z400 > 0]
    if z.size == 0:
        return -70.0
    z1 = z[z > 10.0 ** ((_ABS_GATE + 0.691) / 10.0)]
    if z1.size == 0:
        return -70.0
    loud1 = -0.691 + 10.0 * np.log10(np.mean(z1))
    z2 = z1[z1 > 10.0 ** ((loud1 + _REL_GATE_INT + 0.691) / 10.0)]
    if z2.size == 0:
        return float(loud1)
    return float(-0.691 + 10.0 * np.log10(np.mean(z2)))


def _program_gated(l: np.ndarray) -> np.ndarray:
    """EBU 3342 门控后的节目响度序列（-70 绝对门 + mean-20 相对门），与 _lra_from_short 取样一致。

    用它求 p10 参考电平可避开 measure.lufs_short 尾部「零稀释不完整窗」对参考的拉偏。
    """
    l = np.asarray(l, np.float64)
    keep = np.isfinite(l) & (l > _ABS_GATE)
    if keep.sum() < 3:
        return l[keep]
    z = 10.0 ** ((l[keep] + 0.691) / 10.0)
    mean_l = -0.691 + 10.0 * np.log10(np.mean(z))
    return l[keep][l[keep] > (mean_l + _REL_GATE_LRA)]


def _lra_from_short(l: np.ndarray) -> float:
    """由短时响度序列求 LRA，门控与 measure.lra 完全一致（-70 绝对门 / mean-20 相对门 / p95-p10）。"""
    v = _program_gated(l)
    if v.size < 3:
        return 0.0
    return float(np.percentile(v, 95) - np.percentile(v, 10))


# ---------------------------------------------------------------- 限制器工具

def _rolling_min(v: np.ndarray, w: int) -> np.ndarray:
    """out[i] = min(v[i:i+w])（尾部越界补 +inf），分块前后缀最小值，O(n) 向量化。"""
    n = v.shape[0]
    if w <= 1:
        return v.copy()
    nb = (n + w - 1) // w + 1
    vp = np.pad(v, (0, nb * w - n), constant_values=np.inf).reshape(nb, w)
    suf = np.minimum.accumulate(vp[:, ::-1], axis=1)[:, ::-1]
    pre = np.minimum.accumulate(vp, axis=1)
    grid = np.minimum(suf[:-1], np.roll(pre[1:], 1, axis=1))
    grid[:, 0] = suf[:-1, 0]
    return grid.reshape(-1)[:n]


def _attack_release_smooth(g: np.ndarray, dt: float, attack_s: float, release_s: float) -> np.ndarray:
    """非对称一阶平滑（网格域）：需要下压时按 attack 快速跟随，抬升时按 release 缓慢恢复。"""
    aa = float(np.exp(-dt / max(float(attack_s), 1e-6)))
    ar = float(np.exp(-dt / max(float(release_s), 1e-6)))
    out = np.empty_like(g)
    prev = float(g[0])
    for i in range(g.shape[0]):
        cur = float(g[i])
        a = aa if cur < prev else ar
        prev = a * prev + (1.0 - a) * cur
        out[i] = prev
    return out


def _env_oversampled(x: np.ndarray, sr: float, L: int = 65536) -> np.ndarray:
    """4x 带限插值包络 (4n,) float32：块「rfft -> 谱周期延拓补零去镜像 -> irfft(4L)」得真插值，
    hann 50% 重叠相加重构；两端各补 L 零使全程权重和为 1。"""
    xa = np.asarray(x, np.float64)
    n, ch = xa.shape
    L = min(int(L), 1 << 16)
    L -= L % 2
    hop = L // 2
    w = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(L) / L)).astype(np.float64)
    xp = np.pad(xa, ((L, L), (0, 0)))
    npx = n + 2 * L
    nup = 4 * npx + 8
    acc = np.zeros((nup, ch), np.float32)
    Y = np.zeros((2 * L + 1, ch), np.complex128)
    for s in range(0, npx, hop):
        e = min(s + L, npx)
        seg = np.zeros((L, ch), np.float64)
        seg[:e - s] = xp[s:e]
        W = np.fft.rfft(seg * w[:, None], axis=0)               # (L/2+1, ch)
        Y[:] = 0.0
        Y[:L // 2 + 1] = 2.0 * W                                # 基带正频率（周期延拓去镜像，
        Y[L // 2 + 1:L] = 2.0 * np.conj(W[L // 2 - 1:0:-1])     #  2x 增益恢复采样值）
        Y[L] = np.real(W[0])                                    # 带边半权重
        up = np.fft.irfft(Y, n=4 * L, axis=0)                   # (4L, ch) 真 4x 带限插值
        off = 4 * s
        m = min(4 * L, nup - off)
        acc[off:off + m] += up[:m].astype(np.float32)
    return np.abs(acc[4 * L:4 * L + 4 * n]).max(axis=1)


# ---------------------------------------------------------------- 主流程

def process(x, sr, p, ctx):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    sr = float(sr)
    n, ch = x.shape

    target = float(np.clip(p.get("target_lufs", -16.0), -70.0, -5.0))
    lra_t = float(np.clip(p.get("lra", 6.0), 1.0, 20.0))
    ceil_db = float(np.clip(p.get("ceiling", -1.0), -20.0, 0.0))
    ceil_lin = float(10.0 ** (ceil_db / 20.0))

    base = max(1, int(round(0.1 * sr)))
    nb = n // base

    dyn100 = np.zeros(max(nb, 1), np.float64)
    g_static = 0.0
    lufs_before = -70.0
    lufs_after = -70.0
    lra_before = 0.0
    lra_after = 0.0

    # ---- 1+2：LRA glue 与响度归一（K 加权均方域一次滤波，多轮推演零额外滤波成本）
    if nb >= 4:
        yk = _kfilter_fft(x, sr)
        sq = _mean_square_blocks(yk, base)                 # (nb, ch)
        z_base = sq @ _k_weights(ch)                       # (nb,)
        z400 = _window_means(z_base, 4)
        lufs_before = _gated_integrated(z400)

        zpow = z_base
        SH = 15  # 3s 短时窗（hop 100ms）中心相对窗起点的偏移：增益按窗中心对齐施加到块，
        # 否则前视窗的滞后会让压缩永远打不满目标电平。
        if nb >= 30:
            z3000 = _window_means(z_base, 30)
            l = -0.691 + 10.0 * np.log10(np.maximum(z3000, 1e-16))
            lra_before = _lra_from_short(l)
            l0 = l.copy()
            bidx = np.maximum(np.arange(nb) - SH, 0)
            if lra_before > lra_t + 0.2:
                # 参考轨由输入分布定出（EBU 门控 + 目标 LRA）；比例固定 ratio=2，
                # 压缩量闭式 dyn = -(l-ref)·(1-2^-k)。
                # 2026-10-14 修：原来只试 ref = p10 + lra_t 一个参考点，当目标远低于
                # 素材 LRA 时压不动（实测 lra=1 只把 LRA 降 1.71 LU）。现在沿分位数
                # 扫一遍参考点，取最接近目标的那条曲线 —— 参数才有真正的权威度。
                prog = _program_gated(l)
                if prog.size >= 3:
                    best = None                       # (偏差, dyn, zpow, l)
                    for q in (10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0):
                        ref_q = float(np.percentile(prog, q)) + lra_t
                        dyn_q = np.zeros_like(dyn100)
                        zpow_q = z_base
                        l_q = l0
                        for k in range(1, 13):
                            # 只做向下 glue（ratio=2 的闭式收敛）。**不做向上抬升**：
                            # 把安静段抬上去会让它们穿过 EBU 相对门、进入门控集合，
                            # 实测反而把高动态素材的 LRA 从 ~7 推到 15.8（LRA 的定义使然）。
                            dyn_q = np.clip(-np.maximum(0.0, l0 - ref_q) * (1.0 - 0.5 ** k),
                                            -_GLUE_DOWN_MAX_DB, 0.0)
                            zpow_q = z_base * np.power(10.0, dyn_q[bidx] / 10.0)
                            l_q = -0.691 + 10.0 * np.log10(
                                np.maximum(_window_means(zpow_q, 30), 1e-16))
                            if _lra_from_short(l_q) <= lra_t + 0.5:
                                break
                        cur = _lra_from_short(l_q)
                        if best is None or abs(cur - lra_t) < best[0]:
                            best = (abs(cur - lra_t), dyn_q, zpow_q, l_q)
                        if cur <= lra_t + 0.5:
                            break
                    if best is not None:
                        dyn100, zpow, l = best[1], best[2], best[3]
            lra_after = _lra_from_short(l)
        else:
            bidx = np.arange(nb)

        z400p = z400 if zpow is z_base else _window_means(zpow, 4)
        lufs_mid = _gated_integrated(z400p)
        g_static = float(np.clip(target - lufs_mid, -24.0, 24.0))
        lufs_after = lufs_mid + g_static

        tpos = np.arange(nb) * base + base * 0.5
        gain_db = np.interp(np.arange(n), tpos, dyn100[bidx] + g_static)
        y = x.astype(np.float64) * np.power(10.0, gain_db / 20.0)[:, None]
    else:
        y = x.astype(np.float64)

    # ---- 3：真峰限制器
    env4 = _env_oversampled(y, sr)                                   # (4n,) float32
    req4 = np.minimum(np.float32(1.0), np.float32(ceil_lin) / np.maximum(env4, np.float32(1e-9)))
    la4 = max(4, int(round(0.005 * sr)) * 4)                         # 5ms 前瞻（4x 域）
    pre4 = _rolling_min(req4, la4)
    g1 = pre4.reshape(n, 4).min(axis=1)                              # 每个采样的 4 个子样取最严
    step = max(1, int(round(sr * 0.001)))
    ngs = n // step
    if ngs >= 2:
        gg = g1[:ngs * step].reshape(ngs, step).min(axis=1)          # 1ms 网格（取格内最小，保守）
        gg = _attack_release_smooth(gg.astype(np.float64), step / sr, 0.001, 0.080)
        gsm = np.interp(np.arange(n), np.arange(ngs) * step + step * 0.5, gg).astype(np.float32)
        g1 = np.minimum(g1, gsm)                                     # 平滑永不高于瞬时需求（硬保证）
    y *= g1[:, None]

    y32 = y.astype(np.float32)
    if not np.isfinite(y32).all():
        np.nan_to_num(y32, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        stats = ctx.setdefault("stats", {}) if isinstance(ctx, dict) else {}
        stats["m6"] = {
            "lufs_before": round(float(lufs_before), 2),
            "lufs_after": round(float(lufs_after), 2),
            "lra_before": round(float(lra_before), 2),
            "lra_after": round(float(lra_after), 2),
            "gain_db": round(float(g_static), 2),
        }
        if isinstance(ctx, dict):
            ctx.setdefault("log", []).append(
                "m6: LUFS %.1f->%.1f, LRA %.1f->%.1f, 静态增益 %+.1fdB, 天花板 %.1fdB"
                % (lufs_before, lufs_after, lra_before, lra_after, g_static, ceil_db))
    except Exception:
        pass
    return y32


def info():
    return {"key": KEY, "name": "母带响度",
            "algo": "BS.1770 K 加权 · 短时 glue 压缩 · 4x 过采样前瞻限制器"}
