"""LiveFix 处理引擎：装配处理链、分级路由、计时与统计。

链顺序（固定）：sep → ai → m5 → m1 → m2 → m3 → m4 → m6 → dyneq
分级路由：sep.on 且 routing=="graded" 时，m1 只作用于 backing，m3 只作用于 vocal，
m2 作用于两者后合回；其余模块作用于全轨。m6 不分段（需要全局响度）。
"""
from __future__ import annotations

import importlib
import time
from pathlib import Path

import numpy as np

from . import core

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "livefix_models"

CHAIN = ["sep", "ai", "m5", "m1", "m2", "m3", "m4", "m6", "dyneq"]

DEFAULT_CONFIG = {
    # ⚠ 本默认集是**用户冻结版**（2026-10-14）：逐项抄自用户在 UI 里调好的那一版
    #   （截图逐像素核对：m5/m1/m2/m3/m4 开关为深蓝=ON，m6 为浅灰蓝=OFF）。
    #   改动这里的任何数值都会改变「每次打开应用时的初始参数」——
    #   tests/test_engine.py::test_defaults_frozen 会逐项断言，误改会立刻测试失败。
    #
    # 历史：更早的默认值是拿 origin/ 三份基准（音源前/源1/调整1）的 96 点频谱距离
    # 反推出来的「保守修复」版，那套判据看不到响度、且目标是另一份混了外接麦的
    # 录音，结论偏保守。现在以用户实听调好的这一版为准。
    #
    # 截图可见的 6 个模块（sep / ai / dyneq 未出现在截图里，维持原值）：
    #   m5  降噪0.00 噪底门限-60dB 交流声0.20 市电50Hz 削波重建0.00 泵感0.00
    #   m1  重量感0.20 密度0.25 次谐波0.15 冲击力0.50
    #   m2  空气感0.45 细节0.50 嘶声抑制0.10 延伸到15100Hz
    #   m3  清晰度0.35 房间感0.50 刺耳控制0.00 齿音控制0.00
    #   m4  低频宽度0.56 中频宽度1.75 高频宽度1.75 纵深0.40 声像旋转0.00
    #   m6  目标响度-16.0LUFS 动态范围6LRA 峰值上限-1.0dB  —— 开关 OFF
    # 注：m6 关闭只影响处理链内的母带归一；导出时 exporter 仍会按 export.loudnorm
    #     做响度归一，所以「关掉 m6」不等于导出不归一。
    #
    # 每个模块都带 "mix" 干湿比（0..100，独立于模块自身参数）：
    #   out = (1-mix)·输入 + mix·处理结果，由引擎在段边界做线性混合。
    #   sep 特例：mix 混合的是四个 stem 增益（g_eff = 1-mix+mix·g），
    #   这样 mix=0 等于完全不分离、且与分级路由的 stems 保持一致。
    #   mix=0 时引擎直接跳过该模块（省时间，等效旁通）。
    "modules": {
        # sep 未出现在截图里 —— 维持此前校准结果：
        #   hpss 优于 nmf；low_gain 衰减是灾难（0.0 -> -4.855），必须保持 1.0；
        #   other 轨是房间反射与观众噪声，压低有收益（实测 other_gain=0.7 净贡献 +0.300）
        "sep": {"on": True, "mix": 100, "mode": "hpss", "routing": "graded",
                "vocal_gain": 1.0, "low_gain": 1.0, "perc_gain": 1.0, "other_gain": 0.6},
        "ai": {"on": False, "mix": 100, "model": "none", "focus": "low", "strength": 0.7},
        # 用户版 m5：只留交流声抑制。dereverb（场地去混响 / 中低频抑共振）默认 0，由用户
        # 按场地在预览中试听决定 —— 2026-10-14 起原 denoise + floor_db 已删除，位置由 dereverb 取代，
        # 因两者原来都是 0，默认听感不变（见 CHANGELOG.md）。
        "m5": {"on": True, "mix": 100, "dereverb": 0.0, "hum": 0.20,
               "mains": 50, "declip": 0.0, "pump": 0.0},
        # 用户版 m1：冲击力拉到 0.50；tame 未在 UI 暴露，保持 0（此前实测为净损失）
        # 新增 ring（低频混响衰减）/ dive（下潜重建，20-50Hz）：scan_m1_extra 在冻结基准上
        # 均为净负（频谱距离不感知嗡嗡声/下潜），故默认 0，由用户按场地在预览中试听决定。
        "m1": {"on": True, "mix": 100, "weight": 0.20, "density": 0.25, "sub": 0.15,
               "punch": 0.50, "tame": 0.0, "tame_f": 63.0, "tame_q": 2.0,
               "ring": 0.0, "dive": 0.0},
        # 用户版 m2：细节开到 0.50（相位已修正为真二次谐波）、延伸到 15100Hz
        "m2": {"on": True, "mix": 100, "air": 0.45, "detail": 0.50, "dehiss": 0.10,
               "ext_hz": 15100},
        # 用户版 m3：只留清晰度与房间感
        "m3": {"on": True, "mix": 100, "clarity": 0.35, "room": 0.50,
               "harsh": 0.0, "ess": 0.0},
        # 用户版 m4：宽度给得比较开（低频收窄到 0.56 保低频/mono，中高频 1.75 加宽）
        "m4": {"on": True, "mix": 100, "width_low": 0.56, "width_mid": 1.75,
               "width_high": 1.75, "depth": 0.40, "rotation": 0.0},
        # 用户版 m6：开关 OFF（截图逐像素核对为浅灰蓝轨道），参数保留
        "m6": {"on": False, "mix": 100, "target_lufs": -16, "lra": 6, "ceiling": -1.0},
        # dyneq 未出现在截图里 —— 维持关闭；thr_db 按实测检测电平刻度（3kHz 带中位约 -32dB）取 -30
        "dyneq": {"on": False, "mix": 100, "bands": [
            {"f": 3000, "q": 1.0, "gain_db": 0, "thr_db": -30, "ratio": 3.0,
             "range_db": 6, "mode": "down"}]},
    },
    "export": {"channels": "stereo", "bit_depth": 24, "loudnorm": True},
}

_BLOCKED = {"m6", "sep"}
# m6 需要全局响度统计；sep 内部已自带 stems 跨段拼接逻辑（_stems_push），
# 若引擎再套 block_apply 会造成双重分段、stems 被重复累加而超长；
# 且 HPSS 的滑动中值需要全轨上下文，分段会在段边界引入掩码不连续。
_cache: dict[str, object] = {}


def default_config():
    import copy

    return copy.deepcopy(DEFAULT_CONFIG)


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(cfg: dict | None) -> dict:
    """默认配置与用户配置深合并（用户缺省的字段用默认值）。"""
    base = default_config()
    if not cfg:
        return base
    merged = dict(base)
    merged["modules"] = {k: _merge(base["modules"].get(k, {}), cfg.get("modules", {}).get(k, {}))
                         for k in CHAIN}
    if "export" in cfg:
        merged["export"] = _merge(base["export"], cfg["export"])
    return merged


def _mod(key: str):
    """延迟加载 livefix.<key> 模块（开发期内核文件可能尚未写完）。"""
    mod = _cache.get(key)
    if mod is None:
        mod = importlib.import_module(f".{key}", __package__)
        _cache[key] = mod
    return mod


def list_models() -> list[str]:
    names = ["none"]
    if MODELS_DIR.is_dir():
        names += sorted(p.name for p in MODELS_DIR.glob("*.npz"))
    return names


def mix_of(m: dict) -> float:
    """模块干湿比 0..1（配置里是 0..100）。"""
    try:
        return max(0.0, min(100.0, float(m.get("mix", 100)))) / 100.0
    except (TypeError, ValueError):
        return 1.0


def is_active(mods: dict, key: str) -> bool:
    """该模块在本次处理中是否真正消耗算力（on 且 mix>0；ai 永远不消耗）。"""
    m = mods.get(key, {})
    if key == "ai":
        return False
    return bool(m.get("on")) and mix_of(m) > 0.0


def plan(mods: dict) -> list:
    """预计的计算计划 [(key, weight)]，weight≈该段的相对运行次数/成本权重。

    供 api 做 ETA 估算：分级路由下 m2/m5 各跑两趟（vocal+backing），weight=2。
    与 process() 的实际执行路径严格一致。
    """
    items = []
    graded = is_active(mods, "sep") and mods["sep"].get("routing") == "graded"
    if is_active(mods, "sep"):
        items.append(("sep", 1))
    if graded:
        if is_active(mods, "m5"):
            items.append(("m5", 2))
        for k, w in (("m1", 1), ("m2", 2), ("m3", 1)):
            if is_active(mods, k):
                items.append((k, w))
    else:
        for k in ("m5", "m1", "m2", "m3"):
            if is_active(mods, k):
                items.append((k, 1))
    for k in ("m4", "m6", "dyneq"):
        if is_active(mods, k):
            items.append((k, 1))
    return items


def process(x, sr, cfg: dict, seed: int = 20260930, progress_cb=None,
            capture_stems=False):
    """执行处理链。x: float32 (n,ch)。返回 (y, stages, ctx)。

    progress_cb(frac, stage_key, stage_frac)：frac=总进度 0..1（单调不减），
    stage_key=当前阶段，stage_frac=阶段内进度 0..1。回调异常不中断处理。

    capture_stems=True 时，ctx["chain_stems"] = {key: float16 (n,ch)}，
    存放每个模块的**贡献量** delta = out - in（同 key 多次运行按元素累加）。
    满足望远镜恒等式：输入 + Σ 全部 delta == 输出（f16 量化误差级）。
    调音台（UI 实时推子）用它做 out = 输入 + Σ g_k·delta_k，无需重跑处理链。
    仅在 api 判定窗口不太长（≤150s）时开启，控制内存。
    """
    cfg = load_config(cfg)
    mods = cfg["modules"]
    ctx = {"rng": np.random.default_rng(seed), "stats": {}, "log": [], "stems": None,
           "note": {}, "chain_stems": {} if capture_stems else None}
    stages = []
    y = np.asarray(x, np.float32)
    planned = plan(mods)
    total_w = float(sum(w for _, w in planned)) or 1.0
    state = {"done": 0.0}

    def _emit(key, sfrac, w):
        if progress_cb is None or w <= 0:
            return
        try:
            progress_cb(min(1.0, (state["done"] + w * float(sfrac)) / total_w), key, float(sfrac))
        except Exception:
            pass

    def _stage(key, on, ms, note=None, mix=100.0):
        # 分级路由下同一模块可能处理 vocal/backing 各一次，合并为一条（ms 累加），
        # 保证 UI 的阶段表恒为 9 行。
        for s in stages:
            if s["key"] == key:
                s["ms"] = round(s["ms"] + ms, 1)
                s["on"] = s["on"] or bool(on)
                if note:
                    s["note"] = note
                return
        st = {"key": key, "on": bool(on), "ms": round(ms, 1), "mix": round(mix)}
        if note:
            st["note"] = note
        stages.append(st)

    def run(key, sig, w=None):
        """执行一段。w=该次调用在进度计划里的权重（默认：激活为 1，否则 0）。"""
        m = mods[key]
        mix = mix_of(m)
        active = is_active(mods, key)
        if w is None:
            w = 1.0 if active else 0.0
        base_done = state["done"]
        t0 = time.perf_counter()
        note = None

        def _blk_cb(i, n):
            _emit(key, i / max(n, 1), w)

        if not m.get("on"):
            out = sig
        elif key == "ai":
            out = sig  # 接口预留：透传
            ctx["note"]["ai"] = "interface_only"
            note = "interface_only"
        elif mix <= 0.0:
            out = sig  # 干湿比 0：等效旁通，直接跳过计算
        else:
            mod = _mod(key)
            params = {k: v for k, v in m.items() if k not in ("on", "mix")}
            if key == "sep":
                # sep 的干湿比混合的是四个 stem 增益（dry=全 1=不分离），
                # 这样 mix=0 时分级路由的 stems 与输出严格一致（=原始信号）。
                for gk in ("vocal_gain", "low_gain", "perc_gain", "other_gain"):
                    g0 = float(params.get(gk, 1.0))
                    params[gk] = (1.0 - mix) + mix * g0
                _emit(key, 0.0, w)
                out = mod.process(sig, sr, params, ctx)  # sep 在 _BLOCKED（整轨单次 STFT）
            else:
                if key in _BLOCKED:
                    _emit(key, 0.0, w)
                    proc = mod.process(sig, sr, params, ctx)
                else:
                    proc = core.block_apply(sig, sr,
                                            lambda seg, _sr: mod.process(seg, _sr, params, ctx),
                                            block_s=60.0, xfade_s=1.0, cb=_blk_cb)
                if mix >= 1.0:
                    out = proc
                else:  # 干湿比线性混合（各段均为保相位处理，lerp 语义正确）
                    out = (sig.astype(np.float32) * np.float32(1.0 - mix)
                           + np.asarray(proc, np.float32) * np.float32(mix))
        # 贡献量捕获（调音台用）：delta = out - in，f16 省内存；同 key 多次运行累加
        if capture_stems and active and key != "ai" and ctx.get("chain_stems") is not None:
            d = np.asarray(out, np.float32) - np.asarray(sig, np.float32)
            cs = ctx["chain_stems"]
            if key in cs:
                cs[key] = cs[key] + d.astype(np.float16)
            else:
                cs[key] = d.astype(np.float16)
            del d
        ms = (time.perf_counter() - t0) * 1000.0
        _stage(key, m.get("on"), ms, note, mix * 100.0)
        state["done"] = base_done + w
        if w > 0 and progress_cb is not None:
            _emit(key, 1.0, 0.0)  # 阶段收尾（state 已推进，w=0 → 纯 done 占比）
        return np.asarray(out, np.float32)

    y = run("sep", y)
    y = run("ai", y)

    # 分级路由：sep 已把 vocal/backing 存入 ctx；分级下 m5/m1/m2/m3 按轨作用
    stems = ctx.get("stems") if is_active(mods, "sep") and mix_of(mods["sep"]) > 0 else None
    graded = bool(stems) and mods["sep"].get("routing") == "graded"

    if graded:
        vocal = stems["vocal"]
        backing = stems["backing"]
        # 修复：旧版 m5 在合轨上跑完即被 stems 覆盖（结果被静默丢弃）。
        # 分级语义下 m5 改为按轨执行，保住削波重建/降噪的效果。
        if is_active(mods, "m5"):
            vocal = run("m5", vocal, 1.0)
            backing = run("m5", backing, 1.0)
        else:
            _stage("m5", mods["m5"].get("on"), 0.0, mix=mix_of(mods["m5"]) * 100.0)
        if is_active(mods, "m1"):
            backing = run("m1", backing, 1.0)
        else:
            _stage("m1", mods["m1"].get("on"), 0.0, mix=mix_of(mods["m1"]) * 100.0)
        if is_active(mods, "m2"):
            vocal = run("m2", vocal, 1.0)
            backing = run("m2", backing, 1.0)
        else:
            _stage("m2", mods["m2"].get("on"), 0.0, mix=mix_of(mods["m2"]) * 100.0)
        if is_active(mods, "m3"):
            vocal = run("m3", vocal, 1.0)
        else:
            _stage("m3", mods["m3"].get("on"), 0.0, mix=mix_of(mods["m3"]) * 100.0)
        y = vocal + backing
    else:
        y = run("m5", y)
        y = run("m1", y)
        y = run("m2", y)
        y = run("m3", y)

    y = run("m4", y)
    y = run("m6", y)
    y = run("dyneq", y)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if progress_cb is not None:
        try:
            progress_cb(1.0, "done", 1.0)
        except Exception:
            pass
    return y, stages, ctx


# 保持链顺序的 stages 排序（分级路由下某些 key 会重复/乱序，这里统一）
def order_stages(stages):
    seen = {}
    for s in stages:
        seen[s["key"]] = s
    return [seen[k] for k in CHAIN if k in seen]
