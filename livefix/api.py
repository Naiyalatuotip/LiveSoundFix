"""LiveFix API 胶水层：server.py 唯一依赖的业务接口。

按 spec/API.md §1 实现：state / list_dir / analyze / process / export_wav / models，
外加进度作业：start_job / job_status（process 的异步版，带 frac/eta）。
处理结果缓存在内存（供导出）并写入 cache/（供播放）。

进度与 ETA 模型：
  总成本 = 引擎成本(Σ 每段 ms/秒音频 × 时长 × 权重) + 开销(读文件 + LUFS/真峰/频谱统计)
  两部分各自有先验，并在每次作业完成后按实测 EWMA 自更新；
  eta = 剩余模型时间 × 实际/模型速率（消耗 >3s 后启用速率缩放）。
"""
from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from . import analysis, core, engine, measure, wavio

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
STEMS_MAX_S = 150.0  # 贡献量捕获上限（秒）；窗口更长则调音台不可用
CLIP_DUR_S = 3.0     # 即时 A/B 片段默认时长（秒）
CLIP_MAX_S = 10.0    # 即时 A/B 片段时长上限
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".mov", ".mp4", ".wma", ".ogg", ".opus"}
_ID_OK = __import__("re").compile(r"^[po]_[a-z0-9]{4,16}$")

_lock = threading.Lock()
# id -> (float32 (n,ch), sr)。**有界 LRU**：旧版是无上限 dict，长开服务必然涨爆内存。
# 淘汰只丢内存引用（磁盘 wav 还在），后续读取会自动回退到读盘，不会失败。
_audio: "OrderedDict[str, tuple]" = OrderedDict()
_AUDIO_MEM_MAX_BYTES = 1_000_000_000   # 内存里最多驻留约 1GB 音频
_AUDIO_MEM_KEEP_MIN = 6                # 但至少保留最近 6 条（当前预览/原片/remix 结果）
_jobs: dict[str, dict] = {}


def _audio_put(aid, arr, sr):
    """写入内存音频缓存并做 LRU 淘汰。不要持 _lock 调用（内部自己加锁）。"""
    with _lock:
        _audio[aid] = (arr, sr)
        _audio.move_to_end(aid)
        total = sum(int(v[0].nbytes) for v in _audio.values())
        while total > _AUDIO_MEM_MAX_BYTES and len(_audio) > _AUDIO_MEM_KEEP_MIN:
            _k, (v, _s) = _audio.popitem(last=False)
            total -= int(v.nbytes)
        return total


def audio_mem_info():
    with _lock:
        return {"entries": len(_audio),
                "bytes": sum(int(v[0].nbytes) for v in _audio.values()),
                "max_bytes": _AUDIO_MEM_MAX_BYTES,
                "keep_min": _AUDIO_MEM_KEEP_MIN}

# ---------------------------------------------------------------- ETA 成本模型
# 引擎先验（本机 judge_final.py，60s 素材）：毫秒 / 每秒音频；任务完成后 EWMA 自更新。
_MODEL = {"sep": 52.6, "m5": 1.5, "m1": 19.2, "m2": 49.4, "m3": 30.8,
          "m4": 37.5, "m6": 40.6, "dyneq": 10.0}
# 开销 = 读文件 + 统计（LUFS×2、真峰 4x 过采样、频谱×2），随音频时长线性增长。
# 实测 180s 素材：统计阶段 ≈ 24s ≈ 133ms/s；读 WAV ≈ 4ms/s。
_OH = {"rate": 138.0, "fixed": 1500.0}   # ms/s 音频, ms
_MODEL_LOCK = threading.Lock()

# 引擎完成后统计阶段的子步骤（用于把最后 10% 进度切细）
_STAT_STEPS = ("lufs_in", "lufs_out", "true_peak", "spec_before", "spec_after")


def _plan_cost_ms(cfg, dur_s):
    mods = cfg["modules"]
    try:
        items = engine.plan(mods)
    except Exception:
        items = []
    dur_s = max(float(dur_s), 1.0)
    with _MODEL_LOCK:
        total = _OH["fixed"] + _OH["rate"] * dur_s
        for key, w in items:
            total += _MODEL.get(key, 20.0) * dur_s * w
    return total


def _model_update(stages, dur_s, engine_ms, overhead_ms):
    if dur_s < 5:
        return
    with _MODEL_LOCK:
        for s in stages:
            k, ms = s.get("key"), float(s.get("ms", 0) or 0)
            if k in _MODEL and ms > 50:
                rate = ms / dur_s
                _MODEL[k] = max(0.2, min(500.0, 0.5 * _MODEL[k] + 0.5 * rate))
        if overhead_ms > 0:
            rate = overhead_ms / dur_s
            _OH["rate"] = max(20.0, min(500.0, 0.5 * _OH["rate"] + 0.5 * rate))


_inited = False


def _init():
    global _inited
    CACHE.mkdir(parents=True, exist_ok=True)
    if not _inited:
        _inited = True
        # 启动时按默认上限回收一次（cache/ 历史产物可能已经很大）
        cache_auto_clean(force=True)


def state():
    return {
        "version": "1.1",
        "config": engine.default_config(),
        "models": engine.list_models(),
        "chain": list(engine.CHAIN),
        "cache": audio_cache_info(),
    }


def models():
    return {"models": engine.list_models()}


def list_dir(path=None):
    p = Path(path) if path else ROOT
    try:
        p = p.resolve()
        if not p.is_dir():
            p = p.parent
    except OSError:
        p = ROOT
    dirs, files = [], []
    try:
        for c in sorted(p.iterdir(), key=lambda q: (not q.is_dir(), q.name.lower())):
            if c.is_dir():
                dirs.append({"name": c.name, "path": str(c)})
            elif c.suffix.lower() in AUDIO_EXT:
                files.append({"name": c.name, "path": str(c)})
    except PermissionError:
        pass
    parent = str(p.parent) if p.parent != p else None
    return {"dir": str(p), "parent": parent, "dirs": dirs, "files": files}


def _load(path):
    x, sr, _cached = _load_cached(path)
    return x, sr


# ---------------------------------------------------------------- 音频 LRU 缓存
# 同一文件反复处理（调参数、重算干湿比、来回跑作业）不再重新解码。
# 键 = (绝对路径, mtime_ns, size)：文件被改动会自然失效；按总字节上限做 LRU 淘汰。
_AUDIO_CACHE_MAX_BYTES = 1_500_000_000     # ~1.5GB（约 4 份 8 分钟立体声 48k）
_audio_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
_audio_cache_bytes = 0
_audio_cache_lock = threading.Lock()
_LOAD_STATS = {"hits": 0, "misses": 0, "read_ms": 0.0}


def _file_key(path):
    """文件身份：(绝对路径, mtime_ns, size)。文件不存在时返回 None。"""
    try:
        p = Path(path).resolve()
        st = p.stat()
        return (str(p), int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        return None


def _load_cached(path):
    """读文件 -> (float32 (n,ch), sr, 是否命中缓存)。命中时不再碰磁盘。"""
    global _audio_cache_bytes
    key = _file_key(path)
    if key is not None:
        with _audio_cache_lock:
            hit = _audio_cache.get(key)
            if hit is not None:
                _audio_cache.move_to_end(key)
                _LOAD_STATS["hits"] += 1
                return hit[0], hit[1], True
    t0 = time.time()
    wav = wavio.ensure_readable(path, cache_dir=CACHE)
    x, sr = wavio.read_wav(wav)
    _LOAD_STATS["read_ms"] = round((time.time() - t0) * 1000.0, 1)
    if key is not None:
        # 输入数组全程只读（各模块都是 X=stft(x) / x.copy() 的用法，不会原地改写），
        # 因此多个作业可以安全共享同一份数组。
        nbytes = int(x.nbytes)
        with _audio_cache_lock:
            _LOAD_STATS["misses"] += 1
            if nbytes <= _AUDIO_CACHE_MAX_BYTES:
                if key in _audio_cache:
                    _audio_cache_bytes -= int(_audio_cache[key][0].nbytes)
                _audio_cache[key] = (x, sr)
                _audio_cache.move_to_end(key)
                _audio_cache_bytes += nbytes
                while _audio_cache_bytes > _AUDIO_CACHE_MAX_BYTES and len(_audio_cache) > 1:
                    _k, (v, _s) = _audio_cache.popitem(last=False)
                    _audio_cache_bytes -= int(v.nbytes)
    return x, sr, False


def audio_cache_info():
    with _audio_cache_lock:
        return {"files": len(_audio_cache), "bytes": int(_audio_cache_bytes),
                "max_bytes": _AUDIO_CACHE_MAX_BYTES,
                "hits": int(_LOAD_STATS["hits"]), "misses": int(_LOAD_STATS["misses"]),
                "last_read_ms": _LOAD_STATS["read_ms"]}


def audio_cache_clear():
    """清空「解码缓存 + 输入统计缓存」（换素材/排障用；不碰磁盘上的 cache/ 产物）。"""
    global _audio_cache_bytes
    with _audio_cache_lock:
        _audio_cache.clear()
        _audio_cache_bytes = 0
    with _stats_lock:
        _stats_cache.clear()
    return {"ok": True}


# ---------------------------------------------------------------- cache/ 产物清理
# cache/ 里全是**可再生**中间产物：处理后预览 p_*、原片 o_*、贡献量 stems_*.npz、
# ffmpeg 转码 <stem>_<10hex>.wav。旧版只保留最近 3 份 stems，p_/o_ 只增不减
# （实测涨到 9.6GB / 467 文件）。这里按类统计 + 按容量/天数清理 + 自动回收。
_CACHE_AUTO_TRIGGER = 3_000_000_000     # 超过 3GB 时作业结束自动触发一次回收
_CACHE_DEFAULT_MAX_MB = 2048            # 目标上限 2GB
_CACHE_KEEP_RECENT = 8                  # 每个类别至少保留最新的 N 个（保护当前会话）
_CLIP_RE = __import__("re").compile(r"^[po]_[a-z0-9]{4,16}\.wav$")
_STEMS_RE = __import__("re").compile(r"^stems_job_[a-z0-9]{4,16}\.npz$")
_TRANSCODE_RE = __import__("re").compile(r"^.+_[0-9a-f]{10}\.wav$")

_CACHE_GROUPS = [
    ("preview", "处理后预览 p_*", True),
    ("original", "原片 o_*", True),
    ("stems", "贡献量 stems_*.npz", True),
    ("transcode", "ffmpeg 转码缓存", True),
    ("other", "其它（不自动删）", False),
]


def _cache_entries(cache_dir=None):
    """扫描 cache/，按类别归集候选文件 -> [(path, size, mtime, group)]。"""
    d = Path(cache_dir) if cache_dir else CACHE
    out = []
    if not d.is_dir():
        return out
    for p in d.iterdir():
        try:
            if not p.is_file():
                continue
            st = p.stat()
        except OSError:
            continue
        n = p.name
        if _STEMS_RE.match(n):
            g = "stems"
        elif _CLIP_RE.match(n):
            g = "preview" if n.startswith("p_") else "original"
        elif _TRANSCODE_RE.match(n):
            g = "transcode"
        else:
            g = "other"
        out.append((p, int(st.st_size), float(st.st_mtime), g))
    return out


def _protected_ids():
    """不允许被清理的 id：内存里驻留的 + 在跑/最近作业引用的。"""
    with _lock:
        ids = set(_audio.keys())
        for j in _jobs.values():
            for k in ("_orig_id",):
                if j.get(k):
                    ids.add(j[k])
            res = j.get("result") or {}
            for k in ("id", "orig_id"):
                if res.get(k):
                    ids.add(res[k])
        keep_files = {j["_stems_file"] for j in _jobs.values() if j.get("_stems_file")}
    return ids, keep_files


def _protected_paths(ents, keep_recent=None, allowed=None):
    """保护集（绝对路径字符串集合）：内存/作业引用的 + 每类最新 keep_recent 个。

    cache_info 与 cache_clean 共用同一套规则 —— 否则「可清理量」会虚高。
    """
    keep_recent = _CACHE_KEEP_RECENT if keep_recent is None else max(0, int(keep_recent))
    ids, keep_files = _protected_ids()
    prot = set()
    by_group = {}
    for e in ents:
        by_group.setdefault(e[3], []).append(e)
    for _g, lst in by_group.items():
        lst.sort(key=lambda e: e[2], reverse=True)
        for e in lst[:keep_recent]:
            prot.add(str(e[0]))
    for p, _s, _m, g in ents:
        if (allowed is not None and g not in allowed) or p.stem in ids or str(p) in keep_files:
            prot.add(str(p))
    return prot


def cache_info(cache_dir=None, keep_recent=None):
    """cache/ 占用分类统计 + 实际可清理量 + 内存/解码缓存状态（只读，不删任何东西）。"""
    ents = _cache_entries(cache_dir)
    allowed = {k for k, _l, dele in _CACHE_GROUPS if dele}
    prot = _protected_paths(ents, keep_recent, allowed)
    groups = {k: {"label": lbl, "deletable": dele, "files": 0, "bytes": 0,
                  "protected": 0, "reclaimable_files": 0, "reclaimable_bytes": 0}
              for k, lbl, dele in _CACHE_GROUPS}
    total_f = total_b = 0
    rec_f = rec_b = 0
    for p, size, _mt, g in ents:
        gi = groups[g]
        gi["files"] += 1
        gi["bytes"] += size
        total_f += 1
        total_b += size
        if str(p) in prot:
            gi["protected"] += 1
        elif gi["deletable"]:
            gi["reclaimable_files"] += 1
            gi["reclaimable_bytes"] += size
            rec_f += 1
            rec_b += size
    return {
        "dir": str(cache_dir or CACHE),
        "total": {"files": total_f, "bytes": total_b},
        "reclaimable": {"files": rec_f, "bytes": rec_b, "keep_recent": keep_recent
                        if keep_recent is not None else _CACHE_KEEP_RECENT},
        "groups": groups,
        "auto_trigger_bytes": _CACHE_AUTO_TRIGGER,
        "default_max_mb": _CACHE_DEFAULT_MAX_MB,
        "decode_cache": audio_cache_info(),
        "audio_memory": audio_mem_info(),
        "clip_ring": len(_clip_files),
    }


def cache_clean(max_mb=None, max_age_days=None, keep_recent=None, include_other=False,
                dry_run=True, cache_dir=None):
    """清理 cache/ 里的可再生产物。**默认 dry_run=True（只报告不删）。**

    规则（三条可叠加，先按天数、再按容量）：
      1. 保护：内存里驻留 / 在跑与最近作业引用的 id；每类最新 keep_recent 个；
      2. max_age_days 给定 -> 删掉比它更旧的；
      3. max_mb 给定（或取默认 2GB）-> 从最旧的开始删到总量达标。
    `include_other=False`（默认）时「其它」类别一律不动。
    只删 cache/ 目录内的**文件**，绝不动目录与匹配不到的文件。
    """
    d = (Path(cache_dir) if cache_dir else CACHE).resolve()
    if not d.is_dir():
        return {"ok": False, "error": f"cache 目录不存在：{d}"}
    ents = _cache_entries(d)
    keep_recent = _CACHE_KEEP_RECENT if keep_recent is None else max(0, int(keep_recent))
    if max_mb is None and max_age_days is None:
        max_mb = _CACHE_DEFAULT_MAX_MB
    budget = None if max_mb is None else max(0, int(float(max_mb) * 1e6))
    now = time.time()
    allowed = {k for k, _l, dele in _CACHE_GROUPS if dele}
    if include_other:
        allowed.add("other")

    # 与 cache_info 共用同一套保护规则
    prot = _protected_paths(ents, keep_recent, allowed)
    cand = [e for e in ents if str(e[0]) not in prot]
    cand.sort(key=lambda e: e[2])

    delete, freed = [], 0
    # 规则 2：按天数
    if max_age_days is not None:
        cutoff = now - float(max_age_days) * 86400.0
        for p, size, mt, _g in cand:
            if mt < cutoff:
                delete.append((p, size))
                freed += size
        done = {str(p) for p, _ in delete}
        cand = [e for e in cand if str(e[0]) not in done]
    # 规则 3：按容量（对剩余总量算，含保留下来的）
    if budget is not None:
        kept = sum(s for _p, s, _m, _g in ents if str(_p) not in {str(p) for p, _ in delete})
        for p, size, _mt, _g in cand:
            if kept <= budget:
                break
            delete.append((p, size))
            freed += size
            kept -= size

    if not dry_run:
        for p, _size in delete:
            try:
                # 双保险：解析后的父目录必须就是 cache/
                if p.resolve().parent == d:
                    p.unlink(missing_ok=True)
            except OSError:
                pass
        with _clip_lock:
            gone = {str(p) for p, _ in delete}
            _clip_files[:] = [f for f in _clip_files if f not in gone]

    per_group = {}
    for p, size in delete:
        # 用扫描时的分类回填
        g = next((e[3] for e in ents if e[0] == p), "other")
        cur = per_group.setdefault(g, {"files": 0, "bytes": 0})
        cur["files"] += 1
        cur["bytes"] += size
    return {
        "ok": True, "dry_run": bool(dry_run), "dir": str(d),
        "candidates": len(delete), "freed_bytes": freed,
        "deleted": len(delete) if not dry_run else 0,
        "per_group": per_group,
        "protected": len(prot),
        "remain_bytes": sum(s for _p, s, _m, _g in ents) - (0 if dry_run else freed),
        "params": {"max_mb": max_mb, "max_age_days": max_age_days,
                   "keep_recent": keep_recent, "include_other": bool(include_other)},
    }


def cache_auto_clean(force=False):
    """作业结束后调用：占用超阈值时按默认上限回收一次（幂等、不抛异常）。"""
    try:
        info = cache_info()
        if not force and info["total"]["bytes"] < _CACHE_AUTO_TRIGGER:
            return None
        return cache_clean(dry_run=False)
    except Exception:  # noqa: BLE001 - 清理失败绝不能影响主流程
        return None


# ---------------------------------------------------------------- 输入侧统计缓存
# lufs_in / corr_in / spec_before 只是输入的函数，同一 (文件, 窗口) 结果必然相同。
# 实测统计开销约 138ms/秒音频 —— 8 分钟素材每次重算要 ~70s，比读文件贵得多。
_STATS_MAX = 8
_stats_cache: "OrderedDict[tuple, dict]" = OrderedDict()
_stats_lock = threading.Lock()


def _stats_key(path, window, cfg=None):
    """统计只依赖音频本身（LUFS/相关度/频谱都与配置无关），故键里不放 cfg。"""
    fk = _file_key(path)
    if fk is None:
        return None
    w = (round(float(window[0]), 3), round(float(window[1]), 3)) if window else None
    return (fk, w)


def _stats_get(key):
    if key is None:
        return None
    with _stats_lock:
        v = _stats_cache.get(key)
        if v is not None:
            _stats_cache.move_to_end(key)
        return v


def _stats_put(key, val):
    if key is None:
        return
    with _stats_lock:
        _stats_cache[key] = val
        _stats_cache.move_to_end(key)
        while len(_stats_cache) > _STATS_MAX:
            _stats_cache.popitem(last=False)


def analyze(path):
    x, sr = _load(path)
    return analysis.analyze(x, sr)


def _new_id(prefix):
    return f"{prefix}_{secrets.token_hex(4)}"


def _slice_window(x, sr, window):
    if not window:
        return x
    s = max(0, int(float(window[0]) * sr))
    d = int(float(window[1]) * sr)
    if d <= 0:
        return x
    seg = x[s:s + d]
    return np.ascontiguousarray(seg) if seg.shape[0] else x


def _run_core(path, cfg, window=None, progress_cb=None, phase_cb=None, preload=None,
              capture=False, load_cached=None, s_key=None, pre_sliced=False):
    """读文件 -> 切窗 -> 跑链 -> 统计。返回 (result_dict, y, x_slice, sr, ctx)。

    phase_cb(kind, k, n)：阶段进度上报。kind="load"（读文件）或 "stats"（统计子步骤 k/n）。
    preload=(x, sr)：调用方已读好的音频，避免重复 IO。
    pre_sliced=True：preload 传进来的已经是切好的窗口，不要再切一次
      （旧版会二次切片：window=[30,120] 被切成 90s）。
    capture=True：捕获各模块贡献量（调音台用，内存见 STEMS_MAX_S）。
    load_cached/s_key：解码缓存命中标记与输入统计缓存键（同一文件重算时几乎零开销）。
    """
    def _ph(kind, k, n):
        if phase_cb is not None:
            try:
                phase_cb(kind, k, n)
            except Exception:
                pass

    _ph("load", 0, 1)
    if preload is not None:
        x, sr = preload
    else:
        x, sr, load_cached = _load_cached(path)
    if not pre_sliced:
        x = _slice_window(x, sr, window)
    dur = x.shape[0] / float(sr)
    y, stages, ctx = engine.process(x, sr, cfg, progress_cb=progress_cb,
                                    capture_stems=capture)
    stages = engine.order_stages(stages)
    pid, oid = _new_id("p"), _new_id("o")
    _audio_put(pid, y, sr)
    _audio_put(oid, x, sr)
    bits_out = _write_cache_wav(CACHE / f"{pid}.wav", y, sr)
    _write_cache_wav(CACHE / f"{oid}.wav", x, sr)

    # 输入侧统计（LUFS/相关度/频谱）只是输入的函数，同 (文件, 窗口) 直接复用。
    # ⚠ 顺序必须保持在 engine.process 之后：进度回调的 frac 必须单调不减。
    n = len(_STAT_STEPS)
    _ph("stats", 1, n)
    cached_stats = _stats_get(s_key)
    lufs_in = cached_stats["lufs_in"] if cached_stats else measure.lufs_integrated(x, sr)
    corr_in = cached_stats["corr_in"] if cached_stats else measure.corr(x)
    spec_before = cached_stats["spec_before"] if cached_stats else measure.spectrum_db(x, sr, 96)
    if cached_stats is None:
        _stats_put(s_key, {"lufs_in": lufs_in, "corr_in": corr_in, "spec_before": spec_before})

    _ph("stats", 2, n)
    lufs_out = measure.lufs_integrated(y, sr)
    _ph("stats", 3, n)
    tp_out = measure.true_peak_db(y, sr)
    _ph("stats", 4, n)
    spec_after = measure.spectrum_db(y, sr, 96)
    _ph("stats", 5, n)

    result = {
        "id": pid,
        "orig_id": oid,
        "url": f"/api/audio?id={pid}",
        "orig_url": f"/api/audio?id={oid}",
        "sr": int(sr),
        "channels": int(y.shape[1]),
        "duration": round(dur, 3),
        "windowed": bool(window),
        "window": [round(float(window[0]), 2), round(float(window[1]), 2)] if window else None,
        "stats": {
            "lufs_in": round(lufs_in, 2),
            "lufs_out": round(lufs_out, 2),
            "peak_db": round(measure.peak_db(y), 2),
            "true_peak_db": round(tp_out, 2),
            "corr_in": round(corr_in, 3),
            "corr_out": round(measure.corr(y), 3),
            "stages": stages,
            "notes": dict(ctx.get("note", {})),
        },
        "load": {
            "source_cached": bool(load_cached),
            "stats_cached": cached_stats is not None,
            "last_read_ms": _LOAD_STATS["read_ms"],
        },
        "spec_before": spec_before,
        "spec_after": spec_after,
    }
    stems = ctx.get("chain_stems") or {}
    if stems:
        result["mixer"] = {"available": True, "keys": list(stems.keys()), "job": None}
    else:
        result["mixer"] = {"available": False, "keys": [], "job": None,
                           "reason": f"未捕获贡献量（渲染时长需 ≤{int(STEMS_MAX_S)}s 才能启用调音台）"}
    return result, y, x, sr, ctx


def process(path, cfg, window=None, capture=False):
    """同步处理（测试与旧客户端用）。带进度请用 start_job/job_status。"""
    _init()
    result, _y, _x, _sr, _ctx = _run_core(path, cfg, window=window, capture=capture)
    return result


# ---------------------------------------------------------------- 异步进度作业

def start_job(path, cfg, window=None):
    """启动后台处理作业 -> {"job": id}。window=[start_s, dur_s] 为快速试听窗口。"""
    _init()
    jid = "job_" + secrets.token_hex(3)
    job = {"id": jid, "state": "running", "frac": 0.0, "stage": "load",
           "stage_frac": 0.0, "eta_s": None, "error": None, "result": None,
           "t0": time.time(), "path": path, "window": window,
           "_total_ms": None, "_eng_ms": None, "_oh_ms": None, "_dur_s": None,
           "_cons": 0.0, "_cur_key": None, "_cur_frac": 0.0,
           "_stats_k": 0, "_stats_n": len(_STAT_STEPS), "_stats_added": False}
    with _lock:
        _jobs[jid] = job
        # 只保留最近 8 个作业，防内存膨胀（不碰在跑的）
        if len(_jobs) > 8:
            for k in sorted(_jobs, key=lambda q: _jobs[q]["t0"]):
                if len(_jobs) <= 8:
                    break
                if _jobs[k]["state"] != "running":
                    _jobs.pop(k, None)
    cfg = engine.load_config(cfg)

    def _flush_stage():
        """把当前阶段按整段模型成本计入已消耗（每段费率不同，必须逐段计账）。"""
        k = job.get("_cur_key")
        if k and job.get("_dur_s"):
            job["_cons"] += _MODEL.get(k, 20.0) * float(job["_dur_s"])
        job["_cur_key"] = None
        job["_cur_frac"] = 0.0

    def _recompute_eta():
        total = job.get("_total_ms")
        if not total:
            job["eta_s"] = None
            return
        dur_s = float(job.get("_dur_s") or 300.0)
        cons = float(job["_cons"])
        if job.get("_cur_key"):
            cons += _MODEL.get(job["_cur_key"], 20.0) * dur_s * float(job["_cur_frac"])
        if job.get("_stats_k"):
            cons += float(job.get("_oh_ms") or 0.0) * (
                job["_stats_k"] / max(job["_stats_n"], 1))
        consumed = 1500.0 + cons  # 1500ms 固定起步（建线程/读文件头）
        remaining = max(0.0, total - consumed)
        elapsed = (time.time() - job["t0"]) * 1000.0
        if consumed > 3000:
            speed = min(5.0, max(0.2, elapsed / consumed))
            remaining *= speed
        job["eta_s"] = round(remaining / 1000.0, 1)

    def _engine_cb(frac, stage, sfrac):
        # 每个激活段是一次独立 run（w=1）；换段或 sfrac 回退 = 上一段已结束
        if stage == "done":
            _flush_stage()
        else:
            if stage != job.get("_cur_key") or float(sfrac) < float(job.get("_cur_frac", 0.0)) - 1e-9:
                if job.get("_cur_key") is not None:
                    _flush_stage()
                job["_cur_key"] = stage
                job["_cur_frac"] = 0.0
            job["_cur_frac"] = max(float(job.get("_cur_frac", 0.0)), float(sfrac))
        job["frac"] = round(0.05 + 0.85 * float(min(max(frac, 0.0), 1.0)), 4)
        job["stage"] = stage
        job["stage_frac"] = round(float(sfrac), 4)
        _recompute_eta()

    def _phase_cb(kind, k, n):
        if kind == "load":
            job["stage"] = "load"
            job["frac"] = 0.03
            job["stage_frac"] = 0.0
        elif kind == "stats":
            if not job.get("_stats_added"):
                _flush_stage()  # 引擎收尾：计入最后一个未结段
                job["_stats_added"] = True
            job["stage"] = "stats"
            job["_stats_k"], job["_stats_n"] = k, n
            job["stage_frac"] = round(k / max(n, 1), 3)
            job["frac"] = round(0.90 + 0.09 * k / max(n, 1), 4)
        _recompute_eta()

    def _worker():
        try:
            t_load0 = time.time()
            # 命中解码缓存时这里几乎是零成本（不再碰磁盘，也不再跑 ffmpeg）
            x0, sr0, src_cached = _load_cached(path)
            load_ms = (time.time() - t_load0) * 1000.0
            full_dur = float(job["window"][1]) if job.get("window") else x0.shape[0] / float(sr0)
            # 切窗后再算模型（窗口模式只处理窗口长度，进度/ETA 都按窗口时长计账）
            xw = _slice_window(x0, sr0, job.get("window"))
            eff_dur = xw.shape[0] / float(sr0)
            job["_dur_s"] = eff_dur
            job["_full_dur"] = full_dur
            job["_eng_ms"] = sum(_MODEL.get(k, 20.0) * eff_dur * w
                                 for k, w in engine.plan(cfg["modules"]))
            job["_oh_ms"] = _OH["fixed"] + _OH["rate"] * eff_dur
            job["_total_ms"] = 1500.0 + job["_eng_ms"] + job["_oh_ms"]
            _phase_cb("load", 0, 1)

            t_eng0 = time.time()
            capture = eff_dur <= STEMS_MAX_S
            result, _y, _x, _sr, ctx = _run_core(path, cfg, window=job.get("window"),
                                                 progress_cb=_engine_cb,
                                                 phase_cb=_phase_cb,
                                                 preload=(xw, sr0),
                                                 capture=capture,
                                                 load_cached=src_cached,
                                                 s_key=_stats_key(path, job.get("window")),
                                                 pre_sliced=True)
            # 贡献量落盘：{key: f16 (n,ch)}，供 /api/remix 快速混音（不重跑处理链）
            stems = ctx.get("chain_stems") or {}
            if stems:
                sf = CACHE / f"stems_{jid}.npz"
                np.savez(sf, **{k: np.asarray(v, np.float16) for k, v in stems.items()})
                job["_stems_file"] = str(sf)
                job["_orig_id"] = result["orig_id"]
                result["mixer"] = {"available": True, "keys": list(stems.keys()), "job": jid}
            else:
                result["mixer"]["job"] = jid
            _prune_stems()
            eng_and_stats_ms = (time.time() - t_eng0) * 1000.0
            # EWMA：统计开销 = 总时长 - 引擎实测（引擎实测从 stages 汇总）
            engine_actual = sum(float(s.get("ms", 0) or 0) for s in result["stats"]["stages"])
            overhead_actual = max(0.0, eng_and_stats_ms - engine_actual)
            _model_update(result["stats"]["stages"], eff_dur, engine_actual, overhead_actual)
            job["result"] = result
            job["state"] = "done"
            job["frac"] = 1.0
            job["stage"] = "done"
            job["eta_s"] = 0.0
            cache_auto_clean()      # 占用超阈值时按默认上限回收（失败不影响作业）
        except Exception as e:  # noqa: BLE001
            job["state"] = "error"
            job["error"] = f"{type(e).__name__}: {e}"
            job["eta_s"] = None

    threading.Thread(target=_worker, daemon=True).start()
    return {"job": jid}


def job_status(job_id):
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise FileNotFoundError("作业不存在或已过期")
        return {
            "job": job["id"],
            "state": job["state"],
            "frac": job["frac"],
            "stage": job["stage"],
            "stage_frac": job["stage_frac"],
            "eta_s": job["eta_s"],
            "error": job["error"],
            "result": job["result"],
        }


def _prune_stems():
    """贡献量文件最多保留最近 3 份（旧作业的 npz 删除并标记不可混音）。"""
    with _lock:
        holders = [j for j in _jobs.values() if j.get("_stems_file")]
        holders.sort(key=lambda j: j.get("t0", 0.0))
        excess = holders[:-3] if len(holders) > 3 else []
        for j in excess:
            sf = j.pop("_stems_file", None)
            j["_stems_pruned"] = True
            if sf:
                try:
                    Path(sf).unlink(missing_ok=True)
                except OSError:
                    pass


def _mix_array(job, gains):
    """调音台快速混音：x0 + Σ g_k·d_k（f16 贡献量）。不重跑处理链。"""
    sf = job.get("_stems_file")
    if not sf or not Path(sf).exists():
        raise RuntimeError("当前预览没有贡献量数据（渲染时长需 ≤150s），"
                           "请改用快速窗口重新渲染后再调调音台")
    orig = job.get("_orig_id")
    if not orig:
        raise RuntimeError("缺少原片缓存，无法混音")
    x0, sr = wavio.read_wav(CACHE / f"{orig}.wav")
    out = x0.astype(np.float32, copy=True)
    with np.load(sf) as data:
        for k in data.files:
            g = float((gains or {}).get(k, 1.0))
            if abs(g) < 1e-6:
                continue
            out += data[k].astype(np.float32) * np.float32(g)
    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out, sr


def remix(job_id, gains=None):
    """调音台：按各模块贡献量增益快速混音 -> {"id","url"}（约 0.3-0.8s，不重跑引擎）。"""
    _init()
    with _lock:
        job = _jobs.get(str(job_id))
    if job is None:
        raise FileNotFoundError("作业不存在或已过期")
    if job.get("state") != "done":
        raise RuntimeError("渲染尚未完成，无法混音")
    x, sr = _mix_array(job, gains)
    rid = _new_id("p")
    _audio_put(rid, x, sr)
    _write_cache_wav(CACHE / f"{rid}.wav", x, sr)
    return {"id": rid, "url": f"/api/audio?id={rid}"}


def _apply_eq(x, sr, eq_bands):
    """导出烘焙：静态 peaking EQ 级联（RBJ 复频响连乘 + overlap-save 精确滤波）。"""
    bands = []
    for b in (eq_bands or [])[:8]:
        try:
            g = float(b.get("gain_db", 0.0))
        except (TypeError, ValueError):
            continue
        if abs(g) > 0.05:
            bands.append(b)
    if not bands:
        return x
    N = 65536
    freqs = np.fft.rfftfreq(N, 1.0 / float(sr))
    w = 2.0 * np.pi * freqs / float(sr)
    z1 = np.exp(-1j * w)
    z2 = z1 * z1
    H = np.ones(N // 2 + 1, np.complex128)
    for b in bands:
        bb, aa = core.biquad_coeffs("peak", sr, float(b.get("f", 1000.0)),
                                    float(b.get("q", 1.0)), float(b.get("gain_db", 0.0)))
        H *= (bb[0] + bb[1] * z1 + bb[2] * z2) / (aa[0] + aa[1] * z1 + aa[2] * z2)
    return core.freq_filt(x, H, fft_n=N)


def _cache_bits(arr):
    """缓存/中转 wav 的位深：样本超出 ±1 时用 32bit float。

    链上没开限制器时（例如冻结默认把 m6 关掉了）输出完全可能过 0 dBFS，
    24bit 定点写入会 `np.clip` 掉峰值 —— 那不只是试听失真：调音台的
    「原片 + Σ贡献量 == 处理后」恒等式、以及 /api/clips 的「片段 == 整轨同区间」
    都会因此破裂（实测被削 18 个样本、偏差 0.10）。
    float32 WAV（WAVE_FORMAT_IEEE_FLOAT）浏览器可直接播放。
    """
    return 32 if float(np.max(np.abs(arr))) > 1.0 else 24


def _write_cache_wav(path, arr, sr):
    """写预览/中转缓存 wav，返回实际用的位深。"""
    bits = _cache_bits(arr)
    wavio.write_wav(path, arr, sr, bits)
    return bits


def audio_bytes(audio_id):
    """供 /api/audio 直接读取缓存 wav 文件路径。"""
    _init()
    if not audio_id or not _ID_OK.match(audio_id):
        raise ValueError("非法 id")
    f = CACHE / f"{audio_id}.wav"
    if not f.exists():
        raise FileNotFoundError("音频不存在或已过期，请重新处理")
    return f


# ---------------------------------------------------------------- 即时 A/B 片段
# 设计：UI 的 A/B 对比不再靠「切换整轨 src + seek」（145MB WAV 换 src 后浏览器
# 只能缓冲很小一段，赋 currentTime 会被夹回 0 = 从头播放）。改为服务端切出
# 「当前播放位置起 N 秒」的双侧小片段（~0.9MB），前端用 AudioBuffer 双路并行
# 播放、靠增益切换实现零延迟无缝 A/B。片段来自已渲染缓存，不需要重跑 DSP。
_CLIP_KEEP = 24                 # 保留的片段数（LRU，防 cache 膨胀）
_clip_lock = threading.Lock()
_clip_files: list[str] = []
_POOL = None                    # 常驻进程池（多核并行准备双侧片段）
_POOL_FAILED = False


def _clip_worker(job):
    """子进程任务：写一个片段文件。必须是模块级函数才能被 pickle。"""
    arr, sr, path, bits = job
    wavio.write_wav(path, arr, sr, bits)
    return str(path)


def _pool():
    global _POOL, _POOL_FAILED
    if _POOL is None and not _POOL_FAILED:
        try:
            from concurrent.futures import ProcessPoolExecutor
            _POOL = ProcessPoolExecutor(max_workers=2)
        except Exception:  # noqa: BLE001 - 任何失败都退回串行
            _POOL_FAILED = True
    return _POOL


def _prune_clips():
    with _clip_lock:
        while len(_clip_files) > _CLIP_KEEP:
            p = _clip_files.pop(0)
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass


def _edge_fade(a: np.ndarray, sr: int, fade_s: float = 0.005) -> np.ndarray:
    """片段两端做 5ms 淡入淡出（避免切点爆音）。"""
    k = min(int(fade_s * sr), a.shape[0] // 2)
    if k > 1:
        r = np.linspace(0.0, 1.0, k, dtype=np.float32)[:, None]
        a[:k] *= r
        a[-k:] *= r[::-1]
    return a


def _slice_at(x: np.ndarray, sr: int, start: float, dur: float):
    s = max(0, int(round(start * sr)))
    e = min(x.shape[0], s + int(round(dur * sr)))
    if e <= s:
        raise ValueError("请求的时间位置超出音频长度")
    return _edge_fade(np.ascontiguousarray(x[s:e]), sr)


def clips(audio_id, start=0.0, dur=CLIP_DUR_S, orig_id=None, job_id=None, gains=None):
    """即时 A/B：切出 [start, start+dur] 的「处理后」与「原片」双侧片段。

    返回 {"start","dur","sr","a":{id,url},"b":{id,url},"parallel":bool}。
    两侧的写盘用常驻进程池并行（多核）；池不可用时自动串行回退。
    """
    global _POOL, _POOL_FAILED
    _init()
    dur = float(min(max(float(dur or CLIP_DUR_S), 0.5), CLIP_MAX_S))
    start = max(0.0, float(start or 0.0))

    a_arr = b_arr = None
    sr = None
    # 原片
    if orig_id and _ID_OK.match(str(orig_id)):
        with _lock:
            ent = _audio.get(str(orig_id))
        if ent is not None:
            b_arr, sr = ent[0], ent[1]
        else:
            b_arr, sr = wavio.read_wav(CACHE / f"{orig_id}.wav")
    elif job_id:
        with _lock:
            job = _jobs.get(str(job_id))
        oid = (job or {}).get("_orig_id")
        if oid and (CACHE / f"{oid}.wav").exists():
            b_arr, sr = wavio.read_wav(CACHE / f"{oid}.wav")
    # 处理后（带调音台增益时按混音结果切）
    if gains and job_id:
        with _lock:
            job = _jobs.get(str(job_id))
        if job is None:
            raise FileNotFoundError("作业不存在或已过期")
        a_arr, sr = _mix_array(job, gains)
    elif audio_id and _ID_OK.match(str(audio_id)):
        with _lock:
            ent = _audio.get(str(audio_id))
        if ent is not None:
            a_arr, sr = ent[0], ent[1]
        else:
            a_arr, sr = wavio.read_wav(CACHE / f"{audio_id}.wav")
    if a_arr is None or sr is None:
        raise ValueError("缺少处理后音频（需要 audio_id 或 gains+job）")
    if b_arr is None:
        b_arr = a_arr

    seg_a = _slice_at(a_arr, sr, start, dur)
    seg_b = _slice_at(b_arr, sr, start, dur)

    pa = CACHE / f"p_{secrets.token_hex(4)}.wav"
    pb = CACHE / f"p_{secrets.token_hex(4)}.wav"
    # 位深与整轨缓存保持同一规则：超 ±1 用 32bit float，否则 24bit。
    # 否则「片段 == 整轨同区间」会在过载素材上失效（一侧被削顶、一侧没有）。
    jobs = [(seg_a, sr, pa, _cache_bits(seg_a)), (seg_b, sr, pb, _cache_bits(seg_b))]
    parallel = False
    pool = _pool()
    if pool is not None:
        try:
            list(pool.map(_clip_worker, jobs))
            parallel = True
        except Exception:  # noqa: BLE001 - 池坏了就永久退回串行
            _POOL_FAILED = True
            try:
                pool.shutdown(wait=False)
            except Exception:  # noqa: BLE001
                pass
            _POOL = None
    if not parallel:
        for j in jobs:
            _clip_worker(j)

    with _clip_lock:
        _clip_files.extend([str(pa), str(pb)])
    _prune_clips()
    aid, bid = pa.stem, pb.stem
    return {"start": round(start, 3), "dur": round(dur, 3), "sr": int(sr),
            "samples": int(seg_a.shape[0]), "parallel": parallel,
            "a": {"id": aid, "url": f"/api/audio?id={aid}"},
            "b": {"id": bid, "url": f"/api/audio?id={bid}"}}


def export_wav(audio_id, channels="stereo", bit_depth=24, loudnorm=False, out_path=None,
               eq_bands=None, gains=None, job_id=None):
    """导出。可选烘焙：调音台增益（需 job 的贡献量）与实时 EQ（静态 peaking 级联）。"""
    _init()
    from . import exporter  # 延迟导入（exporter 由功能子代理实现）

    x = sr = None
    if gains and job_id:
        with _lock:
            job = _jobs.get(str(job_id))
        if job is None:
            raise FileNotFoundError("作业不存在或已过期，无法按调音台混音导出")
        x, sr = _mix_array(job, gains)
    elif audio_id and _ID_OK.match(str(audio_id)):
        with _lock:
            entry = _audio.get(str(audio_id))
        if entry is None:
            f = CACHE / f"{audio_id}.wav"
            if not f.exists():
                raise FileNotFoundError("音频不存在或已过期，请重新处理")
            x, sr = wavio.read_wav(f)
        else:
            x, sr = entry
    else:
        raise ValueError("非法 id")

    if eq_bands:
        x = _apply_eq(x, sr, eq_bands)
    if not out_path:
        suffix = audio_id or "mix"
        out_path = str(CACHE / f"{suffix}_{channels.replace('.', '')}.wav")
    return exporter.export(x, sr, out_path, channels=channels, bit_depth=int(bit_depth),
                           loudnorm=bool(loudnorm))
