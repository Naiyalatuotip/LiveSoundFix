"""LiveFix 假接口（开发期联调 UI 管线用）。

用法：LIVEFIX_FAKE_API=E:/作品集/音频修复v1/tests/_fake_api.py python server.py
server.py 会用 importlib 加载本文件替代 livefix.api。只用标准库。
process 用 wave 在 cache/ 生成 3 秒正弦 wav，返回 spec/API.md §3 形状的假数据；
start_job 用后台线程模拟 frac 0→1 的进度（供 UI 轮询联调）。
"""
from __future__ import annotations

import copy
import math
import secrets
import struct
import threading
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".mov", ".mp4", ".wma", ".ogg", ".opus"}
CHAIN = ["sep", "ai", "m5", "m1", "m2", "m3", "m4", "m6", "dyneq"]

DEFAULT_CONFIG = {
    "modules": {
        "sep": {"on": False, "mix": 100, "mode": "hpss", "routing": "graded",
                "vocal_gain": 1.0, "low_gain": 1.0, "perc_gain": 1.0, "other_gain": 1.0},
        "ai": {"on": False, "mix": 100, "model": "none", "focus": "low", "strength": 0.7},
        "m5": {"on": True, "mix": 100, "dereverb": 0.5, "hum": 0.6,
               "mains": 50, "declip": 0.6, "pump": 0.4},
        "m1": {"on": True, "mix": 100, "weight": 0.6, "density": 0.5, "sub": 0.4, "punch": 0.5},
        "m2": {"on": True, "mix": 100, "air": 0.5, "detail": 0.5, "dehiss": 0.4, "ext_hz": 11000},
        "m3": {"on": True, "mix": 100, "clarity": 0.6, "room": 0.4, "harsh": 0.4, "ess": 0.4},
        "m4": {"on": True, "mix": 100, "width_low": 0.10, "width_mid": 1.20, "width_high": 1.50,
               "depth": 0.4, "rotation": 0.0},
        "m6": {"on": True, "mix": 100, "target_lufs": -16, "lra": 6, "ceiling": -1.0},
        "dyneq": {"on": False, "mix": 100, "bands": [
            {"f": 3000, "q": 1.0, "gain_db": 0, "thr_db": -24, "ratio": 3.0,
             "range_db": 6, "mode": "down"}]},
    },
    "export": {"channels": "stereo", "bit_depth": 24, "loudnorm": True},
}

_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def state():
    return {"version": "1.1.1-fake", "config": copy.deepcopy(DEFAULT_CONFIG),
            "models": ["none"], "chain": list(CHAIN)}


def models():
    names = ["none"]
    if (ROOT / "livefix_models").is_dir():
        names += sorted(p.name for p in (ROOT / "livefix_models").glob("*.npz"))
    return {"models": names}


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


def analyze(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")
    octaves = [(63, "63"), (125, "125"), (250, "250"), (500, "500"), (1000, "1k"),
               (2000, "2k"), (4000, "4k"), (8000, "8k"), (12000, "12k"),
               (14000, "14k"), (16000, "16k")]
    bands = [{"f": label, "db": round(-14 - 6 * math.log10(f / 63) + (i % 3) * 1.1, 1)}
             for i, (f, label) in enumerate(octaves)]
    return {
        "path": str(p), "sr": 48000, "channels": 2, "duration": 532.1,
        "diagnosis": {
            "lufs": -19.4, "lra": 9.2, "peak_db": -0.2, "clipped_pct": 0.31,
            "corr": 0.92, "noise_floor_db": -58.0, "cutoff_hz": 11200, "hum_hz": 50,
            "bands": bands,
            "issues": [
                {"level": "info", "text": "整体响度偏低，建议归一到 -16 LUFS"},
                {"level": "warn", "text": "检测到 50Hz 市电哼声残留"},
                {"level": "bad", "text": "峰值 -0.2 dBFS，存在轻微削波风险"},
            ],
            "suggest": {"m1": {"weight": 0.7}, "m4": {"width_mid": 1.3},
                        "m5": {"hum": 0.8}, "m6": {"target_lufs": -14}},
        },
    }


def _new_id(prefix):
    return f"{prefix}_{secrets.token_hex(4)}"


def _write_sine(path, seconds=3.0, sr=44100, freq=440.0, amp=0.5):
    n = int(seconds * sr)
    frames = bytearray()
    for i in range(n):
        v = int(amp * math.sin(2 * math.pi * freq * i / sr) * 32767)
        frames += struct.pack("<hh", v, v)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))


def process(path, cfg, window=None):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")
    CACHE.mkdir(parents=True, exist_ok=True)
    pid, oid = _new_id("p"), _new_id("o")
    _write_sine(CACHE / f"{pid}.wav")
    _write_sine(CACHE / f"{oid}.wav", amp=0.4)
    mods = (cfg or {}).get("modules", DEFAULT_CONFIG["modules"])
    stages = [{"key": k, "on": bool(mods.get(k, {}).get("on")),
               "ms": round(4.7 + i * 13.9, 1),
               "mix": round(float(mods.get(k, {}).get("mix", 100))),
               **({"note": "interface_only"} if k == "ai" else {})}
              for i, k in enumerate(CHAIN)]
    spec_before = [round(-3 - 34 * math.log10(1 + i * 0.55) + (i % 5) * 0.8, 2)
                   for i in range(96)]
    spec_after = [round(-1.5 - 30 * math.log10(1 + i * 0.5) + (i % 4) * 0.6, 2)
                  for i in range(96)]
    duration = round(float(window[1]), 3) if window else 3.0
    mixer = {"available": True,
             "keys": ["sep", "m5", "m1", "m2", "m3", "m4", "m6"],
             "job": "job_fake01"}
    return {
        "id": pid, "orig_id": oid,
        "url": f"/api/audio?id={pid}", "orig_url": f"/api/audio?id={oid}",
        "sr": 44100, "channels": 2, "duration": duration,
        "windowed": bool(window),
        "window": [round(float(window[0]), 2), round(float(window[1]), 2)] if window else None,
        "stats": {
            "lufs_in": -19.4, "lufs_out": -16.0, "peak_db": -1.1, "true_peak_db": -0.8,
            "corr_in": 0.92, "corr_out": 0.41,
            "stages": stages,
            "notes": {"ai": "interface_only"},
        },
        "spec_before": spec_before,
        "spec_after": spec_after,
        "mixer": mixer,
    }


# ---------------------------------------------------------------- 异步进度作业（模拟）
def start_job(path, cfg, window=None):
    """起后台线程，约 1.6s 内 frac 从 0 走到 1，再填入 process 的假结果。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")
    jid = "job_" + secrets.token_hex(3)
    job = {"id": jid, "state": "running", "frac": 0.0, "stage": "load",
           "stage_frac": 0.0, "eta_s": None, "error": None, "result": None,
           "t0": time.time(), "path": path, "window": window}
    with _lock:
        _jobs[jid] = job
        if len(_jobs) > 6:  # 只清已完成的，防膨胀
            for k in sorted(_jobs, key=lambda q: _jobs[q]["t0"]):
                if len(_jobs) <= 6:
                    break
                if _jobs[k]["state"] != "running":
                    _jobs.pop(k, None)

    stages_order = ["load", "sep", "ai", "m5", "m1", "m2", "m3", "m4", "m6", "dyneq", "stats"]
    total_s = 1.6

    def _worker():
        try:
            per = total_s / len(stages_order)
            for i, st in enumerate(stages_order):
                for k in range(4):  # 每阶段 4 步平滑推进
                    time.sleep(per / 4)
                    with _lock:
                        job["frac"] = round(min(0.99, (i + (k + 1) / 4) / len(stages_order)), 4)
                        job["stage"] = st
                        job["stage_frac"] = round((k + 1) / 4, 4)
                        job["eta_s"] = round(max(0.0, total_s - (time.time() - job["t0"])), 1)
            with _lock:
                job["result"] = process(path, cfg, window)
                # 让 mixer.job 指向真实作业 id（/api/remix 用）
                if isinstance(job["result"], dict) and isinstance(job["result"].get("mixer"), dict):
                    job["result"]["mixer"]["job"] = jid
                job["stage"] = "done"
                job["stage_frac"] = 1.0
                job["frac"] = 1.0
                job["eta_s"] = 0.0
                job["state"] = "done"
        except Exception as e:  # noqa: BLE001
            with _lock:
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


def remix(job_id, gains=None):
    """调音台假实现：校验作业存在后生成新 id（复制结果 wav），约 0.1s。"""
    _init_cache()
    with _lock:
        job = _jobs.get(str(job_id))
    if job is None:
        raise FileNotFoundError("作业不存在或已过期")
    if job.get("state") != "done":
        raise RuntimeError("渲染尚未完成，无法混音")
    res = job.get("result") or {}
    src = CACHE / f"{res.get('id', '')}.wav"
    rid = _new_id("p")
    dst = CACHE / f"{rid}.wav"
    if src.is_file():
        import shutil

        shutil.copyfile(src, dst)
    else:                                   # 兜底：生成正弦
        _write_sine(dst)
    _ = gains                               # 假混音不真正应用增益
    return {"id": rid, "url": f"/api/audio?id={rid}"}


def _init_cache():
    CACHE.mkdir(parents=True, exist_ok=True)


def export_wav(audio_id, channels="stereo", bit_depth=24, loudnorm=False, out_path=None,
               eq_bands=None, gains=None, job_id=None):
    """导出假实现：接受 eq_bands/gains/job_id 新参数；带 gains+job 时按作业结果导出。"""
    import shutil

    _init_cache()
    if gains and job_id:
        with _lock:
            job = _jobs.get(str(job_id))
        if job is None:
            raise FileNotFoundError("作业不存在或已过期，无法按调音台混音导出")
        res = job.get("result") or {}
        audio_id = res.get("id")
    if not audio_id or not __import__("re").match(r"^[po]_[a-z0-9]{4,16}$", str(audio_id)):
        raise ValueError("非法 id")
    src = CACHE / f"{audio_id}.wav"
    if not src.is_file():
        raise FileNotFoundError("音频不存在或已过期，请重新处理")
    out = Path(out_path) if out_path else CACHE / f"{audio_id}_{channels.replace('.', '')}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out)
    return {"ok": True, "path": str(out), "channels": 2, "duration": 3.0,
            "baked_eq": len(eq_bands or []), "applied_gains": bool(gains)}
