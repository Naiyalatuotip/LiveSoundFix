"""LiveFix 音频文件读写（纯标准库 + numpy）。

read_wav 支持 PCM 8/16/24/32 位与 IEEE float32/64，返回 float32 (n, ch)。
write_wav 支持 16/24 位整型与 32 位 IEEE float。
非 WAV 文件尝试用本机 ffmpeg 转码（若可用）。
"""
from __future__ import annotations

import hashlib
import os
import struct
import subprocess
from pathlib import Path

import numpy as np

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".mov", ".mp4", ".wma", ".ogg", ".opus"}


def read_wav(path):
    """-> (float32 (n,ch), sr)。抛出清晰的 ValueError/RuntimeError。"""
    raw = Path(path).read_bytes()
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"不是标准 WAV 文件：{path}")
    pos = 12
    fmt = None
    data = None
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        size = int.from_bytes(raw[pos + 4:pos + 8], "little")
        body_start = pos + 8
        if size == 0xFFFFFFFF or body_start + size > len(raw):
            body = raw[body_start:]
        else:
            body = raw[body_start:body_start + size]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            data = body
        pos = body_start + max(size, 0) + (size & 1)
        if size == 0xFFFFFFFF:
            break
    if fmt is None or len(fmt) < 16:
        raise ValueError(f"WAV 缺少 fmt 块：{path}")
    tag = int.from_bytes(fmt[0:2], "little")
    ch = int.from_bytes(fmt[2:4], "little")
    sr = int.from_bytes(fmt[4:8], "little")
    bits = int.from_bytes(fmt[14:16], "little")
    if tag == 0xFFFE:  # WAVE_FORMAT_EXTENSIBLE：子格式在偏移 24（GUID 前 2 字节）
        if len(fmt) >= 26:
            tag = int.from_bytes(fmt[24:26], "little")
        else:
            raise ValueError("WAV EXTENSIBLE 格式块过短")
    if data is None:
        raise ValueError(f"WAV 缺少 data 块：{path}")
    if ch < 1 or sr < 1:
        raise ValueError("WAV 参数非法")
    if tag == 1:  # PCM
        if bits == 16:
            a = np.frombuffer(data, "<i2").astype(np.float32) / 32768.0
        elif bits == 8:
            a = (np.frombuffer(data, np.uint8).astype(np.float32) - 128.0) / 128.0
        elif bits == 24:
            u = np.frombuffer(data, np.uint8)
            u = u[: (len(u) // 3) * 3].reshape(-1, 3).astype(np.int32)
            v = u[:, 0] + (u[:, 1] << 8) + (u[:, 2] << 16)
            v = np.where(v & 0x800000, v - 0x1000000, v)
            a = v.astype(np.float32) / 8388608.0
        elif bits == 32:
            a = np.frombuffer(data, "<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"不支持的 PCM 位深：{bits}")
    elif tag == 3:  # IEEE float
        if bits == 32:
            a = np.frombuffer(data, "<f4").astype(np.float32)
        elif bits == 64:
            a = np.frombuffer(data, "<f8").astype(np.float32)
        else:
            raise ValueError(f"不支持的 float 位深：{bits}")
    else:
        raise ValueError(f"不支持的 WAV 编码 tag={tag}（可用 ffmpeg 转为 PCM）")
    if ch > 1:
        a = a[: (len(a) // ch) * ch].reshape(-1, ch)
    else:
        a = a.reshape(-1, 1)
    if not np.all(np.isfinite(a)):
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.ascontiguousarray(a, np.float32), int(sr)


def write_wav(path, x, sr, bit_depth=24):
    """写 PCM16/24 或 IEEE float32（bit_depth=32）。x: (n,ch) 或 (n,)。"""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    ch = x.shape[1]
    interleaved = x.reshape(-1)
    if bit_depth == 16:
        body = (np.clip(interleaved, -1.0, 1.0 - 1 / 32768) * 32767.0).astype("<i2").tobytes()
        tag, bits = 1, 16
    elif bit_depth == 24:
        v = (np.clip(interleaved, -1.0, 1.0 - 1 / 8388608) * 8388607.0).astype("<i4")
        b0 = (v & 0xFF).astype(np.uint8)
        b1 = ((v >> 8) & 0xFF).astype(np.uint8)
        b2 = ((v >> 16) & 0xFF).astype(np.uint8)
        body = np.stack([b0, b1, b2], axis=1).tobytes()
        tag, bits = 1, 24
    elif bit_depth == 32:
        body = interleaved.astype("<f4").tobytes()
        tag, bits = 3, 32
    else:
        raise ValueError(f"不支持的位深：{bit_depth}")
    block_align = ch * bits // 8
    byte_rate = int(sr) * block_align
    fmt = struct.pack("<HHIIHH", tag, ch, int(sr), byte_rate, block_align, bits)
    hdr = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    dat = b"data" + struct.pack("<I", len(body)) + body
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b"RIFF" + struct.pack("<I", 4 + len(hdr) + len(dat)) + b"WAVE" + hdr + dat)
    return str(Path(path).resolve())


def transcode_tag(path):
    """转码缓存名后缀 = sha1(绝对路径|mtime_ns|size) 前 10 位。

    必须是**稳定**的：旧实现用 `abs(hash(path))`，而 Python 的 str hash 默认随机化
    （PYTHONHASHSEED），于是每次重启服务都会为同一个 mp3/mp4 重新转码一份新 wav。
    把 mtime/size 纳入签名，源文件被替换时会自动失效重转。
    """
    p = Path(path)
    try:
        st = p.stat()
        sig = f"{p.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode("utf-8")
    except OSError:  # pragma: no cover - stat 失败时退回仅按路径
        sig = str(p.resolve()).encode("utf-8")
    return hashlib.sha1(sig).hexdigest()[:10]


def ensure_readable(path, cache_dir=None):
    """非 WAV 时尝试用 ffmpeg 转码到 cache_dir；返回可 read_wav 的路径。

    转码结果是**持久复用**的：缓存名由 transcode_tag 生成（稳定、可复现），
    已存在就直接返回，不再调 ffmpeg。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    ext = p.suffix.lower()
    if ext == ".wav":
        try:
            read_wav(p)
            return str(p.resolve())
        except Exception as exc:  # noqa: BLE001 - 统一走 ffmpeg 兜底
            if shutil_which("ffmpeg") is None:
                raise RuntimeError(f"WAV 读取失败：{exc}") from exc
    out_dir = Path(cache_dir) if cache_dir else p.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{p.stem}_{transcode_tag(p)}.wav"
    if out.exists() and out.stat().st_size > 44:
        return str(out.resolve())          # 已经转码过：直接复用，不调 ffmpeg
    if shutil_which("ffmpeg") is None:
        raise RuntimeError(
            f"读取 {ext or '该格式'} 需要本机安装 ffmpeg（未在 PATH 中找到）；"
            "请安装 ffmpeg，或将音频先转成 WAV。"
        )
    cmd = [shutil_which("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(p), "-map", "0:a:0", "-c:a", "pcm_f32le", str(out)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"ffmpeg 转码失败：{exc}") from exc
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"ffmpeg 转码失败：{(r.stderr or '').strip()[:400]}")
    return str(out.resolve())


def shutil_which(name):
    import shutil

    return shutil.which(name)


def probe(path):
    """轻量读取头信息 -> {"sr","channels","duration","format"}。"""
    x, sr = read_wav(path)
    return {
        "sr": sr,
        "channels": int(x.shape[1]),
        "duration": round(float(x.shape[0]) / sr, 3),
        "format": "wav",
    }
