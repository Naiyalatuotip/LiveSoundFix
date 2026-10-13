"""LiveFix 本地 HTTP 服务器：唯一职责是把 livefix.api 暴露成 HTTP 端点 + 托管 web/index.html。

- 只用标准库；livefix.api 在每个请求处理函数内部延迟导入（内核可能仍在开发中）。
- 业务异常一律包成 HTTP 200 + {"ok": false, "error": "..."}；服务器自身崩溃除外。
- /api/audio 只从 cache/ 目录取文件，id 白名单正则防目录穿越。
- 开发期钩子：环境变量 LIVEFIX_FAKE_API=<某 .py 绝对路径> 时，用 importlib 加载它替代 livefix.api。
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import mimetypes
import re
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
INDEX = ROOT / "web" / "index.html"
ID_RE = re.compile(r"^[po]_[a-z0-9]{4,16}$")
JOB_RE = re.compile(r"^job_[a-z0-9]{4,16}$")
RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")   # 只支持单区间；多区间/不合法一律回退 200
STREAM_CHUNK = 1 << 20

_api_lock = threading.Lock()
_api_mod = None  # 缓存已加载的 api 模块（真或假）


# ---------------------------------------------------------------- api 加载
def _load_api():
    """加载 livefix.api；设置了 LIVEFIX_FAKE_API 则加载假接口（仅联调）。"""
    fake = __import__("os").environ.get("LIVEFIX_FAKE_API")
    if fake:
        spec = importlib.util.spec_from_file_location("livefix_fake_api", fake)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    from livefix import api  # 延迟导入：请求时才触碰内核

    return api


def _api():
    """线程安全地取得 api 模块（首次请求时加载）。"""
    global _api_mod
    if _api_mod is None:
        with _api_lock:
            if _api_mod is None:
                _api_mod = _load_api()
    return _api_mod


# ---------------------------------------------------------------- 处理器
class Handler(BaseHTTPRequestHandler):
    server_version = "LiveFix/1.0"
    protocol_version = "HTTP/1.1"

    # ---- 基础发送 ----
    def _send(self, code: int, content_type: str, body: bytes = b"", extra=None,
              length=None, head_only: bool = False):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length",
                         str(len(body) if length is None else int(length)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if head_only or not body:
            return
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code: int = 200, head_only: bool = False):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(code, "application/json; charset=utf-8", body, head_only=head_only)

    def _err(self, exc, code: int = 200):
        """业务异常 → HTTP 200 + {"ok":false,...}。"""
        self._json({"ok": False, "error": f"{exc}"}, code)

    def _body(self):
        """读取并解析 POST 的 UTF-8 JSON 体；空体返回 {}。"""
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            raise ValueError(f"请求体不是合法 JSON：{e}") from e
        if not isinstance(data, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return data

    # ---- GET ----
    def do_GET(self):  # noqa: N802（http.server 命名约定）
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        try:
            if route == "/":
                self._serve_index()
            elif route == "/api/state":
                self._json(_api().state())
            elif route == "/api/models":
                self._json(_api().models())
            elif route == "/api/cache":
                # 分类占用统计（磁盘 cache/ + 内存 + 解码缓存）
                self._json({"ok": True, **_api().cache_info()})
            elif route == "/api/job":
                jid = urllib.parse.parse_qs(parsed.query).get("id", [""])[0]
                if not JOB_RE.match(jid or ""):
                    raise ValueError("非法 job id")
                self._json({"ok": True, **_api().job_status(jid)})
            elif route == "/api/audio":
                self._serve_audio(urllib.parse.parse_qs(parsed.query).get("id", [""])[0])
            else:
                self._json({"ok": False, "error": f"未知路径：{route}"}, 404)
        except Exception as e:  # noqa: BLE001——业务异常统一包装
            self._err(e)

    def do_HEAD(self):  # noqa: N802
        """HEAD：媒体元素（尤其 Safari）会先探测；不支持会拿到 501。只发头，不发体。"""
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        try:
            if route == "/":
                self._serve_index(head_only=True)
            elif route == "/api/audio":
                self._serve_audio(
                    urllib.parse.parse_qs(parsed.query).get("id", [""])[0], head_only=True)
            else:
                self._json({"ok": False, "error": f"未知路径：{route}"}, 404, head_only=True)
        except Exception as e:  # noqa: BLE001
            self._json({"ok": False, "error": f"{e}"}, 200, head_only=True)

    def _serve_index(self, head_only: bool = False):
        if not INDEX.is_file():
            self._json({"ok": False, "error": f"UI 文件不存在：{INDEX}"}, 500,
                       head_only=head_only)
            return
        ctype = mimetypes.guess_type(str(INDEX))[0] or "text/html"
        self._send(200, f"{ctype}; charset=utf-8", INDEX.read_bytes(), head_only=head_only)

    def _serve_audio(self, audio_id: str, head_only: bool = False):
        """只从 cache/ 取 wav；id 必须过白名单正则，防目录穿越。

        必须支持 HTTP Range（206 + Content-Range + Accept-Ranges）：浏览器换 <audio>.src
        时只缓冲很小一段，若服务端忽略 Range，赋 currentTime 会被静默夹回 0
        —— 这正是「A/B 切换后从头播放」的根因。按块流式发送，避免整文件读进内存。
        """
        if not ID_RE.match(audio_id or ""):
            self._json({"ok": False, "error": "非法 id"}, 400, head_only=head_only)
            return
        f = (CACHE / f"{audio_id}.wav").resolve()
        if f.parent != CACHE.resolve() or not f.is_file():
            self._json({"ok": False, "error": "音频不存在或已过期，请重新处理"}, 404,
                       head_only=head_only)
            return
        size = f.stat().st_size
        start, end, partial = 0, size - 1, False
        m = RANGE_RE.match((self.headers.get("Range") or "").strip())
        if size > 0 and m:
            g1, g2 = m.group(1), m.group(2)
            if g1 == "" and g2 == "":            # "bytes=-" 不合法 -> 回退整文件
                pass
            else:
                if g1 == "":                     # bytes=-N：末尾 N 字节
                    n = int(g2)
                    start = max(0, size - n) if n > 0 else size
                else:
                    start = int(g1)
                    if g2 != "":
                        end = min(int(g2), size - 1)
                if start >= size or start > end:
                    self._send(416, "application/json; charset=utf-8",
                               json.dumps({"ok": False, "error": "Range 不可满足"},
                                          ensure_ascii=False).encode("utf-8"),
                               {"Content-Range": f"bytes */{size}"}, head_only=head_only)
                    return
                partial = True
        n = 0 if size == 0 else end - start + 1
        extra = {"Accept-Ranges": "bytes"}
        if partial:
            extra["Content-Range"] = f"bytes {start}-{end}/{size}"
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(n))
        self.send_header("Cache-Control", "no-store")
        for k, v in extra.items():
            self.send_header(k, v)
        self.end_headers()
        if head_only or n <= 0:
            return
        try:
            with f.open("rb") as fh:
                fh.seek(start)
                left = n
                while left > 0:
                    chunk = fh.read(min(STREAM_CHUNK, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    # ---- POST ----
    def do_POST(self):  # noqa: N802
        route = urllib.parse.urlparse(self.path).path
        try:
            body = self._body()
        except ValueError as e:
            self._err(e, 400)
            return
        try:
            if route == "/api/list":
                self._json(_api().list_dir(body.get("dir")))
            elif route == "/api/analyze":
                path = body.get("path")
                if not path:
                    raise ValueError("缺少参数 path")
                self._json({"ok": True, **_api().analyze(path)})
            elif route == "/api/process":
                path, cfg = body.get("path"), body.get("config")
                if not path:
                    raise ValueError("缺少参数 path")
                # window 可选：[start_s, dur_s] 快速试听窗口；缺省 None = 完整文件
                self._json({"ok": True, **_api().process(path, cfg or {}, body.get("window"))})
            elif route == "/api/job":
                path, cfg = body.get("path"), body.get("config")
                if not path:
                    raise ValueError("缺少参数 path")
                # 立即返回 {"ok":True,"job":"job_xxx"}，后台线程执行
                self._json({"ok": True, **_api().start_job(path, cfg or {}, body.get("window"))})
            elif route == "/api/remix":
                job_id = body.get("job")
                if not job_id:
                    raise ValueError("缺少参数 job")
                # 调音台快速混音 -> {"ok":true,"id","url"}（不重跑引擎）
                self._json({"ok": True, **_api().remix(job_id, body.get("gains") or {})})
            elif route == "/api/clips":
                # 即时 A/B：并行切出「当前时刻起 N 秒」的双侧小片段
                self._json({"ok": True, **_api().clips(
                    body.get("id"), body.get("start") or 0.0, body.get("dur"),
                    orig_id=body.get("orig_id"), job_id=body.get("job"),
                    gains=body.get("gains"))})
            elif route == "/api/cache":
                # {"action":"clear"} 清空解码缓存 + 输入统计缓存
                # {"action":"clean"} 清理 cache/ 磁盘产物（默认 dry_run，只看不删）
                act = (body.get("action") or "").lower()
                if act == "clear":
                    _api().audio_cache_clear()
                    self._json({"ok": True, **_api().audio_cache_info()})
                elif act == "clean":
                    self._json(_api().cache_clean(
                        max_mb=body.get("max_mb"), max_age_days=body.get("max_age_days"),
                        keep_recent=body.get("keep_recent"),
                        include_other=bool(body.get("include_other", False)),
                        dry_run=bool(body.get("dry_run", True))))
                elif act == "info" or act == "":
                    self._json({"ok": True, **_api().cache_info()})
                else:
                    raise ValueError(f"未知 action：{act}（可用 clear / clean / info）")
            elif route == "/api/export":
                result = _api().export_wav(
                    body.get("id"),
                    body.get("channels", "stereo"),
                    body.get("bit_depth", 24),
                    bool(body.get("loudnorm", False)),
                    body.get("out_path"),
                    eq_bands=body.get("eq_bands"),
                    gains=body.get("gains"),
                    job_id=body.get("job"),
                )
                # 契约形状：{ok:true,...}（内核个别实现可能漏 ok，此处补全）
                if isinstance(result, dict) and "ok" not in result:
                    result = {"ok": True, **result}
                self._json(result)
            else:
                self._json({"ok": False, "error": f"未知路径：{route}"}, 404)
        except Exception as e:  # noqa: BLE001——业务异常统一包装
            self._err(e)

    # 安静一点的访问日志（保留错误信息）
    def log_message(self, fmt, *args):
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)


# ---------------------------------------------------------------- 启动
def serve(port: int = 8765):
    """启动阻塞式服务（run.py 调用入口）。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    ThreadingHTTPServer.allow_reuse_address = True
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"LiveFix server: http://127.0.0.1:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()


def main():
    ap = argparse.ArgumentParser(description="LiveFix 本地服务器")
    ap.add_argument("--port", type=int, default=8765)
    serve(ap.parse_args().port)


if __name__ == "__main__":
    main()
