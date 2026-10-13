"""LiveFix 入口：启动本地服务器并打开浏览器。

用法：python run.py [--port 8765] [--no-browser]
"""
from __future__ import annotations

import argparse
import threading
import webbrowser


def main():
    ap = argparse.ArgumentParser(description="LiveFix · Livehouse 音频修复")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()

    import server  # 延迟导入：确保工作目录/路径已就绪

    url = f"http://127.0.0.1:{args.port}/"
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"LiveFix 已启动：{url}  （Ctrl+C 退出）")
    server.serve(port=args.port)


if __name__ == "__main__":
    main()
