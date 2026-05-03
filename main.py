"""Project entrypoint: static poker UI lives in poker_page/ (PyScript).

EDA and fold simulators previously hosted in Streamlit are not shipped in this
entrypoint. Use notebooks or standalone scripts under scripts/ for analysis.
"""

from __future__ import annotations

import argparse
import http.server
import os
import socketserver
import webbrowser
from pathlib import Path


def _serve_poker_page(host: str, port: int) -> None:
    root = Path(__file__).resolve().parent / "poker_page"
    if not root.is_dir():
        raise SystemExit(f"Missing poker_page directory: {root}")
    os.chdir(root)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer((host, port), handler) as httpd:
        url = f"http://{host}:{port}/index.html"
        print(f"Serving {root} at {url}")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Gambling / poker tools")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_serve = sub.add_parser("serve-poker", help="HTTP-serve poker_page (PyScript UI)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.cmd == "serve-poker":
        _serve_poker_page(args.host, args.port)


if __name__ == "__main__":
    main()
