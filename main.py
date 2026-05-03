"""Project entrypoint: static poker UI lives in poker_page/ (PyScript).

EDA and fold simulators previously hosted in Streamlit are not shipped in this
entrypoint. Use notebooks or standalone scripts under scripts/ for analysis.
"""

from __future__ import annotations

import argparse
import http.server
import os
import socketserver
import sys
import webbrowser
from pathlib import Path


class _ReusableTCPServer(socketserver.TCPServer):
    """Avoid WinError 10048 / EADDRINUSE when rebinding shortly after a prior run."""

    allow_reuse_address = True


def _serve_poker_page(host: str, port: int) -> None:
    root = Path(__file__).resolve().parent / "poker_page"
    if not root.is_dir():
        raise SystemExit(f"Missing poker_page directory: {root}")
    os.chdir(root)
    handler = http.server.SimpleHTTPRequestHandler
    try:
        httpd = _ReusableTCPServer((host, port), handler)
    except OSError as exc:
        hint = (
            f"Try another port: python main.py serve-poker --port {port + 1}\n"
            "This entrypoint is plain Python (static HTTP), not Streamlit — "
            "use: python main.py serve-poker"
        )
        raise SystemExit(
            f"Could not listen on {host!s}:{port} ({exc}).\n{hint}"
        ) from exc
    with httpd:
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
    if "streamlit" in sys.modules:
        raise SystemExit(
            "This file is not a Streamlit app — it starts a small static HTTP server.\n"
            "From the project root run:\n"
            "  python main.py serve-poker\n"
            "Then open the printed URL (default http://127.0.0.1:8765/index.html)."
        )
    main()
