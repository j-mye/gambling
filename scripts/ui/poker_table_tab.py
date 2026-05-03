"""Playable poker UI moved to poker_page/ (PyScript static app).

The Streamlit implementation was migrated to:
  - poker_page/poker_core.py — engine + view model
  - poker_page/main.py — DOM + async bot loop

Open poker_page/index.html via a static HTTP server (see repository README).
"""

from __future__ import annotations


def render_playable_poker_tab() -> None:
    raise RuntimeError(
        "Streamlit poker tab has been removed. Serve poker_page/index.html with PyScript "
        "(e.g. `python -m http.server` from the poker_page directory)."
    )
