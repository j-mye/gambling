"""Visualize heuristic bot decision regions (facing a bet, no random bluff).

Run from repo root::

    py -3.12 scripts/viz/bot_threshold_panel.py

Writes ``scripts/viz/bot_threshold_facing_bet.png`` (fold / call / raise map in
s vs pot_odds space, deterministic policy without BLUFF_RAISE_PROB).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "poker_page"))

from bot_backend import AGGRESSION_RAISE, WEAK_FOLD  # noqa: E402


def _action_facing_bet(s: float, po: float) -> int:
    """0 = fold, 1 = call, 2 = raise. Matches HeuristicAggressionBot without bluff or jitter."""
    if s < WEAK_FOLD:
        return 0
    if s >= AGGRESSION_RAISE:
        return 2
    if s > po:
        return 1
    return 0


def main() -> None:
    import matplotlib.pyplot as plt

    n = 240
    s_axis = np.linspace(0.0, 1.0, n)
    po_axis = np.linspace(0.0, 1.0, n)
    grid = np.zeros((n, n), dtype=np.int8)
    for i, s in enumerate(s_axis):
        for j, po in enumerate(po_axis):
            grid[i, j] = _action_facing_bet(float(s), float(po))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(
        grid,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=2,
    )
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Fold", "Call", "Raise"])
    ax.set_xlabel("Pot odds (call / (pot + call))")
    ax.set_ylabel("Normalized strength s")
    ax.axhline(WEAK_FOLD, color="w", linestyle="--", linewidth=0.8, alpha=0.7, label=f"WEAK_FOLD={WEAK_FOLD}")
    ax.axhline(AGGRESSION_RAISE, color="w", linestyle="-.", linewidth=0.8, alpha=0.7, label=f"AGGRESSION_RAISE={AGGRESSION_RAISE}")
    ax.set_title("Heuristic regions (facing bet, no bluff jitter)")
    ax.legend(loc="upper left", fontsize=8)
    out = Path(__file__).resolve().parent / "bot_threshold_facing_bet.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
