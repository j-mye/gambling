"""Coach copy for the analytical footer (browser-safe stub)."""

from __future__ import annotations

from typing import Any


def coach_message(view: dict[str, Any]) -> str:
    """Return short coach text; extend with real GTO logic later."""
    if view.get("hand_complete"):
        return "Hand complete. Deal the next hand when ready."
    if not view.get("is_hero_turn"):
        return "Waiting for opponents."
    facing = view.get("facing_bet")
    if facing:
        return "Facing a bet: consider pot odds and your win probability before continuing."
    return "In position with no bet: checking preserves showdown value; betting builds the pot."


def gto_bar_html(view: dict[str, Any]) -> str:
    """Placeholder GTO distribution strip (fold / call / raise)."""
    wp = view.get("win_probability")
    if wp is None:
        fold_w, call_w, raise_w = 20, 50, 30
    else:
        fold_w = max(5, int((1.0 - float(wp)) * 35))
        raise_w = max(5, int(float(wp) * 40))
        call_w = 100 - fold_w - raise_w
    return (
        f'<div class="bg-red-500/60 h-full flex items-center justify-center border-r border-slate-900" '
        f'style="width:{fold_w}%">FOLD</div>'
        f'<div class="bg-slate-500/60 h-full flex items-center justify-center border-r border-slate-900" '
        f'style="width:{call_w}%">CALL</div>'
        f'<div class="bg-emerald-500/60 h-full flex items-center justify-center" style="width:{raise_w}%">RAISE</div>'
    )
