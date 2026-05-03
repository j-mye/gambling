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
