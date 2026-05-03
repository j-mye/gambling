"""PokerKit seat bet totals (no pandas; safe for PyScript)."""

from __future__ import annotations

from typing import Any


def seat_commitment_and_raises(state: Any, seat: int) -> tuple[float, float, int]:
    """Return (street_bet, total_hand_bet, raise_count) for a seat from PokerKit ops."""
    bets = list(getattr(state, "bets", []) or [])
    street_bet = float(bets[seat]) if seat < len(bets) else 0.0
    total = 0.0
    raises = 0
    for op in list(getattr(state, "operations", []) or []):
        if getattr(op, "player_index", None) != seat:
            continue
        op_name = type(op).__name__
        if op_name == "CompletionBettingOrRaisingTo":
            raises += 1
        amount = getattr(op, "amount", None)
        if amount is not None and op_name in {
            "AntePosting",
            "BlindOrStraddlePosting",
            "CheckingOrCalling",
            "CompletionBettingOrRaisingTo",
        }:
            total += float(amount)
    return street_bet, total, raises
