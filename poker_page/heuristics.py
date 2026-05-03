"""Lightweight win / bluff proxies when ML models are unavailable (PyScript browser)."""

from __future__ import annotations

from typing import Any

from live_seat_metrics import seat_commitment_and_raises

_RANK = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "10": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}


def _hole_rank_value(rank: str) -> float:
    return float(_RANK.get(str(rank).upper(), 0))


def hero_win_probability_proxy(
    hole: list[tuple[str, str]],
    board: list[tuple[str, str]],
    villains_active: int,
) -> float:
    """Rough [0,1] win chance from hole + board shape and number of live villains."""
    nv = max(1, int(villains_active))
    base = 1.0 / (1.0 + float(nv))
    if len(hole) < 2:
        return min(0.9, max(0.08, base))

    r1, r2 = _hole_rank_value(hole[0][0]), _hole_rank_value(hole[1][0])
    high, low = max(r1, r2), min(r1, r2)
    bonus = 0.0
    if r1 == r2:
        bonus += 0.14 + 0.004 * (r1 - 2.0)
    else:
        bonus += 0.0012 * (high + low) + (0.04 if high >= 12 else 0.0) + (0.025 if high == 14 else 0.0)
        if abs(r1 - r2) <= 2:
            bonus += 0.02

    all_ranks = [_hole_rank_value(c[0]) for c in hole] + [_hole_rank_value(c[0]) for c in board]
    if all_ranks:
        top3 = sum(sorted(all_ranks, reverse=True)[:3])
        made = 0.0015 * top3
        if len(board) >= 3:
            made *= 1.15
        bonus += min(0.22, made)

    return min(0.9, max(0.09, base + bonus))


def bluff_probability_proxy(state: Any, seat: int, pot_total: float) -> float:
    """Observable-style bluff score from aggression vs pot (not a trained model)."""
    _, total_bet, raises = seat_commitment_and_raises(state, seat)
    pot = max(float(pot_total), 1.0)
    bet_ratio = min(float(total_bet) / pot, 4.0) / 4.0
    raise_part = min(float(raises), 5.0) / 5.0
    raw = 0.1 + 0.5 * bet_ratio + 0.32 * raise_part
    return min(0.9, max(0.05, raw))
