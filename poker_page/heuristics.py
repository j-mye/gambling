"""Lightweight win / bluff proxies when ML models are unavailable (PyScript browser)."""

from __future__ import annotations

from typing import Any

from hand_eval import board_texture_risk, hand_strength_index, tuples_to_tokens
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

# Best-hand category index (hand_eval.COMBINATION_MAP) -> rough river equity vs one random hand.
_TIER_RIVER_EQUITY: dict[int, float] = {
    1: 0.46,
    2: 0.58,
    3: 0.66,
    4: 0.74,
    5: 0.80,
    6: 0.84,
    7: 0.92,
    8: 0.96,
    9: 0.99,
}


def _hole_rank_value(rank: str) -> float:
    return float(_RANK.get(str(rank).upper(), 0))


def _preflop_win_proxy(hole: list[tuple[str, str]], villains_active: int) -> float:
    nv = max(1, int(villains_active))
    hole_t = tuples_to_tokens(hole)
    if len(hole_t) < 2:
        return min(0.55, max(0.09, 1.0 / (1.0 + float(nv))))
    mult = 1.0 / (1.0 + 0.24 * max(0, nv - 1))
    hs = hand_strength_index(hole_t)
    r1, r2 = _hole_rank_value(hole[0][0]), _hole_rank_value(hole[1][0])
    high, low = max(r1, r2), min(r1, r2)
    if hs >= 2:
        pair_rank = r1 if r1 == r2 else high
        score = 0.48 + 0.024 * max(0.0, pair_rank - 2.0)
        return min(0.58, max(0.12, score * mult))
    score = 0.22 + 0.018 * (high + low) + (0.06 if high >= 12 else 0.0) + (0.04 if high == 14 else 0.0)
    if abs(r1 - r2) <= 3:
        score += 0.03
    return min(0.52, max(0.1, score * mult))


def hero_win_probability_proxy(
    hole: list[tuple[str, str]],
    board: list[tuple[str, str]],
    villains_active: int,
) -> float:
    """Win proxy from actual 7-card best hand, villain count, street, and board texture."""
    nv = max(1, int(villains_active))
    if len(hole) < 2:
        return min(0.55, max(0.08, 1.0 / (1.0 + float(nv))))

    hole_t = tuples_to_tokens(hole)
    board_t = tuples_to_tokens(board)
    if len(hole_t) < 2:
        return min(0.55, max(0.08, 1.0 / (1.0 + float(nv))))

    if not board_t:
        return _preflop_win_proxy(hole, nv)

    all_t = hole_t + board_t
    strength = hand_strength_index(all_t)
    river_eq = _TIER_RIVER_EQUITY.get(strength, 0.45)

    board_n = len(board_t)
    if board_n <= 2:
        street_factor = 0.74 + 0.08 * board_n
    elif board_n == 3:
        street_factor = 0.88
    elif board_n == 4:
        street_factor = 0.94
    else:
        street_factor = 1.0

    extra = max(0, nv - 1)
    if strength >= 8:
        mult = 1.0 / (1.0 + 0.14 * extra)
    elif strength >= 7:
        mult = 1.0 / (1.0 + 0.18 * extra)
    elif strength >= 5:
        mult = 1.0 / (1.0 + 0.3 * extra)
    else:
        mult = 1.0 / (1.0 + 0.48 * extra)

    risk = board_texture_risk(board_t)
    tex_scale = max(0.0, 1.0 - 0.1 * max(0, strength - 5))
    texture_mult = 1.0 - (0.34 * risk * tex_scale)

    board_only = hand_strength_index(board_t) if len(board_t) >= 3 else 1
    hole_lift = max(0.0, float(strength - board_only))
    kicker_nudge = min(0.08, 0.02 * hole_lift)

    nuts_bonus = 0.0
    if board_n >= 5 and strength >= 7:
        nuts_bonus = 0.06 + 0.04 * float(strength - 7)

    raw = river_eq * street_factor * mult * texture_mult + kicker_nudge + nuts_bonus
    return min(0.97, max(0.08, raw))


def bluff_probability_proxy(state: Any, seat: int, pot_total: float) -> float:
    """Observable-style bluff score from aggression vs pot (not a trained model)."""
    _, total_bet, raises = seat_commitment_and_raises(state, seat)
    pot = max(float(pot_total), 1.0)
    bet_ratio = min(float(total_bet) / pot, 4.0) / 4.0
    raise_part = min(float(raises), 5.0) / 5.0
    raw = 0.1 + 0.5 * bet_ratio + 0.32 * raise_part
    return min(0.9, max(0.05, raw))
