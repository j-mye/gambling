"""Observable-at-table features for bluff probability (no opponent hole cards).

Uses community cards, stacks, pot, seat bets / aggression — same information a live player
could write down without peeking at villain's hand.
"""

from __future__ import annotations

import pandas as pd

from scripts.features.poker_hand_strength import build_stage_feature_payload, parse_cards as pk_parse_cards


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        v = float(x)
        return default if v != v else v  # nan
    except Exception:
        return default


def _stage_from_board_row(row: pd.Series) -> str:
    br = row.get("board_river")
    bt = row.get("board_turn")
    bf = row.get("board_flop")
    if br not in (None, "", "0", 0) and str(br).strip() not in {"--", "nan"}:
        return "river"
    if bt not in (None, "", "0", 0) and str(bt).strip() not in {"--", "nan"}:
        return "turn"
    if bf not in (None, "", "0", 0) and str(bf).strip() not in {"--", "nan"}:
        return "flop"
    return "preflop"


def _parse_board_chain(row: pd.Series, stage: str) -> list[str]:
    parts: list[str] = []
    if stage in {"flop", "turn", "river"}:
        parts.extend(pk_parse_cards(row.get("board_flop")))
    if stage in {"turn", "river"}:
        parts.extend(pk_parse_cards(row.get("board_turn")))
    if stage == "river":
        parts.extend(pk_parse_cards(row.get("board_river")))
    return parts


def _raises_in_actions(row: pd.Series, stage: str) -> int:
    texts: list[str] = [str(row.get("action_pre", ""))]
    if stage in {"flop", "turn", "river"}:
        texts.append(str(row.get("action_flop", "")))
    if stage in {"turn", "river"}:
        texts.append(str(row.get("action_turn", "")))
    if stage == "river":
        texts.append(str(row.get("action_river", "")))
    joined = " ".join(texts).lower()
    return joined.count("raise")


def _bet_total_for_stage(row: pd.Series, stage: str) -> float:
    pre = _safe_float(row.get("bet_pre"))
    fl = _safe_float(row.get("bet_flop"))
    tu = _safe_float(row.get("bet_turn"))
    ri = _safe_float(row.get("bet_river"))
    if stage == "preflop":
        return pre
    if stage == "flop":
        return pre + fl
    if stage == "turn":
        return pre + fl + tu
    return pre + fl + tu + ri


def _pot_for_stage(row: pd.Series, stage: str) -> float:
    if stage == "preflop":
        return _safe_float(row.get("pot_pre"))
    if stage == "flop":
        return _safe_float(row.get("pot_flop")) or _safe_float(row.get("pot_pre"))
    if stage == "turn":
        return _safe_float(row.get("pot_turn")) or _safe_float(row.get("pot_flop"))
    return _safe_float(row.get("pot_river")) or _safe_float(row.get("pot_turn"))


def vector_from_csv_row(row: pd.Series) -> dict[str, float]:
    """Training-only: build observable vector from one cleaned/hand-history row."""
    stage = _stage_from_board_row(row)
    board = _parse_board_chain(row, stage)
    blind_text = str(row.get("blinds", "1/2"))
    bb = 2.0
    if "/" in blind_text:
        try:
            bb = float(blind_text.split("/")[-1].strip())
        except ValueError:
            bb = 2.0
    stack = _safe_float(row.get("stack"))
    ts = [stack] * 6
    total_bet = _bet_total_for_stage(row, stage)
    pot = _pot_for_stage(row, stage)
    pos = str(row.get("position", ""))
    payload = build_stage_feature_payload(
        stage,
        [],
        board,
        total_bet=total_bet,
        current_pot=pot,
        position=pos,
        hero_stack=stack,
        table_stacks=ts,
        big_blind=bb,
    )
    rc = _raises_in_actions(row, stage)
    bet_to_pot = total_bet / max(pot, 1.0)
    phase = {"preflop": 0.0, "flop": 0.33, "turn": 0.67, "river": 1.0}[stage]
    keys = VISIBLE_BLUFF_FEATURE_KEYS
    vec = {k: float(payload.get(k, 0.0)) for k in keys if k in payload}
    vec["seat_bet_to_pot"] = float(min(bet_to_pot, 6.0))
    vec["seat_raise_count"] = float(min(rc, 12)) / 12.0
    vec["street_phase"] = float(phase)
    return vec


def vector_from_live_state(
    *,
    stage: str,
    board_cards: list[tuple[str, str]],
    seat_stack: float,
    seat_total_bet: float,
    pot_total: float,
    position_token: str,
    table_stacks: list[float],
    big_blind: float,
    seat_raise_count: int,
) -> dict[str, float]:
    """Inference: same contract without hole cards."""
    board_tokens = [f"{r}{s.lower()}" for r, s in board_cards]
    payload = build_stage_feature_payload(
        stage,
        [],
        board_tokens,
        total_bet=float(seat_total_bet),
        current_pot=float(pot_total),
        position=str(position_token),
        hero_stack=float(seat_stack),
        table_stacks=table_stacks,
        big_blind=float(big_blind),
    )
    bet_to_pot = float(seat_total_bet) / max(float(pot_total), 1.0)
    phase = {"preflop": 0.0, "flop": 0.33, "turn": 0.67, "river": 1.0}[stage.lower()]
    keys = VISIBLE_BLUFF_FEATURE_KEYS
    vec = {k: float(payload.get(k, 0.0)) for k in keys if k in payload}
    vec["seat_bet_to_pot"] = float(min(bet_to_pot, 6.0))
    vec["seat_raise_count"] = float(min(seat_raise_count, 12)) / 12.0
    vec["street_phase"] = float(phase)
    return vec


# Keys pulled from build_stage_feature_payload (hole_cards=[] → broadcast-strong fields only)
VISIBLE_BLUFF_FEATURE_KEYS: tuple[str, ...] = (
    "position_value",
    "hand_strength",
    "pair_count",
    "trips_count",
    "flush_draw",
    "open_ended_straight_draw",
    "gutshot_draw",
    "board_high_rank",
    "board_is_paired",
    "board_four_flush",
    "board_straight_present",
    "board_straight_4liner",
    "board_pair_count",
    "board_trips_present",
    "board_full_house_present",
    "board_shared_strength_risk",
    "board_only_strength",
    "hero_stack_bb",
    "effective_stack_bb",
    "spr",
)

VISIBLE_BLUFF_FEATURES: list[str] = list(VISIBLE_BLUFF_FEATURE_KEYS) + [
    "seat_bet_to_pot",
    "seat_raise_count",
    "street_phase",
]


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
