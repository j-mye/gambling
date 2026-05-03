"""Session schema helpers for legacy notebooks (no Streamlit)."""

from __future__ import annotations

import random
import uuid
from typing import Any

POSITIONS_6MAX = ["Dealer", "Small Blind", "Big Blind", "UTG", "Hijack", "Cutoff"]
RANKS = list("23456789TJQKA")
SUITS = list("SHDC")


def _draw_card(used: set[str]) -> tuple[str, str]:
    while True:
        c = f"{random.choice(RANKS)}{random.choice(SUITS)}"
        if c not in used:
            used.add(c)
            rank = "10" if c[0] == "T" else c[0]
            return rank, c[1]


def init_hand_state(
    hand: dict[str, Any],
    *,
    seat_count: int = 6,
    starting_stack: int = 200,
) -> None:
    """Populate hand dict for hand lifecycle (mutates `hand` in place)."""
    hero_seat = random.randint(0, seat_count - 1)
    used: set[str] = set()
    hero_cards = [_draw_card(used), _draw_card(used)]
    villain_cards = {
        seat: [_draw_card(used), _draw_card(used)] for seat in range(seat_count) if seat != hero_seat
    }
    hand.clear()
    hand.update(
        {
            "game_initialized": True,
            "hand_id": str(uuid.uuid4())[:8],
            "hero_seat": hero_seat,
            "turn_index": hero_seat,
            "seat_count": seat_count,
            "button_seat": random.randint(0, seat_count - 1),
            "seat_stacks": {seat: float(starting_stack) for seat in range(seat_count)},
            "seat_folded": {seat: False for seat in range(seat_count)},
            "seat_street_bet": {seat: 0.0 for seat in range(seat_count)},
            "seat_total_bet": {seat: 0.0 for seat in range(seat_count)},
            "last_action_by_seat": {seat: "" for seat in range(seat_count)},
            "pot_total": 0.0,
            "board_cards": [],
            "hero_cards": hero_cards,
            "villain_cards": villain_cards,
            "active_street": "Preflop",
            "action_log": [],
            "pending_actor": hero_seat,
            "hand_complete": False,
            "last_action": "Hand initialized",
            "bot_action_feed": [],
            "newly_dealt_count": 0,
            "animation_tick": 0,
            "board_deal_stage": "preflop",
            "round_start_delay_consumed": {
                "Preflop": False,
                "Flop": False,
                "Turn": False,
                "River": False,
            },
            "showdown_persisted": False,
            "payoffs_by_seat": {seat: 0.0 for seat in range(seat_count)},
            "final_stacks_by_seat": {seat: float(starting_stack) for seat in range(seat_count)},
            "hand_end_reason": "",
            "legal_snapshot": {},
            "engine_payload": {},
            "seat_map_ui_to_engine": {i: i for i in range(seat_count)},
        }
    )


def reset_hand(hand: dict[str, Any]) -> None:
    init_hand_state(hand, seat_count=int(hand.get("seat_count", 6)))


def hero_position_label(hand: dict[str, Any]) -> str:
    idx = (hand["hero_seat"] - hand["button_seat"]) % hand["seat_count"]
    return POSITIONS_6MAX[idx] if hand["seat_count"] == 6 else f"Seat {hand['hero_seat'] + 1}"
