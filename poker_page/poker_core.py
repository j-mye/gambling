"""Playable Poker tab.

Architecture (read this before touching anything):
  PILLAR 1 – Setup (run once per session)
      _ensure_initialized() creates the PokerKit state with maximum automations and
      a shuffled session deck.  The Dealer is called immediately so hole cards are
      dealt before the first render.

  PILLAR 2 – The Dealer (_run_dealer)
      One function, one job: give PokerKit cards whenever it asks for them.
      It handles can_deal_hole and can_deal_board.  Nothing else.

  PILLAR 3 – Strict top-to-bottom execution order inside render_playable_poker_tab:
      Step 1  Process the hero's pending action (if any was queued by a button click).
      Step 2  Run the Dealer (clears any deal requests triggered by the hero's action).
      Step 2b Pre-fill table + disabled hero before the bot loop (rigid DOM before time.sleep).
      Step 3  Run bots + Dealer in alternation until it is the hero's turn or the hand ends.
      Step 4  Render – read-only; zero engine mutations below this line.
"""

from __future__ import annotations

import random
import re
from typing import Any

from pokerkit import Automation, NoLimitTexasHoldem

from bot_backend import BasePokerBot, build_bot_instances, normalize_bot_response
from cards_html import render_face_down_card, render_face_up_card
from game_state import GameState

# ── Constants ────────────────────────────────────────────────────────────────

_AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    # CARD_BURNING intentionally absent – the Dealer handles burns from our deck
    # so the session deck and PokerKit's internal deck stay in sync.
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    # HOLE_DEALING and BOARD_DEALING are intentionally absent so the Dealer
    # function controls exactly which cards go where.
)
_N_PLAYERS = 6
_STARTING_STACK = 200
_BLINDS = (1, 2)
_MIN_BET = 2


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 1 – SETUP
# ══════════════════════════════════════════════════════════════════════════════


def _normalize_bankrolls(raw: Any) -> list[int]:
    """Return a safe bankroll list sized to _N_PLAYERS."""
    base = [_STARTING_STACK] * _N_PLAYERS
    if not isinstance(raw, (list, tuple)):
        return base

    out: list[int] = []
    for i in range(_N_PLAYERS):
        if i < len(raw):
            try:
                out.append(int(round(float(raw[i]))))
            except Exception:
                out.append(base[i])
        else:
            out.append(base[i])

    # Prevent invalid negative stack values from poisoning new hands.
    return [max(0, v) for v in out]


def _blinds_vector_from_button(button_index: int, player_count: int) -> tuple[int, ...]:
    """Build per-seat blind vector for static seats + moving button."""
    blinds = [0] * player_count
    if player_count <= 0:
        return tuple(blinds)
    sb_seat = (button_index + 1) % player_count
    bb_seat = (button_index + 2) % player_count
    blinds[sb_seat] = int(_BLINDS[0])
    blinds[bb_seat] = int(_BLINDS[1])
    return tuple(blinds)

def _deck_from_state(state: Any) -> list[str]:
    """Extract PokerKit's pre-shuffled internal deck as our session deck.

    Using PokerKit's own deck guarantees that when the Dealer burns a card it
    never conflicts with any card PokerKit is tracking internally.
    """
    out: list[str] = []
    for card in state.deck_cards:          # deque, leftmost = bottom
        s = str(card)
        m = re.search(r"\((\w+)\)", s)
        out.append(m.group(1) if m else s.strip())
    return out                             # pop() from right = top of deck


def _new_hand(gs: GameState, *, advance_button: bool) -> None:
    """Create a fresh game and deal hole cards."""
    bankrolls = _normalize_bankrolls(
        gs.master_bankrolls or [_STARTING_STACK] * _N_PLAYERS
    )
    gs.master_bankrolls = bankrolls
    button_index = int(gs.button_index) % _N_PLAYERS
    if advance_button:
        button_index = (button_index + 1) % _N_PLAYERS
    gs.button_index = button_index
    blinds_vector = _blinds_vector_from_button(button_index, _N_PLAYERS)

    state = NoLimitTexasHoldem.create_state(
        _AUTOMATIONS,
        True,                           # uniform_antes
        0,                              # raw_antes
        blinds_vector,                  # raw_blinds_or_straddles
        _MIN_BET,                       # min_bet
        tuple(bankrolls),
        _N_PLAYERS,
    )
    gs.poker_state = state
    gs.deck = _deck_from_state(state)
    gs.last_action = {}
    gs.terminal_payload = None
    gs.action_error = ""
    gs.bot_error = ""
    gs.hand_resolved = False
    gs.resolved_hand_id = None
    gs.bot_instances = build_bot_instances(_N_PLAYERS, int(gs.hero))
    _run_dealer(gs)


def _ensure_initialized(gs: GameState) -> None:
    gs.master_bankrolls = _normalize_bankrolls(
        gs.master_bankrolls or [_STARTING_STACK] * _N_PLAYERS
    )
    if gs.hero is None:
        gs.hero = random.randrange(_N_PLAYERS)
    gs.bot_error = gs.bot_error or ""
    if gs.poker_state is None:
        _new_hand(gs, advance_button=False)
    if not gs.bot_instances:
        gs.bot_instances = build_bot_instances(_N_PLAYERS, int(gs.hero))


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 2 – THE DEALER
# ══════════════════════════════════════════════════════════════════════════════

def _pop(gs: GameState, n: int) -> list[str]:
    """Pop n cards from the session deck.  Returns [] if the deck is too short."""
    deck = gs.deck
    if len(deck) < n:
        return []
    return [deck.pop() for _ in range(n)]


def _run_dealer(gs: GameState) -> None:
    """Give PokerKit cards when it asks.  Does nothing else – ever."""
    state = gs.poker_state
    if state is None:
        return
    for _ in range(30):
        if state.can_deal_hole():
            cards = _pop(gs, 2)
            if not cards:
                break
            state.deal_hole(cards[0] + cards[1])
        elif state.can_burn_card():
            card = _pop(gs, 1)
            if not card:
                break
            state.burn_card(card[0])
        elif state.can_deal_board():
            n = 3 if not state.board_cards else 1
            cards = _pop(gs, n)
            if not cards:
                break
            state.deal_board("".join(cards))
        else:
            break


# ══════════════════════════════════════════════════════════════════════════════
# BOT TURN (sync; async sleeps live in main.py)
# ══════════════════════════════════════════════════════════════════════════════


def _extract_bot_hole_cards(state: Any, seat: int) -> list[str]:
    """Return active bot seat hole cards as normalized strings."""
    try:
        seat_cards = list((state.hole_cards or [])[seat] or [])
    except Exception:
        return []
    cards: list[str] = []
    for card in seat_cards:
        parsed = _parse_card(card)
        if parsed:
            rank, suit = parsed
            cards.append(f"{rank}{suit}")
    return cards


def _legal_moves_snapshot(state: Any, seat: int) -> dict[str, Any]:
    """Capture legal move surface for the active bot seat."""
    call_amount = 0
    min_raise_to: int | None = None
    max_raise_to: int | None = None
    try:
        call_amount = int(getattr(state, "checking_or_calling_amount", 0) or 0)
    except Exception:
        call_amount = 0
    try:
        val = getattr(state, "min_completion_betting_or_raising_to_amount", None)
        min_raise_to = int(val) if val is not None else None
    except Exception:
        min_raise_to = None
    try:
        val = getattr(state, "max_completion_betting_or_raising_to_amount", None)
        max_raise_to = int(val) if val is not None else None
    except Exception:
        max_raise_to = None
    return {
        "seat": seat,
        "turn_index": getattr(state, "turn_index", None),
        "facing_bet": call_amount > 0,
        "call_amount": call_amount,
        "can_fold": bool(state.can_fold()),
        "can_check_or_call": bool(state.can_check_or_call()),
        "can_raise": min_raise_to is not None and max_raise_to is not None,
        "min_raise_to": min_raise_to,
        "max_raise_to": max_raise_to,
    }


def _execute_bot_action(
    state: Any,
    seat: int,
    action: str,
    amount: int,
    legal_moves: dict[str, Any],
) -> str:
    """Execute validated bot action against PokerKit state and return action label."""
    if action == "FOLD":
        if state.can_fold():
            state.fold()
            return "Fold"
        raise ValueError("Illegal FOLD")

    if action in {"CALL", "CHECK"}:
        if state.can_check_or_call():
            call_amount = int(legal_moves.get("call_amount", 0) or 0)
            state.check_or_call()
            return f"Call ${call_amount}" if call_amount > 0 else "Check"
        raise ValueError("Illegal CHECK/CALL")

    if action in {"RAISE", "BET"}:
        min_raise_to = legal_moves.get("min_raise_to")
        max_raise_to = legal_moves.get("max_raise_to")
        if min_raise_to is None or max_raise_to is None:
            raise ValueError("Raise bounds unavailable")
        target = int(amount)
        if target < int(min_raise_to) or target > int(max_raise_to):
            raise ValueError("Raise amount out of bounds")
        if state.can_complete_bet_or_raise_to(target):
            state.complete_bet_or_raise_to(target)
            facing = bool(legal_moves.get("facing_bet", False))
            return f"Raise ${target}" if facing else f"Bet ${target}"
        raise ValueError("Illegal RAISE/BET")

    raise ValueError(f"Unsupported action {action}")


def _force_safe_bot_fallback(state: Any) -> str:
    """Non-crashing fallback if bot returns illegal/hallucinated output."""
    if state.can_fold():
        state.fold()
        return "Fold"
    if state.can_check_or_call():
        call_amount = int(getattr(state, "checking_or_calling_amount", 0) or 0)
        state.check_or_call()
        return f"Call ${call_amount}" if call_amount > 0 else "Check"
    return "NoOp"


def run_one_bot_turn(gs: GameState) -> None:
    """Execute one bot action when it is a non-hero seat's turn."""
    state = gs.poker_state
    if state is None:
        return
    hero = int(gs.hero)
    ti = state.turn_index
    if ti is None or ti == hero:
        return
    legal_moves = _legal_moves_snapshot(state, ti)
    hole_cards = _extract_bot_hole_cards(state, ti)
    bots: dict[int, BasePokerBot] = gs.bot_instances
    bot = bots.get(ti)
    if bot is None:
        gs.bot_instances = build_bot_instances(_N_PLAYERS, hero)
        bot = gs.bot_instances.get(ti)
    try:
        if bot is None:
            raise ValueError("Missing bot instance")
        raw_response = bot.calculate_action(state, hole_cards, legal_moves)
        action, amount = normalize_bot_response(raw_response)
        label = _execute_bot_action(state, ti, action, amount, legal_moves)
        gs.last_action[ti] = label
    except Exception as exc:
        gs.bot_error = f"Seat {ti + 1} hallucinated illegal move: {exc}"
        fallback_label = _force_safe_bot_fallback(state)
        gs.last_action[ti] = fallback_label


def _apply_hero_action(gs: GameState, action: dict[str, Any]) -> None:
    """Execute the hero's action against the live PokerKit state."""
    state = gs.poker_state
    hero = int(gs.hero)
    if state is None:
        return
    gs.action_error = ""
    try:
        kind = action["type"]
        if kind == "fold":
            if state.can_fold():
                state.fold()
                gs.last_action[hero] = "Fold"
            else:
                gs.action_error = "Cannot fold right now."

        elif kind == "call":
            if state.can_check_or_call():
                facing = state.checking_or_calling_amount > 0
                label = (
                    f"Call ${state.checking_or_calling_amount}" if facing else "Check"
                )
                state.check_or_call()
                gs.last_action[hero] = label
            else:
                gs.action_error = "Cannot check/call right now."

        elif kind == "raise":
            amount = int(action["amount"])
            if state.can_complete_bet_or_raise_to(amount):
                facing = state.checking_or_calling_amount > 0
                label = f"Raise ${amount}" if facing else f"Bet ${amount}"
                state.complete_bet_or_raise_to(amount)
                gs.last_action[hero] = label
            else:
                gs.action_error = f"${amount} is not a legal raise size."
    except Exception as exc:
        gs.action_error = str(exc)


# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL STATE
# ══════════════════════════════════════════════════════════════════════════════

def _capture_terminal(gs: GameState) -> None:
    """Persist payoffs once when the hand ends."""
    state = gs.poker_state
    if state is None:
        return
    hand_id = id(state)

    try:
        payoffs = list(state.payoffs or [])
        stacks = list(state.stacks or [])
    except Exception:
        payoffs, stacks = [], []

    already_resolved = bool(gs.hand_resolved)
    resolved_hand_id = gs.resolved_hand_id
    if (not already_resolved) and (resolved_hand_id != hand_id):
        bankrolls = _normalize_bankrolls(
            gs.master_bankrolls or [_STARTING_STACK] * _N_PLAYERS
        )
        updated = bankrolls[:]
        for i in range(_N_PLAYERS):
            delta = 0.0
            if i < len(payoffs):
                try:
                    delta = float(payoffs[i] or 0)
                except Exception:
                    delta = 0.0
            updated[i] = max(0, int(round(updated[i] + delta)))
        gs.master_bankrolls = updated
        gs.hand_resolved = True
        gs.resolved_hand_id = hand_id

    if gs.terminal_payload is not None:
        return
    gs.terminal_payload = {"payoffs": payoffs, "stacks": stacks}


# ══════════════════════════════════════════════════════════════════════════════
# VIEW MODEL  (read-only – no mutations)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_card(card_obj: Any) -> tuple[str, str] | None:
    """Convert a card-like value into normalized (rank, suit) tuple.

    Accepts PokerKit Card objects and card strings in either verbose form
    ('ACE OF SPADES (As)') or compact token form ('As', '10h'), including
    noisy wrappers like list/tuple string repr fragments.
    """
    s = str(card_obj or "")
    token: str | None = None

    # Preferred: verbose PokerKit representation with token in parentheses.
    m = re.search(r"\((\w+)\)", s)
    if m:
        token = m.group(1)
    else:
        # Fallback: compact token anywhere in the string (handles noisy reprs).
        m2 = re.search(r"(?i)\b(10|[2-9tjqka])([cdhs])\b", s)
        if m2:
            token = f"{m2.group(1)}{m2.group(2)}"

    if not token:
        return None

    rank = token[:-1].upper()
    suit = token[-1].upper()
    if rank == "T":
        rank = "10"
    if suit not in {"C", "D", "H", "S"} or rank not in {
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "J",
        "Q",
        "K",
        "A",
    }:
        return None
    return rank, suit


def _parse_card_list(cards: Any) -> list[tuple[str, str]]:
    """Parse a sequence of card-like values to normalized (rank, suit) tuples."""
    parsed: list[tuple[str, str]] = []
    for card in list(cards or []):
        p = _parse_card(card)
        if p:
            parsed.append(p)
    return parsed


def _phase_label(board_len: int) -> str:
    return {0: "Pre-Flop", 3: "Flop", 4: "Turn", 5: "River"}.get(
        board_len, "Pre-Flop"
    )


def _phase_key_from_board(board_len: int) -> str:
    return {0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(
        board_len, "preflop"
    )


def _street_metric_label(phase_key: str) -> str:
    return {
        "preflop": "Pre-Flop Bets",
        "flop": "Flop Bets",
        "turn": "Turn Bets",
        "river": "River Bets",
    }.get(phase_key, "Street Bets")


def _street_pot_breakdown(state: Any) -> dict[str, float]:
    """Estimate pot contributions per street from PokerKit operation history."""
    out = {"preflop": 0.0, "flop": 0.0, "turn": 0.0, "river": 0.0}
    current_street = "preflop"
    street_order = ["preflop", "flop", "turn", "river"]

    for op in list(getattr(state, "operations", []) or []):
        op_name = type(op).__name__
        amount = getattr(op, "amount", None)
        if amount is not None and op_name in {
            "AntePosting",
            "BlindOrStraddlePosting",
            "CheckingOrCalling",
            "CompletionBettingOrRaisingTo",
        }:
            out[current_street] += float(amount)

        if op_name == "BoardDealing":
            idx = street_order.index(current_street)
            if idx < len(street_order) - 1:
                current_street = street_order[idx + 1]

    return out


def _seat_role_map(button_index: int, seat_count: int) -> dict[int, str]:
    """Return seat role labels for the current hand: D, SB, BB."""
    if seat_count <= 0:
        return {}
    dealer = int(button_index) % seat_count
    sb_seat = (dealer + 1) % seat_count
    bb_seat = (dealer + 2) % seat_count
    return {dealer: "D", sb_seat: "SB", bb_seat: "BB"}


def _hero_commitment_metrics(state: Any, hero: int) -> tuple[float, float]:
    """Return (street_bet, total_hand_bet) for the hero seat."""
    bets = list(getattr(state, "bets", []) or [])
    street_bet = float(bets[hero]) if hero < len(bets) else 0.0
    total = 0.0
    for op in list(getattr(state, "operations", []) or []):
        if getattr(op, "player_index", None) != hero:
            continue
        amount = getattr(op, "amount", None)
        if amount is not None and type(op).__name__ in {
            "AntePosting",
            "BlindOrStraddlePosting",
            "CheckingOrCalling",
            "CompletionBettingOrRaisingTo",
        }:
            total += float(amount)
    return street_bet, total


def _hero_raise_count(state: Any, hero: int) -> int:
    """Count hero raises/bets (completion operations) so far in hand."""
    count = 0
    for op in list(getattr(state, "operations", []) or []):
        if getattr(op, "player_index", None) != hero:
            continue
        if type(op).__name__ == "CompletionBettingOrRaisingTo":
            count += 1
    return count


def _hero_street_bets_count(state: Any, hero: int) -> int:
    """Count how many streets the hero has invested chips in."""
    streets = ["preflop", "flop", "turn", "river"]
    current_street = "preflop"
    invested = {street: 0.0 for street in streets}

    for op in list(getattr(state, "operations", []) or []):
        op_name = type(op).__name__
        if op_name == "BoardDealing":
            idx = streets.index(current_street)
            if idx < len(streets) - 1:
                current_street = streets[idx + 1]
            continue
        if getattr(op, "player_index", None) != hero:
            continue
        amount = getattr(op, "amount", None)
        if amount is None:
            continue
        if op_name in {
            "AntePosting",
            "BlindOrStraddlePosting",
            "CheckingOrCalling",
            "CompletionBettingOrRaisingTo",
        }:
            invested[current_street] += float(amount)

    return sum(1 for value in invested.values() if value > 0)


def _live_pot_metrics(state: Any) -> tuple[float, float]:
    """Compute (total_pot, current_street_pot) from live PokerKit state."""
    active_bets = float(sum(list(getattr(state, "bets", []) or [])))
    finalized_pots = 0.0

    # Prefer explicit pot objects if available.
    pots = getattr(state, "pots", None)
    if pots is not None:
        try:
            for pot in pots:
                finalized_pots += float(getattr(pot, "amount", 0) or 0)
        except Exception:
            finalized_pots = 0.0

    # Fallback for engines exposing pushed amounts directly.
    if finalized_pots <= 0.0:
        try:
            finalized_pots = float(getattr(state, "total_pushed_amount", 0) or 0)
        except Exception:
            finalized_pots = 0.0

    total_pot = finalized_pots + active_bets
    current_street_pot = active_bets
    return total_pot, current_street_pot


def _build_view(gs: GameState) -> dict[str, Any]:
    state = gs.poker_state
    hero = int(gs.hero)
    if state is None:
        return {
            "hero": hero,
            "seat_count": _N_PLAYERS,
            "stacks": [_STARTING_STACK] * _N_PLAYERS,
            "bets": [0] * _N_PLAYERS,
            "pot_total": 0.0,
            "board_cards": [],
            "hole_by_seat": [[] for _ in range(_N_PLAYERS)],
            "folded": [False] * _N_PLAYERS,
            "turn_index": None,
            "is_hero_turn": False,
            "hand_complete": False,
            "payoffs": [],
            "last_action": {},
            "facing_bet": False,
            "call_amount": 0,
            "min_raise": _MIN_BET * 2,
            "max_raise": _STARTING_STACK,
            "action_error": "",
            "phase": "Pre-Flop",
            "street_pots": {"preflop": 0.0, "flop": 0.0, "turn": 0.0, "river": 0.0},
            "current_street_pot": 0.0,
            "current_street_label": "Pre-Flop Bets",
            "hero_street_bet": 0.0,
            "hero_total_bet": 0.0,
            "seat_roles": {},
            "button_index": 0,
            "sb_seat": 1,
            "bb_seat": 2,
            "bot_error": "",
            "win_probability": None,
            "prediction_error": "",
            "prediction_stage": "preflop",
            "bluff_prob_by_seat": [None] * _N_PLAYERS,
            "bluff_prediction_error": "",
        }

    hand_complete = not bool(getattr(state, "status", True))
    terminal = gs.terminal_payload

    stacks: list[int] = list(state.stacks or [])
    bets: list[int] = list(state.bets or [])

    display_stacks = terminal["stacks"] if (terminal and hand_complete) else stacks

    board_cards = _parse_card_list(state.board_cards)

    hole_by_seat: list[list[tuple[str, str]]] = []
    for seat_cards in state.hole_cards or []:
        hole_by_seat.append(_parse_card_list(seat_cards))

    folded = [not bool(s) for s in (state.statuses or [])]

    ti = state.turn_index
    is_hero_turn = bool(ti == hero and state.status)

    live_total_pot, live_current_street_pot = _live_pot_metrics(state)

    call_amount = 0
    min_raise = _MIN_BET * 2
    max_raise = _STARTING_STACK
    if is_hero_turn:
        try:
            call_amount = int(state.checking_or_calling_amount or 0)
        except Exception:
            pass
        try:
            v = state.min_completion_betting_or_raising_to_amount
            if v is not None:
                min_raise = int(v)
        except Exception:
            pass
        try:
            v = state.max_completion_betting_or_raising_to_amount
            if v is not None:
                max_raise = int(v)
        except Exception:
            pass

    facing_bet = call_amount > 0
    street_pots = _street_pot_breakdown(state)
    hero_street_bet, hero_total_bet = _hero_commitment_metrics(state, hero)
    phase_key = _phase_key_from_board(len(board_cards))
    hero_cards = hole_by_seat[hero] if hero < len(hole_by_seat) else []
    hole_tokens = [f"{rank}{suit.lower()}" for rank, suit in hero_cards]
    board_tokens = [f"{rank}{suit.lower()}" for rank, suit in board_cards]
    button_index = int(gs.button_index) % _N_PLAYERS
    seat_roles = _seat_role_map(button_index, _N_PLAYERS)
    hero_position = seat_roles.get(hero, "")
    hero_stack = float(stacks[hero]) if hero < len(stacks) else 0.0
    table_stacks = [float(s) for s in stacks]
    big_blind = float(_BLINDS[1]) if len(_BLINDS) > 1 else 2.0

    win_probability: float | None = None
    prediction_error = ""
    bluff_prob_by_seat: list[float | None] = [None] * _N_PLAYERS
    bluff_prediction_error = ""

    if gs.ml_enabled:
        try:
            from scripts.features.poker_hand_strength import build_stage_feature_payload
            from scripts.features.visible_bluff_features import (
                seat_commitment_and_raises,
                vector_from_live_state,
            )
            from scripts.models.stage_win_predictor import predict_stage_win_probability
            from scripts.models.visible_bluff_predictor import predict_visible_bluff_probability

            model_features = build_stage_feature_payload(
                phase_key,
                hole_tokens,
                board_tokens,
                total_bet=float(hero_total_bet),
                current_pot=float(live_total_pot),
                position=hero_position,
                hero_stack=hero_stack,
                table_stacks=table_stacks,
                big_blind=big_blind,
            )
            try:
                win_probability = predict_stage_win_probability(phase_key, model_features)
            except Exception as exc:
                prediction_error = str(exc)

            live_bb = float(_BLINDS[1]) if len(_BLINDS) > 1 else 2.0
            ts_float = [float(s) for s in stacks]
            for seat in range(_N_PLAYERS):
                if seat >= len(stacks):
                    continue
                try:
                    _, total_bet, raises = seat_commitment_and_raises(state, seat)
                    pos_tok = str(seat_roles.get(seat, "") or "")
                    vec = vector_from_live_state(
                        stage=phase_key,
                        board_cards=board_cards,
                        seat_stack=float(stacks[seat]),
                        seat_total_bet=float(total_bet),
                        pot_total=float(live_total_pot),
                        position_token=pos_tok,
                        table_stacks=ts_float,
                        big_blind=live_bb,
                        seat_raise_count=raises,
                    )
                    bluff_prob_by_seat[seat] = predict_visible_bluff_probability(vec)
                except Exception as exc:
                    if not bluff_prediction_error:
                        bluff_prediction_error = str(exc)
                    if isinstance(exc, (FileNotFoundError, ValueError)):
                        break
        except Exception as exc:
            prediction_error = str(exc)
            bluff_prediction_error = str(exc)

    if win_probability is None:
        try:
            from heuristics import hero_win_probability_proxy

            villains_active = sum(
                1
                for s in range(_N_PLAYERS)
                if s != hero and s < len(folded) and not folded[s]
            )
            win_probability = hero_win_probability_proxy(
                hero_cards, board_cards, villains_active
            )
        except Exception:
            pass

    try:
        from heuristics import bluff_probability_proxy

        for seat in range(_N_PLAYERS):
            if bluff_prob_by_seat[seat] is None:
                bluff_prob_by_seat[seat] = bluff_probability_proxy(
                    state, seat, live_total_pot
                )
    except Exception:
        pass

    current_street_pot = live_current_street_pot
    current_street_label = _street_metric_label(phase_key)

    return {
        "hero": hero,
        "seat_count": _N_PLAYERS,
        "stacks": display_stacks,
        "bets": bets,
        "pot_total": live_total_pot,
        "board_cards": board_cards,
        "hole_by_seat": hole_by_seat,
        "folded": folded,
        "turn_index": ti,
        "is_hero_turn": is_hero_turn,
        "hand_complete": hand_complete,
        "payoffs": terminal["payoffs"] if terminal else [],
        "last_action": dict(gs.last_action),
        "facing_bet": facing_bet,
        "call_amount": call_amount,
        "min_raise": min_raise,
        "max_raise": max_raise,
        "action_error": gs.action_error,
        "phase": _phase_label(len(board_cards)),
        "street_pots": street_pots,
        "current_street_pot": current_street_pot,
        "current_street_label": current_street_label,
        "hero_street_bet": hero_street_bet,
        "hero_total_bet": hero_total_bet,
        "seat_roles": seat_roles,
        "button_index": button_index,
        "sb_seat": (button_index + 1) % _N_PLAYERS,
        "bb_seat": (button_index + 2) % _N_PLAYERS,
        "bot_error": gs.bot_error,
        "win_probability": win_probability,
        "prediction_error": prediction_error,
        "prediction_stage": phase_key,
        "bluff_prob_by_seat": bluff_prob_by_seat,
        "bluff_prediction_error": bluff_prediction_error,
    }


# ══════════════════════════════════════════════════════════════════════════════
# RENDER LAYER  (zero engine mutations – read-only)
# ══════════════════════════════════════════════════════════════════════════════

def _outcome_class(seat: int, view: dict[str, Any]) -> str:
    if not view["hand_complete"]:
        return ""
    payoffs = view["payoffs"]
    if seat >= len(payoffs):
        return ""
    p = payoffs[seat]
    if p > 0:
        return " winner"
    if p < 0:
        return " loser"
    return ""


def _seat_html(seat: int, view: dict[str, Any]) -> str:
    hero = view["hero"]
    is_hero = seat == hero
    folded = view["folded"][seat] if seat < len(view["folded"]) else False
    is_active = (view["turn_index"] == seat) and not view["hand_complete"]

    stack = view["stacks"][seat] if seat < len(view["stacks"]) else 0
    bet = view["bets"][seat] if seat < len(view["bets"]) else 0
    hole = view["hole_by_seat"][seat] if seat < len(view["hole_by_seat"]) else []
    last = view["last_action"].get(seat, "")
    role = view.get("seat_roles", {}).get(seat, "")

    seat_class = "player-seat"
    if is_active:
        seat_class += " active-turn"
    if folded:
        seat_class += " seat-folded"
    seat_class += _outcome_class(seat, view)

    badge = f"<div class='action-badge'>{last}</div>" if last else ""
    role_badge = ""
    if role:
        role_color = {"D": "#F1C40F", "SB": "#3498DB", "BB": "#9B59B6"}.get(role, "#BCCCDC")
        role_badge = (
            "<div style='display:inline-flex;align-items:center;justify-content:center;"
            "margin-left:6px;padding:2px 7px;border-radius:999px;font-size:10px;"
            f"font-weight:800;background:{role_color};color:#102A43;'>{role}</div>"
        )
    name = f"YOU (Seat {seat + 1})" if is_hero else f"Seat {seat + 1}"

    bps = view.get("bluff_prob_by_seat") or []
    metrics_parts = [f"Stack ${stack}", f"Bet ${bet}"]
    if seat < len(bps) and bps[seat] is not None:
        metrics_parts.append(f"Bluff ~{float(bps[seat]) * 100.0:.0f}%")
    metrics_line = " | ".join(metrics_parts)

    if is_hero:
        cards_html = "".join(render_face_up_card(r, s) for r, s in hole)
    else:
        # Showdown reveal rule:
        # - Hand complete + opponent not folded => reveal real hole cards.
        # - Folded opponents remain face-down (and are already grayed by seat class).
        reveal_at_showdown = bool(view["hand_complete"] and not folded)
        if reveal_at_showdown:
            cards_html = "".join(render_face_up_card(r, s) for r, s in hole)
        else:
            cards_html = "".join(render_face_down_card() for _ in range(len(hole) or 2))

    return (
        f"<div class='{seat_class}' style='position:relative;'>"
        f"{badge}"
        f"<div style='font-weight:700;font-size:13px;'>{name}{role_badge}</div>"
        f"<div class='seat-metrics'>{metrics_line}</div>"
        f"<div style='display:flex;gap:6px;margin-top:8px;'>{cards_html}</div>"
        f"</div>"
    )


def _board_html(view: dict[str, Any]) -> str:
    board = view["board_cards"]
    slot_html: list[str] = []
    # 5-slot rule: always render exactly five board positions.
    for i in range(5):
        if i < len(board):
            rank, suit = board[i]
            slot_html.append(
                "<div class='board-slot card' style='width:72px;height:104px;'>"
                + render_face_up_card(
                    rank, suit, classes=f"community-card deal-animate deal-delay-{i}"
                )
                + "</div>"
            )
        else:
            slot_html.append(
                "<div class='board-slot'>"
                "<div class='card placeholder' "
                "style='width:72px;height:104px;border:1px dashed rgba(130,154,177,0.85);"
                "border-radius:12px;background:rgba(15,23,42,0.25);"
                "box-shadow:inset 0 0 0 1px rgba(188,204,220,0.2);'></div>"
                "</div>"
            )
    return (
        "<div class='board-row-fixed' "
        "style='display:flex;justify-content:center;gap:10px;width:100%;align-items:center;'>"
        + "".join(slot_html)
        + "</div>"
    )


_CALL_LABEL_DISPLAY_LEN = 14


def stable_call_button_label(view: dict[str, Any]) -> str:
    facing = view["facing_bet"]
    call_amount = view["call_amount"]
    raw = f"Call ${call_amount}" if facing else "Check"
    if len(raw) >= _CALL_LABEL_DISPLAY_LEN:
        return raw[:_CALL_LABEL_DISPLAY_LEN]
    return raw + "\u2007" * (_CALL_LABEL_DISPLAY_LEN - len(raw))


def ensure_initialized(gs: GameState) -> None:
    _ensure_initialized(gs)


def new_hand(gs: GameState, *, advance_button: bool) -> None:
    _new_hand(gs, advance_button=advance_button)


def run_dealer(gs: GameState) -> None:
    _run_dealer(gs)


def build_view(gs: GameState) -> dict[str, Any]:
    return _build_view(gs)


def apply_hero_action(gs: GameState, action: dict[str, Any]) -> None:
    _apply_hero_action(gs, action)


def capture_terminal_if_needed(gs: GameState) -> None:
    if gs.poker_state and not gs.poker_state.status:
        _capture_terminal(gs)


def hero_payoff_chips(view: dict[str, Any]) -> int:
    payoffs = view["payoffs"]
    hero = view["hero"]
    if hero >= len(payoffs):
        return 0
    return int(payoffs[hero])

