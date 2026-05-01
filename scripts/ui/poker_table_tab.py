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
      Step 3  Run bots + Dealer in alternation until it is the hero's turn or the hand ends.
      Step 4  Render – read-only; zero engine mutations below this line.
"""

from __future__ import annotations

import random  # used for hero seat selection and bot 80/20 fold decision
import re
import time
from typing import Any

import streamlit as st
from pokerkit import Automation, NoLimitTexasHoldem

from scripts.features.poker_hand_strength import build_stage_feature_payload
from scripts.game.bot_backend import (
    BasePokerBot,
    build_bot_instances,
    normalize_bot_response,
)
from scripts.models.stage_win_predictor import predict_stage_win_probability
from scripts.ui import theme
from scripts.ui.card_html import (
    poker_table_css,
    render_face_down_card,
    render_face_up_card,
)

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


def _new_hand(*, advance_button: bool) -> None:
    """Create a fresh game and deal hole cards."""
    bankrolls = _normalize_bankrolls(
        st.session_state.get("master_bankrolls", [_STARTING_STACK] * _N_PLAYERS)
    )
    st.session_state.master_bankrolls = bankrolls
    button_index = int(st.session_state.get("button_index", 0)) % _N_PLAYERS
    if advance_button:
        button_index = (button_index + 1) % _N_PLAYERS
    st.session_state.button_index = button_index
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
    st.session_state.poker_state = state
    st.session_state.deck = _deck_from_state(state)
    st.session_state.last_action = {}        # {seat_index: str} action badge text
    st.session_state.terminal_payload = None
    st.session_state.action_error = ""
    st.session_state.bot_error = ""
    st.session_state.pop("pending_action", None)
    # Reset per-hand settlement guards for the newly created hand.
    st.session_state.hand_resolved = False
    st.session_state.resolved_hand_id = None
    st.session_state.bot_instances = build_bot_instances(_N_PLAYERS, st.session_state.hero)
    # Deal hole cards immediately so the first render shows a live hand.
    _run_dealer(state)


def _ensure_initialized() -> None:
    st.session_state.master_bankrolls = _normalize_bankrolls(
        st.session_state.get("master_bankrolls", [_STARTING_STACK] * _N_PLAYERS)
    )
    if "hand_resolved" not in st.session_state:
        st.session_state.hand_resolved = False
    if "resolved_hand_id" not in st.session_state:
        st.session_state.resolved_hand_id = None
    if "hero" not in st.session_state:
        st.session_state.hero = random.randrange(_N_PLAYERS)
    if "button_index" not in st.session_state:
        st.session_state.button_index = 0
    if "bot_error" not in st.session_state:
        st.session_state.bot_error = ""
    if "poker_state" not in st.session_state:
        _new_hand(advance_button=False)
    if "bot_instances" not in st.session_state:
        st.session_state.bot_instances = build_bot_instances(_N_PLAYERS, st.session_state.hero)


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 2 – THE DEALER
# ══════════════════════════════════════════════════════════════════════════════

def _pop(n: int) -> list[str]:
    """Pop n cards from the session deck.  Returns [] if the deck is too short."""
    deck: list[str] = st.session_state.deck
    if len(deck) < n:
        return []
    return [deck.pop() for _ in range(n)]


def _run_dealer(state: Any) -> None:
    """Give PokerKit cards when it asks.  Does nothing else – ever.

    Order of checks: hole cards first, then burns (between streets), then board.
    Each iteration handles exactly one dealing action and loops back to re-check,
    ensuring the correct sequence without manual street-tracking.
    """
    for _ in range(30):
        if state.can_deal_hole():
            cards = _pop(2)
            if not cards:
                break
            state.deal_hole(cards[0] + cards[1])
        elif state.can_burn_card():
            card = _pop(1)
            if not card:
                break
            state.burn_card(card[0])
        elif state.can_deal_board():
            n = 3 if not state.board_cards else 1
            cards = _pop(n)
            if not cards:
                break
            state.deal_board("".join(cards))
        else:
            break


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 3 – BOT LOOP  (legal moves only – this is what was broken before)
# ══════════════════════════════════════════════════════════════════════════════

def _render_live_snapshot(
    state: Any,
    hero: int,
    table_placeholder: Any,
    controls_placeholder: Any,
    status_placeholder: Any | None = None,
    status_text: str = "",
    include_controls: bool = False,
) -> None:
    """Render an intermediate table snapshot during bot playback."""
    view = _build_view(state, hero)
    with table_placeholder.container():
        _render_table(view)
    with controls_placeholder.container():
        st.markdown("---")
        _render_hero_dashboard(view, include_controls=include_controls)
    if status_placeholder is not None:
        if status_text:
            status_placeholder.info(status_text)
        else:
            status_placeholder.empty()


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


def _run_bots_to_hero(
    state: Any,
    hero: int,
    table_placeholder: Any | None = None,
    controls_placeholder: Any | None = None,
    status_placeholder: Any | None = None,
) -> None:
    """Alternate Dealer → Bot until the hero must act or the hand ends.

    The bot only ever asks PokerKit "what can I do?" before acting.
    It never executes an action that has not been verified legal.
    """
    for _cycle in range(120):
        # Always run the Dealer first so any end-of-street dealing happens
        # before we inspect the turn index.
        _run_dealer(state)

        if not state.status:
            _capture_terminal(state)
            break

        ti = state.turn_index
        if ti is None:
            # Dealer ran and turn is still None – structural stall; break safely.
            break
        if ti == hero:
            # Hero's turn – let the UI render controls.
            break

        # Show explicit hand-off before the bot mutates state.
        if (
            table_placeholder is not None
            and controls_placeholder is not None
            and status_placeholder is not None
        ):
            _render_live_snapshot(
                state,
                hero,
                table_placeholder,
                controls_placeholder,
                status_placeholder=status_placeholder,
                status_text=f"Seat {ti + 1} is thinking...",
                include_controls=False,
            )
            time.sleep(1.5)

        # ── Bot's turn via backend interface ─────────────────────────────────
        action_text = ""
        legal_moves = _legal_moves_snapshot(state, ti)
        hole_cards = _extract_bot_hole_cards(state, ti)
        bots: dict[int, BasePokerBot] = st.session_state.get("bot_instances", {})
        bot = bots.get(ti)
        if bot is None:
            bots = build_bot_instances(_N_PLAYERS, hero)
            st.session_state.bot_instances = bots
            bot = bots.get(ti)

        try:
            if bot is None:
                raise ValueError("Missing bot instance")
            raw_response = bot.calculate_action(state, hole_cards, legal_moves)
            action, amount = normalize_bot_response(raw_response)
            label = _execute_bot_action(state, ti, action, amount, legal_moves)
            st.session_state.last_action[ti] = label
            action_text = f"Seat {ti + 1} {label.lower()}."
        except Exception as exc:
            st.session_state.bot_error = (
                f"Seat {ti + 1} hallucinated illegal move: {exc}"
            )
            fallback_label = _force_safe_bot_fallback(state)
            st.session_state.last_action[ti] = fallback_label
            action_text = f"Seat {ti + 1} {fallback_label.lower()}."

        # Render the action result, then brief pause so users can register it.
        if (
            table_placeholder is not None
            and controls_placeholder is not None
            and status_placeholder is not None
        ):
            _render_live_snapshot(
                state,
                hero,
                table_placeholder,
                controls_placeholder,
                status_placeholder=status_placeholder,
                status_text=action_text,
                include_controls=False,
            )
            time.sleep(0.5)


# ══════════════════════════════════════════════════════════════════════════════
# HERO ACTION EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def _apply_hero_action(state: Any, action: dict[str, Any], hero: int) -> None:
    """Execute the hero's queued action against the live PokerKit state."""
    st.session_state.action_error = ""
    try:
        kind = action["type"]
        if kind == "fold":
            if state.can_fold():
                state.fold()
                st.session_state.last_action[hero] = "Fold"
            else:
                st.session_state.action_error = "Cannot fold right now."

        elif kind == "call":
            if state.can_check_or_call():
                facing = state.checking_or_calling_amount > 0
                # Read label before the action mutates the state.
                label = (
                    f"Call ${state.checking_or_calling_amount}" if facing else "Check"
                )
                state.check_or_call()
                st.session_state.last_action[hero] = label
            else:
                st.session_state.action_error = "Cannot check/call right now."

        elif kind == "raise":
            amount = int(action["amount"])
            if state.can_complete_bet_or_raise_to(amount):
                facing = state.checking_or_calling_amount > 0
                label = f"Raise ${amount}" if facing else f"Bet ${amount}"
                state.complete_bet_or_raise_to(amount)
                st.session_state.last_action[hero] = label
            else:
                st.session_state.action_error = (
                    f"${amount} is not a legal raise size."
                )
    except Exception as exc:
        st.session_state.action_error = str(exc)


# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL STATE
# ══════════════════════════════════════════════════════════════════════════════

def _capture_terminal(state: Any) -> None:
    """Persist payoffs once when the hand ends."""
    hand_id = id(state)

    try:
        payoffs = list(state.payoffs or [])
        stacks = list(state.stacks or [])
    except Exception:
        payoffs, stacks = [], []

    # Apply payout exactly once per hand across reruns.
    already_resolved = bool(st.session_state.get("hand_resolved", False))
    resolved_hand_id = st.session_state.get("resolved_hand_id")
    if (not already_resolved) and (resolved_hand_id != hand_id):
        bankrolls = _normalize_bankrolls(
            st.session_state.get("master_bankrolls", [_STARTING_STACK] * _N_PLAYERS)
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
        st.session_state.master_bankrolls = updated
        st.session_state.hand_resolved = True
        st.session_state.resolved_hand_id = hand_id

    if st.session_state.get("terminal_payload") is not None:
        return
    st.session_state.terminal_payload = {"payoffs": payoffs, "stacks": stacks}


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
    if rank == "10":
        rank = "T"
    if suit not in {"C", "D", "H", "S"} or rank not in {
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "T",
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


def _risk_band(value: float) -> str:
    if value >= 0.65:
        return "High"
    if value >= 0.35:
        return "Medium"
    return "Low"


def _build_view(state: Any, hero: int) -> dict[str, Any]:
    hand_complete = not bool(getattr(state, "status", True))
    terminal = st.session_state.get("terminal_payload")

    stacks: list[int] = list(state.stacks or [])
    bets: list[int] = list(state.bets or [])

    # Use final stacks from terminal snapshot when hand is over.
    display_stacks = terminal["stacks"] if (terminal and hand_complete) else stacks

    # Board and hole cards use the same parser pipeline to keep formats aligned.
    board_cards = _parse_card_list(state.board_cards)

    hole_by_seat: list[list[tuple[str, str]]] = []
    for seat_cards in state.hole_cards or []:
        hole_by_seat.append(_parse_card_list(seat_cards))

    # statuses: True = still active, False = folded/eliminated.
    folded = [not bool(s) for s in (state.statuses or [])]

    ti = state.turn_index
    is_hero_turn = bool(ti == hero and state.status)

    live_total_pot, live_current_street_pot = _live_pot_metrics(state)

    # Legal snapshot (only meaningful when it is the hero's turn).
    call_amount = 0
    min_raise = _MIN_BET * 2          # sensible default
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
    button_index = int(st.session_state.get("button_index", 0)) % _N_PLAYERS
    seat_roles = _seat_role_map(button_index, _N_PLAYERS)
    hero_position = seat_roles.get(hero, "")
    hero_stack = float(stacks[hero]) if hero < len(stacks) else 0.0
    table_stacks = [float(s) for s in stacks]
    big_blind = float(_BLINDS[1]) if len(_BLINDS) > 1 else 2.0
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
    win_probability: float | None = None
    prediction_error = ""
    try:
        win_probability = predict_stage_win_probability(phase_key, model_features)
    except Exception as exc:
        prediction_error = str(exc)
    board_risk = float(model_features.get("board_shared_strength_risk", 0.0))
    adjusted_aggression = (
        float(win_probability) - (0.35 * board_risk)
        if win_probability is not None
        else None
    )
    # Bind the visible metric to live active bets so it always updates with the
    # current betting round. Keep street breakdown as secondary/debug context.
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
        "last_action": dict(st.session_state.get("last_action", {})),
        "facing_bet": facing_bet,
        "call_amount": call_amount,
        "min_raise": min_raise,
        "max_raise": max_raise,
        "action_error": st.session_state.get("action_error", ""),
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
        "bot_error": st.session_state.get("bot_error", ""),
        "win_probability": win_probability,
        "prediction_error": prediction_error,
        "prediction_stage": phase_key,
        "board_shared_strength_risk": board_risk,
        "board_risk_band": _risk_band(board_risk),
        "adjusted_aggression": adjusted_aggression,
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
    name = f"You – Seat {seat + 1}" if is_hero else f"Seat {seat + 1}"

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
        f"<div class='seat-metrics'>Stack ${stack} | Bet ${bet}</div>"
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


def _render_table(view: dict[str, Any]) -> None:
    hero = view["hero"]
    opponents = [s for s in range(view["seat_count"]) if s != hero]

    # Opponent row
    cols = st.columns(len(opponents))
    for i, seat in enumerate(opponents):
        with cols[i]:
            st.markdown(_seat_html(seat, view), unsafe_allow_html=True)

    # Community board
    st.markdown(
        "<div style='margin:10px 0;'>" + _board_html(view) + "</div>",
        unsafe_allow_html=True,
    )


def _render_hero_dashboard(
    view: dict[str, Any],
    include_controls: bool = True,
) -> None:
    hero = view["hero"]
    with st.container(border=True):
        left, right = st.columns(2)

        # Left half: same hero information block as before.
        with left:
            st.markdown(_seat_html(hero, view), unsafe_allow_html=True)
            st.caption(
                f"Street Bet: ${view['hero_street_bet']:.2f} | Total Hand Bet: ${view['hero_total_bet']:.2f}"
            )
            win_probability = view.get("win_probability")
            if win_probability is not None:
                st.metric(
                    "Win Chance",
                    f"{float(win_probability):.2%}",
                    help=f"Model stage: {view.get('prediction_stage', 'preflop')}",
                )
                st.caption(
                    f"Shared-board risk: {view.get('board_risk_band', 'Low')} ({float(view.get('board_shared_strength_risk', 0.0)):.2f})"
                )
                adj = view.get("adjusted_aggression")
                if adj is not None:
                    style = "Aggressive" if float(adj) >= 0.6 else ("Cautious" if float(adj) <= 0.4 else "Balanced")
                    st.caption(f"Risk-adjusted posture: {style}")
            elif view.get("prediction_error"):
                st.caption(f"Win chance unavailable: {view['prediction_error']}")

        # Right half: nested action controls.
        with right:
            if include_controls:
                _render_controls(view)
            else:
                _render_controls_readonly(view)


def _render_controls_readonly(view: dict[str, Any]) -> None:
    """Render a non-interactive controls layout for bot-loop snapshots.

    Keeps actions and pot metrics visible without creating Streamlit widgets,
    which prevents duplicate widget key crashes during repeated snapshot renders.
    """
    facing = view["facing_bet"]
    call_amount = view["call_amount"]
    min_raise = view["min_raise"]
    max_raise = view["max_raise"]
    call_label = f"Call ${call_amount}" if facing else "Check"
    bet_label = "Raise To" if facing else "Bet"
    default_raise = max(min_raise, min(min_raise, max_raise))
    submit_label = (
        f"{bet_label} ${int(st.session_state.get('raise_input', default_raise))}"
    )

    # Reuse the exact same action-control CSS for visual parity.
    st.markdown(
        """
<style>
/* Scoped only to this action panel via marker + :has */
div[data-testid="stVerticalBlock"]:has(> div > .hero-action-controls-scope) {
  min-height: 240px;
}
div[data-testid="stVerticalBlock"]:has(> div > .hero-action-controls-scope) div[data-testid="column"] {
  min-height: 240px;
}
div[data-testid="stVerticalBlock"]:has(> div > .hero-action-controls-scope) button[aria-label="Fold"] {
  height: 120px !important;         /* 50% of passive column */
  width: 100% !important;
}
div[data-testid="stVerticalBlock"]:has(> div > .hero-action-controls-scope) button[aria-label="Check / Call"] {
  height: 120px !important;         /* 50% of passive column */
  width: 100% !important;
}
div[data-testid="stVerticalBlock"]:has(> div > .hero-action-controls-scope) button[aria-label="Submit Bet / Raise"] {
  height: 192px !important;         /* 80% of aggressive column */
  width: 100% !important;
}
div[data-testid="stVerticalBlock"]:has(> div > .hero-action-controls-scope) div[data-testid="stNumberInput"] {
  width: 100% !important;
}
</style>
<div class="hero-action-controls-scope"></div>
""",
        unsafe_allow_html=True,
    )

    seq = int(st.session_state.get("_readonly_controls_seq", 0)) + 1
    st.session_state["_readonly_controls_seq"] = seq

    passive_col, aggressive_col = st.columns([1, 1])
    with passive_col:
        st.button(
            "Fold",
            use_container_width=True,
            key=f"ro_btn_fold_{seq}",
            disabled=True,
        )
        st.button(
            call_label,
            use_container_width=True,
            key=f"ro_btn_call_{seq}",
            disabled=True,
        )
        st.metric("Total Pot", f"${float(view.get('pot_total', 0.0)):,.2f}")

    with aggressive_col:
        st.button(
            submit_label,
            use_container_width=True,
            key=f"ro_btn_raise_{seq}",
            disabled=True,
        )
        st.number_input(
            bet_label,
            min_value=min_raise,
            max_value=max(min_raise, max_raise),
            value=int(st.session_state.get("raise_input", default_raise)),
            step=1,
            key=f"ro_raise_input_{seq}",
            label_visibility="collapsed",
            disabled=True,
        )
        st.metric(
            view.get("current_street_label", "Street Bets"),
            f"${float(view.get('current_street_pot', 0.0)):,.2f}",
        )
        if not view["is_hero_turn"] and not view["hand_complete"]:
            ti = view["turn_index"]
            label = f"Seat {ti + 1}" if ti is not None else "the engine"
            st.caption(f"Waiting for {label} to act…")


def _render_controls(view: dict[str, Any]) -> None:
    # ── Hand over ────────────────────────────────────────────────────────────
    if view["hand_complete"]:
        payoffs = view["payoffs"]
        hero = view["hero"]
        hp = payoffs[hero] if hero < len(payoffs) else 0
        msg = "You won!" if hp > 0 else ("You lost." if hp < 0 else "Chopped pot.")
        color = "#2ECC71" if hp > 0 else "#E74C3C"
        st.markdown(
            (
                "<div style='padding:10px 12px;border-radius:10px;"
                f"border:1px solid {color};background:rgba(16, 42, 67, 0.35);"
                f"color:{color};font-weight:700;'>Hand over — {msg}  ({hp:+.0f} chips)</div>"
            ),
            unsafe_allow_html=True,
        )
        if st.button("Deal Next Hand", key="next_hand_btn"):
            _new_hand(advance_button=True)
            st.rerun()
        return

    # ── Hero's controls ───────────────────────────────────────────────────────
    if view["action_error"]:
        st.error(view["action_error"])

    facing = view["facing_bet"]
    call_amount = view["call_amount"]
    min_raise = view["min_raise"]
    max_raise = view["max_raise"]

    call_label = f"Call ${call_amount}" if facing else "Check"
    bet_label = "Raise To" if facing else "Bet"
    controls_disabled = (not view["is_hero_turn"]) or view["hand_complete"]

    st.markdown(
        """
<style>
/* Scoped only to this action panel via marker + :has */
div[data-testid="stVerticalBlock"]:has(> div > .hero-action-controls-scope) {
  min-height: 240px;
}
div[data-testid="stVerticalBlock"]:has(> div > .hero-action-controls-scope) div[data-testid="column"] {
  min-height: 240px;
}
div[data-testid="stVerticalBlock"]:has(> div > .hero-action-controls-scope) button[aria-label="Fold"] {
  height: 120px !important;         /* 50% of passive column */
  width: 100% !important;
}
div[data-testid="stVerticalBlock"]:has(> div > .hero-action-controls-scope) button[aria-label="Check / Call"] {
  height: 120px !important;         /* 50% of passive column */
  width: 100% !important;
}
div[data-testid="stVerticalBlock"]:has(> div > .hero-action-controls-scope) button[aria-label="Submit Bet / Raise"] {
  height: 192px !important;         /* 80% of aggressive column */
  width: 100% !important;
}
div[data-testid="stVerticalBlock"]:has(> div > .hero-action-controls-scope) div[data-testid="stNumberInput"] {
  width: 100% !important;
}
</style>
<div class="hero-action-controls-scope"></div>
""",
        unsafe_allow_html=True,
    )

    passive_col, aggressive_col = st.columns([1, 1])

    with passive_col:
        if st.button(
            "Fold",
            use_container_width=True,
            key="btn_fold",
            disabled=controls_disabled,
        ):
            st.session_state.pending_action = {"type": "fold"}
            st.rerun()
        if st.button(
            call_label,
            use_container_width=True,
            key="btn_call",
            disabled=controls_disabled,
        ):
            st.session_state.pending_action = {"type": "call"}
            st.rerun()
        st.metric("Total Pot", f"${float(view.get('pot_total', 0.0)):,.2f}")

    with aggressive_col:
        # Clamp default value inside [min_raise, max_raise] to avoid Streamlit error.
        default_raise = max(min_raise, min(min_raise, max_raise))
        submit_label = (
            f"{bet_label} ${int(st.session_state.get('raise_input', default_raise))}"
        )
        if st.button(
            submit_label,
            use_container_width=True,
            key="btn_raise",
            disabled=controls_disabled,
        ):
            st.session_state.pending_action = {
                "type": "raise",
                "amount": int(st.session_state.get("raise_input", default_raise)),
            }
            st.rerun()
        st.number_input(
            bet_label,
            min_value=min_raise,
            max_value=max(min_raise, max_raise),
            value=default_raise,
            step=1,
            key="raise_input",
            label_visibility="collapsed",
            disabled=controls_disabled,
        )
        st.metric(
            view.get("current_street_label", "Street Bets"),
            f"${float(view.get('current_street_pot', 0.0)):,.2f}",
        )
        if not view["is_hero_turn"] and not view["hand_complete"]:
            ti = view["turn_index"]
            label = f"Seat {ti + 1}" if ti is not None else "the engine"
            st.caption(f"Waiting for {label} to act…")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def render_playable_poker_tab() -> None:
    st.markdown(poker_table_css(), unsafe_allow_html=True)

    # ── PILLAR 1: Setup ───────────────────────────────────────────────────────
    _ensure_initialized()
    state: Any = st.session_state.poker_state
    hero: int = st.session_state.hero
    table_placeholder = st.empty()
    controls_placeholder = st.empty()
    status_placeholder = st.empty()

    # ── PILLAR 3: Strict execution order ──────────────────────────────────────

    # Step 1 – Process pending hero action (queued by a button click).
    pending = st.session_state.pop("pending_action", None)
    if pending is not None and state.turn_index == hero and state.status:
        _apply_hero_action(state, pending, hero)

    # Step 2 – Dealer: clears any dealing requests triggered by the hero's action.
    _run_dealer(state)

    # Step 3 – Bots + Dealer loop until hero's turn or hand ends.
    _run_bots_to_hero(
        state,
        hero,
        table_placeholder=table_placeholder,
        controls_placeholder=controls_placeholder,
        status_placeholder=status_placeholder,
    )

    # ── Step 4: Render (read-only from here down) ─────────────────────────────
    view = _build_view(state, hero)
    with table_placeholder.container():
        _render_table(view)
    with controls_placeholder.container():
        st.markdown("---")
        _render_hero_dashboard(view)
    status_placeholder.empty()

    # Collapsible debug panel
    with st.expander("Debug", expanded=False):
        st.json(
            {
                "hero_seat": hero,
                "button_index": view.get("button_index"),
                "sb_seat": view.get("sb_seat"),
                "bb_seat": view.get("bb_seat"),
                "turn_index": state.turn_index,
                "is_hero_turn": view["is_hero_turn"],
                "hand_complete": view["hand_complete"],
                "phase": view["phase"],
                "board_len": len(view["board_cards"]),
                "deck_remaining": len(st.session_state.get("deck", [])),
                "stacks": view["stacks"],
                "bets": view["bets"],
                "pot": view["pot_total"],
                "payoffs": view["payoffs"],
                "last_action": view["last_action"],
                "bot_error": view.get("bot_error", ""),
            }
        )

    if st.button("Restart (New Hand)", key="restart_btn"):
        _new_hand(advance_button=True)
        st.rerun()
