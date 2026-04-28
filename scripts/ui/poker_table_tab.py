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
from typing import Any

import streamlit as st
from pokerkit import Automation, NoLimitTexasHoldem

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


def _new_hand() -> None:
    """Create a fresh game and deal hole cards."""
    state = NoLimitTexasHoldem.create_state(
        _AUTOMATIONS,
        True,                           # uniform_antes
        0,                              # raw_antes
        _BLINDS,                        # raw_blinds_or_straddles
        _MIN_BET,                       # min_bet
        (_STARTING_STACK,) * _N_PLAYERS,
        _N_PLAYERS,
    )
    st.session_state.poker_state = state
    st.session_state.deck = _deck_from_state(state)
    st.session_state.hero = random.randrange(_N_PLAYERS)
    st.session_state.last_action = {}        # {seat_index: str} action badge text
    st.session_state.terminal_payload = None
    st.session_state.action_error = ""
    st.session_state.pop("pending_action", None)
    # Deal hole cards immediately so the first render shows a live hand.
    _run_dealer(state)


def _ensure_initialized() -> None:
    if "poker_state" not in st.session_state:
        _new_hand()


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

def _run_bots_to_hero(state: Any, hero: int) -> None:
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

        # ── Bot's turn ───────────────────────────────────────────────────────
        # Ask PokerKit what is legal, then pick from that exact list.
        # can_check_or_call() covers both "check" (no bet to face) and "call"
        # (facing a bet), so it handles every non-aggressive betting action.
        if state.can_check_or_call():
            facing = state.checking_or_calling_amount > 0
            # 80 % call / check, 20 % fold (only when actually facing a bet).
            if facing and random.random() < 0.20 and state.can_fold():
                state.fold()
                st.session_state.last_action[ti] = "Fold"
            else:
                state.check_or_call()
                st.session_state.last_action[ti] = (
                    f"Call ${state.checking_or_calling_amount}"
                    if facing
                    else "Check"
                )
        elif state.can_fold():
            # Fallback – should rarely trigger but guarantees forward progress.
            state.fold()
            st.session_state.last_action[ti] = "Fold"
        else:
            # No legal betting action available on a live turn.
            # This should not happen with the chosen automations, but break
            # rather than spin forever.
            st.session_state.action_error = (
                f"Seat {ti + 1} has no legal action – structural stall."
            )
            break


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
    if st.session_state.get("terminal_payload") is not None:
        return
    try:
        payoffs = list(state.payoffs or [])
        stacks = list(state.stacks or [])
    except Exception:
        payoffs, stacks = [], []
    st.session_state.terminal_payload = {"payoffs": payoffs, "stacks": stacks}


# ══════════════════════════════════════════════════════════════════════════════
# VIEW MODEL  (read-only – no mutations)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_card(card_obj: Any) -> tuple[str, str] | None:
    """Convert a PokerKit card object to (rank, suit) for HTML rendering."""
    s = str(card_obj)
    # Verbose format: 'SIX OF HEARTS (6h)' → extract token in parens.
    m = re.search(r"\((\w+)\)", s)
    token = m.group(1) if m else s.strip()
    if len(token) < 2:
        return None
    rank = token[:-1].upper()
    suit = token[-1].upper()
    if rank == "10":
        rank = "T"
    return rank, suit


def _phase_label(board_len: int) -> str:
    return {0: "Pre-Flop", 3: "Flop", 4: "Turn", 5: "River"}.get(
        board_len, "Pre-Flop"
    )


def _build_view(state: Any, hero: int) -> dict[str, Any]:
    hand_complete = not bool(getattr(state, "status", True))
    terminal = st.session_state.get("terminal_payload")

    stacks: list[int] = list(state.stacks or [])
    bets: list[int] = list(state.bets or [])

    # Use final stacks from terminal snapshot when hand is over.
    display_stacks = terminal["stacks"] if (terminal and hand_complete) else stacks

    board_cards: list[tuple[str, str]] = []
    for card in state.board_cards or []:
        p = _parse_card(card)
        if p:
            board_cards.append(p)

    hole_by_seat: list[list[tuple[str, str]]] = []
    for seat_cards in state.hole_cards or []:
        parsed: list[tuple[str, str]] = []
        for card in seat_cards:
            p = _parse_card(card)
            if p:
                parsed.append(p)
        hole_by_seat.append(parsed)

    # statuses: True = still active, False = folded/eliminated.
    folded = [not bool(s) for s in (state.statuses or [])]

    ti = state.turn_index
    is_hero_turn = bool(ti == hero and state.status)

    pot = float(sum(bets))
    try:
        pot += float(getattr(state, "total_pushed_amount", 0) or 0)
    except Exception:
        pass

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

    return {
        "hero": hero,
        "seat_count": _N_PLAYERS,
        "stacks": display_stacks,
        "bets": bets,
        "pot_total": pot,
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

    seat_class = "player-seat"
    if is_active:
        seat_class += " active-turn"
    if folded:
        seat_class += " seat-folded"
    seat_class += _outcome_class(seat, view)

    badge = f"<div class='action-badge'>{last}</div>" if last else ""
    name = f"You – Seat {seat + 1}" if is_hero else f"Seat {seat + 1}"

    if is_hero:
        cards_html = "".join(render_face_up_card(r, s) for r, s in hole)
    else:
        cards_html = "".join(render_face_down_card() for _ in range(len(hole) or 2))

    return (
        f"<div class='{seat_class}' style='position:relative;'>"
        f"{badge}"
        f"<div style='font-weight:700;font-size:13px;'>{name}</div>"
        f"<div class='seat-metrics'>Stack ${stack} | Bet ${bet}</div>"
        f"<div style='display:flex;gap:6px;margin-top:8px;'>{cards_html}</div>"
        f"</div>"
    )


def _board_html(view: dict[str, Any]) -> str:
    board = view["board_cards"]
    slots: list[str] = []
    for i in range(5):
        if i < len(board):
            rank, suit = board[i]
            inner = render_face_up_card(
                rank, suit, classes=f"community-card deal-animate deal-delay-{i}"
            )
            slots.append(f"<div class='board-slot'>{inner}</div>")
        else:
            slots.append("<div class='board-slot placeholder'></div>")
    return "<div class='board-row-fixed'>" + "".join(slots) + "</div>"


def _render_table(view: dict[str, Any]) -> None:
    hero = view["hero"]
    opponents = [s for s in range(view["seat_count"]) if s != hero]

    # Opponent row
    cols = st.columns(len(opponents))
    for i, seat in enumerate(opponents):
        with cols[i]:
            st.markdown(_seat_html(seat, view), unsafe_allow_html=True)

    # Pot / phase banner
    st.markdown(
        f"<div style='text-align:center;margin:12px 0;padding:10px;"
        f"border:1px solid {theme.PANEL_BORDER};border-radius:12px;"
        f"background:{theme.PANEL_BG};color:{theme.TEXT_LIGHT};'>"
        f"<div style='font-size:13px;color:{theme.TEXT_MUTED};'>{view['phase']}</div>"
        f"<div style='font-size:26px;font-weight:700;'>Pot ${view['pot_total']:.0f}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Community board
    st.markdown(
        "<div style='margin:10px 0;'>" + _board_html(view) + "</div>",
        unsafe_allow_html=True,
    )

    # Hero seat
    st.markdown(_seat_html(hero, view), unsafe_allow_html=True)


def _render_controls(view: dict[str, Any]) -> None:
    # ── Hand over ────────────────────────────────────────────────────────────
    if view["hand_complete"]:
        payoffs = view["payoffs"]
        hero = view["hero"]
        hp = payoffs[hero] if hero < len(payoffs) else 0
        msg = "You won!" if hp > 0 else ("You lost." if hp < 0 else "Chopped pot.")
        st.success(f"Hand over — {msg}  ({hp:+.0f} chips)")
        if st.button("Deal Next Hand", key="next_hand_btn"):
            _new_hand()
            st.rerun()
        return

    # ── Waiting for a bot ────────────────────────────────────────────────────
    if not view["is_hero_turn"]:
        ti = view["turn_index"]
        label = f"Seat {ti + 1}" if ti is not None else "the engine"
        st.info(f"Waiting for {label} to act…")
        return

    # ── Hero's controls ───────────────────────────────────────────────────────
    if view["action_error"]:
        st.error(view["action_error"])

    facing = view["facing_bet"]
    call_amount = view["call_amount"]
    min_raise = view["min_raise"]
    max_raise = view["max_raise"]

    call_label = f"Call  ${call_amount}" if facing else "Check"
    bet_label = "Raise To" if facing else "Bet"

    col_fold, col_call, col_raise = st.columns([1, 1, 2])

    with col_fold:
        if st.button("Fold", use_container_width=True, key="btn_fold"):
            st.session_state.pending_action = {"type": "fold"}
            st.rerun()

    with col_call:
        if st.button(call_label, use_container_width=True, key="btn_call"):
            st.session_state.pending_action = {"type": "call"}
            st.rerun()

    with col_raise:
        # Clamp default value inside [min_raise, max_raise] to avoid Streamlit error.
        default_raise = max(min_raise, min(min_raise, max_raise))
        raise_amount = st.number_input(
            bet_label,
            min_value=min_raise,
            max_value=max(min_raise, max_raise),
            value=default_raise,
            step=1,
            key="raise_input",
        )
        if st.button(
            f"{bet_label}  ${raise_amount}", use_container_width=True, key="btn_raise"
        ):
            st.session_state.pending_action = {"type": "raise", "amount": raise_amount}
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def render_playable_poker_tab() -> None:
    st.markdown(poker_table_css(), unsafe_allow_html=True)

    # ── PILLAR 1: Setup ───────────────────────────────────────────────────────
    _ensure_initialized()
    state: Any = st.session_state.poker_state
    hero: int = st.session_state.hero

    # ── PILLAR 3: Strict execution order ──────────────────────────────────────

    # Step 1 – Process pending hero action (queued by a button click).
    pending = st.session_state.pop("pending_action", None)
    if pending is not None and state.turn_index == hero and state.status:
        _apply_hero_action(state, pending, hero)

    # Step 2 – Dealer: clears any dealing requests triggered by the hero's action.
    _run_dealer(state)

    # Step 3 – Bots + Dealer loop until hero's turn or hand ends.
    _run_bots_to_hero(state, hero)

    # ── Step 4: Render (read-only from here down) ─────────────────────────────
    view = _build_view(state, hero)
    _render_table(view)
    st.markdown("---")
    _render_controls(view)

    # Collapsible debug panel
    with st.expander("Debug", expanded=False):
        st.json(
            {
                "hero_seat": hero,
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
            }
        )

    if st.button("Restart (New Hand)", key="restart_btn"):
        _new_hand()
        st.rerun()
