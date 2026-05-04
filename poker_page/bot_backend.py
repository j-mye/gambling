"""Bot backend: heuristic aggression engine (PokerKit strength + pot odds)."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
import random
from typing import Any

from pokerkit import StandardHighHand

USE_CUSTOM_MODEL = False
USE_HEURISTIC_BOT = True

VALID_ACTIONS = {"FOLD", "CALL", "CHECK", "RAISE", "BET"}

# Heuristic policy (tunable)
# Marginal "like the hand" threshold (facing: used for flat-call band, not auto 3-bet).
AGGRESSION_RAISE = 0.70
# Unchecked / unopened: open or isolate when strength clears this bar.
OPENING_RAISE_STRENGTH = 0.75
WEAK_FOLD = 0.30
BLUFF_RAISE_PROB = 0.10
AGGRESSION_JITTER = 0.02

# Scaling fear: cost_to_call = call_amount / (pot_total + call_amount) — same as legal_moves["pot_odds"].
SCALING_FEAR_RATIO = 0.40
SCALING_FEAR_PREMIUM_STRENGTH = 0.85
# Facing a bet: no re-raise unless strength strictly exceeds this (stops raise/call/raise loops).
FACE_RERAISE_MIN_STRENGTH = 0.90
# All-in raise safety: only commit full stack on calculated raise if strength exceeds this.
ALL_IN_RAISE_PREMIUM_STRENGTH = 0.98
# Trapping: fraction of time a planned raise/bet becomes flat call/check.
TRAP_FLAT_CALL_PROB = 0.30

# StandardHighHand: weakest 7-high … strongest royal; indices from PokerKit 0.7.3.
_STANDARD_HIGH_MAX_INDEX = 7461


def _split_hole_token(tok: str) -> tuple[int, str] | None:
    t = tok.strip()
    if not t:
        return None
    m = re.match(r"(?i)^([2-9]|10|t|j|q|k|a)([cdhs])$", t)
    if not m:
        return None
    r_raw, suit = m.group(1).upper(), m.group(2).lower()
    if r_raw == "T":
        r_raw = "10"
    rmap = {
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "10": 10,
        "J": 11,
        "Q": 12,
        "K": 13,
        "A": 14,
    }
    rv = rmap.get(r_raw)
    if rv is None:
        return None
    return rv, suit


def _preflop_strength(hole_cards: list[str]) -> float:
    """Map preflop hole cards to ~[0, 1] without a full board (Option A)."""
    if len(hole_cards) < 2:
        return 0.35
    a = _split_hole_token(hole_cards[0])
    b = _split_hole_token(hole_cards[1])
    if a is None or b is None:
        return 0.35
    ra, sa = a
    rb, sb = b
    high, low = max(ra, rb), min(ra, rb)
    base = (high + low - 4.0) / 26.0
    bonus = 0.0
    if ra == rb:
        bonus += 0.12 + 0.003 * (ra - 2.0)
    if sa == sb:
        bonus += 0.045
    if abs(ra - rb) <= 2 and ra != rb:
        bonus += 0.025
    if high >= 12:
        bonus += 0.04
    if high == 14:
        bonus += 0.03
    return min(0.92, max(0.06, base + bonus))


def _board_compact(state: Any) -> str:
    """Concatenate board card tokens in PokerKit compact form (e.g. KhJdTh)."""
    parts: list[str] = []
    for card in list(getattr(state, "board_cards", None) or []):
        s = str(card)
        m = re.search(r"\((\w+)\)", s)
        if m:
            token = m.group(1)
        else:
            m2 = re.search(r"(?i)\b(10|[2-9tjqka])([cdhs])\b", s)
            if not m2:
                continue
            r, su = m2.group(1), m2.group(2)
            token = f"{r}{su}"
        rank = token[:-1].upper()
        suit = token[-1].lower()
        if rank == "T":
            rank = "10"
        parts.append(f"{rank}{suit}")
    return "".join(parts)


def _postflop_strength_normalized(hole_compact: str, board_compact: str) -> float:
    hand = StandardHighHand.from_game(hole_compact, board_compact)
    return float(hand.entry.index) / float(_STANDARD_HIGH_MAX_INDEX)


def _normalized_strength(hole_cards: list[str], state: Any) -> float:
    """Hand strength in [0, 1]: preflop chart; postflop PokerKit best 5 of 7."""
    board = _board_compact(state)
    if len(board) < 3:
        return _preflop_strength(hole_cards)
    hole_compact = "".join(hole_cards)
    try:
        return _postflop_strength_normalized(hole_compact, board)
    except Exception:
        return _preflop_strength(hole_cards)


def _seat_stack(state: Any, seat: int) -> int:
    try:
        stacks = list(getattr(state, "stacks", None) or [])
        if 0 <= seat < len(stacks):
            return int(stacks[seat])
    except Exception:
        pass
    return 0


def _seat_street_bet(state: Any, seat: int) -> int:
    try:
        bets = list(getattr(state, "bets", None) or [])
        if 0 <= seat < len(bets):
            return int(bets[seat])
    except Exception:
        return 0


def _chips_to_complete_to(state: Any, seat: int, target_to: int) -> int:
    """Extra chips from this seat's stack needed to complete a raise/bet to target_to."""
    return max(0, int(target_to) - _seat_street_bet(state, seat))


def _cost_to_call_ratio(legal_moves: dict[str, Any]) -> float:
    """call_amount / (pot_total + call_amount); mirrors poker_core._legal_moves_snapshot pot_odds."""
    cached = legal_moves.get("pot_odds")
    if cached is not None:
        return float(cached)
    ca = int(legal_moves.get("call_amount", 0) or 0)
    if ca <= 0:
        return 0.0
    pt = float(legal_moves.get("pot_total", 0.0) or 0.0)
    return float(ca) / max(pt + float(ca), 1e-9)


def _raise_complete_to(legal_moves: dict[str, Any]) -> int:
    """Target completion = half pot (chips), clamped to legal [min_raise_to, max_raise_to]."""
    mn = legal_moves.get("min_raise_to")
    mx = legal_moves.get("max_raise_to")
    if mn is None or mx is None:
        raise ValueError("Raise bounds unavailable")
    mn_i, mx_i = int(mn), int(mx)
    pot = float(legal_moves.get("pot_total", 0.0) or 0.0)
    raw = int(pot * 0.5)
    return int(min(mx_i, max(mn_i, raw)))


def _raise_would_commit_stack_or_more(
    state: Any, seat: int, target_to: int, strength: float
) -> bool:
    """True if we should revert raise → call (keep shove only when strength >= ALL_IN threshold)."""
    stack = _seat_stack(state, seat)
    need = _chips_to_complete_to(state, seat, target_to)
    return need >= stack and strength < ALL_IN_RAISE_PREMIUM_STRENGTH


def _passive_fallback(
    legal_moves: dict[str, Any],
) -> tuple[str, int]:
    """Conservative call/fold when heuristic fails."""
    facing_bet = bool(legal_moves.get("facing_bet", False))
    can_call = bool(legal_moves.get("can_check_or_call", False))
    can_fold = bool(legal_moves.get("can_fold", False))
    if can_call:
        if facing_bet and can_fold and random.random() < 0.20:
            return normalize_bot_response(("FOLD", 0))
        return normalize_bot_response(("CALL", 0))
    if can_fold:
        return normalize_bot_response(("FOLD", 0))
    return normalize_bot_response(("CHECK", 0))


class BasePokerBot(ABC):
    """Abstract interface all poker bots must follow."""

    def format_response(self, action: str, amount: int) -> tuple[str, int]:
        """Normalize action response to strict (ACTION, AMOUNT) contract."""
        return normalize_bot_response((action, amount))

    @abstractmethod
    def calculate_action(
        self,
        state: Any,
        hole_cards: list[str],
        legal_moves: dict[str, Any],
    ) -> tuple[str, int]:
        """Return bot action as (ACTION, AMOUNT)."""


def normalize_bot_response(raw: Any) -> tuple[str, int]:
    """Validate and normalize bot output to strict tuple contract."""
    if not isinstance(raw, (tuple, list)) or len(raw) != 2:
        raise ValueError("Bot response must be a 2-item tuple/list")
    action = str(raw[0]).strip().upper()
    if action not in VALID_ACTIONS:
        raise ValueError(f"Unsupported bot action: {action}")
    try:
        amount = int(raw[1])
    except Exception as exc:
        raise ValueError("Bot amount must be an integer") from exc
    return action, amount


class DefaultBot(BasePokerBot):
    """Safe fallback bot that mirrors current baseline behavior."""

    def calculate_action(
        self,
        state: Any,
        hole_cards: list[str],
        legal_moves: dict[str, Any],
    ) -> tuple[str, int]:
        _ = (state, hole_cards)
        return _passive_fallback(legal_moves)


class HeuristicAggressionBot(BasePokerBot):
    """Strength + pot-odds heuristic with occasional aggressive bluffs."""

    def _aggressive_raise_or_flat(
        self,
        state: Any,
        lm: dict[str, Any],
        seat: int,
        strength: float,
        *,
        facing: bool,
        can_raise: bool,
        can_cc: bool,
    ) -> tuple[str, int] | None:
        """Planned raise/bet with trapping (30% flat) and all-in sizing safety; None = caller decides."""
        if not can_raise:
            return None
        if random.random() < TRAP_FLAT_CALL_PROB and can_cc:
            return self.format_response("CALL" if facing else "CHECK", 0)
        target = _raise_complete_to(lm)
        if _raise_would_commit_stack_or_more(state, seat, target, strength):
            if can_cc:
                return self.format_response("CALL" if facing else "CHECK", 0)
            return None
        return self.format_response("RAISE" if facing else "BET", target)

    def calculate_action(
        self,
        state: Any,
        hole_cards: list[str],
        legal_moves: dict[str, Any],
    ) -> tuple[str, int]:
        try:
            lm = legal_moves
            seat = int(lm.get("seat", 0))
            facing = bool(lm.get("facing_bet", False))
            can_fold = bool(lm.get("can_fold", False))
            can_cc = bool(lm.get("can_check_or_call", False))
            can_raise = bool(lm.get("can_raise", False))
            po = float(lm.get("pot_odds", 0.0) or 0.0)

            s = float(_normalized_strength(hole_cards, state))
            if AGGRESSION_JITTER > 0:
                s = min(1.0, max(0.0, s + random.uniform(-AGGRESSION_JITTER, AGGRESSION_JITTER)))

            cost_ratio = _cost_to_call_ratio(lm)
            if (
                facing
                and cost_ratio > SCALING_FEAR_RATIO
                and s <= SCALING_FEAR_PREMIUM_STRENGTH
                and can_fold
            ):
                return self.format_response("FOLD", 0)

            medium_bluff = (
                WEAK_FOLD <= s < OPENING_RAISE_STRENGTH
                and random.random() < BLUFF_RAISE_PROB
            )
            likes_hand = (s >= AGGRESSION_RAISE) or medium_bluff
            wants_open = (s >= OPENING_RAISE_STRENGTH) or medium_bluff

            if facing:
                if s < WEAK_FOLD and can_fold:
                    return self.format_response("FOLD", 0)
                # Re-raise / 3-bet only with true premiums; otherwise flat and stop ping-pong.
                if s > FACE_RERAISE_MIN_STRENGTH and likes_hand and can_raise:
                    agg = self._aggressive_raise_or_flat(
                        state, lm, seat, s, facing=True, can_raise=can_raise, can_cc=can_cc
                    )
                    if agg is not None:
                        return agg
                if likes_hand and can_cc:
                    return self.format_response("CALL", 0)
                if s > po and can_cc:
                    return self.format_response("CALL", 0)
                if can_fold:
                    return self.format_response("FOLD", 0)
                if can_cc:
                    return self.format_response("CALL", 0)
                return _passive_fallback(lm)

            if wants_open:
                agg = self._aggressive_raise_or_flat(
                    state, lm, seat, s, facing=False, can_raise=can_raise, can_cc=can_cc
                )
                if agg is not None:
                    return agg
            if can_cc:
                return self.format_response("CHECK", 0)
            return _passive_fallback(lm)
        except Exception:
            return _passive_fallback(legal_moves)


class CustomBot(BasePokerBot):
    """Placeholder custom bot implementation for developer extension."""

    def calculate_action(
        self,
        state: Any,
        hole_cards: list[str],
        legal_moves: dict[str, Any],
    ) -> tuple[str, int]:
        _ = (state, hole_cards)
        if legal_moves.get("can_check_or_call", False):
            return self.format_response("CALL", 0)
        if legal_moves.get("can_fold", False):
            return self.format_response("FOLD", 0)
        return self.format_response("CHECK", 0)


def build_bot_instances(
    player_count: int,
    hero_seat: int,
    use_custom_model: bool = USE_CUSTOM_MODEL,
) -> dict[int, BasePokerBot]:
    """Build all-or-nothing bot seat map for all non-hero seats."""
    if use_custom_model:
        bot_cls: type[BasePokerBot] = CustomBot
    elif USE_HEURISTIC_BOT:
        bot_cls = HeuristicAggressionBot
    else:
        bot_cls = DefaultBot
    return {
        seat: bot_cls()
        for seat in range(player_count)
        if seat != hero_seat
    }
