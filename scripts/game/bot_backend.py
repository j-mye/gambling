"""Bot backend scaffolding for plug-and-play poker AI models.

This module intentionally contains contract and integration scaffolding only.
Custom strategy logic should be implemented by developers in CustomBot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import Any

USE_CUSTOM_MODEL = False
VALID_ACTIONS = {"FOLD", "CALL", "CHECK", "RAISE", "BET"}


class BasePokerBot(ABC):
    """Abstract interface all poker bots must follow."""

    def parse_game_state(self, state: Any, legal_moves: dict[str, Any]) -> dict[str, Any]:
        """Transform raw state into model-friendly features."""
        return {
            "turn_index": getattr(state, "turn_index", None),
            "street_index": getattr(state, "street_index", None),
            "legal_moves": dict(legal_moves),
        }

    def evaluate_hand_strength(self, hole_cards: list[str], state: Any) -> float:
        """Return rough hand strength estimate in [0, 1]."""
        _ = (hole_cards, state)
        return 0.5

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
        facing_bet = bool(legal_moves.get("facing_bet", False))
        can_call = bool(legal_moves.get("can_check_or_call", False))
        can_fold = bool(legal_moves.get("can_fold", False))

        if can_call:
            if facing_bet and can_fold and random.random() < 0.20:
                return self.format_response("FOLD", 0)
            return self.format_response("CALL", 0)
        if can_fold:
            return self.format_response("FOLD", 0)
        return self.format_response("CHECK", 0)


class CustomBot(BasePokerBot):
    """Placeholder custom bot implementation for developer extension."""

    def calculate_action(
        self,
        state: Any,
        hole_cards: list[str],
        legal_moves: dict[str, Any],
    ) -> tuple[str, int]:
        # Conservative placeholder until custom model logic is implemented.
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
    bot_cls: type[BasePokerBot] = CustomBot if use_custom_model else DefaultBot
    return {
        seat: bot_cls()
        for seat in range(player_count)
        if seat != hero_seat
    }
