"""Browser-global poker session state (replaces Streamlit session_state)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GameState:
    poker_state: Any | None = None
    deck: list[str] = field(default_factory=list)
    master_bankrolls: list[int] = field(default_factory=list)
    button_index: int = 0
    hero: int | None = None
    bot_instances: dict[int, Any] = field(default_factory=dict)
    last_action: dict[int, str] = field(default_factory=dict)
    terminal_payload: dict[str, Any] | None = None
    hand_resolved: bool = False
    resolved_hand_id: int | None = None
    action_error: str = ""
    bot_error: str = ""
    raise_amount: int = 0
    bots_running: bool = False
    ml_enabled: bool = False
    # Fingerprint of visible cards; skip innerHTML when unchanged (avoids flicker).
    ui_cards_sig: object | None = None
    # id(poker_state) when ui_cards_sig was last reset (new hand).
    ui_hand_pyid: int | None = None
