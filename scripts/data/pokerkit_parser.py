"""Parser layer that translates cleaned rows into replayable poker states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class HandReplayResult:
    hand_id: str
    actions_applied: int
    terminal_pot: float
    status: str
    error: str = ""


def _extract_actions(row: pd.Series) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for street in ["preflop", "flop", "turn", "river"]:
        actions.extend(row.get(f"{street}_parsed_actions", []))
    return actions


def row_to_state(row: pd.Series) -> dict[str, Any]:
    """
    Build a canonical pseudo-state object.

    The project can swap this dict with a true PokerKit state object once the
    dataset format is finalized. Keeping this shape stable allows downstream
    simulation and validation to proceed immediately.
    """
    players = int(row.get("table_size", 2) or 2)
    stacks = {f"p{i + 1}": float(row.get("buyin", 0.0) or 0.0) for i in range(players)}
    return {
        "hand_id": str(row.get("hand_id", "")),
        "tournament_id": str(row.get("tournament_id", "")),
        "actions": _extract_actions(row),
        "players": players,
        "stacks": stacks,
        "target_pot": float(row.get("pot_size", row.get("pot", 0.0)) or 0.0),
    }


def replay_actions(state: dict[str, Any]) -> HandReplayResult:
    """Replay parsed actions and produce comparable terminal metrics."""
    pot = 0.0
    actions_applied = 0
    try:
        for action in state["actions"]:
            act = action.get("action", "unknown")
            amt = max(float(action.get("amount", 0.0) or 0.0), 0.0)
            if act in {"bet", "raise", "call"}:
                pot += amt
            actions_applied += 1
        return HandReplayResult(
            hand_id=state["hand_id"],
            actions_applied=actions_applied,
            terminal_pot=pot,
            status="ok",
        )
    except Exception as exc:  # defensive parser path
        return HandReplayResult(
            hand_id=state.get("hand_id", ""),
            actions_applied=actions_applied,
            terminal_pot=pot,
            status="error",
            error=str(exc),
        )


def validate_replays(df: pd.DataFrame, tolerance: float = 2.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay hand subset and compare computed pot with dataset pot."""
    results: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        state = row_to_state(row)
        replay = replay_actions(state)
        observed = float(row.get("pot_size", row.get("pot", 0.0)) or 0.0)
        delta = replay.terminal_pot - observed
        payload = {
            "hand_id": replay.hand_id,
            "actions_applied": replay.actions_applied,
            "computed_pot": replay.terminal_pot,
            "observed_pot": observed,
            "delta": delta,
            "status": replay.status,
            "error": replay.error,
        }
        results.append(payload)
        if replay.status != "ok" or abs(delta) > tolerance:
            payload["mismatch_reason"] = "replay_error" if replay.status != "ok" else "pot_delta"
            mismatches.append(payload)
    return pd.DataFrame(results), pd.DataFrame(mismatches)


def categorize_mismatches(mismatches: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mismatch causes for iteration tracking."""
    if mismatches.empty:
        return pd.DataFrame(columns=["mismatch_reason", "count"])
    return (
        mismatches.groupby("mismatch_reason", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )