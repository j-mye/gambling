"""Feature engineering for strength and bluff labeling."""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd


def _stable_float(seed_text: str) -> float:
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def evaluate_strength_by_street(df: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic proxy strengths per street (0 to 1)."""
    out = df.copy()
    base_key = (
        out.get("hand_id", pd.Series(out.index, index=out.index)).astype(str)
        + "|"
        + out.get("player_id", pd.Series("player", index=out.index)).astype(str)
    )
    out["strength_preflop"] = base_key.apply(lambda x: _stable_float(f"{x}|preflop"))
    out["strength_flop"] = (out["strength_preflop"] + base_key.apply(lambda x: _stable_float(f"{x}|flop"))) / 2
    out["strength_turn"] = (out["strength_flop"] + base_key.apply(lambda x: _stable_float(f"{x}|turn"))) / 2
    out["strength_river"] = (out["strength_turn"] + base_key.apply(lambda x: _stable_float(f"{x}|river"))) / 2
    out["strength_mean"] = out[["strength_preflop", "strength_flop", "strength_turn", "strength_river"]].mean(axis=1)
    return out


def validate_strength_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Diagnostics for strength drift by street."""
    checks = pd.DataFrame(index=df.index)
    checks["street_jump_max"] = df[["strength_preflop", "strength_flop", "strength_turn", "strength_river"]].diff(axis=1).abs().max(axis=1)
    checks["strength_out_of_bounds"] = (
        (df[["strength_preflop", "strength_flop", "strength_turn", "strength_river"]] < 0).any(axis=1)
        | (df[["strength_preflop", "strength_flop", "strength_turn", "strength_river"]] > 1).any(axis=1)
    )
    checks["weak_signal_flag"] = checks["street_jump_max"] < 1e-6
    return checks


def _aggression_proxy(df: pd.DataFrame, action_columns: Iterable[str]) -> pd.Series:
    text = pd.Series("", index=df.index, dtype="string")
    for col in action_columns:
        if col in df.columns:
            text = text + " " + df[col].astype("string").fillna("")
    lower = text.str.lower()
    raises = lower.str.count("raise").fillna(0)
    bets = lower.str.count("bet").fillna(0)
    calls = lower.str.count("call").fillna(0)
    return raises * 1.5 + bets * 1.2 + calls * 0.4


def label_bluffs(
    df: pd.DataFrame,
    strength_threshold: float = 0.35,
    aggression_threshold: float = 2.0,
) -> pd.DataFrame:
    """Label bluffs based on low strength and aggressive actions."""
    out = df.copy()
    out["aggression_score"] = _aggression_proxy(out, ["preflop", "flop", "turn", "river"])
    out["is_bluffing"] = (out["strength_mean"] <= strength_threshold) & (out["aggression_score"] >= aggression_threshold)
    dist = out["is_bluffing"].value_counts(normalize=True, dropna=False)
    imbalance = abs(float(dist.get(True, 0.0)) - 0.5)
    out["bluff_label_quality"] = np.where(imbalance > 0.4, "low", "good")
    out["bluff_confidence"] = (
        (aggression_threshold - out["aggression_score"]).abs() + (strength_threshold - out["strength_mean"]).abs()
    ).clip(0, 1)
    return out