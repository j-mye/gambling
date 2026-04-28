"""Monte Carlo simulators for folded hand opportunity cost."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any

import numpy as np
import pandas as pd


def filter_folded_hands(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    actions = (
        out.get("preflop", "").astype(str)
        + " "
        + out.get("flop", "").astype(str)
        + " "
        + out.get("turn", "").astype(str)
        + " "
        + out.get("river", "").astype(str)
    ).str.lower()
    out["fold_filter_reason"] = np.where(actions.str.contains("fold"), "ok", "no_fold_detected")
    folded = out[out["fold_filter_reason"] == "ok"].copy()
    excluded = out[out["fold_filter_reason"] != "ok"][["fold_filter_reason"]].copy()
    return folded, excluded


def _seeded_strength(row: pd.Series, seed: int) -> float:
    text = f"{row.get('hand_id', '')}|{row.get('player_id', '')}|{seed}"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def simulate_folded_hand(row: pd.Series, runs: int = 1000, seed: int = 42) -> dict[str, Any]:
    rng = random.Random(seed)
    wins = 0
    probs: list[float] = []
    base = _seeded_strength(row, seed)
    for _ in range(runs):
        p = max(0.01, min(0.99, base + rng.uniform(-0.2, 0.2)))
        probs.append(p)
        wins += int(rng.random() < p)
    win_rate = wins / max(runs, 1)
    stderr = math.sqrt((win_rate * (1 - win_rate)) / max(runs, 1))
    ci95 = 1.96 * stderr
    pot = float(row.get("pot_size", row.get("pot", 0.0)) or 0.0)
    invested = float(row.get("invested", 0.0) or 0.0)
    ev_sacrificed = (win_rate * pot) - invested
    return {
        "hand_id": row.get("hand_id", ""),
        "runs": runs,
        "win_rate": win_rate,
        "ci95_low": max(0.0, win_rate - ci95),
        "ci95_high": min(1.0, win_rate + ci95),
        "ev_sacrificed": ev_sacrificed,
    }


def simulate_folded_dataset(df: pd.DataFrame, runs: int = 1000, seed: int = 42) -> pd.DataFrame:
    records = [simulate_folded_hand(row, runs=runs, seed=seed) for _, row in df.iterrows()]
    result = pd.DataFrame(records)
    if not result.empty:
        result["buyin_bucket"] = pd.cut(pd.to_numeric(df.get("buyin", 0), errors="coerce").fillna(0), bins=4, duplicates="drop")
    return result


def aggregate_ev(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["segment", "ev_sacrificed_mean", "win_rate_mean"])
    segments = []
    for col in ["player_id", "buyin_bucket", "win_streak", "loss_streak"]:
        if col in df.columns:
            grouped = (
                df.groupby(col, dropna=False)[["ev_sacrificed", "win_rate"]]
                .mean()
                .rename(columns={"ev_sacrificed": "ev_sacrificed_mean", "win_rate": "win_rate_mean"})
                .reset_index()
            )
            grouped["segment"] = col
            segments.append(grouped)
    return pd.concat(segments, ignore_index=True) if segments else pd.DataFrame()