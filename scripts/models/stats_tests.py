"""Statistical tests and grouped profile analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def streak_impact_tests(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    agg = pd.to_numeric(df.get("aggression_score", 0), errors="coerce").fillna(0.0)
    win = agg[(pd.to_numeric(df.get("win_streak", 0), errors="coerce").fillna(0) >= 2)]
    base = agg[(pd.to_numeric(df.get("win_streak", 0), errors="coerce").fillna(0) == 0)]
    if len(win) > 1 and len(base) > 1:
        t, p = stats.ttest_ind(win, base, equal_var=False)
        out["t_stat_win_streak"] = float(t)
        out["p_value_win_streak"] = float(p)
        out["cohens_d_win_streak"] = float((win.mean() - base.mean()) / (np.sqrt((win.std() ** 2 + base.std() ** 2) / 2) + 1e-9))
    return out


def bankroll_impact(df: pd.DataFrame) -> dict[str, float]:
    stack = pd.to_numeric(df.get("starting_stack", 0), errors="coerce").fillna(0.0)
    aggression = pd.to_numeric(df.get("aggression_score", 0), errors="coerce").fillna(0.0)
    corr, p = stats.pearsonr(stack, aggression) if len(stack) > 2 else (0.0, 1.0)
    return {"stack_aggression_correlation": float(corr), "stack_aggression_p_value": float(p)}


def allin_profiles(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["strength_mean"] = pd.to_numeric(data.get("strength_mean", 0), errors="coerce").fillna(0.0)
    data["net_result"] = pd.to_numeric(data.get("net_result", 0), errors="coerce").fillna(0.0)
    filt = data[data.get("is_all_in", False).astype(bool)]
    if filt.empty:
        return pd.DataFrame(columns=["all_in_type", "avg_strength", "avg_net_result", "count"])
    grouped = (
        filt.assign(all_in_type=np.where(filt["net_result"] > 0, "all_in_win", "all_in_loss"))
        .groupby("all_in_type", dropna=False)
        .agg(avg_strength=("strength_mean", "mean"), avg_net_result=("net_result", "mean"), count=("net_result", "size"))
        .reset_index()
    )
    return grouped