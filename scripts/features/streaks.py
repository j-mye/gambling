"""Stack and streak features for recurring players."""

from __future__ import annotations

import pandas as pd


def add_starting_stacks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["player_id", "hand_datetime", "hand_id"], na_position="last").reset_index(drop=True)
    buyin = pd.to_numeric(out.get("buyin", 0), errors="coerce").fillna(0.0)
    pnl = pd.to_numeric(out.get("net_result", 0), errors="coerce").fillna(0.0)
    out["starting_stack"] = buyin.groupby(out["player_id"]).cumsum() - pnl.groupby(out["player_id"]).shift(fill_value=0).cumsum()
    out["stack_transition_error"] = out["starting_stack"] < 0
    return out


def add_streaks(df: pd.DataFrame, session_break_hours: int = 12) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["player_id", "hand_datetime", "hand_id"], na_position="last").reset_index(drop=True)
    pnl = pd.to_numeric(out.get("net_result", 0), errors="coerce").fillna(0.0)
    out["is_win"] = pnl > 0
    out["is_loss"] = pnl < 0
    time_gap = out.groupby("player_id")["hand_datetime"].diff().dt.total_seconds().fillna(0) / 3600
    session_reset = time_gap > session_break_hours
    out["session_reset"] = session_reset

    win_streak: list[int] = []
    loss_streak: list[int] = []
    current_win = 0
    current_loss = 0
    prev_player = None
    for row in out.itertuples():
        if prev_player != row.player_id or row.session_reset:
            current_win = 0
            current_loss = 0
        if row.is_win:
            current_win += 1
            current_loss = 0
        elif row.is_loss:
            current_loss += 1
            current_win = 0
        else:
            current_win = 0
            current_loss = 0
        win_streak.append(current_win)
        loss_streak.append(current_loss)
        prev_player = row.player_id

    out["win_streak"] = win_streak
    out["loss_streak"] = loss_streak
    out["streak_short"] = (out["win_streak"] >= 2) | (out["loss_streak"] >= 2)
    out["streak_medium"] = (out["win_streak"] >= 4) | (out["loss_streak"] >= 4)
    out["streak_long"] = (out["win_streak"] >= 7) | (out["loss_streak"] >= 7)
    return out


def add_allin_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text = (
        out.get("preflop", "").astype(str)
        + " "
        + out.get("flop", "").astype(str)
        + " "
        + out.get("turn", "").astype(str)
        + " "
        + out.get("river", "").astype(str)
    ).str.lower()
    out["is_all_in"] = text.str.contains("all-in|all in", regex=True)
    out["all_in_type"] = "none"
    out.loc[out["is_all_in"] & (pd.to_numeric(out.get("net_result", 0), errors="coerce").fillna(0.0) > 0), "all_in_type"] = "winning_all_in"
    out.loc[out["is_all_in"] & (pd.to_numeric(out.get("net_result", 0), errors="coerce").fillna(0.0) < 0), "all_in_type"] = "losing_all_in"
    return out