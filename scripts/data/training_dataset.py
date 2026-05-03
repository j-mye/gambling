"""Build `cleanedGambling.csv`: schema aligned with `model_train` and dashboard models."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from scripts.features.hand_eval import evaluate_strength_by_street, label_bluffs
from scripts.features.streaks import add_allin_flags, add_streaks
from scripts.simulation.equity import add_preflop_equity


def _project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def _stable_player_id(name: object) -> str:
    return hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:16]


def _encode_table_position(pos: object) -> float:
    """Same scale as `POSITION_MAP` in poker_hand_strength (for win_model numeric feature)."""
    p = str(pos).strip().lower()
    mapping = {
        "btn": 1.0,
        "button": 1.0,
        "bu": 1.0,
        "co": 0.8,
        "cutoff": 0.8,
        "hj": 0.65,
        "mp": 0.65,
        "utg": 0.5,
        "sb": 0.25,
        "bb": 0.0,
    }
    return float(mapping.get(p, 0.5))


def _coerce_numeric_money(series: pd.Series) -> pd.Series:
    text = (
        series.astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .str.replace("nan", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(text, errors="coerce")


def _annotate_action_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """`label_bluffs` / `add_allin_flags` expect columns preflop, flop, turn, river."""
    out = df.copy()
    pairs = [
        ("action_pre", "preflop"),
        ("action_flop", "flop"),
        ("action_turn", "turn"),
        ("action_river", "river"),
    ]
    for src, dst in pairs:
        if src in out.columns:
            out[dst] = out[src].astype(str)
    return out


def build_cleaned_gambling_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Single source of truth for notebook export and reproducible training rows.

    Preserves raw columns required by `model_train.build_training_frame` (cards, boards,
    bets, pots, stack, blinds, position, hand_id, result, balance, …) and appends
    dashboard-model features (streaks, bluff labels, equity proxy, encoded position).

    Does **not** duplicate z-scored stage feature matrices — those are recomputed at
    train/inference time via `build_stage_feature_payload`.
    """
    df = raw.copy()

    # Normalized ids / typing used across the project
    if "hand_id" in df.columns:
        df["hand_id"] = df["hand_id"].astype(str).str.strip()
    if "name" in df.columns:
        df["player_id"] = df["name"].map(_stable_player_id)
    else:
        df["player_id"] = "unknown"

    # Datetime for streak ordering
    if {"date", "time"}.issubset(df.columns):
        df["hand_datetime"] = pd.to_datetime(
            df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
            errors="coerce",
            utc=True,
        )
    elif "hand_datetime" not in df.columns:
        df["hand_datetime"] = pd.NaT

    # Core numerics (names match `model_train` / PokerKit UI)
    for c in [
        "stack",
        "balance",
        "pot_pre",
        "pot_flop",
        "pot_turn",
        "pot_river",
        "bet_pre",
        "bet_flop",
        "bet_turn",
        "bet_river",
        "table_size",
        "ante",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "buyin" in df.columns:
        df["buyin"] = _coerce_numeric_money(df["buyin"]).astype("Float64")

    # Outcome targets
    df["net_result"] = pd.to_numeric(df.get("balance"), errors="coerce").fillna(0.0)
    result_s = df.get("result", "").fillna("").astype(str)
    df["won_flag"] = result_s.str.contains("took chips|won", case=False, regex=True).astype(int)

    # Win / money / bluff models
    df["starting_stack"] = pd.to_numeric(df.get("stack"), errors="coerce").fillna(0.0)
    df["table_position"] = df.get("position", "").map(_encode_table_position)

    df = _annotate_action_aliases(df)
    df = evaluate_strength_by_street(df)
    df = label_bluffs(df)
    df = add_streaks(df)
    df = add_allin_flags(df)
    df = add_preflop_equity(df, hand_col="cards")

    if "is_bluffing" in df.columns:
        df["is_bluffing"] = df["is_bluffing"].astype(int)

    # Drop raw screen name from export (keep hashed player_id)
    if "name" in df.columns:
        df = df.drop(columns=["name"])

    # Stable column order: identifiers first
    priority = [
        "hand_id",
        "player_id",
        "hand_datetime",
        "tourn_id",
        "table",
        "table_size",
        "seat",
        "position",
        "table_position",
        "cards",
        "board_flop",
        "board_turn",
        "board_river",
        "stack",
        "starting_stack",
        "blinds",
        "bet_pre",
        "bet_flop",
        "bet_turn",
        "bet_river",
        "pot_pre",
        "pot_flop",
        "pot_turn",
        "pot_river",
        "result",
        "balance",
        "net_result",
        "won_flag",
    ]
    rest = [c for c in sorted(df.columns) if c not in priority]
    ordered = [c for c in priority if c in df.columns] + rest
    return df[ordered].reset_index(drop=True)


def export_cleaned_gambling_csv(
    raw_path: str | Path | None = None,
    out_path: str | Path | None = None,
) -> Path:
    raw_path = Path(raw_path) if raw_path is not None else _project_root_from_here() / "data" / "gambling.csv"
    out_path = Path(out_path) if out_path is not None else _project_root_from_here() / "data" / "cleanedGambling.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(raw_path)
    cleaned = build_cleaned_gambling_dataframe(raw)
    cleaned.to_csv(out_path, index=False)
    return out_path


def export_model_ready_csv(
    cleaned: pd.DataFrame,
    out_path: str | Path | None = None,
) -> Path:
    """Slim EDA export for Streamlit (see ``feature_contracts.eda_export_columns``)."""
    from scripts.models.feature_contracts import eda_export_columns

    out_path = Path(out_path) if out_path is not None else _project_root_from_here() / "artifacts" / "processed" / "model_ready.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    want = eda_export_columns()
    cols = [c for c in want if c in cleaned.columns]
    cleaned.reindex(columns=cols).to_csv(out_path, index=False)
    return out_path


def describe_training_alignment() -> str:
    """Short reference for notebooks."""
    from scripts.models import feature_contracts as fc

    return (
        "Stage win models (`model_train.py` → `poker_models.pkl`): "
        "`scripts/models/feature_contracts.STAGE_FEATURES`, built via "
        "`build_stage_feature_payload` (same contract as the poker UI).\n"
        "Visible-bluff model (`visible_bluff_train.py` → `visible_bluff_model.pkl`): "
        "observable table features only — see `scripts/features/visible_bluff_features.py`.\n"
        f"Ignored duplicate street payload keys: {fc.DROPPED_FROM_STAGE_PAYLOAD_KEYS}."
    )
