"""Data cleaning utilities for poker behavioral analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleaningArtifacts:
    """Container for cleaned data and audits."""

    cleaned: pd.DataFrame
    parse_failures: pd.DataFrame
    malformed_actions: pd.DataFrame
    schema_snapshot: pd.DataFrame


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    """Load dataset and keep original columns untouched."""
    return pd.read_csv(path)


def snapshot_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return reproducible schema profile for notebook logging."""
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "null_count": [int(df[c].isna().sum()) for c in df.columns],
            "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )


def freeze_sample(
    df: pd.DataFrame,
    out_path: str | Path,
    n: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Create stable random sample used for parser iteration."""
    n = min(n, len(df))
    sampled = df.sample(n=n, random_state=seed) if n else df.copy()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out, index=False)
    return sampled


def _lookup_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    for col in df.columns:
        if any(alias.lower() in col.lower() for alias in aliases):
            return col
    return None


def _to_numeric(series: pd.Series) -> pd.Series:
    text = (
        series.astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .str.replace("nan", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(text, errors="coerce")


def clean_core_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize ids, buyin, datetime, and table size with parse flags."""
    cleaned = df.copy()
    parse_notes: list[dict[str, Any]] = []

    mappings = {
        "buyin": ["buyin", "buy_in", "buy-in"],
        "tournament_id": ["tournament_id", "tournamentid", "tour_id", "tournament"],
        "hand_id": ["hand_id", "handid", "id_hand", "hand"],
        "table_size": ["table_size", "players", "num_players", "seat_count"],
        "hand_datetime": ["date_time", "datetime", "timestamp", "date"],
    }

    found = {target: _lookup_column(cleaned, aliases) for target, aliases in mappings.items()}
    cleaned["parse_failed_core"] = False
    cleaned["parse_failed_core_reason"] = ""

    if found["buyin"]:
        cleaned["buyin"] = _to_numeric(cleaned[found["buyin"]]).astype("Float64")
    if found["tournament_id"]:
        cleaned["tournament_id"] = cleaned[found["tournament_id"]].astype(str).str.strip()
    if found["hand_id"]:
        cleaned["hand_id"] = cleaned[found["hand_id"]].astype(str).str.strip()
    if found["table_size"]:
        cleaned["table_size"] = _to_numeric(cleaned[found["table_size"]]).astype("Int64")
    if found["hand_datetime"]:
        cleaned["hand_datetime"] = pd.to_datetime(cleaned[found["hand_datetime"]], errors="coerce", utc=True)

    required_targets = ["buyin", "tournament_id", "hand_id", "table_size", "hand_datetime"]
    for target in required_targets:
        if target not in cleaned.columns:
            cleaned[target] = pd.NA
            parse_notes.append({"kind": "missing_column", "target": target})

    missing_mask = (
        cleaned["buyin"].isna()
        | cleaned["tournament_id"].isna()
        | cleaned["hand_id"].isna()
        | cleaned["table_size"].isna()
        | cleaned["hand_datetime"].isna()
    )
    cleaned.loc[missing_mask, "parse_failed_core"] = True
    cleaned.loc[missing_mask, "parse_failed_core_reason"] = "missing_or_invalid_core_value"

    for target, source in found.items():
        if source is None:
            parse_notes.append({"kind": "source_not_found", "target": target})

    failure_df = cleaned.loc[cleaned["parse_failed_core"], ["parse_failed_core", "parse_failed_core_reason"]].copy()
    if parse_notes:
        notes_df = pd.DataFrame(parse_notes)
        notes_df.index = range(len(notes_df))
        failure_df = pd.concat([failure_df, notes_df], axis=0, ignore_index=False)
    return cleaned, failure_df


def parse_action_line(action_text: str) -> list[dict[str, Any]]:
    """Parse compact action text into structured action tokens."""
    if not isinstance(action_text, str) or not action_text.strip():
        return []
    tokens = [t.strip() for t in re.split(r"[;|,]", action_text) if t.strip()]
    parsed: list[dict[str, Any]] = []
    for token in tokens:
        # Expected-ish examples: "P1:raise 12", "P2 call 12", "P3 folds"
        actor_match = re.search(r"(p?\d+|player[_ ]?\d+|sb|bb)", token, flags=re.I)
        amount_match = re.search(r"(-?\d+(\.\d+)?)", token)
        action = "unknown"
        lowered = token.lower()
        if "raise" in lowered:
            action = "raise"
        elif "call" in lowered:
            action = "call"
        elif "bet" in lowered:
            action = "bet"
        elif "fold" in lowered:
            action = "fold"
        elif "check" in lowered:
            action = "check"
        parsed.append(
            {
                "raw": token,
                "actor": actor_match.group(1).lower() if actor_match else "unknown",
                "action": action,
                "amount": float(amount_match.group(1)) if amount_match else 0.0,
            }
        )
    return parsed


def clean_betting_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse street actions and flag malformed betting sequences."""
    cleaned = df.copy()
    malformed_rows: list[dict[str, Any]] = []
    street_aliases = {
        "preflop": ["preflop", "pre_flop", "preflop_action"],
        "flop": ["flop", "flop_action"],
        "turn": ["turn", "turn_action"],
        "river": ["river", "river_action"],
    }

    for street, aliases in street_aliases.items():
        source = _lookup_column(cleaned, aliases)
        target = f"{street}_parsed_actions"
        if source is None:
            cleaned[target] = [[] for _ in range(len(cleaned))]
            continue
        cleaned[target] = cleaned[source].apply(parse_action_line)

    def validate_row(row: pd.Series) -> str:
        total = 0.0
        for street in ["preflop", "flop", "turn", "river"]:
            for action in row.get(f"{street}_parsed_actions", []):
                if action["action"] in {"bet", "raise", "call"}:
                    total += max(action["amount"], 0)
        pot_col = _lookup_column(cleaned, ["pot", "pot_size"])
        if pot_col:
            pot_val = _to_numeric(pd.Series([row.get(pot_col)])).iloc[0]
            if pd.notna(pot_val) and total > pot_val * 10:
                return "implausible_bet_to_pot_ratio"
        return ""

    cleaned["betting_validation_error"] = cleaned.apply(validate_row, axis=1)
    bad = cleaned["betting_validation_error"] != ""
    if bad.any():
        malformed_rows = cleaned.loc[bad, ["betting_validation_error"]].reset_index().to_dict("records")
    return cleaned, pd.DataFrame(malformed_rows)


def clean_dataset(df: pd.DataFrame, sample_out: str | Path | None = None) -> CleaningArtifacts:
    """Run full cleaning flow used across notebooks and modeling scripts."""
    schema = snapshot_schema(df)
    if sample_out:
        freeze_sample(df, sample_out)
    core_cleaned, parse_failures = clean_core_columns(df)
    full_cleaned, malformed_actions = clean_betting_columns(core_cleaned)
    return CleaningArtifacts(
        cleaned=full_cleaned,
        parse_failures=parse_failures,
        malformed_actions=malformed_actions,
        schema_snapshot=schema,
    )