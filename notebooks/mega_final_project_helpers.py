"""Helpers for `00_mega_final_project.ipynb`.

This module holds the long-form functions so the notebook stays readable.
The notebook imports from here; this file lives next to the notebook under `notebooks/`.

**Hand strength / win modeling alignment**

- **Training & this notebook:** `best_combination_from_tokens` and `hand_strength_from_tokens` here follow the
  same category ladder as `scripts/features/poker_hand_strength.py` (used by `model_train.py` and
  `predict_stage_win_probability` when `poker_models.pkl` is present).
- **PyScript poker UI (default):** [`poker_page/hand_eval.py`](../poker_page/hand_eval.py) duplicates that
  ladder for WASM; [`poker_page/heuristics.py`](../poker_page/heuristics.py) maps tier + context to a live
  win *proxy* when `GameState.ml_enabled` is false (typical). Section 6c in the mega notebook demonstrates
  agreement on the best-hand label for a sample board.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

# Project paths (repo root is parent of `notebooks/`)
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "gambling.csv"
CLEAN_PATH = ROOT / "data" / "cleanedGambling.csv"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# -----------------------------
# Generic helpers
# -----------------------------

def safe_float(x: object, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default


def stable_player_id(name: object) -> str:
    return hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:16]


def stable_float(seed_text: str) -> float:
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


# -----------------------------
# Card parsing and poker features
# -----------------------------
RANK_MAP = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}

COMBINATION_ORDER = [
    "unknown", "high card", "pair", "two pair", "three of a kind",
    "straight", "flush", "full house", "four of a kind", "straight flush"
]
COMBINATION_MAP = {c: i for i, c in enumerate(COMBINATION_ORDER)}
POSITION_MAP = {"SB": 0.0, "BB": 0.25, "UTG": 0.5, "MP": 0.65, "CO": 0.8, "BTN": 1.0}

STAGE_FEATURES = {
    "preflop": [
        "hole_rank_high", "hole_rank_low", "is_pair", "is_suited", "rank_gap",
        "has_ace", "has_broadway", "preflop_strength", "hero_stack_bb", "effective_stack_bb", "spr"
    ],
    "flop": [
        "hand_strength", "pair_count", "trips_count", "flush_draw", "open_ended_straight_draw",
        "gutshot_draw", "board_high_rank", "board_is_paired", "hero_stack_bb", "effective_stack_bb", "spr",
        "board_four_flush", "board_straight_present", "board_straight_4liner", "board_pair_count",
        "board_trips_present", "board_full_house_present", "hero_uses_hole_for_best", "shared_strength_gap",
        "board_shared_strength_risk"
    ],
    "turn": [
        "hand_strength", "pair_count", "trips_count", "flush_draw", "open_ended_straight_draw",
        "gutshot_draw", "board_high_rank", "board_is_paired", "hero_stack_bb", "effective_stack_bb", "spr",
        "board_four_flush", "board_straight_present", "board_straight_4liner", "board_pair_count",
        "board_trips_present", "board_full_house_present", "hero_uses_hole_for_best", "shared_strength_gap",
        "board_shared_strength_risk"
    ],
    "river": [
        "hand_strength", "pair_count", "trips_count", "board_high_rank", "board_is_paired", "hero_stack_bb",
        "effective_stack_bb", "spr", "board_four_flush", "board_straight_present", "board_straight_4liner",
        "board_pair_count", "board_trips_present", "board_full_house_present", "hero_uses_hole_for_best",
        "shared_strength_gap", "board_shared_strength_risk"
    ],
}

VISIBLE_BLUFF_FEATURE_KEYS = (
    "position_value", "hand_strength", "pair_count", "trips_count", "flush_draw",
    "open_ended_straight_draw", "gutshot_draw", "board_high_rank", "board_is_paired",
    "board_four_flush", "board_straight_present", "board_straight_4liner", "board_pair_count",
    "board_trips_present", "board_full_house_present", "board_shared_strength_risk",
    "board_only_strength", "hero_stack_bb", "effective_stack_bb", "spr"
)
VISIBLE_BLUFF_FEATURES = list(VISIBLE_BLUFF_FEATURE_KEYS) + ["seat_bet_to_pot", "seat_raise_count", "street_phase"]


def parse_cards(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if text in {"", "0", "--", "nan", "None"}:
        return []
    out: list[str] = []
    for token in text.split():
        t = token.strip()
        if len(t) >= 3 and t[:2].upper() == "10":
            suit = t[2].lower()
            if suit in {"c", "d", "h", "s"}:
                out.append(f"10{suit}")
            continue
        if len(t) < 2:
            continue
        rank = t[0].upper()
        suit = t[1].lower()
        if rank in RANK_MAP and suit in {"c", "d", "h", "s"}:
            out.append(f"{rank}{suit}")
    return out


def card_rank(card: str) -> int | None:
    if not card:
        return None
    if len(card) >= 3 and card[:2].upper() == "10":
        return 10
    if len(card) < 2:
        return None
    return RANK_MAP.get(card[0].upper())


def card_suit(card: str) -> str | None:
    if not card:
        return None
    if len(card) >= 3 and card[:2].upper() == "10":
        s = card[2].lower()
    elif len(card) >= 2:
        s = card[1].lower()
    else:
        return None
    return s if s in {"c", "d", "h", "s"} else None


def ranks(cards: list[str]) -> list[int]:
    return [r for c in cards if (r := card_rank(c)) is not None]


def suits(cards: list[str]) -> list[str]:
    return [s for c in cards if (s := card_suit(c)) is not None]


def is_straight(rank_values: list[int]) -> bool:
    uniq = sorted(set(rank_values))
    if len(uniq) < 5:
        return False
    if 14 in uniq:
        uniq = [1] + uniq
    for i in range(len(uniq) - 4):
        window = uniq[i:i+5]
        if window == list(range(window[0], window[0] + 5)):
            return True
    return False


def best_combination_from_tokens(cards: list[str]) -> str:
    card_ranks = [card_rank(c) for c in cards if card_rank(c) is not None]
    card_suits = [card_suit(c) for c in cards if card_suit(c) is not None]
    if not card_ranks:
        return "unknown"
    counts = Counter(card_ranks)
    cvals = sorted(counts.values(), reverse=True)

    if len(card_ranks) < 5:
        if 3 in cvals:
            return "three of a kind"
        if cvals.count(2) >= 2:
            return "two pair"
        if 2 in cvals:
            return "pair"
        return "high card"

    suit_counts = Counter(card_suits)
    flush_suit = next((s for s, cnt in suit_counts.items() if cnt >= 5), None)
    flush_cards = [c for c in cards if card_suit(c) == flush_suit] if flush_suit else []
    flush_ranks = [card_rank(c) for c in flush_cards if card_rank(c) is not None]

    straight = is_straight(card_ranks)
    straight_flush = flush_suit is not None and is_straight(flush_ranks)

    if straight_flush:
        return "straight flush"
    if 4 in cvals:
        return "four of a kind"
    if 3 in cvals and 2 in cvals:
        return "full house"
    if flush_suit is not None:
        return "flush"
    if straight:
        return "straight"
    if 3 in cvals:
        return "three of a kind"
    if cvals.count(2) >= 2:
        return "two pair"
    if 2 in cvals:
        return "pair"
    return "high card"


def hand_strength_from_tokens(cards: list[str]) -> float:
    return float(COMBINATION_MAP.get(best_combination_from_tokens(cards), 0))


def straight_draw_flags(cards: list[str]) -> tuple[int, int]:
    rs = sorted(set(ranks(cards)))
    if 14 in rs:
        rs = [1] + rs
    gutshot = 0
    open_ended = 0
    rset = set(rs)
    for start in range(1, 11):
        target = set(range(start, start + 5))
        have = len(target.intersection(rset))
        if have >= 4:
            missing = sorted(target.difference(rset))
            if len(missing) == 1:
                if missing[0] in {start, start + 4}:
                    open_ended = 1
                else:
                    gutshot = 1
    return open_ended, gutshot


def board_straight_flags(board_cards: list[str]) -> tuple[float, float]:
    rs = sorted(set(ranks(board_cards)))
    if not rs:
        return 0.0, 0.0
    if 14 in rs:
        rs = [1] + rs
    rset = set(rs)
    straight_present = 0.0
    four_liner = 0.0
    for start in range(1, 11):
        target = set(range(start, start + 5))
        have = len(target.intersection(rset))
        if have == 5:
            straight_present = 1.0
        elif have >= 4:
            four_liner = 1.0
    return straight_present, four_liner


def board_pair_profile(board_cards: list[str]) -> tuple[float, float, float]:
    bcounts = Counter(ranks(board_cards))
    pair_count = float(sum(1 for c in bcounts.values() if c >= 2))
    trips_present = float(any(c >= 3 for c in bcounts.values()))
    full_house = float(any(c >= 3 for c in bcounts.values()) and (sum(1 for c in bcounts.values() if c >= 2) >= 2))
    return pair_count, trips_present, full_house


def hole_card_features(hole_cards: list[str]) -> dict[str, float]:
    if len(hole_cards) < 2:
        return {
            "hole_rank_high": 0.0, "hole_rank_low": 0.0, "is_pair": 0.0, "is_suited": 0.0,
            "rank_gap": 0.0, "has_ace": 0.0, "has_broadway": 0.0, "preflop_strength": 0.0,
        }
    r1 = card_rank(hole_cards[0]) or 0
    r2 = card_rank(hole_cards[1]) or 0
    high = float(max(r1, r2))
    low = float(min(r1, r2))
    is_pair = float(r1 == r2)
    is_suited = float((card_suit(hole_cards[0]) or "") == (card_suit(hole_cards[1]) or ""))
    gap = float(abs(r1 - r2))
    has_ace = float(14 in {r1, r2})
    broadway = float(r1 >= 10 or r2 >= 10)
    preflop_strength = high + 0.6 * low + 2.5 * is_pair + 0.8 * is_suited - 0.15 * gap
    return {
        "hole_rank_high": high,
        "hole_rank_low": low,
        "is_pair": is_pair,
        "is_suited": is_suited,
        "rank_gap": gap,
        "has_ace": has_ace,
        "has_broadway": broadway,
        "preflop_strength": float(preflop_strength),
    }


def build_stage_feature_payload(
    stage: str,
    hole_cards: list[str],
    board_cards: list[str],
    *,
    total_bet: float = 0.0,
    current_pot: float = 0.0,
    position: str = "",
    hero_stack: float = 0.0,
    table_stacks: list[float] | None = None,
    big_blind: float = 2.0,
) -> dict[str, float]:
    cards = list(hole_cards) + list(board_cards)
    rs = ranks(cards)
    ss = suits(cards)
    rank_counts = Counter(rs)
    suit_counts = Counter(ss)
    max_suit = max(suit_counts.values()) if suit_counts else 0

    open_ended, gutshot = straight_draw_flags(cards)
    board_rs = ranks(board_cards)
    board_high = float(max(board_rs) if board_rs else 0.0)
    board_is_paired = float(len(set(board_rs)) < len(board_rs)) if board_rs else 0.0
    board_four_flush = float((max(Counter(suits(board_cards)).values()) if board_cards else 0) >= 4)
    board_straight_present, board_straight_4liner = board_straight_flags(board_cards)
    board_pair_count, board_trips_present, board_full_house_present = board_pair_profile(board_cards)

    hand_strength = hand_strength_from_tokens(cards)
    board_only_strength = hand_strength_from_tokens(board_cards)
    shared_strength_gap = float(hand_strength - board_only_strength)
    hero_uses_hole_for_best = float(shared_strength_gap > 0)

    bb = max(0.1, float(big_blind))
    hero_stack = max(0.0, float(hero_stack))
    stack_pool = [max(0.0, float(s)) for s in (table_stacks or [hero_stack])]
    effective_stack = min(hero_stack, max(stack_pool) if stack_pool else hero_stack)
    hero_stack_bb = hero_stack / bb
    effective_stack_bb = effective_stack / bb
    spr = effective_stack / max(float(current_pot), 1.0)

    risk = min(
        1.0,
        0.3 * board_four_flush
        + 0.25 * board_straight_present
        + 0.15 * board_straight_4liner
        + 0.2 * float(board_pair_count >= 1)
        + 0.15 * float(board_pair_count >= 2)
        + 0.25 * board_trips_present
        + 0.35 * board_full_house_present
        + 0.2 * float(shared_strength_gap <= 1.0)
        + 0.2 * float(hero_uses_hole_for_best == 0.0)
    )

    payload = {
        "hand_strength": float(hand_strength),
        "total_bet": float(total_bet),
        "current_pot": float(current_pot),
        "position_value": float(POSITION_MAP.get(str(position).upper(), 0.5)),
        "pair_count": float(sum(1 for c in rank_counts.values() if c == 2)),
        "trips_count": float(sum(1 for c in rank_counts.values() if c == 3)),
        "flush_draw": float(max_suit == 4),
        "open_ended_straight_draw": float(open_ended),
        "gutshot_draw": float(gutshot),
        "board_high_rank": board_high,
        "board_is_paired": board_is_paired,
        "hero_stack_bb": float(hero_stack_bb),
        "effective_stack_bb": float(effective_stack_bb),
        "spr": float(spr),
        "board_four_flush": float(board_four_flush),
        "board_straight_present": float(board_straight_present),
        "board_straight_4liner": float(board_straight_4liner),
        "board_pair_count": float(board_pair_count),
        "board_trips_present": float(board_trips_present),
        "board_full_house_present": float(board_full_house_present),
        "hero_uses_hole_for_best": float(hero_uses_hole_for_best),
        "shared_strength_gap": float(shared_strength_gap),
        "board_only_strength": float(board_only_strength),
        "board_shared_strength_risk": float(risk),
    }
    payload.update(hole_card_features(hole_cards))

    if stage.lower() == "preflop":
        payload["flush_draw"] = 0.0
        payload["open_ended_straight_draw"] = 0.0
        payload["gutshot_draw"] = 0.0
        payload["board_high_rank"] = 0.0
        payload["board_is_paired"] = 0.0
    return payload


# -----------------------------
# Data cleaning and label engineering
# -----------------------------

def encode_table_position(pos: object) -> float:
    p = str(pos).strip().lower()
    mapping = {
        "btn": 1.0, "button": 1.0, "bu": 1.0,
        "co": 0.8, "cutoff": 0.8,
        "hj": 0.65, "mp": 0.65,
        "utg": 0.5,
        "sb": 0.25,
        "bb": 0.0,
    }
    return float(mapping.get(p, 0.5))


def coerce_numeric_money(series: pd.Series) -> pd.Series:
    text = (
        series.astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .str.replace("nan", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(text, errors="coerce")


def aggression_proxy(df: pd.DataFrame, action_columns: list[str]) -> pd.Series:
    text = pd.Series("", index=df.index, dtype="string")
    for col in action_columns:
        if col in df.columns:
            text = text + " " + df[col].astype("string").fillna("")
    lower = text.str.lower()
    raises = lower.str.count("raise").fillna(0)
    bets = lower.str.count("bet").fillna(0)
    calls = lower.str.count("call").fillna(0)
    return raises * 1.5 + bets * 1.2 + calls * 0.4


def evaluate_strength_by_street(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    base_key = (
        out.get("hand_id", pd.Series(out.index, index=out.index)).astype(str)
        + "|" + out.get("player_id", pd.Series("player", index=out.index)).astype(str)
    )
    out["strength_preflop"] = base_key.apply(lambda x: stable_float(f"{x}|preflop"))
    out["strength_flop"] = (out["strength_preflop"] + base_key.apply(lambda x: stable_float(f"{x}|flop"))) / 2
    out["strength_turn"] = (out["strength_flop"] + base_key.apply(lambda x: stable_float(f"{x}|turn"))) / 2
    out["strength_river"] = (out["strength_turn"] + base_key.apply(lambda x: stable_float(f"{x}|river"))) / 2
    out["strength_mean"] = out[["strength_preflop", "strength_flop", "strength_turn", "strength_river"]].mean(axis=1)
    return out


def label_bluffs(df: pd.DataFrame, strength_threshold: float = 0.35, aggression_threshold: float = 2.0) -> pd.DataFrame:
    out = df.copy()
    out["aggression_score"] = aggression_proxy(out, ["preflop", "flop", "turn", "river"])
    out["is_bluffing"] = ((out["strength_mean"] <= strength_threshold) & (out["aggression_score"] >= aggression_threshold)).astype(int)
    return out


def add_streaks(df: pd.DataFrame, session_break_hours: int = 12) -> pd.DataFrame:
    out = df.copy().sort_values(["player_id", "hand_datetime", "hand_id"], na_position="last").reset_index(drop=True)
    pnl = pd.to_numeric(out.get("net_result", 0), errors="coerce").fillna(0.0)
    out["is_win"] = pnl > 0
    out["is_loss"] = pnl < 0
    time_gap = out.groupby("player_id")["hand_datetime"].diff().dt.total_seconds().fillna(0) / 3600
    out["session_reset"] = time_gap > session_break_hours

    win_streak = []
    loss_streak = []
    w = 0
    l = 0
    prev_player = None
    for row in out.itertuples():
        if prev_player != row.player_id or row.session_reset:
            w = 0
            l = 0
        if row.is_win:
            w += 1
            l = 0
        elif row.is_loss:
            l += 1
            w = 0
        else:
            w = 0
            l = 0
        win_streak.append(w)
        loss_streak.append(l)
        prev_player = row.player_id
    out["win_streak"] = win_streak
    out["loss_streak"] = loss_streak
    return out


def add_allin_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text = (
        out.get("preflop", "").astype(str) + " "
        + out.get("flop", "").astype(str) + " "
        + out.get("turn", "").astype(str) + " "
        + out.get("river", "").astype(str)
    ).str.lower()
    out["is_all_in"] = text.str.contains("all-in|all in", regex=True)
    return out


def preflop_equity_proxy(hand_repr: str, table_size: int = 2) -> float:
    key = f"{hand_repr}|{table_size}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF
    return max(0.01, min(0.99, raw))


def add_preflop_equity(df: pd.DataFrame, hand_col: str = "cards") -> pd.DataFrame:
    out = df.copy()
    out["preflop_equity"] = [
        preflop_equity_proxy(str(hand), int(ts) if pd.notna(ts) else 2)
        for hand, ts in zip(out.get(hand_col, ""), out.get("table_size", 2), strict=False)
    ]
    return out


def build_cleaned_gambling_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    if "hand_id" in df.columns:
        df["hand_id"] = df["hand_id"].astype(str).str.strip()
    if "name" in df.columns:
        df["player_id"] = df["name"].map(stable_player_id)
    else:
        df["player_id"] = "unknown"

    if {"date", "time"}.issubset(df.columns):
        df["hand_datetime"] = pd.to_datetime(
            df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
            errors="coerce",
            utc=True,
        )
    else:
        df["hand_datetime"] = pd.NaT

    for c in [
        "stack", "balance", "pot_pre", "pot_flop", "pot_turn", "pot_river",
        "bet_pre", "bet_flop", "bet_turn", "bet_river", "table_size", "ante"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "buyin" in df.columns:
        df["buyin"] = coerce_numeric_money(df["buyin"]).astype("Float64")

    df["net_result"] = pd.to_numeric(df.get("balance"), errors="coerce").fillna(0.0)
    result_s = df.get("result", "").fillna("").astype(str)
    df["won_flag"] = result_s.str.contains("took chips|won", case=False, regex=True).astype(int)

    # action aliases
    for src, dst in [("action_pre", "preflop"), ("action_flop", "flop"), ("action_turn", "turn"), ("action_river", "river")]:
        if src in df.columns:
            df[dst] = df[src].astype(str)

    df["starting_stack"] = pd.to_numeric(df.get("stack"), errors="coerce").fillna(0.0)
    df["table_position"] = df.get("position", "").map(encode_table_position)

    df = evaluate_strength_by_street(df)
    df = label_bluffs(df)
    df = add_streaks(df)
    df = add_allin_flags(df)
    df = add_preflop_equity(df, hand_col="cards")

    if "name" in df.columns:
        df = df.drop(columns=["name"])
    return df.reset_index(drop=True)


# -----------------------------
# Training-frame builders
# -----------------------------

def stage_context(row: pd.Series, stage: str) -> tuple[list[str], list[str], float, float]:
    hole = parse_cards(row.get("cards"))
    board_flop = parse_cards(row.get("board_flop"))
    board_turn = parse_cards(row.get("board_turn"))
    board_river = parse_cards(row.get("board_river"))

    board_cards: list[str] = []
    if stage in {"flop", "turn", "river"}:
        board_cards += board_flop
    if stage in {"turn", "river"}:
        board_cards += board_turn
    if stage == "river":
        board_cards += board_river

    pre = safe_float(row.get("bet_pre"))
    fl = safe_float(row.get("bet_flop"))
    tu = safe_float(row.get("bet_turn"))
    ri = safe_float(row.get("bet_river"))

    if stage == "preflop":
        total_bet = pre
        current_pot = safe_float(row.get("pot_pre"))
    elif stage == "flop":
        total_bet = pre + fl
        current_pot = safe_float(row.get("pot_flop")) or safe_float(row.get("pot_pre"))
    elif stage == "turn":
        total_bet = pre + fl + tu
        current_pot = safe_float(row.get("pot_turn")) or safe_float(row.get("pot_flop"))
    else:
        total_bet = pre + fl + tu + ri
        current_pot = safe_float(row.get("pot_river")) or safe_float(row.get("pot_turn"))

    return hole, board_cards, float(total_bet), float(current_pot)


def build_stage_training_frame(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    stacks_by_hand = (
        df.groupby("hand_id")["stack"].apply(lambda s: [float(x) for x in s.fillna(0.0).tolist()]).to_dict()
        if "hand_id" in df.columns and "stack" in df.columns
        else {}
    )
    rows = []
    for _, row in df.iterrows():
        hole, board_cards, total_bet, current_pot = stage_context(row, stage)
        hand_id = str(row.get("hand_id", ""))
        table_stacks = stacks_by_hand.get(hand_id, [])
        hero_stack = safe_float(row.get("stack", 0.0))
        blind_text = str(row.get("blinds", "1/2"))
        bb = 2.0
        if "/" in blind_text:
            try:
                bb = float(blind_text.split("/")[-1].strip())
            except Exception:
                bb = 2.0
        payload = build_stage_feature_payload(
            stage,
            hole,
            board_cards,
            total_bet=total_bet,
            current_pot=current_pot,
            position=str(row.get("position", "")),
            hero_stack=hero_stack,
            table_stacks=table_stacks,
            big_blind=bb,
        )
        payload["won_flag"] = int(row.get("won_flag", 0))
        payload["hand_id"] = hand_id
        rows.append(payload)
    return pd.DataFrame(rows)


def stage_from_board_row(row: pd.Series) -> str:
    br = row.get("board_river")
    bt = row.get("board_turn")
    bf = row.get("board_flop")
    if br not in (None, "", "0", 0) and str(br).strip() not in {"--", "nan"}:
        return "river"
    if bt not in (None, "", "0", 0) and str(bt).strip() not in {"--", "nan"}:
        return "turn"
    if bf not in (None, "", "0", 0) and str(bf).strip() not in {"--", "nan"}:
        return "flop"
    return "preflop"


def parse_board_chain(row: pd.Series, stage: str) -> list[str]:
    parts = []
    if stage in {"flop", "turn", "river"}:
        parts.extend(parse_cards(row.get("board_flop")))
    if stage in {"turn", "river"}:
        parts.extend(parse_cards(row.get("board_turn")))
    if stage == "river":
        parts.extend(parse_cards(row.get("board_river")))
    return parts


def raises_in_actions(row: pd.Series, stage: str) -> int:
    texts = [str(row.get("action_pre", ""))]
    if stage in {"flop", "turn", "river"}:
        texts.append(str(row.get("action_flop", "")))
    if stage in {"turn", "river"}:
        texts.append(str(row.get("action_turn", "")))
    if stage == "river":
        texts.append(str(row.get("action_river", "")))
    return " ".join(texts).lower().count("raise")


def bet_total_for_stage(row: pd.Series, stage: str) -> float:
    pre = safe_float(row.get("bet_pre"))
    fl = safe_float(row.get("bet_flop"))
    tu = safe_float(row.get("bet_turn"))
    ri = safe_float(row.get("bet_river"))
    if stage == "preflop":
        return pre
    if stage == "flop":
        return pre + fl
    if stage == "turn":
        return pre + fl + tu
    return pre + fl + tu + ri


def pot_for_stage(row: pd.Series, stage: str) -> float:
    if stage == "preflop":
        return safe_float(row.get("pot_pre"))
    if stage == "flop":
        return safe_float(row.get("pot_flop")) or safe_float(row.get("pot_pre"))
    if stage == "turn":
        return safe_float(row.get("pot_turn")) or safe_float(row.get("pot_flop"))
    return safe_float(row.get("pot_river")) or safe_float(row.get("pot_turn"))


def visible_vector_from_csv_row(row: pd.Series) -> dict[str, float]:
    stage = stage_from_board_row(row)
    board = parse_board_chain(row, stage)

    blind_text = str(row.get("blinds", "1/2"))
    bb = 2.0
    if "/" in blind_text:
        try:
            bb = float(blind_text.split("/")[-1].strip())
        except Exception:
            bb = 2.0

    stack = safe_float(row.get("stack"))
    table_stacks = [stack] * 6
    total_bet = bet_total_for_stage(row, stage)
    pot = pot_for_stage(row, stage)

    payload = build_stage_feature_payload(
        stage,
        [],
        board,
        total_bet=total_bet,
        current_pot=pot,
        position=str(row.get("position", "")),
        hero_stack=stack,
        table_stacks=table_stacks,
        big_blind=bb,
    )

    vec = {k: float(payload.get(k, 0.0)) for k in VISIBLE_BLUFF_FEATURE_KEYS}
    vec["seat_bet_to_pot"] = float(min(total_bet / max(pot, 1.0), 6.0))
    vec["seat_raise_count"] = float(min(raises_in_actions(row, stage), 12)) / 12.0
    vec["street_phase"] = {"preflop": 0.0, "flop": 0.33, "turn": 0.67, "river": 1.0}[stage]
    return vec


def build_visible_bluff_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        vec = visible_vector_from_csv_row(row)
        vec["is_bluffing"] = int(row.get("is_bluffing", 0))
        vec["hand_id"] = str(row.get("hand_id", ""))
        rows.append(vec)
    return pd.DataFrame(rows)


def evaluate_classifier_with_group_split(frame: pd.DataFrame, features: list[str], target: str, group_col: str, label: str) -> tuple[CalibratedClassifierCV, dict, pd.DataFrame]:
    X = frame.reindex(columns=features).fillna(0.0)
    y = frame[target].astype(int)
    groups = frame[group_col].astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    base_model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_SEED,
        max_depth=12,
        min_samples_leaf=5,
    )
    base_model.fit(X_train, y_train)

    model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probs) if y_test.nunique() > 1 else 0.5
    brier = brier_score_loss(y_test, probs)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)

    metrics = {
        "model": label,
        "rows": len(frame),
        "accuracy": float(acc),
        "auc": float(auc),
        "brier": float(brier),
        "f1": float(f1),
    }

    importance = pd.DataFrame({
        "feature": features,
        "importance": base_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="uniform")
    plt.figure(figsize=(5, 4))
    plt.plot(mean_pred, frac_pos, marker="o", label=label)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title(f"Calibration Curve: {label}")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"\n{label} classification report:\n")
    print(classification_report(y_test, preds, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    return model, metrics, importance
