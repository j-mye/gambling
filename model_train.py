from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from scripts.features.poker_hand_strength import build_stage_feature_payload, parse_cards
from scripts.models.feature_contracts import STAGE_FEATURES

STAGES = ["preflop", "flop", "turn", "river"]


def _safe_num(value: object) -> float:
    try:
        if value is None:
            return 0.0
        x = float(value)
        if np.isnan(x):
            return 0.0
        return x
    except Exception:
        return 0.0


def _target_from_row(row: pd.Series) -> int:
    result = str(row.get("result", "")).lower()
    if "took chips" in result or "won" in result:
        return 1
    if "lost" in result or "gave up" in result:
        return 0
    balance = _safe_num(row.get("balance", 0.0))
    return int(balance > 0)


def _stage_context(row: pd.Series, stage: str) -> tuple[list[str], list[str], float, float]:
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

    if stage == "preflop":
        total_bet = _safe_num(row.get("bet_pre"))
        current_pot = _safe_num(row.get("pot_pre"))
    elif stage == "flop":
        total_bet = _safe_num(row.get("bet_pre")) + _safe_num(row.get("bet_flop"))
        current_pot = _safe_num(row.get("pot_flop")) or _safe_num(row.get("pot_pre"))
    elif stage == "turn":
        total_bet = _safe_num(row.get("bet_pre")) + _safe_num(row.get("bet_flop")) + _safe_num(row.get("bet_turn"))
        current_pot = _safe_num(row.get("pot_turn")) or _safe_num(row.get("pot_flop")) or _safe_num(row.get("pot_pre"))
    else:
        total_bet = (
            _safe_num(row.get("bet_pre"))
            + _safe_num(row.get("bet_flop"))
            + _safe_num(row.get("bet_turn"))
            + _safe_num(row.get("bet_river"))
        )
        current_pot = _safe_num(row.get("pot_river")) or _safe_num(row.get("pot_turn")) or _safe_num(row.get("pot_flop"))

    return hole, board_cards, total_bet, current_pot


def build_training_frame(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    stacks_by_hand = (
        df.groupby("hand_id")["stack"]
        .apply(lambda s: [float(x) for x in s.fillna(0.0).tolist()])
        .to_dict()
    )
    rows: list[dict[str, float]] = []
    for _, row in df.iterrows():
        hole, board_cards, total_bet, current_pot = _stage_context(row, stage)
        hand_id = row.get("hand_id")
        table_stacks = stacks_by_hand.get(hand_id, [])
        hero_stack = _safe_num(row.get("stack", 0.0))
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
        payload["won_flag"] = _target_from_row(row)
        payload["hand_id"] = hand_id
        rows.append(payload)
    return pd.DataFrame(rows)


def train_stage_model(stage: str, frame: pd.DataFrame) -> CalibratedClassifierCV:
    cols = STAGE_FEATURES[stage]
    X = frame[cols].fillna(0.0)
    y = frame["won_flag"].astype(int)
    groups = frame["hand_id"].astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    base_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        max_depth=12,
        min_samples_leaf=5,
    )
    base_model.fit(X_train, y_train)
    model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y_test, probs) if y_test.nunique() > 1 else 0.5
    brier = brier_score_loss(y_test, probs)
    accuracy = accuracy_score(y_test, preds)
    print(f"{stage}: accuracy={accuracy:.4f} auc={auc:.4f} brier={brier:.4f}")

    importance_df = pd.DataFrame(
        {"feature": cols, "importance": base_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print(f"Top card-centric features ({stage}):")
    print(importance_df.head(5).to_string(index=False))
    print()
    return model


def main() -> None:
    cleaned = Path("data/cleanedGambling.csv")
    raw = Path("data/gambling.csv")
    path = cleaned if cleaned.exists() else raw
    if path == raw and not raw.exists():
        raise FileNotFoundError(
            f"Missing dataset: expected {cleaned} (run notebooks/01 first) or {raw}"
        )
    df = pd.read_csv(path)
    if len(df) > 50000:
        unique_hands = df["hand_id"].dropna().astype(str).unique()
        take = min(20000, len(unique_hands))
        sampled_hands = pd.Series(unique_hands).sample(n=take, random_state=42).tolist()
        df = df[df["hand_id"].astype(str).isin(sampled_hands)].reset_index(drop=True)

    models: dict[str, CalibratedClassifierCV] = {}
    for stage in STAGES:
        print(f"Training stage model: {stage}")
        stage_frame = build_training_frame(df, stage)
        models[stage] = train_stage_model(stage, stage_frame)

    joblib.dump(models, "poker_models.pkl")
    joblib.dump(STAGE_FEATURES, "feature_names.pkl")
    print("Saved card-centric stage models to poker_models.pkl")


if __name__ == "__main__":
    main()