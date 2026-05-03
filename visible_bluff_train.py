"""Train observable-features bluff classifier → visible_bluff_model.pkl."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from scripts.features.visible_bluff_features import VISIBLE_BLUFF_FEATURES, vector_from_csv_row


def build_visible_bluff_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for _, row in df.iterrows():
        vec = vector_from_csv_row(row)
        vec["is_bluffing"] = int(row.get("is_bluffing", 0))
        vec["hand_id"] = str(row.get("hand_id", ""))
        rows.append(vec)
    return pd.DataFrame(rows)


def main() -> None:
    cleaned = Path("data/cleanedGambling.csv")
    raw = Path("data/gambling.csv")
    path = cleaned if cleaned.exists() else raw
    if not path.exists():
        raise FileNotFoundError(f"Need {cleaned} (notebook 01) or {raw}")

    df = pd.read_csv(path)
    if len(df) > 60000:
        hands = df["hand_id"].dropna().astype(str).unique()
        take = min(25000, len(hands))
        keep = pd.Series(hands).sample(n=take, random_state=42).tolist()
        df = df[df["hand_id"].astype(str).isin(keep)].reset_index(drop=True)

    frame = build_visible_bluff_frame(df)
    cols = VISIBLE_BLUFF_FEATURES
    X = frame.reindex(columns=cols).fillna(0.0)
    y = frame["is_bluffing"].astype(int)
    groups = frame["hand_id"].astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    base = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        max_depth=14,
        min_samples_leaf=8,
    )
    base.fit(X_train, y_train)
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y_test, probs) if y_test.nunique() > 1 else 0.5
    brier = brier_score_loss(y_test, probs)
    acc = accuracy_score(y_test, preds)
    print(f"visible_bluff: accuracy={acc:.4f} auc={auc:.4f} brier={brier:.4f}")

    imp = pd.DataFrame({"feature": cols, "importance": base.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    print(imp.head(8).to_string(index=False))

    joblib.dump(model, "visible_bluff_model.pkl")
    joblib.dump(VISIBLE_BLUFF_FEATURES, "visible_bluff_feature_names.pkl")
    print("Saved visible_bluff_model.pkl + visible_bluff_feature_names.pkl")


if __name__ == "__main__":
    main()
