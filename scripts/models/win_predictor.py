"""Win probability model helpers."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["preflop_equity", "table_position", "starting_stack", "table_size"]


def train_win_model(df: pd.DataFrame) -> dict[str, object]:
    X = df.reindex(columns=FEATURES).fillna(0.0)
    y = (pd.to_numeric(df.get("net_result", 0), errors="coerce").fillna(0) > 0).astype(int)
    temporal_sort = "hand_datetime" in df.columns
    if temporal_sort:
        ordered = df.assign(_y=y).sort_values("hand_datetime")
        split_idx = int(0.8 * len(ordered))
        train = ordered.iloc[:split_idx]
        test = ordered.iloc[split_idx:]
        X_train = train.reindex(columns=FEATURES).fillna(0.0)
        y_train = train["_y"]
        X_test = test.reindex(columns=FEATURES).fillna(0.0)
        y_test = test["_y"]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=200))])
    pipeline.fit(X_train, y_train)
    probs = pipeline.predict_proba(X_test)[:, 1] if y_test.nunique() > 1 else [0.5] * len(y_test)
    auc = roc_auc_score(y_test, probs) if y_test.nunique() > 1 else 0.5
    brier = brier_score_loss(y_test, probs)
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="uniform")
    return {
        "model": pipeline,
        "metrics": {"auc": float(auc), "brier": float(brier)},
        "calibration": {"fraction_positives": frac_pos.tolist(), "mean_predicted_value": mean_pred.tolist()},
    }


def persist_win_artifacts(bundle: dict[str, object], out_dir: str | Path = "artifacts/models") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle["model"], out / "win_model.joblib")
    joblib.dump(bundle["metrics"], out / "win_metrics.joblib")
    joblib.dump(bundle["calibration"], out / "win_calibration.joblib")


def predict_win_probability(features: dict[str, float], model: object) -> float:
    frame = pd.DataFrame([features]).reindex(columns=FEATURES).fillna(0.0)
    probabilities = np.asarray(model.predict_proba(frame))
    return float(probabilities[:, 1][0])
