"""Bluff classification training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["aggression_score", "strength_mean", "starting_stack", "win_streak", "loss_streak"]


def _prepare_xy(df: pd.DataFrame, features: Sequence[str] = FEATURES) -> tuple[pd.DataFrame, pd.Series]:
    X = df.reindex(columns=features).fillna(0.0)
    y = df["is_bluffing"].astype(int)
    return X, y


def train_bluff_models(df: pd.DataFrame) -> dict[str, object]:
    X, y = _prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)
    logit = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=200)),
        ]
    )
    forest = RandomForestClassifier(n_estimators=200, random_state=42)
    logit.fit(X_train, y_train)
    forest.fit(X_train, y_train)
    logit_probs = logit.predict_proba(X_test)[:, 1] if y_test.nunique() > 1 else pd.Series([0.5] * len(y_test))
    forest_probs = forest.predict_proba(X_test)[:, 1] if y_test.nunique() > 1 else pd.Series([0.5] * len(y_test))
    auc_logit = roc_auc_score(y_test, logit_probs) if y_test.nunique() > 1 else 0.5
    auc_forest = roc_auc_score(y_test, forest_probs) if y_test.nunique() > 1 else 0.5
    precision, recall, thresholds = precision_recall_curve(y_test, forest_probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    idx = int(f1.argmax()) if len(f1) else 0
    threshold = float(thresholds[max(idx - 1, 0)]) if len(thresholds) else 0.5
    return {
        "baseline_model": logit,
        "alternate_model": forest,
        "metrics": {
            "auc_logit": float(auc_logit),
            "auc_forest": float(auc_forest),
            "selected_threshold": threshold,
        },
    }


def persist_bluff_artifacts(bundle: dict[str, object], out_dir: str | Path = "artifacts/models") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle["baseline_model"], out / "bluff_logit.joblib")
    joblib.dump(bundle["alternate_model"], out / "bluff_forest.joblib")
    joblib.dump(bundle["metrics"], out / "bluff_metrics.joblib")


def predict_bluff(features: dict[str, float], model: object, threshold: float = 0.5) -> dict[str, float | bool]:
    frame = pd.DataFrame([features]).reindex(columns=FEATURES).fillna(0.0)
    probabilities = np.asarray(model.predict_proba(frame))
    probability = float(probabilities[:, 1][0])
    return {"bluff_probability": probability, "is_bluffing": probability >= threshold}