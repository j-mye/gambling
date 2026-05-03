"""Calibrated bluff probability from observable table features only."""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def _project_root() -> Path:
    """Repo root (directory that contains main.py), regardless of process cwd."""
    return Path(__file__).resolve().parents[2]


def default_visible_bluff_paths() -> tuple[str, str]:
    root = _project_root()
    return str(root / "visible_bluff_model.pkl"), str(root / "visible_bluff_feature_names.pkl")


DEFAULT_MODEL_PATH, DEFAULT_FEATURE_PATH = default_visible_bluff_paths()


@lru_cache(maxsize=1)
def load_visible_bluff_model(path: str = DEFAULT_MODEL_PATH) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p.name} at project root (run `python visible_bluff_train.py` "
            f"after `data/cleanedGambling.csv` exists)."
        )
    return joblib.load(p)


@lru_cache(maxsize=1)
def load_visible_bluff_features(path: str = DEFAULT_FEATURE_PATH) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p.name} at project root (same train command as the model)."
        )
    names = joblib.load(p)
    return [str(x) for x in list(names)]


def predict_visible_bluff_probability(
    feature_values: dict[str, float],
    model_path: str = DEFAULT_MODEL_PATH,
    feature_path: str = DEFAULT_FEATURE_PATH,
) -> float:
    """Return P(bluff heuristic) in [0, 1]."""
    feats = load_visible_bluff_features(feature_path)
    missing = [f for f in feats if f not in feature_values]
    if missing:
        raise ValueError(f"Missing visible-bluff features: {missing}")
    values_row = [float(feature_values[f]) for f in feats]
    model = load_visible_bluff_model(model_path)
    row = pd.DataFrame([values_row], columns=feats)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*valid feature names.*",
            category=UserWarning,
        )
        return float(model.predict_proba(row)[0][1])
