"""Inference adapter for stage-based win probability models."""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

VALID_STAGES = {"preflop", "flop", "turn", "river"}


def _project_root() -> Path:
    """Repo root (directory that contains main.py), regardless of process cwd."""
    return Path(__file__).resolve().parents[2]


def default_stage_artifact_paths() -> tuple[str, str]:
    root = _project_root()
    return str(root / "poker_models.pkl"), str(root / "feature_names.pkl")


DEFAULT_STAGE_MODEL_PATH, DEFAULT_STAGE_FEATURE_PATH = default_stage_artifact_paths()


@lru_cache(maxsize=1)
def load_stage_models(model_path: str = DEFAULT_STAGE_MODEL_PATH) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name} under {path.parent} — not on GitHub (too large). "
            "From the project root run: python model_train.py "
            "(needs data/cleanedGambling.csv or data/gambling.csv)."
        )
    loaded = joblib.load(path)
    if not isinstance(loaded, dict):
        raise ValueError("Stage model artifact must be a stage->model dictionary")
    return loaded


@lru_cache(maxsize=1)
def load_stage_feature_map(feature_path: str = DEFAULT_STAGE_FEATURE_PATH) -> dict[str, list[str]]:
    path = Path(feature_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name} under {path.parent}. Run python model_train.py from project root."
        )
    loaded = joblib.load(path)
    if not isinstance(loaded, dict):
        raise ValueError("Feature artifact must be a stage->feature-list dictionary")
    out: dict[str, list[str]] = {}
    for stage, features in loaded.items():
        out[str(stage)] = [str(f) for f in list(features)]
    return out


def predict_stage_win_probability(
    stage: str,
    feature_values: dict[str, float | int],
    model_path: str = DEFAULT_STAGE_MODEL_PATH,
    feature_path: str = DEFAULT_STAGE_FEATURE_PATH,
) -> float:
    stage_key = str(stage).strip().lower()
    if stage_key not in VALID_STAGES:
        raise ValueError(f"Invalid stage '{stage}'. Expected one of {sorted(VALID_STAGES)}")

    feature_map = load_stage_feature_map(feature_path=feature_path)
    expected_features = feature_map.get(stage_key)
    if not expected_features:
        raise KeyError(f"No feature list found for stage '{stage_key}'")

    missing = [f for f in expected_features if f not in feature_values]
    if missing:
        raise ValueError(f"Missing required model features: {missing}")

    vector = pd.DataFrame(
        [[float(feature_values[name]) for name in expected_features]],
        columns=expected_features,
    )

    models = load_stage_models(model_path=model_path)
    model = models.get(stage_key)
    if model is None:
        raise KeyError(f"No model found for stage '{stage_key}' in artifact")
    # CalibratedClassifierCV forwards ndarray to the inner RF; sklearn warns even when
    # callers pass a DataFrame — suppress only this noisy mismatch.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*valid feature names.*",
            category=UserWarning,
        )
        return float(model.predict_proba(vector)[0][1])
