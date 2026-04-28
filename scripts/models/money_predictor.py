"""Money outcome regression models."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["aggression_score", "strength_mean", "starting_stack", "preflop_equity", "table_size"]


def train_money_models(df: pd.DataFrame) -> dict[str, object]:
    X = df.reindex(columns=FEATURES).fillna(0.0)
    y = pd.to_numeric(df.get("net_result", 0), errors="coerce").fillna(0.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    baseline = RandomForestRegressor(n_estimators=200, random_state=42)
    robust = Pipeline([("scaler", StandardScaler()), ("model", HuberRegressor())])
    baseline.fit(X_train, y_train)
    robust.fit(X_train, y_train)

    pred_base = baseline.predict(X_test)
    pred_robust = robust.predict(X_test)
    metrics = {
        "baseline_mae": float(mean_absolute_error(y_test, pred_base)),
        "baseline_rmse": float(mean_squared_error(y_test, pred_base) ** 0.5),
        "robust_mae": float(mean_absolute_error(y_test, pred_robust)),
        "robust_rmse": float(mean_squared_error(y_test, pred_robust) ** 0.5),
    }
    err_df = pd.DataFrame({"actual": y_test, "pred": pred_base})
    err_df["abs_err"] = (err_df["actual"] - err_df["pred"]).abs()
    if "buyin" in df.columns:
        err_df["stake"] = pd.to_numeric(df.loc[err_df.index, "buyin"], errors="coerce").fillna(0)
    else:
        err_df["stake"] = 0
    if "table_size" in df.columns:
        err_df["table_size"] = pd.to_numeric(df.loc[err_df.index, "table_size"], errors="coerce").fillna(0)
    else:
        err_df["table_size"] = 0
    segmented = (
        err_df.groupby(["stake", "table_size"], dropna=False)["abs_err"]
        .mean()
        .reset_index(name="mean_abs_err")
    )
    return {"baseline_model": baseline, "robust_model": robust, "metrics": metrics, "segmented_errors": segmented}


def persist_money_artifacts(bundle: dict[str, object], out_dir: str | Path = "artifacts/models") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle["baseline_model"], out / "money_baseline.joblib")
    joblib.dump(bundle["robust_model"], out / "money_robust.joblib")
    joblib.dump(bundle["metrics"], out / "money_metrics.joblib")
    bundle["segmented_errors"].to_csv(out / "money_segmented_errors.csv", index=False)


def predict_money(features: dict[str, float], model: object) -> float:
    frame = pd.DataFrame([features]).reindex(columns=FEATURES).fillna(0.0)
    return float(model.predict(frame)[0])
