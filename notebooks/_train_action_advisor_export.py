"""Train multinomial logistic action model + write poker_page/action_model.json.

Run from repo root:  py -3 notebooks/_train_action_advisor_export.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "poker_page" / "action_model.json"

CLASS_ORDER = ("fold", "call", "raise")


def _oracle_row(row: np.ndarray) -> int:
    """Rule-based labels for synthetic training (fold=0, call=1, raise=2)."""
    wp, call_frac, pot_frac, facing, min_r_frac, brd, nv_frac, h_stack, h_street, h_tot, vb, min_max = row[:12]
    facing_b = facing > 0.5
    if wp < 0.2 and facing_b and call_frac > 0.12:
        return 0
    if wp < 0.16 and facing_b:
        return 0
    if wp < 0.3 and facing_b and call_frac > 0.4:
        return 0
    if facing_b and wp < 0.26 and call_frac > 0.04:
        return 0
    if facing_b and wp < 0.33 and call_frac > 0.11:
        return 0
    if facing_b and wp < 0.40 and call_frac > 0.20:
        return 0
    # --- Aggression / opens (live UI often sits in wp 0.38–0.58; old oracle rarely labeled raise there.) ---
    if not facing_b and wp >= 0.48 and brd < 0.06 and pot_frac < 0.2:
        return 2
    if not facing_b and wp >= 0.52 and brd < 0.12 and pot_frac < 0.28:
        return 2
    if not facing_b and wp >= 0.44 and nv_frac <= 0.42 and brd < 0.06 and pot_frac < 0.15:
        return 2
    if not facing_b and wp >= 0.43 and brd < 0.08 and pot_frac < 0.22:
        return 2
    if facing_b and wp >= 0.56 and call_frac < 0.1:
        return 2
    if facing_b and 0.38 <= wp <= 0.52 and brd >= 0.5 and call_frac < 0.14:
        return 2
    if wp > 0.7 and not facing_b and pot_frac < 0.35:
        return 2
    if wp > 0.62 and facing_b and call_frac < 0.06:
        return 2
    if wp > 0.55 and not facing_b and brd < 0.45:
        return 2
    if wp >= 0.68 and facing_b:
        return 2
    if wp >= 0.62 and facing_b and call_frac < 0.35:
        return 2
    return 1


def _synthetic_matrix(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Mix uniform coverage with **live-shaped** draws (low pot/call fractions, wp ~ heuristic)."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 16), dtype=np.float64)
    half = n // 2
    # --- Live-like half: matches typical 200bb browser spots (small pots, modest call prices). ---
    X[:half, 0] = np.clip(rng.normal(0.44, 0.12, half), 0.08, 0.92)
    X[:half, 1] = rng.beta(1.2, 5.0, half) * 0.55
    X[:half, 2] = rng.beta(1.5, 6.0, half) * 0.55
    X[:half, 3] = rng.choice([0.0, 1.0], half, p=[0.42, 0.58])
    X[:half, 4] = rng.uniform(0.02, 0.22, half)
    X[:half, 5] = rng.choice([0.0, 0.4, 0.6, 0.8, 1.0], half, p=[0.28, 0.22, 0.22, 0.18, 0.10])
    X[:half, 6] = rng.uniform(0.2, 1.0, half)
    X[:half, 7] = rng.uniform(0.55, 1.05, half)
    X[:half, 8] = rng.beta(1.5, 8.0, half) * 0.35
    X[:half, 9] = rng.beta(2.0, 7.0, half) * 0.45
    X[:half, 10] = rng.uniform(0.08, 0.55, half)
    X[:half, 11] = rng.beta(2.0, 3.0, half) * 0.85 + 0.05
    street = rng.integers(0, 4, half)
    for i in range(4):
        X[:half, 12 + i] = (street == i).astype(np.float64)
    # --- Uniform half: preserves tail coverage for OOD robustness. ---
    X[half:, 0] = rng.uniform(0.08, 0.92, n - half)
    X[half:, 1] = rng.uniform(0, 0.55, n - half)
    X[half:, 2] = rng.uniform(0, 0.9, n - half)
    X[half:, 3] = rng.choice([0.0, 1.0], n - half, p=[0.35, 0.65])
    X[half:, 4] = rng.uniform(0.02, 0.45, n - half)
    X[half:, 5] = rng.choice([0.0, 0.4, 0.6, 0.8, 1.0], n - half)
    X[half:, 6] = rng.uniform(0.15, 1.0, n - half)
    X[half:, 7] = rng.uniform(0.15, 1.1, n - half)
    X[half:, 8] = rng.uniform(0, 0.35, n - half)
    X[half:, 9] = rng.uniform(0, 0.5, n - half)
    X[half:, 10] = rng.uniform(0.05, 0.85, n - half)
    X[half:, 11] = rng.uniform(0.02, 0.95, n - half)
    street2 = rng.integers(0, 4, n - half)
    for i in range(4):
        X[half:, 12 + i] = (street2 == i).astype(np.float64)
    y = np.array([_oracle_row(X[i]) for i in range(n)], dtype=np.int64)
    return X, y


def main() -> None:
    n = 120_000
    X, y = _synthetic_matrix(n, seed=42)
    for c, name in enumerate(CLASS_ORDER):
        print(f"oracle label {name}: {(y == c).mean():.3f}")
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2500,
                    C=1.15,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    clf = pipe.named_steps["clf"]
    sc = pipe.named_steps["scaler"]
    payload = {
        "version": 3,
        "feature_dim": int(X.shape[1]),
        "class_order": list(CLASS_ORDER),
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "scaler_mean": sc.mean_.tolist(),
        "scaler_scale": sc.scale_.tolist(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pred = clf.predict(sc.transform(X))
    acc = float((pred == y).mean())
    print(f"Wrote {OUT} (train acc on synthetic oracle ~{acc:.3f})")
    for c, name in enumerate(CLASS_ORDER):
        m = y == c
        if int(m.sum()) > 0:
            rec = float((pred[m] == y[m]).mean())
            print(f"  recall {name}: {rec:.3f}")


if __name__ == "__main__":
    main()
    sys.exit(0)
