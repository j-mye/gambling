"""Pre-flop equity helpers."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import pandas as pd


@lru_cache(maxsize=5000)
def preflop_equity(hand_repr: str, table_size: int = 2) -> float:
    """Deterministic equity proxy with caching."""
    key = f"{hand_repr}|{table_size}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF
    return max(0.01, min(0.99, raw))


def add_preflop_equity(df: pd.DataFrame, hand_col: str = "hole_cards") -> pd.DataFrame:
    out = df.copy()
    out["preflop_equity"] = [
        preflop_equity(str(hand), int(ts) if pd.notna(ts) else 2)
        for hand, ts in zip(out.get(hand_col, ""), out.get("table_size", 2), strict=False)
    ]
    return out


def equity_cache_stats() -> dict[str, int]:
    info = preflop_equity.cache_info()
    return {"hits": info.hits, "misses": info.misses, "currsize": info.currsize}