"""Feature lists for stage win models (see ``model_train.py``).

Street-level reductions are documented in ``notebooks/02_eda_stage_models_visual.ipynb``.

**Retrain stage models** after edits: ``python model_train.py`` (writes ``feature_names.pkl``).

The **visible-bluff** classifier (``visible_bluff_train.py``, ``scripts/features/visible_bluff_features.py``)
uses a separate feature list — board/pot/stack aggression only, no hole cards — documented in
``notebooks/03_visible_bluff_model.ipynb``.
"""

from __future__ import annotations

STAGE_FEATURES: dict[str, list[str]] = {
    "preflop": [
        "hole_rank_high",
        "hole_rank_low",
        "is_pair",
        "is_suited",
        "rank_gap",
        "has_ace",
        "has_broadway",
        "preflop_strength",
        "hero_stack_bb",
        "effective_stack_bb",
        "spr",
    ],
    "flop": [
        "hand_strength",
        "pair_count",
        "trips_count",
        "flush_draw",
        "open_ended_straight_draw",
        "gutshot_draw",
        "board_high_rank",
        "board_is_paired",
        "hero_stack_bb",
        "effective_stack_bb",
        "spr",
        "board_four_flush",
        "board_straight_present",
        "board_straight_4liner",
        "board_pair_count",
        "board_trips_present",
        "board_full_house_present",
        "hero_uses_hole_for_best",
        "shared_strength_gap",
        "board_shared_strength_risk",
    ],
    "turn": [
        "hand_strength",
        "pair_count",
        "trips_count",
        "flush_draw",
        "open_ended_straight_draw",
        "gutshot_draw",
        "board_high_rank",
        "board_is_paired",
        "hero_stack_bb",
        "effective_stack_bb",
        "spr",
        "board_four_flush",
        "board_straight_present",
        "board_straight_4liner",
        "board_pair_count",
        "board_trips_present",
        "board_full_house_present",
        "hero_uses_hole_for_best",
        "shared_strength_gap",
        "board_shared_strength_risk",
    ],
    "river": [
        "hand_strength",
        "pair_count",
        "trips_count",
        "board_high_rank",
        "board_is_paired",
        "hero_stack_bb",
        "effective_stack_bb",
        "spr",
        "board_four_flush",
        "board_straight_present",
        "board_straight_4liner",
        "board_pair_count",
        "board_trips_present",
        "board_full_house_present",
        "hero_uses_hole_for_best",
        "shared_strength_gap",
        "board_shared_strength_risk",
    ],
}

DROPPED_FROM_STAGE_PAYLOAD_KEYS = (
    "hero_stack_percentile",
    "is_short_stack",
    "board_connected",
)


def eda_export_columns() -> list[str]:
    """Columns for Streamlit EDA / ``artifacts/processed/model_ready.csv``."""
    return [
        "player_id",
        "hand_id",
        "hand_datetime",
        "net_result",
        "is_bluffing",
        "is_all_in",
        "aggression_score",
        "strength_mean",
        "starting_stack",
        "win_streak",
        "preflop_equity",
        "table_position",
    ]
