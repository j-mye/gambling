"""Streamlit dashboard for poker behavior analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from scripts.simulation.monte_carlo import simulate_folded_hand
from scripts.ui.poker_table_tab import render_playable_poker_tab


def _load_data() -> pd.DataFrame:
    for path in (
        Path("artifacts/processed/model_ready.csv"),
        Path("data/cleanedGambling.csv"),
    ):
        if path.exists():
            df = pd.read_csv(path)
            preferred = [
                "player_id",
                "aggression_score",
                "starting_stack",
                "win_streak",
                "strength_mean",
                "preflop_equity",
                "table_position",
                "net_result",
                "is_all_in",
            ]
            if path.name == "cleanedGambling.csv":
                keep = [c for c in preferred if c in df.columns]
                if keep:
                    return df[keep].copy()
                # Wrong schema — don't return raw df; try next path or built-in fallback.
                continue
            return df
    return pd.DataFrame(
        {
            "player_id": ["p1", "p2", "p3"],
            "aggression_score": [1.2, 3.5, 2.3],
            "starting_stack": [100, 125, 80],
            "win_streak": [0, 2, 1],
            "table_position": [1.0, 0.25, 0.5],
            "strength_mean": [0.62, 0.28, 0.47],
            "preflop_equity": [0.55, 0.38, 0.44],
            "net_result": [12, -20, 5],
            "is_all_in": [False, True, False],
        }
    )


def render_eda_tab(df: pd.DataFrame):
    st.subheader("EDA & Streaks")
    filtered = df.copy()
    players = ["all"] + sorted(filtered["player_id"].astype(str).unique().tolist()) if "player_id" in filtered.columns else ["all"]
    player = st.selectbox("Player filter", players)
    if player != "all":
        filtered = filtered[filtered["player_id"].astype(str) == player]
    if {"starting_stack", "net_result"}.issubset(filtered.columns):
        fig = px.scatter(filtered, x="starting_stack", y="net_result", color="player_id" if "player_id" in filtered.columns else None, title="Stack vs Net Result")
        st.plotly_chart(fig, use_container_width=True)
    st.metric("Rows", len(filtered))
    if "win_streak" in filtered.columns:
        st.metric("Avg win streak", f"{filtered['win_streak'].mean():.2f}")


def render_fold_tab():
    st.subheader("Fold Simulator")
    hand_id = st.text_input("Hand ID", value="demo_hand")
    runs = st.slider("Simulation runouts", 100, 5000, 1000, step=100)
    pot_size = st.number_input("Pot size", min_value=0.0, value=50.0)
    invested = st.number_input("Your invested amount", min_value=0.0, value=10.0)
    result = simulate_folded_hand(pd.Series({"hand_id": hand_id, "player_id": "user", "pot_size": pot_size, "invested": invested}), runs=runs, seed=42)
    st.progress(min(runs / 5000.0, 1.0))
    st.metric("Estimated fold win rate", f"{result['win_rate']:.2%}")
    st.metric("95% CI", f"{result['ci95_low']:.2%} - {result['ci95_high']:.2%}")
    st.metric("EV sacrificed", f"{result['ev_sacrificed']:.2f}")


def main():
    st.set_page_config(page_title="Poker Behavioral Analysis", layout="wide")
    st.title("Online Poker Behavioral Analysis")
    df = _load_data()
    tab1, tab2, tab3 = st.tabs(["Poker Simulator", "EDA & Streaks", "Fold Simulator"])
    with tab1:
        render_playable_poker_tab()
    with tab2:
        render_eda_tab(df)
    with tab3:
        render_fold_tab()


if __name__ == "__main__":
    main()
