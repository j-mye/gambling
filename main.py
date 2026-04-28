"""Streamlit dashboard for poker behavior analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from scripts.models.bluff_predictor import FEATURES as BLUFF_FEATURES
from scripts.models.bluff_predictor import predict_bluff
from scripts.models.money_predictor import FEATURES as MONEY_FEATURES
from scripts.models.money_predictor import predict_money
from scripts.models.win_predictor import FEATURES as WIN_FEATURES
from scripts.models.win_predictor import predict_win_probability
from scripts.simulation.monte_carlo import simulate_folded_hand
from scripts.ui.poker_table_tab import render_playable_poker_tab


class FallbackClassifier:
    def predict_proba(self, frame: pd.DataFrame):
        value = min(max(float(frame.mean(axis=1).iloc[0]) / 10.0, 0.05), 0.95)
        return [[1 - value, value]]


class FallbackRegressor:
    def predict(self, frame: pd.DataFrame):
        return [float(frame.sum(axis=1).iloc[0]) * 0.1]


def _load_data() -> pd.DataFrame:
    path = Path("artifacts/processed/model_ready.csv")
    if path.exists():
        return pd.read_csv(path)
    # Keep app usable without artifacts.
    return pd.DataFrame(
        {
            "player_id": ["p1", "p2", "p3"],
            "aggression_score": [1.2, 3.5, 2.3],
            "starting_stack": [100, 125, 80],
            "win_streak": [0, 2, 1],
            "loss_streak": [1, 0, 0],
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


def render_bluff_tab():
    st.subheader("Bluff Predictor")
    inputs = {f: st.slider(f, 0.0, 10.0, 1.0) for f in BLUFF_FEATURES}
    result = predict_bluff(inputs, FallbackClassifier(), threshold=0.5)
    st.metric("Bluff probability", f"{result['bluff_probability']:.2%}")
    st.write("Likely bluffing" if result["is_bluffing"] else "Likely value betting")
    baseline = result["bluff_probability"]
    sensitivity = {k: predict_bluff({**inputs, k: v + 1.0}, FallbackClassifier())["bluff_probability"] - baseline for k, v in inputs.items()}
    st.bar_chart(pd.Series(sensitivity), use_container_width=True)


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
    tab1, tab2, tab3, tab4 = st.tabs(["Poker Simulator", "EDA & Streaks", "Bluff Predictor", "Fold Simulator"])
    with tab1:
        render_playable_poker_tab()
    with tab2:
        render_eda_tab(df)
    with tab3:
        render_bluff_tab()
    with tab4:
        render_fold_tab()


if __name__ == "__main__":
    main()