from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

folder = Path(__file__).resolve().parent

st.set_page_config(
    page_title="ATP Match Predictor",
    page_icon="🎾",
    layout="centered",
)

st.title("ATP Match Predictor")
st.write("Choose two players and a court surface.")

model_path = folder / "tennis_model.joblib"
players_path = folder / "players_snapshot.csv"

if not model_path.exists() or not players_path.exists():
    st.error(
        "Missing model or player snapshot. "
        "Run build_elo.py and train_model.py first."
    )
    st.stop()

# Load only the model file you generated yourself.
bundle = joblib.load(model_path)
players = pd.read_csv(
    players_path,
    dtype={"player_id": str},
    parse_dates=["record_date", "snapshot_date"],
)

for column in ["overall_k", "surface_k"]:
    if not players[column].eq(bundle[column]).all():
        st.error("Player ratings and model settings do not match.")
        st.stop()

# A ranking is required by this model.
players = players[
    players["rank"].gt(0) & players["name"].notna()
].copy()
players = players.sort_values(["name", "player_id"])

if len(players) < 2:
    st.error("At least two players with valid rankings are required.")
    st.stop()

snapshot_date = players["snapshot_date"].max()

st.info(
    f"Historical demo · Data snapshot: {snapshot_date:%d %B %Y}. "
    "Predictions use stored rankings and ratings, not live player form."
)

players = players.set_index("player_id")
player_ids = players.index.tolist()


def player_label(player_id):
    row = players.loc[player_id]
    return f"{row['name']} · {player_id}"


with st.form("prediction_form"):
    left, right = st.columns(2)

    with left:
        player_a = st.selectbox(
            "Player A — type to search",
            player_ids,
            format_func=player_label,
            index=0,
        )

    with right:
        player_b = st.selectbox(
            "Player B — type to search",
            player_ids,
            format_func=player_label,
            index=1,
        )

    surface = st.selectbox("Court surface", ["Hard", "Clay", "Grass"])
    submitted = st.form_submit_button("Predict match")

if submitted:
    if player_a == player_b:
        st.error("Please select two different players.")
        st.stop()

    a = players.loc[player_a]
    b = players.loc[player_b]

    features = pd.DataFrame([{
        "ranking_strength_diff": np.log(b["rank"] / a["rank"]),
        "overall_elo_diff": a["overall_elo"] - b["overall_elo"],
        "surface_elo_diff": (
            a[f"{surface}_elo"] - b[f"{surface}_elo"]
        ),
    }])[bundle["feature_columns"]]

    if not np.isfinite(features.to_numpy()).all():
        st.error("These players have incomplete prediction features.")
        st.stop()

    score = float(bundle["model"].decision_function(features)[0])
    calibrated_score = score / bundle["temperature"]

    probability_a = float(
        np.exp(-np.logaddexp(0, -calibrated_score))
    )
    probability_b = 1 - probability_a

    st.divider()

    left, right = st.columns(2)
    left.metric(str(a["name"]), f"{probability_a:.1%} chance to win")
    right.metric(str(b["name"]), f"{probability_b:.1%} chance to win")

    if abs(probability_a - 0.5) < 0.0005:
        st.write("**Too close to call.**")
    else:
        favourite = a["name"] if probability_a > 0.5 else b["name"]
        st.write(f"**Predicted winner: {favourite}**")

    st.caption(
        "Probabilities assume the match is completed. "
        "Retirements and walkovers are not modelled."
    )

    with st.expander("Player information used"):
        details = pd.DataFrame({
            "Player": [a["name"], b["name"]],
            "Stored ranking": [int(a["rank"]), int(b["rank"])],
            "Overall Elo": [a["overall_elo"], b["overall_elo"]],
            f"{surface} Elo": [
                a[f"{surface}_elo"], b[f"{surface}_elo"]
            ],
            "Last record": [
                a["record_date"].strftime("%Y-%m-%d"),
                b["record_date"].strftime("%Y-%m-%d"),
            ],
        })
        st.dataframe(details.round(1), hide_index=True)
        st.caption(
            "Ratings use delayed result updates. A surface rating "
            "starts at 1500 when no eligible history is available."
        )