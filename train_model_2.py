from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

folder = Path(__file__).resolve().parent

# Load the previously selected parameters.
settings = pd.read_csv(folder / "tuning_results.csv").sort_values(
    ["accuracy_pct", "log_loss", "overall_k", "surface_k", "C"],
    ascending=[False, True, True, True, True],
).iloc[0]

overall_k = int(settings["overall_k"])
surface_k = int(settings["surface_k"])
c = float(settings["C"])

data = pd.read_csv(
    folder / "matches_elo.csv",
    parse_dates=["tourney_date"],
)

# Check that Elo was built with matching parameters.
for column, expected in [
    ("overall_k", overall_k),
    ("surface_k", surface_k),
]:
    if column not in data or not data[column].eq(expected).all():
        raise ValueError(
            "Elo settings do not match. Run build_elo.py first."
        )

# Confirm that the earlier years are present.
required_years = {2022, 2023, 2024, 2025}
available_years = set(data["tourney_date"].dt.year.unique())

if not required_years.issubset(available_years):
    raise ValueError(
        "Earlier history is missing. Include 2022 and 2023 in "
        "prepare_data.py, then run prepare_data.py and build_elo.py."
    )

valid = (
    data["winner_rank"].gt(0)
    & data["loser_rank"].gt(0)
    & data["winner_rank"].ne(data["loser_rank"])
)
data = data.loc[valid].copy()

# 2022 contributes through the Elo history already calculated.
# Logistic regression trains on 2023 and 2024.
train = data[
    (data["tourney_date"] >= "2023-01-01")
    & (data["tourney_date"] < "2025-01-01")
].copy()

calibration = data[
    (data["tourney_date"] >= "2025-01-01")
    & (data["tourney_date"] < "2025-07-01")
].copy()

evaluation = data[
    (data["tourney_date"] >= "2025-07-01")
    & (data["tourney_date"] < "2026-01-01")
].copy()

if any(frame.empty for frame in [train, calibration, evaluation]):
    raise ValueError("A required training/evaluation period is empty.")


def make_features(matches):
    return pd.DataFrame({
        "ranking_strength_diff": np.log(
            matches["loser_rank"] / matches["winner_rank"]
        ),
        "overall_elo_diff": (
            matches["winner_elo_before"]
            - matches["loser_elo_before"]
        ),
        "surface_elo_diff": (
            matches["winner_surface_elo_before"]
            - matches["loser_surface_elo_before"]
        ),
    })


train_features = make_features(train)
calibration_features = make_features(calibration)
evaluation_features = make_features(evaluation)

for features in [
    train_features,
    calibration_features,
    evaluation_features,
]:
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Features contain missing or infinite values.")

# Represent each training match in both player orientations.
X_train = pd.concat(
    [train_features, -train_features],
    ignore_index=True,
)
y_train = np.concatenate([
    np.ones(len(train), dtype=int),
    np.zeros(len(train), dtype=int),
])

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        C=c,
        fit_intercept=False,
        max_iter=1000,
    ),
)
model.fit(X_train, y_train)

# Fit temperature using January–June 2025 only.
# Scores are oriented toward the historical winner.
calibration_scores = model.decision_function(
    calibration_features
)


def temperature_loss(log_temperature):
    scaled = calibration_scores / np.exp(log_temperature)
    return np.logaddexp(0, -scaled).mean()


fit = minimize_scalar(
    temperature_loss,
    bounds=(np.log(0.25), np.log(4.0)),
    method="bounded",
)

if not fit.success:
    raise RuntimeError("Temperature fitting failed.")

temperature = float(np.exp(fit.x))

# Evaluate each July–December match once.
scores = model.decision_function(evaluation_features)
calibrated_scores = scores / temperature


def metrics(decision_scores):
    probabilities = np.exp(
        -np.logaddexp(0, -decision_scores)
    )
    return {
        "accuracy_pct": 100 * (
            (decision_scores > 0).mean()
            + 0.5 * (decision_scores == 0).mean()
        ),
        "log_loss": np.logaddexp(0, -decision_scores).mean(),
        "brier": np.mean((1 - probabilities) ** 2),
    }


comparison = pd.DataFrame({
    "Uncalibrated": metrics(scores),
    "Calibrated": metrics(calibrated_scores),
}).T

ranking_accuracy = (
    evaluation["winner_rank"] < evaluation["loser_rank"]
).mean()

print(
    f"\nSettings: overall K={overall_k}, "
    f"surface K={surface_k}, C={c}"
)
print("Elo history includes 2022.")
print("Training period: 2023–2024")
print("Training matches:", len(train))
print("Calibration matches:", len(calibration))
print("Evaluation matches:", len(evaluation))
print(f"Temperature: {temperature:.4f}")

print("\nJULY–DECEMBER 2025 RESULTS")
print(comparison.round(4))
print(f"\nATP ranking accuracy: {ranking_accuracy:.2%}")

# Export predictions without saving the model.
p = np.exp(-np.logaddexp(0, -calibrated_scores))

results = evaluation[
    ["tourney_date", "winner_name", "loser_name", "surface"]
].copy()

results["probability_of_actual_winner"] = p
results["predicted_winner"] = np.where(
    p > 0.5,
    results["winner_name"],
    np.where(p < 0.5, results["loser_name"], "Tie"),
)
results["confidence_pct"] = 100 * np.maximum(p, 1 - p)
results["prediction_credit"] = np.where(
    p > 0.5, 1.0, np.where(p < 0.5, 0.0, 0.5)
)

output = folder / "predictions_2025_h2_train_2023_2024.csv"
results.to_csv(output, index=False)

print(f"\nPredictions saved to {output.name}")
print("Model trained in memory; model file not saved.")