from pathlib import Path
from collections import defaultdict
from itertools import product
import heapq

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

folder = Path(__file__).resolve().parent

OVERALL_K_VALUES = [16, 32, 48]
SURFACE_K_VALUES = [8, 16, 32]
C_VALUES = [0.01, 0.1, 1.0, 10.0]

data = pd.read_csv(
    folder / "matches_clean.csv",
    parse_dates=["tourney_date"],
)

# Train on 2024; select settings on January–June 2025.
data = data[
    data["tourney_level"].isin(["250", "500", "M", "G"])
    & (data["tourney_date"] < "2025-07-01")
].sort_values("tourney_date", kind="stable")


def win_probability(a, b):
    return 1 / (1 + 10 ** ((b - a) / 400))


def build_features(overall_k, surface_k):
    overall = defaultdict(lambda: 1500.0)
    surface_elo = defaultdict(lambda: 1500.0)
    pending = []
    records = []

    for sequence, (_, match) in enumerate(data.iterrows()):
        date = match["tourney_date"]

        # Apply updates after the existing 21-day buffer.
        while pending and pending[0][0] <= date:
            _, _, w, l, surface, change, surface_change = (
                heapq.heappop(pending)
            )
            overall[w] += change
            overall[l] -= change
            surface_elo[(w, surface)] += surface_change
            surface_elo[(l, surface)] -= surface_change

        w = match["winner_id"]
        l = match["loser_id"]
        surface = match["surface"]

        w_elo, l_elo = overall[w], overall[l]
        w_surface = surface_elo[(w, surface)]
        l_surface = surface_elo[(l, surface)]

        w_rank = match["winner_rank"]
        l_rank = match["loser_rank"]

        if (
            pd.notna(w_rank)
            and pd.notna(l_rank)
            and w_rank > 0
            and l_rank > 0
            and w_rank != l_rank
        ):
            records.append({
                "date": date,
                "ranking_strength_diff": np.log(l_rank / w_rank),
                "overall_elo_diff": w_elo - l_elo,
                "surface_elo_diff": w_surface - l_surface,
            })

        # Matches without rankings still contribute to Elo history.
        change = overall_k * (1 - win_probability(w_elo, l_elo))
        surface_change = surface_k * (
            1 - win_probability(w_surface, l_surface)
        )

        heapq.heappush(
            pending,
            (
                date + pd.Timedelta(days=21),
                sequence, w, l, surface,
                change, surface_change,
            ),
        )

    return pd.DataFrame(records)


columns = [
    "ranking_strength_diff",
    "overall_elo_diff",
    "surface_elo_diff",
]

results = []
total = (
    len(OVERALL_K_VALUES)
    * len(SURFACE_K_VALUES)
    * len(C_VALUES)
)

for overall_k, surface_k in product(
    OVERALL_K_VALUES, SURFACE_K_VALUES
):
    features = build_features(overall_k, surface_k)

    train = features[features["date"].dt.year.eq(2024)]
    validation = features[features["date"].dt.year.eq(2025)]

    if train.empty or validation.empty:
        raise ValueError("Training or validation matches are missing.")

    # Both player orientations, within the training period only.
    X_train = pd.concat(
        [train[columns], -train[columns]],
        ignore_index=True,
    )
    y_train = np.concatenate([
        np.ones(len(train), dtype=int),
        np.zeros(len(train), dtype=int),
    ])

    for c in C_VALUES:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c,
                fit_intercept=False,
                max_iter=1000,
            ),
        )
        model.fit(X_train, y_train)

        # Scores refer to each match's actual winner.
        scores = model.decision_function(validation[columns])
        probabilities = np.exp(-np.logaddexp(0, -scores))

        correct = int((scores > 0).sum())
        ties = int((scores == 0).sum())
        accuracy = (correct + 0.5 * ties) / len(validation)

        results.append({
            "overall_k": overall_k,
            "surface_k": surface_k,
            "C": c,
            "matches": len(validation),
            "correct": correct,
            "ties": ties,
            "accuracy_pct": 100 * accuracy,
            "log_loss": np.logaddexp(0, -scores).mean(),
            "brier": np.mean((1 - probabilities) ** 2),
        })

    print(f"Completed {len(results)}/{total} combinations")


# Select by accuracy; use log loss to break ties.
summary = pd.DataFrame(results).sort_values(
    ["accuracy_pct", "log_loss", "overall_k", "surface_k", "C"],
    ascending=[False, True, True, True, True],
).reset_index(drop=True)

summary.to_csv(folder / "tuning_results.csv", index=False)

print("\nTOP 10 — HIGHEST VALIDATION ACCURACY")
print(summary.head(10).round(5).to_string(index=False))

original = summary[
    summary["overall_k"].eq(32)
    & summary["surface_k"].eq(32)
    & summary["C"].eq(1.0)
].iloc[0]

best = summary.iloc[0]

print("\nSELECTED SETTINGS")
print(f"Overall K: {int(best['overall_k'])}")
print(f"Surface K: {int(best['surface_k'])}")
print(f"C: {best['C']}")
print(f"Validation accuracy: {best['accuracy_pct']:.2f}%")
print(f"Original accuracy:   {original['accuracy_pct']:.2f}%")
print(
    "Accuracy improvement: "
    f"{best['accuracy_pct'] - original['accuracy_pct']:+.2f} "
    "percentage points"
)
print(f"Correct picks: {int(best['correct'])}/{int(best['matches'])}")
print(f"Equal-probability matches: {int(best['ties'])}")
print(f"Log loss: {best['log_loss']:.5f}")
print("\nSaved comparison to tuning_results.csv")