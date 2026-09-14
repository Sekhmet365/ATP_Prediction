from pathlib import Path
from collections import defaultdict
import heapq

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

folder = Path(__file__).resolve().parent
K_VALUES = [16, 24, 32, 48, 64]

data = pd.read_csv(
    folder / "matches_clean.csv",
    parse_dates=["tourney_date"]
)

# Only use training and tuning periods
data = data[
    data["tourney_level"].isin(["250", "500", "M", "G"])
    & (data["tourney_date"] < "2025-07-01")
].sort_values("tourney_date", kind="stable")


def win_probability(a, b):
    return 1 / (1 + 10 ** ((b - a) / 400))


def build_features(k):
    # Fresh ratings for every candidate
    overall = defaultdict(lambda: 1500.0)
    surface_elo = defaultdict(lambda: 1500.0)
    pending = []
    records = []

    for sequence, (_, match) in enumerate(data.iterrows()):
        date = match["tourney_date"]

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

        # Only ranked matches become classifier examples.
        # Other matches still contribute to Elo history.
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

        change = k * (1 - win_probability(w_elo, l_elo))
        surface_change = k * (
            1 - win_probability(w_surface, l_surface)
        )

        heapq.heappush(
            pending,
            (
                date + pd.Timedelta(days=21),
                sequence, w, l, surface,
                change, surface_change,
            )
        )

    return pd.DataFrame(records)


columns = [
    "ranking_strength_diff",
    "overall_elo_diff",
    "surface_elo_diff",
]

results = []

for k in K_VALUES:
    features = build_features(k)

    train = features[features["date"].dt.year.eq(2024)]
    validation = features[
        features["date"].dt.year.eq(2025)
    ]

    X_train = pd.concat(
        [train[columns], -train[columns]],
        ignore_index=True,
    )
    y_train = np.concatenate([
        np.ones(len(train), dtype=int),
        np.zeros(len(train), dtype=int),
    ])

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(fit_intercept=False, max_iter=1000),
    )
    model.fit(X_train, y_train)

    scores = model.decision_function(validation[columns])
    probabilities = np.exp(-np.logaddexp(0, -scores))

    results.append({
        "k": k,
        "matches": len(validation),
        "accuracy_pct": 100 * (
            (scores > 0).mean() + 0.5 * (scores == 0).mean()
        ),
        "log_loss": np.logaddexp(0, -scores).mean(),
        "brier": np.mean((1 - probabilities) ** 2),
    })

    print(f"Finished k={k}")

summary = pd.DataFrame(results).sort_values(
    ["log_loss", "k"]
).reset_index(drop=True)

print("\nJanuary–June 2025: sorted by lowest log loss")
print(summary.round(5).to_string(index=False))

best_k = int(summary.iloc[0]["k"])
print(f"\nBest candidate by validation log loss: k={best_k}")

summary.to_csv(folder / "k_comparison.csv", index=False)