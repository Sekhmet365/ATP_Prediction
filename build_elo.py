from pathlib import Path
from collections import defaultdict
import heapq

import pandas as pd


def win_probability(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


folder = Path(__file__).resolve().parent

data = pd.read_csv(
    folder / "matches_clean.csv",
    parse_dates=["tourney_date"]
)

# Start with standard tour events and Grand Slams
data = data[
    data["tourney_level"].isin(["250", "500", "M", "G"])
].copy()

data = data.sort_values("tourney_date", kind="stable")

# Everyone starts equally because we have no earlier history yet
overall = defaultdict(lambda: 1500.0)
surface_ratings = defaultdict(lambda: 1500.0)

#overall_k = 48
#surface_k = 8

#print(f"Using overall K={overall_k}, surface K={surface_k}")

settings = pd.read_csv(folder / "tuning_results.csv").sort_values(
    ["accuracy_pct", "log_loss", "overall_k", "surface_k", "C"],
    ascending=[False, True, True, True, True],
).iloc[0]

overall_k = int(settings["overall_k"])
surface_k = int(settings["surface_k"])

print(f"Using overall K={overall_k}, surface K={surface_k}")
pending = []
records = []

for sequence, (_, match) in enumerate(data.iterrows()):
    date = match["tourney_date"]

    # Apply results whose waiting period has passed
    while pending and pending[0][0] <= date:
        _, _, winner, loser, surface, change, surface_change = (
            heapq.heappop(pending)
        )

        overall[winner] += change
        overall[loser] -= change

        surface_ratings[(winner, surface)] += surface_change
        surface_ratings[(loser, surface)] -= surface_change

    winner = match["winner_id"]
    loser = match["loser_id"]
    surface = match["surface"]

    # Store ratings BEFORE this match's result becomes available
    winner_elo = overall[winner]
    loser_elo = overall[loser]

    winner_surface = surface_ratings[(winner, surface)]
    loser_surface = surface_ratings[(loser, surface)]

    record = match.to_dict()
    record.update({
        "winner_elo_before": winner_elo,
        "loser_elo_before": loser_elo,
        "winner_surface_elo_before": winner_surface,
        "loser_surface_elo_before": loser_surface,
    })
    records.append(record)

    # Calculate updates, but delay applying them
    change = overall_k * (
        1 - win_probability(winner_elo, loser_elo)
    )
    surface_change = surface_k * (
        1 - win_probability(winner_surface, loser_surface)
    )

    available_date = date + pd.Timedelta(days=21)

    heapq.heappush(
        pending,
        (
            available_date, sequence, winner, loser,
            surface, change, surface_change
        )
    )

features = pd.DataFrame(records)
features["overall_k"] = overall_k
features["surface_k"] = surface_k
features.to_csv(folder / "matches_elo.csv", index=False)

print("Matches saved:", len(features))
print("\nExample pre-match ratings:")
print(features[
    [
        "winner_name",
        "loser_name",
        "winner_elo_before",
        "loser_elo_before",
    ]
].tail(10).round(1).to_string(index=False))


# Export a player snapshot for the interface.
snapshot_date = data["tourney_date"].max()

player_records = []

for side in ["winner", "loser"]:
    part = data[
        [
            f"{side}_id",
            f"{side}_name",
            f"{side}_rank",
            "tourney_date",
        ]
    ].copy()

    part.columns = ["player_id", "name", "rank", "record_date"]
    player_records.append(part)

players = pd.concat(player_records, ignore_index=True)
players = players.sort_values("record_date", kind="stable")
players = players.drop_duplicates("player_id", keep="last")

players["overall_elo"] = players["player_id"].map(
    lambda player: overall.get(player, 1500.0)
)

for court in ["Hard", "Clay", "Grass"]:
    players[f"{court}_elo"] = players["player_id"].map(
        lambda player: surface_ratings.get((player, court), 1500.0)
    )

players["snapshot_date"] = snapshot_date
players["overall_k"] = overall_k
players["surface_k"] = surface_k

players.to_csv(folder / "players_snapshot.csv", index=False)
print("Saved: players_snapshot.csv")