"""a benchmark: how often does the higher-ranked player win?"""

from pathlib import Path
import pandas as pd

folder = Path(__file__).resolve().parent

data = pd.read_csv(
    folder / "matches_clean.csv",
    parse_dates=["tourney_date"]
)

# Both players need a valid ranking for this comparison
valid = (
    data["winner_rank"].gt(0)
    & data["loser_rank"].gt(0)
    & data["winner_rank"].ne(data["loser_rank"])
)

ranked = data.loc[valid].copy()

# A smaller ranking number means a higher-ranked player
ranked["correct"] = (
    ranked["winner_rank"] < ranked["loser_rank"]
)

print("Matches with usable rankings:", len(ranked))
print("Excluded from benchmark:", len(data) - len(ranked))
print(f"Overall accuracy: {ranked['correct'].mean():.1%}")

print("\nAccuracy by surface:")
surface_summary = ranked.groupby("surface")["correct"].agg(
    matches="size",
    accuracy="mean"
)
surface_summary["accuracy"] *= 100
print(surface_summary.round(1))

print("\nAccuracy by tournament-date year:")
year_summary = ranked.groupby(
    ranked["tourney_date"].dt.year
)["correct"].agg(matches="size", accuracy="mean")

year_summary["accuracy"] *= 100
print(year_summary.round(1))