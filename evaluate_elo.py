from pathlib import Path
import pandas as pd

folder = Path(__file__).resolve().parent
data = pd.read_csv(
    folder / "matches_elo.csv",
    parse_dates=["tourney_date"]
)

# Use 2024 to build history; compare methods on 2025
test = data[
    data["tourney_date"].dt.year.eq(2025)
    & data["winner_rank"].gt(0)
    & data["loser_rank"].gt(0)
    & data["winner_rank"].ne(data["loser_rank"])
].copy()

# An initial equal blend; we haven't tuned this weight
test["winner_blend"] = (
    test["winner_elo_before"]
    + test["winner_surface_elo_before"]
) / 2

test["loser_blend"] = (
    test["loser_elo_before"]
    + test["loser_surface_elo_before"]
) / 2

comparisons = {
    # Smaller ranking number is better
    "ATP ranking": test["loser_rank"] - test["winner_rank"],

    # Higher Elo is better
    "Overall Elo": (
        test["winner_elo_before"] - test["loser_elo_before"]
    ),
    "Surface Elo": (
        test["winner_surface_elo_before"]
        - test["loser_surface_elo_before"]
    ),
    "Blended Elo": test["winner_blend"] - test["loser_blend"],
}

print("2025 evaluation matches:", len(test))

for name, difference in comparisons.items():
    wins = difference.gt(0).sum()
    ties = difference.eq(0).sum()

    # Half credit represents a random choice when ratings are equal
    accuracy = (wins + 0.5 * ties) / len(test)

    print(
        f"{name}: {accuracy:.1%}"
        f" | Equal-rating matches: {ties}"
    )