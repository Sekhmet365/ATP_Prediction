from pathlib import Path
import pandas as pd

folder = Path(__file__).resolve().parent

data = pd.concat(
    [
        pd.read_csv(folder / f"{year}.csv")
        for year in [2024, 2025, 2026]
    ],
    ignore_index=True
)

# Convert tournament dates into real datetime values
data["tourney_date"] = pd.to_datetime(
    data["tourney_date"].astype(str),
    format="%Y%m%d",
    errors="raise"
)

# Identify records we cannot use in the first version
missing_id = data[["winner_id", "loser_id"]].isna().any(axis=1)

unfinished = data["score"].str.contains(
    r"RET|W/O|DEF|ABD",
    case=False,
    na=False
)

missing_score = (
    data["score"].isna()
    | data["score"].str.strip().eq("")
)

exclude = missing_id | unfinished | missing_score

clean = data.loc[~exclude].copy()
clean = clean.sort_values("tourney_date", kind="stable")
clean = clean.reset_index(drop=True)

# Save separately, preserving the original downloads
output = folder / "matches_clean.csv"
clean.to_csv(output, index=False)

print("Original matches:", len(data))
print("Missing player IDs:", missing_id.sum())
print("Flagged unfinished matches:", unfinished.sum())
print("Missing scores:", missing_score.sum())
print("Total excluded:", exclude.sum())
print("Remaining matches:", len(clean))
print("Saved to:", output)