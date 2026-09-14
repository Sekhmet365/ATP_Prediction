from pathlib import Path
import pandas as pd

folder = Path(__file__).resolve().parent

for year in [2024, 2025, 2026]:
    df = pd.read_csv(folder / f"{year}.csv")

    print(f"\n--- {year} ---")
    print(f"Matches: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Tournament date range:",
          df["tourney_date"].min(),
          "to",
          df["tourney_date"].max())
    print("Surfaces:")
    print(df["surface"].value_counts(dropna=False))

print("\nColumn names:")
print(df.columns.tolist())


# Combine the three files
data = pd.concat(
    [
        pd.read_csv(folder / f"{year}.csv")
        for year in [2024, 2025, 2026]
    ],
    ignore_index=True
)

print("\nTotal rows:", len(data))
print("Exact duplicate rows:", data.duplicated().sum())

# Show only columns that contain missing values
missing = data.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print("\nMissing values per column:")
print(missing)

print("\nTournament levels:")
print(data["tourney_level"].value_counts(dropna=False))

# Inspect unusual match endings
unusual = data["score"].str.contains(
    r"RET|W/O|DEF|ABD",
    case=False,
    na=False
)

print("\nRetirements, walkovers or other flagged endings:")
print(data.loc[unusual, "score"].value_counts())

##calculate overall and surface-specific Elo ratings

print("\nRound values:")
print(data["round"].value_counts(dropna=False))

print("\nExample tournament:")
example = data.loc[
    data["tourney_id"].eq(data.iloc[0]["tourney_id"]),
    [
        "tourney_id",
        "tourney_date",
        "round",
        "match_num",
        "winner_name",
        "loser_name",
    ]
]

print(example.to_string(index=False))