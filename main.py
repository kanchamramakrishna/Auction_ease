from pathlib import Path

import pandas as pd

from auction_requirements import get_auction_requirements
from filter_players import filter_players
from rank_players import rank_players

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"


def main():
    # Load player dataset
    data_path = DATA_DIR / "player_features.csv"
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Error: data/processed/player_features.csv not found.")
        print("Place the processed player feature data in the project data/processed folder.")
        return

    # Verify required columns
    if not {"player", "role"}.issubset(data.columns):
        print("Error: Dataset must contain 'player' and 'role'.")
        return

    # Step 1: Collect requirements
    requirements = get_auction_requirements()

    # Step 2: Filter eligible players
    eligible = filter_players(data, requirements)

    print("\nEligible players:", len(eligible))

    if eligible.empty:
        print("No players match your requirements.")
        return

    # Step 3: Rank eligible players
    ranked = rank_players(
        eligible,
        requirements["priority"]
    )

    if ranked.empty:
        print("No ranked players available.")
        return

    # Step 4: Select columns available in the dataset
    display_columns = [
        "rank",
        "player",
        "role",
        "ranking_score",
        "career_runs",
        "career_strike_rate",
        "career_wickets",
        "recent_runs_per_match",
        "recent_wickets_per_match"
    ]

    display_columns = [
        column
        for column in display_columns
        if column in ranked.columns
    ]

    # Step 5: Display recommendations
    print("\n===== AUCTION EASE: TOP 15 PLAYERS =====")
    print("Required role:", requirements["required_role"])
    print("Bowling type:", requirements["bowling_type"])
    print("Performance priority:", requirements["priority"])

    print(
        ranked[display_columns]
        .head(15)
        .round(2)
        .to_string(index=False)
    )

    print("\nTotal eligible players ranked:", len(ranked))


if __name__ == "__main__":
    main()
