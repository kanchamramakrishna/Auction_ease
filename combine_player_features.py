from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load player profiles and LSTM embeddings
profiles = pd.read_csv(DATA_DIR / "player_profiles.csv")
embeddings = pd.read_csv(DATA_DIR / "lstm_player_embeddings.csv")

print("Players in profiles:", profiles["player"].nunique())
print("Players with LSTM embeddings:", embeddings["player"].nunique())

# Combine using player name
player_features = profiles.merge(
    embeddings,
    on="player",
    how="left"
)

# Identify LSTM embedding columns
embedding_columns = [
    col for col in player_features.columns
    if col.startswith("lstm_embedding_")
]

# Players without sufficient history have missing embeddings.
# Use a separate indicator so the ranking model can distinguish
# missing embeddings from genuine zero-valued embeddings.
player_features["has_lstm_embedding"] = (
    player_features[embedding_columns].notna().all(axis=1)
).astype(int)

# Keep all players; fill missing embeddings with zero
player_features[embedding_columns] = (
    player_features[embedding_columns].fillna(0)
)

# Save combined dataset
player_features.to_csv(DATA_DIR / "player_features.csv", index=False)

print("\nPlayer feature dataset created successfully!")
print("Total players:", player_features["player"].nunique())
print("Total columns:", len(player_features.columns))
print(
    "Players with embeddings:",
    player_features["has_lstm_embedding"].sum()
)
print(
    "Players without embeddings:",
    (player_features["has_lstm_embedding"] == 0).sum()
)
print("\nSaved as: data/processed/player_features.csv")
