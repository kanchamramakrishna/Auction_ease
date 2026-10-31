from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, Model

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD PLAYER HISTORY
# ============================================================

df = pd.read_csv(DATA_DIR / "player_history.csv")

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.sort_values(
    ["player", "date", "match_id"]
).reset_index(drop=True)


# ============================================================
# 2. FEATURES USED BY LSTM
# ============================================================

features = [
    "runs",
    "balls",
    "strike_rate",
    "fours",
    "sixes",
    "wickets",
    "runs_conceded",
    "economy"
]

X_data = df[features].copy()

X_data = X_data.replace(
    [np.inf, -np.inf],
    np.nan
)

X_data = X_data.fillna(0)


# ============================================================
# 3. SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_data)


scaled_df = df[["player"]].copy()

for i, feature in enumerate(features):
    scaled_df[feature] = X_scaled[:, i]


# ============================================================
# 4. CREATE SEQUENCES
# ============================================================

SEQUENCE_LENGTH = 5

sequences = []

players = []

for player, group in scaled_df.groupby("player", sort=False):

    values = group[features].values

    # Need at least 5 matches
    if len(values) < SEQUENCE_LENGTH:
        continue

    for end in range(SEQUENCE_LENGTH, len(values) + 1):

        sequence = values[
            end - SEQUENCE_LENGTH:end
        ]

        sequences.append(sequence)
        players.append(player)


X = np.array(sequences, dtype=np.float32)


print("Sequence shape:", X.shape)
print("Number of players:", len(set(players)))


# ============================================================
# 5. BUILD LSTM AUTOENCODER
# ============================================================

input_layer = layers.Input(
    shape=(SEQUENCE_LENGTH, len(features))
)

# Encoder
lstm_output = layers.LSTM(
    64,
    return_sequences=False
)(input_layer)

embedding = layers.Dense(
    32,
    activation="relu",
    name="temporal_embedding"
)(lstm_output)


# Decoder
decoder = layers.RepeatVector(
    SEQUENCE_LENGTH
)(embedding)

decoder = layers.LSTM(
    64,
    return_sequences=True
)(decoder)

output_layer = layers.TimeDistributed(
    layers.Dense(len(features))
)(decoder)


# Complete autoencoder
autoencoder = Model(
    input_layer,
    output_layer
)


# Separate encoder
encoder = Model(
    input_layer,
    embedding
)


# ============================================================
# 6. COMPILE
# ============================================================

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse"
)


# ============================================================
# 7. TRAIN
# ============================================================

autoencoder.fit(
    X,
    X,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    shuffle=False
)


# ============================================================
# 8. CREATE FINAL PLAYER REPRESENTATION
# ============================================================

embeddings = []

for player, group in scaled_df.groupby(
    "player",
    sort=False
):

    values = group[features].values

    if len(values) < SEQUENCE_LENGTH:
        continue

    # Latest 5 matches
    latest_sequence = values[
        -SEQUENCE_LENGTH:
    ]

    latest_sequence = latest_sequence.reshape(
        1,
        SEQUENCE_LENGTH,
        len(features)
    )

    vector = encoder.predict(
        latest_sequence,
        verbose=0
    )[0]

    row = {
        "player": player
    }

    for i, value in enumerate(vector):

        row[f"lstm_embedding_{i+1}"] = float(value)

    embeddings.append(row)


# ============================================================
# 9. SAVE EMBEDDINGS
# ============================================================

embedding_df = pd.DataFrame(
    embeddings
)

embedding_df.to_csv(
    DATA_DIR / "lstm_player_embeddings.csv",
    index=False
)


# ============================================================
# 10. SAVE MODEL
# ============================================================

encoder.save(
    MODELS_DIR / "lstm_temporal_encoder.keras"
)


print("\nLSTM training completed.")

print(
    "Player embeddings:",
    len(embedding_df)
)

print(
    "Embedding dimensions:",
    len(embedding_df.columns) - 1
)

print(
    "\nSaved:"
)

print(
    "data/processed/lstm_player_embeddings.csv"
)

print(
    "models/lstm_temporal_encoder.keras"
)