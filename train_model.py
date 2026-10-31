from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# This file must be the role-labelled match-level dataset for the supported 215 players.
SOURCE = PROCESSED_DIR / "player_match_final_roles.csv"
if not SOURCE.exists():
    raise FileNotFoundError(
        f"Missing {SOURCE}. Put player_match_final_roles.csv in data/processed/."
    )

raw = pd.read_csv(SOURCE)
required = {"player", "season", "date", "role", "runs", "balls", "wickets", "runs_conceded"}
missing = required - set(raw.columns)
if missing:
    raise ValueError(f"Source CSV is missing columns: {sorted(missing)}")

player_pool = set(raw["player"].dropna().astype(str).str.strip())
if len(player_pool) != 215:
    raise ValueError(
        f"Expected the 215 role-labelled players, but found {len(player_pool)}. "
        "Check that SOURCE is player_match_final_roles.csv, not the full IPL dataset."
    )

raw["player"] = raw["player"].astype(str).str.strip()
raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
for col in ["runs", "balls", "wickets", "runs_conceded", "fours", "sixes"]:
    if col not in raw.columns:
        raw[col] = 0.0
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

# Determine season order using the first actual match date for each season.
season_dates = raw.groupby("season")["date"].min().sort_values()
season_order = [str(s) for s in season_dates.index]
season_order_map = {s: i for i, s in enumerate(season_order)}
raw["season"] = raw["season"].astype(str)

season_rows = []
for (player, season), group in raw.groupby(["player", "season"], sort=False):
    runs = group["runs"].fillna(0).sum()
    balls = group["balls"].fillna(0).sum()
    wickets = group["wickets"].fillna(0).sum()
    matches = group["match_id"].nunique() if "match_id" in group.columns else len(group)
    role = group["role"].dropna().astype(str).iloc[-1] if group["role"].notna().any() else "Unknown"
    role_text = role.lower()
    if "wicketkeeper" in role_text:
        role_group = "Wicketkeeper"
    elif "all-rounder" in role_text or "all rounder" in role_text:
        role_group = "All-Rounder"
    elif "bowler" in role_text:
        role_group = "Bowler"
    elif "batter" in role_text:
        role_group = "Batter"
    else:
        role_group = "Unknown"

    season_rows.append({
        "player": player, "season": season,
        "season_order": season_order_map[season],
        "role_group": role_group, "matches": int(matches),
        "runs": float(runs), "balls": float(balls), "wickets": float(wickets),
        "runs_per_match": runs / matches if matches else np.nan,
        "strike_rate": runs / balls * 100 if balls > 0 else np.nan,
        "wickets_per_match": wickets / matches if matches else np.nan,
    })

season_stats = pd.DataFrame(season_rows).sort_values(
    ["player", "season_order"]
).reset_index(drop=True)

numeric_features = [
    "prev_matches", "prev_runs_per_match", "prev_strike_rate",
    "prev_wickets_per_match", "prev_runs", "prev_wickets",
    "career_matches_before", "career_runs_before", "career_balls_before",
    "career_wickets_before", "career_runs_per_match_before",
    "career_strike_rate_before", "career_wickets_per_match_before",
    "mean_last3_runs_per_match", "mean_last3_strike_rate",
    "mean_last3_wickets_per_match", "seasons_played_before",
]
target_cols = [
    "target_runs_per_match", "target_strike_rate", "target_wickets_per_match"
]
previous_col = {
    "target_runs_per_match": "prev_runs_per_match",
    "target_strike_rate": "prev_strike_rate",
    "target_wickets_per_match": "prev_wickets_per_match",
}

examples = []
for player, group in season_stats.groupby("player", sort=False):
    group = group.sort_values("season_order").reset_index(drop=True)
    for idx in range(1, len(group)):
        previous = group.iloc[:idx]
        prev = previous.iloc[-1]
        target = group.iloc[idx]
        last3 = previous.tail(3)
        cm = previous["matches"].sum()
        cr = previous["runs"].sum()
        cb = previous["balls"].sum()
        cw = previous["wickets"].sum()
        examples.append({
            "player": player,
            "target_season": target["season"],
            "target_season_order": int(target["season_order"]),
            "role_group": prev["role_group"],  # role known before the target season
            "prev_matches": prev["matches"],
            "prev_runs_per_match": prev["runs_per_match"],
            "prev_strike_rate": prev["strike_rate"],
            "prev_wickets_per_match": prev["wickets_per_match"],
            "prev_runs": prev["runs"],
            "prev_wickets": prev["wickets"],
            "career_matches_before": cm,
            "career_runs_before": cr,
            "career_balls_before": cb,
            "career_wickets_before": cw,
            "career_runs_per_match_before": cr / cm if cm else np.nan,
            "career_strike_rate_before": cr / cb * 100 if cb else np.nan,
            "career_wickets_per_match_before": cw / cm if cm else np.nan,
            "mean_last3_runs_per_match": last3["runs_per_match"].mean(),
            "mean_last3_strike_rate": last3["strike_rate"].mean(),
            "mean_last3_wickets_per_match": last3["wickets_per_match"].mean(),
            "seasons_played_before": len(previous),
            "target_runs_per_match": target["runs_per_match"],
            "target_strike_rate": target["strike_rate"],
            "target_wickets_per_match": target["wickets_per_match"],
        })

ml_data = pd.DataFrame(examples).replace([np.inf, -np.inf], np.nan)
ml_data = ml_data.dropna(subset=target_cols, how="all").reset_index(drop=True)
if ml_data.empty:
    raise ValueError("No historical player-season training examples were created.")

dataset_path = PROCESSED_DIR / "auction_ml_training_dataset.csv"
ml_data.to_csv(dataset_path, index=False)

# Chronological holdout: most recent two seasons, rather than a random split.
last_two = season_order[-2:]
train_df = ml_data[~ml_data["target_season"].isin(last_two)].copy()
test_df = ml_data[ml_data["target_season"].isin(last_two)].copy()
if train_df.empty or test_df.empty:
    raise ValueError("Unable to create a chronological train/test split.")

feature_cols = numeric_features + ["role_group"]
preprocessor = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), ["role_group"]),
])

def make_model():
    return Pipeline([
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=350, min_samples_leaf=3, max_features=0.9,
            random_state=42, n_jobs=-1
        )),
    ])

metrics, predictions = [], []
for target in target_cols:
    train = train_df.dropna(subset=[target])
    test = test_df.dropna(subset=[target])
    if len(train) < 20 or len(test) < 5:
        continue

    evaluation_model = make_model()
    evaluation_model.fit(train[feature_cols], train[target])
    pred = evaluation_model.predict(test[feature_cols])
    baseline_raw = test[previous_col[target]].to_numpy(dtype=float)
    baseline = np.where(np.isfinite(baseline_raw), baseline_raw, train[target].median())

    model_mae = mean_absolute_error(test[target], pred)
    baseline_mae = mean_absolute_error(test[target], baseline)
    model_rmse = mean_squared_error(test[target], pred) ** 0.5
    baseline_rmse = mean_squared_error(test[target], baseline) ** 0.5
    metrics.append({
        "target": target, "held_out_seasons": ", ".join(last_two),
        "train_rows": len(train), "test_rows": len(test),
        "model_MAE": model_mae, "baseline_MAE": baseline_mae,
        "model_RMSE": model_rmse, "baseline_RMSE": baseline_rmse,
        "model_R2": r2_score(test[target], pred) if len(test) > 1 else np.nan,
        "MAE_improvement_pct": (baseline_mae - model_mae) / baseline_mae * 100
            if baseline_mae > 0 else np.nan,
    })
    for (_, row), predicted, base_pred in zip(test.iterrows(), pred, baseline):
        predictions.append({
            "player": row["player"], "target_season": row["target_season"],
            "role_group": row["role_group"], "target": target,
            "actual": row[target], "model_prediction": predicted,
            "baseline_prediction": base_pred,
        })

# Final models trained on all available labelled historical examples.
final_models = {}
for target in target_cols:
    available = ml_data.dropna(subset=[target])
    if len(available) >= 20:
        fitted = make_model()
        fitted.fit(available[feature_cols], available[target])
        final_models[target] = fitted

if not final_models:
    raise ValueError("No target model met the minimum training-data requirement.")

joblib.dump({
    "models": final_models,
    "numeric_features": numeric_features,
    "categorical_features": ["role_group"],
    "target_columns": list(final_models),
    "player_pool_count": len(player_pool),
    "source_file": SOURCE.name,
    "season_order": season_order,
    "held_out_seasons": last_two,
}, MODELS_DIR / "auction_ease_performance_models.joblib")

pd.DataFrame(metrics).to_csv(PROCESSED_DIR / "auction_ml_evaluation_metrics.csv", index=False)
pd.DataFrame(predictions).to_csv(PROCESSED_DIR / "auction_ml_test_predictions.csv", index=False)
summary = {
    "source_file": str(SOURCE.relative_to(BASE_DIR)),
    "role_labelled_players": len(player_pool),
    "source_player_match_rows": len(raw),
    "aggregated_player_season_rows": len(season_stats),
    "training_dataset_rows": len(ml_data),
    "training_dataset_unique_players": int(ml_data["player"].nunique()),
    "training_dataset_seasons": int(ml_data["target_season"].nunique()),
    "held_out_seasons": last_two,
    "train_rows": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "trained_targets": list(final_models),
}
(PROCESSED_DIR / "auction_ml_training_summary.json").write_text(json.dumps(summary, indent=2))

print("Training complete.")
print(json.dumps(summary, indent=2))
print("\nEvaluation metrics:")
print(pd.DataFrame(metrics).round(3).to_string(index=False))
print(f"\nSaved model: {MODELS_DIR / 'auction_ease_performance_models.joblib'}")
print(f"Saved dataset and evaluation files in: {PROCESSED_DIR}")
