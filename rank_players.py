from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "auction_ease_performance_models.joblib"

# Preserve the existing manually weighted ranker as a fallback/baseline.
BATTING_WEIGHTS = {
    "recent_runs_per_match": 0.30,
    "recent_strike_rate": 0.20,
    "runs_per_match": 0.20,
    "career_strike_rate": 0.10,
    "latest_season_runs": 0.10,
    "best_venue_runs_per_match": 0.10,
}
BOWLING_WEIGHTS = {
    "recent_wickets_per_match": 0.30,
    "wickets_per_match": 0.25,
    "career_wickets": 0.20,
    "best_venue_wickets_per_match": 0.15,
    "recent_matches": 0.10,
}
BALANCED_WEIGHTS = {
    "recent_runs_per_match": 0.20,
    "recent_strike_rate": 0.10,
    "runs_per_match": 0.15,
    "recent_wickets_per_match": 0.20,
    "wickets_per_match": 0.20,
    "career_wickets": 0.15,
}

_TARGETS = {
    "target_runs_per_match": "predicted_runs_per_match",
    "target_strike_rate": "predicted_strike_rate",
    "target_wickets_per_match": "predicted_wickets_per_match",
}
_NUMERIC_FEATURES = [
    "prev_matches", "prev_runs_per_match", "prev_strike_rate",
    "prev_wickets_per_match", "prev_runs", "prev_wickets",
    "career_matches_before", "career_runs_before", "career_balls_before",
    "career_wickets_before", "career_runs_per_match_before",
    "career_strike_rate_before", "career_wickets_per_match_before",
    "mean_last3_runs_per_match", "mean_last3_strike_rate",
    "mean_last3_wickets_per_match", "seasons_played_before",
]
_FEATURE_COLUMNS = _NUMERIC_FEATURES + ["role_group"]


def normalize(series):
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if values.notna().any():
        values = values.fillna(values.median())
    else:
        values = values.fillna(0)
    minimum, maximum = values.min(), values.max()
    if maximum == minimum:
        return pd.Series(50.0, index=values.index)
    return (values - minimum) / (maximum - minimum) * 100


def _role_group(role):
    text = str(role).lower()
    if "wicketkeeper" in text:
        return "Wicketkeeper"
    if "all-rounder" in text or "all rounder" in text:
        return "All-Rounder"
    if "bowler" in text:
        return "Bowler"
    if "batter" in text:
        return "Batter"
    return "Unknown"


def _load_model_bundle():
    if not MODEL_PATH.exists():
        return None
    try:
        bundle = joblib.load(MODEL_PATH)
        if not isinstance(bundle, dict) or not isinstance(bundle.get("models"), dict):
            return None
        return bundle
    except Exception as exc:
        warnings.warn(f"Could not load ML model bundle: {exc}")
        return None


def _historical_features(row):
    # At serving time, these are built from the existing player's profile fields.
    # Career values are available in player_features.csv. Recent values use last-five-match summaries.
    career_matches = pd.to_numeric(row.get("career_matches"), errors="coerce")
    career_runs = pd.to_numeric(row.get("career_runs"), errors="coerce")
    career_wickets = pd.to_numeric(row.get("career_wickets"), errors="coerce")
    career_sr = pd.to_numeric(row.get("career_strike_rate"), errors="coerce")
    runs_pm = pd.to_numeric(row.get("runs_per_match"), errors="coerce")
    wickets_pm = pd.to_numeric(row.get("wickets_per_match"), errors="coerce")
    recent_runs_pm = pd.to_numeric(row.get("recent_runs_per_match"), errors="coerce")
    recent_sr = pd.to_numeric(row.get("recent_strike_rate"), errors="coerce")
    recent_wickets_pm = pd.to_numeric(row.get("recent_wickets_per_match"), errors="coerce")
    recent_matches = pd.to_numeric(row.get("recent_matches"), errors="coerce")
    career_balls = career_runs * 100 / career_sr if pd.notna(career_runs) and pd.notna(career_sr) and career_sr > 0 else np.nan

    # Estimate previous-season-style values from available recent/career profile summaries.
    # Model was trained on season-history features; fields unavailable in profile are missing and imputed.
    values = {
        "prev_matches": recent_matches,
        "prev_runs_per_match": recent_runs_pm,
        "prev_strike_rate": recent_sr,
        "prev_wickets_per_match": recent_wickets_pm,
        "prev_runs": recent_runs_pm * recent_matches if pd.notna(recent_runs_pm) and pd.notna(recent_matches) else np.nan,
        "prev_wickets": recent_wickets_pm * recent_matches if pd.notna(recent_wickets_pm) and pd.notna(recent_matches) else np.nan,
        "career_matches_before": career_matches,
        "career_runs_before": career_runs,
        "career_balls_before": career_balls,
        "career_wickets_before": career_wickets,
        "career_runs_per_match_before": runs_pm,
        "career_strike_rate_before": career_sr,
        "career_wickets_per_match_before": wickets_pm,
        "mean_last3_runs_per_match": recent_runs_pm,
        "mean_last3_strike_rate": recent_sr,
        "mean_last3_wickets_per_match": recent_wickets_pm,
        "seasons_played_before": row.get("seasons_played", np.nan),
        "role_group": _role_group(row.get("role", "")),
    }
    return values


def _apply_ml_predictions(ranked, bundle):
    models = bundle.get("models", {})
    feature_columns = bundle.get("numeric_features", _NUMERIC_FEATURES) + bundle.get("categorical_features", ["role_group"])
    feature_frame = pd.DataFrame([_historical_features(row) for _, row in ranked.iterrows()], index=ranked.index)
    for feature in feature_columns:
        if feature not in feature_frame.columns:
            feature_frame[feature] = np.nan

    applied = []
    for target, output_col in _TARGETS.items():
        model = models.get(target)
        if model is None:
            continue
        try:
            ranked[output_col] = model.predict(feature_frame[feature_columns])
            applied.append(output_col)
        except Exception as exc:
            warnings.warn(f"ML prediction failed for {target}; will use baseline ranking: {exc}")

    if not applied:
        return ranked, False
    return ranked, True


def rank_players(eligible, priority):
    if eligible.empty:
        return eligible.copy()

    ranked = eligible.copy()
    bundle = _load_model_bundle()
    ranked, ml_active = _apply_ml_predictions(ranked, bundle) if bundle else (ranked, False)

    # ML predictions drive the scores when available; existing ranking remains a fallback.
    if ml_active:
        if priority == "Batting":
            score_parts = {
                "predicted_runs_per_match": 0.60,
                "predicted_strike_rate": 0.40,
            }
        elif priority == "Bowling":
            score_parts = {"predicted_wickets_per_match": 1.0}
        else:
            score_parts = {
                "predicted_runs_per_match": 0.40,
                "predicted_strike_rate": 0.20,
                "predicted_wickets_per_match": 0.40,
            }

        scores = pd.Series(0.0, index=ranked.index)
        total_weight = 0.0
        for feature, weight in score_parts.items():
            if feature in ranked.columns:
                scores += normalize(ranked[feature]) * weight
                total_weight += weight
        if total_weight:
            ranked["ranking_score"] = scores / total_weight
            ranked["ranking_method"] = "ML predicted performance"
        else:
            ml_active = False

    if not ml_active:
        weights = BATTING_WEIGHTS if priority == "Batting" else BOWLING_WEIGHTS if priority == "Bowling" else BALANCED_WEIGHTS
        scores = pd.Series(0.0, index=ranked.index)
        total_weight = 0.0
        for feature, weight in weights.items():
            if feature in ranked.columns:
                scores += normalize(ranked[feature]) * weight
                total_weight += weight
        ranked["ranking_score"] = scores / total_weight if total_weight else 0.0
        ranked["ranking_method"] = "Weighted-statistics fallback"

    if "has_lstm_embedding" in ranked.columns:
        embedding_available = pd.to_numeric(ranked["has_lstm_embedding"], errors="coerce").fillna(0)
    else:
        embedding_available = pd.Series(0, index=ranked.index)

    ranked["_embedding_available"] = embedding_available
    ranked = ranked.sort_values(
        ["ranking_score", "_embedding_available"],
        ascending=[False, False], kind="stable"
    ).drop(columns="_embedding_available")
    ranked = ranked.reset_index(drop=True)
    if "rank" in ranked.columns:
        ranked = ranked.drop(columns="rank")
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked
