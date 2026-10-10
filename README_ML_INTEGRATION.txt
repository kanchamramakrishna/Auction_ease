Auction Ease — ML-integrated ranking

1. Existing App.py and filter_players.py are preserved.
2. rank_players.py now loads models/auction_ease_performance_models.joblib and uses predictions to rank eligible players.
3. If the model file is absent or predictions fail, it falls back to the original weighted ranking.
4. The model expects the same 215-player role-labelled pool. The training dataset/evaluation files are in data/processed/.

IMPORTANT MODEL INPUT NOTE:
The saved model was trained on lagged season-level features. For ranking current player profiles, the ranker maps available career/recent profile statistics into those feature slots. This is an approximate inference-time mapping because the current player_features.csv does not expose the exact lagged season features used during training. Validate rankings before relying on them for auction decisions.

To run the UI from the project root:
    streamlit run App.py

To retrain using the 215-player source CSV:
    python train_model.py
