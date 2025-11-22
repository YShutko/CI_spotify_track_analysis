# Spotify Track Analytics

Analyze ~114k Spotify tracks to understand what drives popularity, explore mood/genre patterns, and prototype lightweight prediction tools for playlist building or A&R triage.

## Business Problems and Goals
- What audio and metadata signals most influence a track’s popularity score (0–100)?
- Can we quickly triage large catalogs to surface likely hits or candidate tracks for playlists?
- How do energy/valence/danceability differ by macro-genre, and where are outliers worth A&R follow-up?
- Deliver simple, reproducible tooling (notebook widgets + Gradio demo) that product or data teams can test with minimal setup.

## Data
- Source: [Kaggle – Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- Files: `data/spotify_dataset.csv` (raw, ~20 MB) and `data/spotify_cleaned_data.csv` (preprocessed subset used in all notebooks).
- Fields: track_id, artists, track_name, album_name, track_genre, popularity, duration_ms, explicit, danceability, energy, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature, plus derived macro-genre.

## Repository Layout
- `notebooks/Spotify_track_analysis.ipynb` – EDA, cleaning, feature exploration, visualizations (correlations, genre summaries, energy/valence scatter, duration vs popularity).
- `notebooks/ml_models.ipynb` – Feature engineering and model training for popularity prediction.
- `models_widgets/ipywidgets.ipynb` – In-notebook prediction widget (sliders/dropdowns) using downloaded models.
- `models_widgets/gradio.ipynb` – Gradio UI for quick web demos (multiple downloadable models).
- `models_widgets/.gradio/flagged/` – Sample flagged input from a previous Gradio run.

## Models and Comparison (from `notebooks/ml_models.ipynb`)
- Linear Regression baseline: MAE 14.17, RMSE 19.32, R² 0.252 — weak fit, high error.
- Random Forest (300 trees): MAE 4.86, RMSE 9.98, R² 0.801 — strongest performer in this run.
- XGBoost (untuned): MAE 5.92, RMSE 11.03, R² 0.756 — good, but behind RF.
- XGBoost (tuned): MAE 5.11, RMSE 9.99, R² 0.800 — closes the gap with RF after tuning.
- Takeaway: tree ensembles give the best accuracy; tuning XGB nearly matches RF while offering faster inference knobs for deployment.

## Interactive Prediction Tools
- **Notebook widget (`models_widgets/ipywidgets.ipynb`)**  
  Downloads a selected model from the Hugging Face repo `YShutko/spotify-popularity-models`, loads macro-genre options from the cleaned data, and exposes sliders/dropdowns to test popularity predictions inline.
- **Gradio app (`models_widgets/gradio.ipynb`)**  
  Loads multiple Hugging Face models and builds a Gradio UI with sliders and genre dropdown. Use it to share a quick web demo; Gradio handles launching and optional sharing links. (Note: ensure the selected model is passed through in the predict function before production use.)

## Suggested Workflow
1) Use `data/spotify_cleaned_data.csv` to skip heavy preprocessing.
2) Run `notebooks/Spotify_track_analysis.ipynb` to explore distributions, correlations, and genre-level mood/energy patterns.
3) Train and compare models in `notebooks/ml_models.ipynb`; focus on RF vs tuned XGB.
4) Demo predictions with the ipywidgets notebook or the Gradio app for stakeholder feedback.

## Conclusion
- Popularity is predictable with tree-based models using standard audio features; RF and tuned XGB achieve MAE ≈ 5–5.1 and R² ≈ 0.80 on this dataset.
- Macro-genre and energy/valence interactions remain useful signals for triage; additional metadata (artist history, release timing) could further reduce error.
- The Gradio and notebook widgets provide fast, shareable prototypes for product or A&R teams; harden by validating model selection wiring and adding input validation if deploying beyond experiments.
