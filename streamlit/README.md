---
title: "Spotify Popularity App"
emoji: "🎧"
colorFrom: "blue"
colorTo: "green"
sdk: "docker"
sdk_version: "1.0.0"
app_file: "streamlit_app.py"
pinned: false
---

# Spotify Popularity Project – Streamlit App

This Space hosts an interactive **Streamlit** app for exploring Spotify tracks,
running EDA and making **popularity predictions** using several machine learning models.

## Features

- 📁 Dataset explorer with filters (macro-genre, track genre, artist, explicit flag)
- 📊 EDA tab with correlation heatmap and example visualisations
- 🤖 ML prediction tab loading models from **local files or Hugging Face Hub**
- 🎶 Playlist builder that filters tracks by mood, macro-genre, energy and valence

## File structure

```text
Dockerfile
requirements.txt
streamlit_app.py
spotify_cleaned_data.csv        # upload your cleaned dataset here
models/                         # (optional) local model .pkl files
models_widgets/                 # (optional) alternative model locations
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Notes

- The app will look for `spotify_cleaned_data.csv` in the Space root folder.
- If model files are not present locally, the app will try to download them from
  `YShutko/spotify-popularity-models` on Hugging Face Hub.
