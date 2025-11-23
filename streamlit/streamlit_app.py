import os
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from huggingface_hub import hf_hub_download
import joblib

# Optional: Spotify API
try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    SPOTIPY_AVAILABLE = True
except ImportError:
    SPOTIPY_AVAILABLE = False


st.set_page_config(
    page_title="Spotify Popularity Project",
    layout="wide",
    page_icon="🎧",
)


# 1. DATA LOADING + MACRO-GENRE MAPPING

def map_macro_genre(g: str) -> str:
    g = str(g).lower()
    if "pop" in g:
        return "Pop"
    elif "rock" in g:
        return "Rock"
    elif "hip hop" in g or "rap" in g or "trap" in g:
        return "Hip-Hop/Rap"
    elif "r&b" in g or "soul" in g:
        return "R&B/Soul"
    elif "electro" in g or "techno" in g or "house" in g or "edm" in g or "dance" in g:
        return "Electronic/Dance"
    elif "metal" in g or "hardcore" in g:
        return "Metal/Hardcore"
    elif "jazz" in g or "blues" in g:
        return "Jazz/Blues"
    elif "classical" in g or "orchestra" in g or "piano" in g:
        return "Classical"
    elif "latin" in g or "reggaeton" in g or "sertanejo" in g or "samba" in g:
        return "Latin"
    elif "country" in g:
        return "Country"
    elif "folk" in g or "singer-songwriter" in g:
        return "Folk"
    elif "indie" in g or "alternative" in g:
        return "Indie/Alternative"
    else:
        return "Other"


@st.cache_data
def load_data(filename: str = "spotify_cleaned_data.csv") -> pd.DataFrame:
    """Robust loader that searches for the CSV file in common locations."""
    here = os.path.dirname(__file__)
    cwd = os.getcwd()

    search_paths = [
        os.path.join(here, filename),
        os.path.join(cwd, filename),
        filename,
        os.path.join(here, "..", filename),
        os.path.join(here, "data", filename),
        os.path.join(here, "..", "data", filename),
    ]

    for path in search_paths:
        if os.path.isfile(path):
            st.success(f"Loaded dataset from: **{os.path.relpath(path)}**")
            df = pd.read_csv(path)

            # macro_genre mapping
            if "track_genre" in df.columns:
                df["macro_genre"] = df["track_genre"].apply(map_macro_genre)
            elif "macro_genre" not in df.columns:
                df["macro_genre"] = "Other"

            # explicit boolean
            if "explicit" in df.columns:
                df["explicit"] = df["explicit"].astype(bool)

            # Make some columns string-safe
            string_cols = [
                "artists",
                "track_name",
                "track_genre",
                "album_name",
                "macro_genre",
                "mood_energy",
            ]
            for col in string_cols:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: "; ".join(x) if isinstance(x, list) else str(x)
                    )

            return df

    st.error("❌ spotify_cleaned_data.csv not found in expected locations!")
    raise FileNotFoundError("spotify_cleaned_data.csv not found.")


# 2. MODEL LOADER

@st.cache_resource
def load_models():
    REPO = "YShutko/spotify-popularity-models"
    model_files = {
        "Random Forest": "random_forest_model.pkl",
        "XGBoost (Tuned)": "xgb_model_best.pkl",
        "Linear Regression": "linear_regression_model.pkl",
    }

    here = os.path.dirname(__file__)
    local_dirs = [
        here,
        os.path.join(here, "models"),
        os.path.join(here, "models_widgets"),
        os.path.join(here, "..", "models"),
        os.path.join(here, "..", "models_widgets"),
    ]

    models = {}

    for name, f in model_files.items():
        model_obj = None

        # Local load
        for d in local_dirs:
            p = os.path.join(d, f)
            if os.path.isfile(p):
                st.info(f"Loaded {name} from local file `{os.path.relpath(p)}`")
                model_obj = joblib.load(p)
                break

        # HF download fallback
        if model_obj is None:
            try:
                st.info(f"Downloading {name} from HF Hub...")
                p = hf_hub_download(repo_id=REPO, filename=f, token=None)
                model_obj = joblib.load(p)
            except Exception as e:
                st.error(f"Model {name} could not be loaded. {e}")

        if model_obj is not None:
            models[name] = model_obj

    return models


# 3. MAIN APP

def main():
    df = load_data()

    st.sidebar.title("🎧 Spotify Popularity App")

    theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)
    global_min_pop = st.sidebar.slider("Global min popularity", 0, 100, 0)

    if theme == "Dark":
        st.markdown(
            "<style>.stApp { background-color:#0e1117; color:white; }</style>",
            unsafe_allow_html=True,
        )

    df_filtered = df[df["popularity"] >= global_min_pop]

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📁 Dataset", "📊 EDA", "🤖 ML Prediction", "🎶 Playlist Builder"]
    )

    # TAB 1 — DATASET
    with tab1:
        st.title("📁 Spotify Dataset Overview")
        st.dataframe(df_filtered, use_container_width=True)

    # TAB 2 — EDA
    with tab2:
        st.title("📊 Exploratory Data Analysis")
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(df_filtered[numeric_cols].corr(), cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.info("No numeric columns available for correlation heatmap.")

    # TAB 3 — ML PREDICTION
    with tab3:
        st.title("🤖 Popularity Prediction")
        try:
            models = load_models()
            if models:
                st.success("Models loaded.")
            else:
                st.warning("No models could be loaded.")
        except Exception as e:
            st.error("Failed to load models.")
            st.exception(e)
            models = {}

        if models:
            model_choice = st.selectbox("Model", list(models.keys()))
            energy = st.slider("Energy", 0.0, 1.0, 0.6)
            dance = st.slider("Danceability", 0.0, 1.0, 0.6)
            if st.button("Predict"):
                sample = pd.DataFrame([{
                    "energy": energy,
                    "danceability": dance,
                    "instrumentalness": 0,
                    "acousticness": 0,
                    "liveness": 0,
                    "valence": 0.5,
                    "tempo": 120,
                    "loudness": -8,
                    "duration_min": 3,
                }])
                model = models[model_choice]
                pred = model.predict(sample)[0]
                st.success(f"Prediction: {pred:.1f}")

    # TAB 4 — PLAYLIST BUILDER
    with tab4:
        st.title("🎶 Playlist Builder")
        if len(df_filtered) == 0:
            st.info("No tracks after filtering. Adjust the popularity slider.")
        else:
            n = st.slider("Number of tracks", 5, 50, 15)
            playlist = df_filtered.sample(min(n, len(df_filtered)), random_state=42)
            cols = [c for c in ["track_name", "artists", "popularity", "macro_genre"] if c in playlist.columns]
            st.dataframe(playlist[cols], use_container_width=True)


if __name__ == "__main__":
    main()
