import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

# Optional Spotify API use
try:
    import spotipy
    SPOTIPY_AVAILABLE = True
except ImportError:
    SPOTIPY_AVAILABLE = False


st.set_page_config(
    page_title="Spotify Popularity Project",
    layout="wide",
    page_icon="🎧",
)


# ---------------------------
# 1. DATA LOADING
# ---------------------------

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


# ---------------------------
# 2. MODEL LOADING
# ---------------------------

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

# ---------------------------
# 3. MAIN APP
# ---------------------------

def main():
    df = load_data()

    st.sidebar.title("🎧 Spotify Popularity App")
    theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)

    global_min_pop = st.sidebar.slider("Global min popularity", 0, 100, 0)

    if theme == "Dark":
        st.markdown("<style>.stApp{background:#0e1117;color:white}</style>", unsafe_allow_html=True)

    df_filtered = df[df["popularity"] >= global_min_pop]

    tab1, tab2, tab3 = st.tabs(["Dataset", "Prediction", "Playlist Builder"])

    # ---------------------------
    # TAB 1 — DATASET
    # ---------------------------
    with tab1:
        st.title("Dataset Overview")
        st.dataframe(df_filtered, use_container_width=True)

    # ---------------------------
    # TAB 2 — ML PREDICTION
    # ---------------------------
    with tab2:
        st.title("🤖 Predict Popularity")

        models = load_models()
        if not models:
            st.error("No models could be loaded.")
            return

        model_choice = st.selectbox("Choose a model", list(models.keys()))

        st.write("### Enter audio features")

        # BASE FEATURES
        energy = st.slider("Energy", 0.0, 1.0, 0.6)
        dance = st.slider("Danceability", 0.0, 1.0, 0.6)
        speech = st.slider("Speechiness", 0.0, 1.0, 0.05)
        valence = st.slider("Valence", 0.0, 1.0, 0.5)
        acoustic = st.slider("Acousticness", 0.0, 1.0, 0.2)
        instrumental = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.1)
        loudness = st.slider("Loudness (negative dB)", -60.0, 0.0, -8.0)
        tempo = st.slider("Tempo (BPM)", 50, 220, 120)
        duration = st.slider("Duration (minutes)", 1.0, 8.0, 3.0)

        # REQUIRED ENGINEERED FEATURES FOR THE MODEL
        energy_valence = energy * valence
        loudness_dance = loudness * dance

        # ARTIST POPULARITY proxy
        artist_popularity = float(df["popularity"].mean())

        # MACRO GENRE default
        macro_genre = "Pop"

        # Build final feature row for model
        explicit_flag = st.selectbox("Explicit content?", ["No", "Yes"])
        explicit_value = 1 if explicit_flag == "Yes" else 0

        sample = pd.DataFrame([{
            "danceability": dance,
            "energy": energy,
            "speechiness": speech,
            "acousticness": acoustic,
            "instrumentalness": instrumental,
            "liveness": liveness,
            "valence": valence,
            "tempo": tempo,
            "duration_min": duration,
            "loudness": loudness,
    
            # NEW: required by model
            "explicit": explicit_value,

            # engineered features
            "energy_valence": energy_valence,
            "loudness_danceability": loudness_danceability,
            "artist_popularity": artist_popularity,
            "macro_genre": macro_genre
        }])

        st.write("### Features sent to model:")
        st.dataframe(sample)

        if st.button("Predict"):
            try:
                pred = models[model_choice].predict(sample)[0]
                st.success(f"Predicted popularity: {pred:.1f}")
            except Exception as e:
                st.error("Prediction failed.")
                st.exception(e)


    # ---------------------------
    # TAB 3 — PLAYLIST BUILDER
    # ---------------------------
    with tab3:
        st.title("🎶 Playlist Builder with Filters")

        # PERSONAL PLAYLIST (session state)
        if "my_playlist" not in st.session_state:
            st.session_state.my_playlist = pd.DataFrame()

        st.subheader("🔎 Filter tracks by audio features")

        # AVAILABLE SLIDER FILTERS
        numeric_cols = [
            "danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo", "duration_min"
        ]

        filters = {}
        for col in numeric_cols:
            if col in df_filtered.columns:
                min_val, max_val = float(df_filtered[col].min()), float(df_filtered[col].max())
                filters[col] = st.slider(f"{col}", min_val, max_val, (min_val, max_val))

        # FILTER DATAFRAME
        filtered_df = df_filtered.copy()
        for col, (a, b) in filters.items():
            filtered_df = filtered_df[(filtered_df[col] >= a) & (filtered_df[col] <= b)]

        st.write(f"### {len(filtered_df)} tracks match your filters")
        st.dataframe(filtered_df[["track_name", "artists", "popularity"]], use_container_width=True)

        # SELECT TRACKS TO ADD
        st.subheader("➕ Add track to your playlist")

        track_to_add = st.selectbox(
            "Choose a track",
            filtered_df["track_name"] + " — " + filtered_df["artists"]
        )

        if st.button("Add to my playlist"):
            row = filtered_df.iloc[
                (filtered_df["track_name"] + " — " + filtered_df["artists"] == track_to_add).idxmax()
            ]
            st.session_state.my_playlist = pd.concat(
                [st.session_state.my_playlist, pd.DataFrame([row])],
                ignore_index=True
            )
            st.success("Added!")

        # SHOW PLAYLIST
        st.subheader("🎧 My Playlist")
        if len(st.session_state.my_playlist) == 0:
            st.info("Your playlist is empty. Add some tracks!")
        else:
            st.dataframe(
                st.session_state.my_playlist[["track_name", "artists", "popularity"]],
                use_container_width=True
            )


if __name__ == "__main__":
    main()
