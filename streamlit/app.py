# app.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit.components.v1 as components

from huggingface_hub import hf_hub_download
import joblib

# =============== OPTIONAL: Spotify API (Spotipy) ===================
try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    SPOTIPY_AVAILABLE = True
except ImportError:
    SPOTIPY_AVAILABLE = False
# ===================================================================

st.set_page_config(
    page_title="Spotify Popularity Project",
    layout="wide",
    page_icon="🎧"
)

# =========================================================
# 1. DATA LOADING + MACRO-GENRE MAPPING
# =========================================================

def map_macro_genre(g):
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
def load_data():
    df = pd.read_csv("spotify_cleaned_data.csv")

    # Ensure macro_genre exists using your mapping
    if "track_genre" in df.columns:
        df["macro_genre"] = df["track_genre"].apply(map_macro_genre)
    else:
        df["macro_genre"] = "Other"

    return df

df = load_data()

# =========================================================
# 2. LOAD MODELS FROM HUGGING FACE
# =========================================================

@st.cache_resource
def load_models_from_hf():
    REPO = "YShutko/spotify-popularity-models"  # adjust if different

    model_files = {
        "Random Forest": "random_forest_model.pkl",
        "XGBoost (Tuned)": "xgb_model_best.pkl",
        "Linear Regression": "linear_regression_model.pkl",
    }

    models = {}
    for name, fname in model_files.items():
        model_path = hf_hub_download(
            repo_id=REPO,
            filename=fname,
            token=None  # add token here if your files still require auth
        )
        models[name] = joblib.load(model_path)

    return models

models = load_models_from_hf()

# =========================================================
# 3. SIDEBAR (THEME + GLOBAL FILTERS)
# =========================================================

st.sidebar.title("🎧 Spotify Popularity App")

theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)
global_min_pop = st.sidebar.slider("Global min popularity", 0, 100, 0)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Models from **Hugging Face**:\n"
    "- Random Forest\n- XGBoost (tuned)\n- Linear Regression"
)

if theme == "Dark":
    st.markdown(
        """
        <style>
        body { background-color: #0e1117; color: #fafafa; }
        .stApp { background-color: #0e1117; }
        </style>
        """,
        unsafe_allow_html=True,
    )

df_filtered_global = df[df["popularity"] >= global_min_pop].copy()

# =========================================================
# 4. TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["📁 Dataset", "📊 EDA", "🤖 ML Prediction", "🎶 Playlist Builder"]
)

# ---------------------------------------------------------
# TAB 1 – DATASET
# ---------------------------------------------------------
with tab1:
    st.title("📁 Spotify Dataset Overview")

    st.write(
        "Explore the Spotify tracks by filtering on macro-genre, track genre, artist, "
        "and explicit content. The global popularity filter in the sidebar is also applied."
    )

    col1, col2, col3, col4 = st.columns(4)

    macro_filter = col1.selectbox(
        "Macro Genre", ["All"] + sorted(df_filtered_global["macro_genre"].dropna().unique().tolist())
    )
    genre_filter = col2.selectbox(
        "Track Genre", ["All"] + sorted(df_filtered_global["track_genre"].dropna().unique().tolist())
        if "track_genre" in df_filtered_global.columns else ["All"]
    )
    artist_filter = col3.selectbox(
        "Artist", ["All"] + sorted(df_filtered_global["artists"].dropna().unique().tolist())
        if "artists" in df_filtered_global.columns else ["All"]
    )
    explicit_filter = col4.selectbox("Explicit", ["All", True, False])

    df_tab1 = df_filtered_global.copy()
    if macro_filter != "All":
        df_tab1 = df_tab1[df_tab1["macro_genre"] == macro_filter]
    if genre_filter != "All" and "track_genre" in df_tab1.columns:
        df_tab1 = df_tab1[df_tab1["track_genre"] == genre_filter]
    if artist_filter != "All" and "artists" in df_tab1.columns:
        df_tab1 = df_tab1[df_tab1["artists"] == artist_filter]
    if explicit_filter != "All" and "explicit" in df_tab1.columns:
        df_tab1 = df_tab1[df_tab1["explicit"] == explicit_filter]

    st.subheader(f"Filtered Dataset ({len(df_tab1)} rows)")
    st.dataframe(df_tab1, use_container_width=True, height=450)

    with st.expander("Summary statistics"):
        st.write(df_tab1.describe(include="all"))

# ---------------------------------------------------------
# TAB 2 – EDA
# ---------------------------------------------------------
with tab2:
    st.title("📊 Exploratory Data Analysis")

    st.write(
        "Key visualizations to understand how audio features, genres and mood relate "
        "to popularity."
    )

    # 1. Correlation heatmap
    st.subheader("Correlation Heatmap (Numeric Features)")
    numeric_cols = df_filtered_global.select_dtypes(include=[np.number]).columns
    corr = df_filtered_global[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

    st.markdown(
        "- Strong positive correlation between **energy** and **loudness**.\n"
        "- Popularity has only moderate correlations with individual audio features."
    )

    # 2. Popularity by mood/energy (if available)
    if "mood_energy" in df_filtered_global.columns:
        st.subheader("Popularity Distribution by Mood/Energy Group")
        fig_violin = px.violin(
            df_filtered_global,
            x="mood_energy",
            y="popularity",
            color="mood_energy",
            box=True,
            points="all",
        )
        st.plotly_chart(fig_violin, use_container_width=True)
        st.markdown(
            "Different mood–energy combinations show different popularity distributions."
        )

    # 3. Energy vs Loudness by macro-genre
    st.subheader("Energy vs Loudness (colored by Macro Genre)")
    fig_scatter = px.scatter(
        df_filtered_global.sample(min(4000, len(df_filtered_global))),  # sample to keep it light
        x="energy",
        y="loudness",
        color="macro_genre",
        opacity=0.5,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown("Higher energy tracks tend to be louder, with clear genre clusters.")

# ---------------------------------------------------------
# TAB 3 – ML PREDICTION (IN-APP + GRADIO)
# ---------------------------------------------------------
with tab3:
    st.title("🤖 Popularity Prediction")

    st.write(
        "Predict track popularity using different machine learning models. "
        "Below you can either use the in-app prediction form or an embedded Gradio widget."
    )

    # In-app prediction form
    st.subheader("In-app ML Prediction")

    col_a, col_b = st.columns(2)

    with col_a:
        model_choice = st.selectbox("Choose model", list(models.keys()), index=0)
        energy = st.slider("Energy", 0.0, 1.0, 0.6, 0.01)
        danceability = st.slider("Danceability", 0.0, 1.0, 0.6, 0.01)
        valence = st.slider("Valence (positiveness)", 0.0, 1.0, 0.5, 0.01)
        loudness = st.slider("Loudness (dB)", -60.0, 0.0, -8.0, 1.0)

    with col_b:
        tempo = st.slider("Tempo (BPM)", 50, 200, 120, 1)
        explicit_val = st.selectbox("Explicit content", [False, True])
        macro_genre_pred = st.selectbox(
            "Macro Genre", sorted(df["macro_genre"].dropna().unique())
        )
        artist_pop = st.slider("Artist Popularity (baseline)", 0, 100, 50, 1)

    if st.button("Predict popularity"):
        model = models[model_choice]

        sample = pd.DataFrame(
            [
                {
                    "energy": energy,
                    "danceability": danceability,
                    "valence": valence,
                    "loudness": loudness,
                    "tempo": tempo,
                    "explicit": explicit_val,
                    "macro_genre": macro_genre_pred,
                    "artist_popularity": artist_pop,
                    "loudness_danceability": loudness * danceability,
                    "energy_valence": energy * valence,
                    "instrumentalness": 0,
                    "acousticness": 0,
                    "liveness": 0,
                    "speechiness": 0,
                    "duration_min": 3,
                }
            ]
        )

        pred = model.predict(sample)[0]
        st.success(f"🎵 Predicted popularity ({model_choice}): **{pred:.1f}**")

    st.markdown("---")

    # Embedded Gradio app (optional)
    st.subheader("Embedded Gradio Widget")

    st.write(
        "If you have deployed a Gradio app (e.g. on HuggingFace Spaces), "
        "you can embed it below by setting its URL."
    )

    gradio_app_url = "https://<your-gradio-space-url>"  # TODO: replace with your URL
    if "your-gradio-space-url" not in gradio_app_url:
        components.iframe(gradio_app_url, height=900, scrolling=True)
    else:
        st.info("Set `gradio_app_url` in the code to embed your Gradio app here.")

# ---------------------------------------------------------
# TAB 4 – PLAYLIST BUILDER
# ---------------------------------------------------------
with tab4:
    st.title("🎶 Playlist Composer")

    st.write(
        "Build a playlist based on mood, macro-genre, energy, valence, and popularity. "
        "You can download the playlist as CSV, and optionally send it to Spotify if configured."
    )

    c1, c2, c3 = st.columns(3)

    # Mood filter (if exists)
    if "mood_energy" in df_filtered_global.columns:
        mood_choice = c1.selectbox(
            "Mood/Energy Group",
            sorted(df_filtered_global["mood_energy"].dropna().unique()),
        )
        base = df_filtered_global[df_filtered_global["mood_energy"] == mood_choice]
    else:
        mood_choice = None
        base = df_filtered_global.copy()

    macro_choice_pl = c2.selectbox(
        "Macro Genre (optional)",
        ["All"] + sorted(df_filtered_global["macro_genre"].dropna().unique()),
    )
    if macro_choice_pl != "All":
        base = base[base["macro_genre"] == macro_choice_pl]

    min_pop_playlist = c3.slider("Min popularity", 0, 100, 40, 1)
    base = base[base["popularity"] >= min_pop_playlist]

    e_min, e_max = st.slider("Energy range", 0.0, 1.0, (0.3, 0.9), 0.01)
    v_min, v_max = st.slider("Valence range", 0.0, 1.0, (0.3, 0.9), 0.01)

    base = base[(base["energy"] >= e_min) & (base["energy"] <= e_max)]
    base = base[(base["valence"] >= v_min) & (base["valence"] <= v_max)]

    n_tracks = st.slider("Number of tracks", 1, 50, 15, 1)

    if len(base) == 0:
        st.warning("No tracks match these filters. Try relaxing the constraints.")
        playlist = pd.DataFrame()
    else:
        playlist = base.sample(min(n_tracks, len(base)), random_state=42)

    if not playlist.empty:
        st.subheader("Generated Playlist")
        display_cols = [
            c for c in ["track_name", "artists", "popularity", "macro_genre", "mood_energy"]
            if c in playlist.columns
        ]
        st.dataframe(playlist[display_cols], use_container_width=True, height=400)

        # Preview audio if preview_url exists
        if "preview_url" in playlist.columns:
            previews = playlist["preview_url"].dropna()
            if not previews.empty:
                st.subheader("Preview first track")
                st.audio(previews.iloc[0])
            else:
                st.info("No preview URLs available for the current playlist.")
        else:
            st.info("No `preview_url` column available – audio preview disabled.")

        # Download CSV
        csv_bytes = playlist.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download playlist as CSV",
            data=csv_bytes,
            file_name="playlist.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.subheader("Send to Spotify (optional)")

        if not SPOTIPY_AVAILABLE:
            st.info(
                "To enable Spotify export, install `spotipy` and configure "
                "Spotify credentials (SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI)."
            )
        else:
            st.write(
                "Define Spotify credentials as environment variables and log in "
                "to create a playlist directly in your account."
            )

            spotify_username = st.text_input("Spotify username")
            playlist_name = st.text_input("New playlist name", value="CI Spotify ML Playlist")

            # Try to find a column with Spotify IDs
            track_id_col = None
            for cand in ["id", "track_id", "track_uri"]:
                if cand in playlist.columns:
                    track_id_col = cand
                    break

            if track_id_col is None:
                st.info(
                    "No Spotify track ID column (`id`, `track_id` or `track_uri`) found. "
                    "Cannot create Spotify playlist from this dataset."
                )
            else:
                if st.button("Create playlist on my Spotify"):
                    try:
                        scope = "playlist-modify-public"
                        sp_oauth = SpotifyOAuth(scope=scope)
                        sp = spotipy.Spotify(auth_manager=sp_oauth)

                        sp_playlist = sp.user_playlist_create(
                            spotify_username,
                            playlist_name,
                            public=True,
                            description="Created by Streamlit Spotify ML app",
                        )

                        track_ids = playlist[track_id_col].astype(str).tolist()
                        cleaned_ids = []
                        for tid in track_ids:
                            if tid.startswith("spotify:track:"):
                                cleaned_ids.append(tid)
                            else:
                                cleaned_ids.append("spotify:track:" + tid)

                        sp.playlist_add_items(sp_playlist["id"], cleaned_ids)
                        st.success("Playlist created on Spotify 🎉")
                    except Exception as e:
                        st.error("Failed to create playlist. Check credentials and permissions.")
                        st.exception(e)
