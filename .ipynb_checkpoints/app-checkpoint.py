import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎧",
    layout="wide"
)

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background: radial-gradient(circle at top left, #1db95422, #000000 55%);
        color: #ffffff;
    }
    .song-card {
        padding: 0.9rem 1rem;
        margin-bottom: 0.7rem;
        border-radius: 0.8rem;
        border: 1px solid #1db95455;
        background: #101010cc;
        box-shadow: 0 6px 16px rgba(0,0,0,0.4);
        font-size: 0.95rem;
    }
    .song-card b {
        font-size: 1rem;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 0.5rem;
        margin-bottom: 0.3rem;
    }
    .metric-pill {
        display: inline-block;
        padding: 0.1rem 0.6rem;
        margin-right: 0.3rem;
        font-size: 0.75rem;
        border-radius: 999px;
        background: #1db95422;
        border: 1px solid #1db95455;
    }
    .footer {
        margin-top: 2rem;
        text-align: center;
        font-size: 0.85rem;
        color: #ccccccaa;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: visible;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Data loading + preprocessing
# -------------------------------------------------
FEATURE_COLUMNS = [
    "Danceability",
    "Energy",
    "Valence",
    "Tempo",
    "Loudness",
    "Speechiness",
    "Acousticness",
    "Instrumentalness",
    "Liveness",
]

META_COLUMNS = ["Track Name", "Artists", "Popularity"]

# Mood presets to simplify the "vibe" selection
MOOD_PRESETS = {
    "Happy & Energetic": {
        "Danceability": 0.8,
        "Energy": 0.8,
        "Valence": 0.9,
        "Tempo": 130.0,
        "Loudness": -6.0,
        "Speechiness": 0.08,
        "Acousticness": 0.2,
        "Instrumentalness": 0.2,
        "Liveness": 0.3,
    },
    "Chill & Relaxed": {
        "Danceability": 0.6,
        "Energy": 0.4,
        "Valence": 0.6,
        "Tempo": 90.0,
        "Loudness": -12.0,
        "Speechiness": 0.05,
        "Acousticness": 0.6,
        "Instrumentalness": 0.4,
        "Liveness": 0.2,
    },
    "Sad & Emotional": {
        "Danceability": 0.4,
        "Energy": 0.3,
        "Valence": 0.2,
        "Tempo": 80.0,
        "Loudness": -14.0,
        "Speechiness": 0.06,
        "Acousticness": 0.7,
        "Instrumentalness": 0.5,
        "Liveness": 0.2,
    },
    "Workout / Gym": {
        "Danceability": 0.75,
        "Energy": 0.85,
        "Valence": 0.7,
        "Tempo": 140.0,
        "Loudness": -5.0,
        "Speechiness": 0.09,
        "Acousticness": 0.1,
        "Instrumentalness": 0.1,
        "Liveness": 0.3,
    },
    "Party / Dance": {
        "Danceability": 0.9,
        "Energy": 0.9,
        "Valence": 0.8,
        "Tempo": 125.0,
        "Loudness": -4.0,
        "Speechiness": 0.1,
        "Acousticness": 0.15,
        "Instrumentalness": 0.1,
        "Liveness": 0.4,
    },
    "Focus / Study": {
        "Danceability": 0.5,
        "Energy": 0.35,
        "Valence": 0.5,
        "Tempo": 90.0,
        "Loudness": -16.0,
        "Speechiness": 0.04,
        "Acousticness": 0.6,
        "Instrumentalness": 0.8,
        "Liveness": 0.15,
    },
}


@st.cache_data(show_spinner=False)
def load_dataset(path: str = "Spotify_data.csv"):
    df = pd.read_csv(path)

    keep_cols = META_COLUMNS + FEATURE_COLUMNS
    df = df[keep_cols].dropna().reset_index(drop=True)

    # Clip tempo range to avoid extreme outliers
    df["Tempo"] = df["Tempo"].clip(lower=40, upper=220)

    return df


@st.cache_resource(show_spinner=False)
def build_feature_matrix(df: pd.DataFrame):
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(df[FEATURE_COLUMNS])
    return scaler, features_scaled


def get_recommendations_by_song(
    df: pd.DataFrame,
    features_scaled: np.ndarray,
    track_label: str,
    top_n: int = 10,
):
    labels = df["Track Name"] + " - " + df["Artists"]
    if track_label not in labels.values:
        return []

    idx = labels[labels == track_label].index[0]
    sim_scores = cosine_similarity(
        features_scaled[idx].reshape(1, -1), features_scaled
    ).flatten()

    similar_idx = np.argsort(sim_scores)[::-1]
    similar_idx = [i for i in similar_idx if i != idx][:top_n]

    recommendations = df.iloc[similar_idx].copy()
    recommendations["similarity"] = sim_scores[similar_idx]

    return recommendations


def get_recommendations_by_profile(
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    features_scaled: np.ndarray,
    user_profile: dict,
    top_n: int = 10,
):
    user_df = pd.DataFrame([user_profile])[FEATURE_COLUMNS]
    user_scaled = scaler.transform(user_df)

    sim_scores = cosine_similarity(user_scaled, features_scaled).flatten()
    similar_idx = np.argsort(sim_scores)[::-1][:top_n]

    recommendations = df.iloc[similar_idx].copy()
    recommendations["similarity"] = sim_scores[similar_idx]

    return recommendations


# -------------------------------------------------
# Load data & precompute
# -------------------------------------------------
df = load_dataset()
scaler, features_scaled = build_feature_matrix(df)
all_labels = (df["Track Name"] + " - " + df["Artists"]).tolist()

# Precompute popular tracks for the beginner-friendly dropdown
popular_tracks = df.sort_values("Popularity", ascending=False).head(50)
popular_labels = (
    popular_tracks["Track Name"] + " - " + popular_tracks["Artists"]
).tolist()

# -------------------------------------------------
# Sidebar controls
# -------------------------------------------------
with st.sidebar:
    st.title("🎧 Controls")

    mode = st.radio(
        "Recommendation mode",
        ["By Song", "By Mood / Preferences"],
    )

    top_n = st.slider("Number of recommendations", 5, 20, 10, step=1)

    st.markdown("---")
    st.markdown("#### About this app")
    st.write(
        "- Uses **Spotify audio features**\n"
        "- Recommends songs using **cosine similarity**\n"
        "- Two modes: pick a **song** or pick a **vibe**"
    )

# -------------------------------------------------
# Main title
# -------------------------------------------------
st.title("Music Recommender System 🎶")
st.caption(
    "Content-based music recommendation using Spotify audio features and cosine similarity."
)

# -------------------------------------------------
# Mode 1: Recommend based on an existing song
# -------------------------------------------------
if mode == "By Song":
    st.markdown("### 🔍 Find songs similar to a track")

    st.write(
        "If you're using this for the first time, start by choosing a "
        "**popular track** from the dropdown below."
    )

    # Beginner-friendly dropdown: popular songs
    popular_choice = st.selectbox(
        "Quick pick: popular songs from the dataset",
        ["(None)"] + popular_labels,
    )

    st.markdown("##### Or search any song in the dataset")

    search_query = st.text_input(
        "Search by song or artist:",
        placeholder="e.g. Blinding Lights, Taylor Swift, etc.",
    ).strip()

    filtered_labels = all_labels
    manual_choice = None

    if search_query:
        filtered_labels = [
            label
            for label in all_labels
            if search_query.lower() in label.lower()
        ]
        if not filtered_labels:
            st.warning("No songs matched your search. Try a different keyword.")
            filtered_labels = []
        else:
            manual_choice = st.selectbox(
                "Select a track from search results",
                filtered_labels,
            )
    else:
        manual_choice = st.selectbox(
            "Or browse the full list",
            all_labels,
        )

    # Decide which selection to use
    selected_label = None
    if popular_choice != "(None)":
        selected_label = popular_choice
    else:
        selected_label = manual_choice

    if selected_label and st.button("Recommend similar songs 🚀"):
        with st.spinner("Finding similar tracks..."):
            recs = get_recommendations_by_song(
                df, features_scaled, selected_label, top_n=top_n
            )

        if len(recs) == 0:
            st.error("Could not find recommendations. Please try another track.")
        else:
            st.markdown("#### Recommended tracks")
            for _, row in recs.iterrows():
                st.markdown(
                    f"""
                    <div class="song-card">
                        <div class="section-title">{row['Track Name']}</div>
                        <div><i>{row['Artists']}</i></div>
                        <div style="margin-top:0.4rem;">
                            <span class="metric-pill">⭐ Popularity: {int(row['Popularity'])}</span>
                            <span class="metric-pill">🎭 Danceability: {row['Danceability']:.2f}</span>
                            <span class="metric-pill">⚡ Energy: {row['Energy']:.2f}</span>
                            <span class="metric-pill">😊 Valence: {row['Valence']:.2f}</span>
                        </div>
                        <div style="margin-top:0.3rem; font-size:0.8rem;">
                            Similarity score: {row['similarity']:.3f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# -------------------------------------------------
# Mode 2: Recommend based on mood / preferences
# -------------------------------------------------
else:
    st.markdown("### 🎛️ Choose your vibe")

    st.write(
        "Instead of adjusting many sliders, just pick how you feel. "
        "You can optionally fine-tune the mood in the advanced section."
    )

    mood = st.selectbox(
        "Select a mood / scenario",
        list(MOOD_PRESETS.keys()) + ["Custom (advanced)"],
    )

    if mood != "Custom (advanced)":
        base_profile = MOOD_PRESETS[mood].copy()

        # Optional light fine-tuning for advanced users
        with st.expander("Optional: fine-tune this mood"):
            danceability = st.slider(
                "Danceability",
                0.0,
                1.0,
                float(base_profile["Danceability"]),
                step=0.01,
            )
            energy = st.slider(
                "Energy",
                0.0,
                1.0,
                float(base_profile["Energy"]),
                step=0.01,
            )
            valence = st.slider(
                "Valence (Happiness)",
                0.0,
                1.0,
                float(base_profile["Valence"]),
                step=0.01,
            )

            base_profile["Danceability"] = danceability
            base_profile["Energy"] = energy
            base_profile["Valence"] = valence

        user_profile = base_profile

    else:
        st.markdown("#### Advanced custom mode")
        st.write("Control all audio features yourself:")

        col1, col2, col3 = st.columns(3)

        with col1:
            danceability = st.slider(
                "Danceability",
                0.0,
                1.0,
                0.7,
                step=0.01,
            )
            energy = st.slider(
                "Energy",
                0.0,
                1.0,
                0.7,
                step=0.01,
            )
            valence = st.slider(
                "Valence (Happiness)",
                0.0,
                1.0,
                0.8,
                step=0.01,
            )

        with col2:
            tempo = st.slider(
                "Tempo (BPM)",
                40.0,
                220.0,
                120.0,
                step=1.0,
            )
            loudness = st.slider(
                "Loudness (dB)",
                -60.0,
                0.0,
                -8.0,
                step=1.0,
            )
            speechiness = st.slider(
                "Speechiness",
                0.0,
                1.0,
                0.1,
                step=0.01,
            )

        with col3:
            acousticness = st.slider(
                "Acousticness",
                0.0,
                1.0,
                0.2,
                step=0.01,
            )
            instrumentalness = st.slider(
                "Instrumentalness",
                0.0,
                1.0,
                0.5,
                step=0.01,
            )
            liveness = st.slider(
                "Liveness",
                0.0,
                1.0,
                0.2,
                step=0.01,
            )

        user_profile = {
            "Danceability": danceability,
            "Energy": energy,
            "Valence": valence,
            "Tempo": tempo,
            "Loudness": loudness,
            "Speechiness": speechiness,
            "Acousticness": acousticness,
            "Instrumentalness": instrumentalness,
            "Liveness": liveness,
        }

    if st.button("Generate playlist recommendations 🎶"):
        with st.spinner("Matching songs to your vibe..."):
            recs = get_recommendations_by_profile(
                df,
                scaler,
                features_scaled,
                user_profile,
                top_n=top_n,
            )

        if len(recs) == 0:
            st.error("No recommendations found. Try a different mood or settings.")
        else:
            st.markdown("#### Recommended tracks")
            for _, row in recs.iterrows():
                st.markdown(
                    f"""
                    <div class="song-card">
                        <div class="section-title">{row['Track Name']}</div>
                        <div><i>{row['Artists']}</i></div>
                        <div style="margin-top:0.4rem;">
                            <span class="metric-pill">⭐ Popularity: {int(row['Popularity'])}</span>
                            <span class="metric-pill">🎭 Danceability: {row['Danceability']:.2f}</span>
                            <span class="metric-pill">⚡ Energy: {row['Energy']:.2f}</span>
                            <span class="metric-pill">😊 Valence: {row['Valence']:.2f}</span>
                        </div>
                        <div style="margin-top:0.3rem; font-size:0.8rem;">
                            Similarity score: {row['similarity']:.3f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Made with ❤️ using Streamlit & scikit-learn | Music Recommender Mini Project 🎶
    </div>
    """,
    unsafe_allow_html=True,
)
