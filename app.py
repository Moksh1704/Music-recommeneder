import streamlit as st
import streamlit.components.v1 as components
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
# Custom CSS – cream UI, readable text, fixed buttons
# -------------------------------------------------
st.markdown(
    """
    <style>

    /* ===== BACKGROUND ===== */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f7f2e9;
        color: #111827;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f3e9dd;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* ===== SONG CARD ===== */
    .song-card {
        padding: 0.85rem 1rem;
        border-radius: 0.8rem;
        border: 1px solid #e5d7c6;
        background: #fffaf3;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.06);
        font-size: 0.95rem;
        margin-bottom: 0.8rem;
    }

    .song-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.75rem;
    }

    .song-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #111827;
    }

    .similarity-pill {
        display: inline-block;
        padding: 0.15rem 0.65rem;
        border-radius: 999px;
        background: #ede9fe;
        border: 1px solid #ddd6fe;
        font-size: 0.75rem;
        color: #3730a3;
        white-space: nowrap;
    }

    .song-artist {
        font-size: 0.9rem;
        color: #6b7280;
        margin-bottom: 0.35rem;
    }

    .metric-pill {
        display: inline-block;
        padding: 0.12rem 0.55rem;
        margin-right: 0.35rem;
        font-size: 0.75rem;
        border-radius: 999px;
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        color: #374151;
    }

    /* ===== TEXT & LABELS ===== */
    label, .stMarkdown, .stRadio label, .stSlider label, .stSelectbox label,
    h1, h2, h3, h4, h5, h6, p {
        color: #111827 !important;
        font-weight: 600 !important;
    }

    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #111827;
        margin-top: 1.2rem;
        margin-bottom: 0.3rem;
    }

    .subtle-caption {
        font-size: 0.88rem;
        color: #6b7280;
        margin-bottom: 0.6rem;
    }

    /* ===== BUTTONS – FINAL FIX ===== */
    .stButton > button {
        background-color: #111827 !important;
        color: #ffffff !important;
        border-radius: 999px !important;
        border: 1px solid #111827 !important;
        padding: 0.55rem 1.5rem !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        opacity: 1 !important;
        box-shadow: none !important;
    }

    /* also force any text INSIDE the button to white */
    .stButton > button * {
        color: #ffffff !important;
    }

    .stButton > button:hover {
        background-color: #374151 !important;
        border-color: #374151 !important;
        color: #ffffff !important;
    }

    /* ===== INPUTS & DROPDOWNS ===== */
    div[data-baseweb="select"] > div {
        background-color: #fffaf3 !important;
        border-radius: 0.55rem;
        border: 1px solid #e5d7c6;
    }
    div[data-baseweb="select"] * {
        color: #111827 !important;
    }

    input, textarea {
        background-color: #fffaf3 !important;
        border-radius: 0.55rem !important;
        border: 1px solid #e5d7c6 !important;
        color: #111827 !important;
    }

    ul[role="listbox"] {
        background: #fffaf3 !important;
    }
    ul[role="listbox"] li {
        color: #111827 !important;
    }

    [data-testid="stSidebar"] * {
        color: #111827 !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: visible;}

    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Feature and mood settings
# -------------------------------------------------
FEATURE_COLUMNS = [
    "Danceability", "Energy", "Valence", "Tempo", "Loudness",
    "Speechiness", "Acousticness", "Instrumentalness", "Liveness",
]

META_COLUMNS = ["Track Name", "Artists", "Popularity", "Track ID"]

MOOD_PRESETS = {
    "Happy & Energetic": {"Danceability": 0.8, "Energy": 0.8, "Valence": 0.9,
                          "Tempo": 130.0, "Loudness": -6.0, "Speechiness": 0.08,
                          "Acousticness": 0.2, "Instrumentalness": 0.2, "Liveness": 0.3},
    "Chill & Relaxed": {"Danceability": 0.6, "Energy": 0.4, "Valence": 0.6,
                        "Tempo": 90.0, "Loudness": -12.0, "Speechiness": 0.05,
                        "Acousticness": 0.6, "Instrumentalness": 0.4, "Liveness": 0.2},
    "Sad & Emotional": {"Danceability": 0.4, "Energy": 0.3, "Valence": 0.2,
                        "Tempo": 80.0, "Loudness": -14.0, "Speechiness": 0.06,
                        "Acousticness": 0.7, "Instrumentalness": 0.5, "Liveness": 0.2},
    "Workout / Gym": {"Danceability": 0.75, "Energy": 0.85, "Valence": 0.7,
                      "Tempo": 140.0, "Loudness": -5.0, "Speechiness": 0.09,
                      "Acousticness": 0.1, "Instrumentalness": 0.1, "Liveness": 0.3},
    "Party / Dance": {"Danceability": 0.9, "Energy": 0.9, "Valence": 0.8,
                      "Tempo": 125.0, "Loudness": -4.0, "Speechiness": 0.1,
                      "Acousticness": 0.15, "Instrumentalness": 0.1, "Liveness": 0.4},
    "Focus / Study": {"Danceability": 0.5, "Energy": 0.35, "Valence": 0.5,
                      "Tempo": 90.0, "Loudness": -16.0, "Speechiness": 0.04,
                      "Acousticness": 0.6, "Instrumentalness": 0.8, "Liveness": 0.15},
}

# -------------------------------------------------
# Load Data
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset(path="Spotify_data.csv"):
    df = pd.read_csv(path)
    df = df[META_COLUMNS + FEATURE_COLUMNS]
    df = df.dropna().reset_index(drop=True)
    df["Tempo"] = df["Tempo"].clip(40, 220)
    return df


@st.cache_resource
def build_feature_matrix(df):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[FEATURE_COLUMNS])
    return scaler, X

df = load_dataset()
scaler, X = build_feature_matrix(df)
labels = df["Track Name"] + " - " + df["Artists"]

popular_labels = labels[df["Popularity"].nlargest(50).index].tolist()

# -------------------------------------------------
# Recommendation Functions
# -------------------------------------------------
def get_by_song(song, top_n=10):
    idx = labels[labels == song].index[0]
    sims = cosine_similarity(X[idx].reshape(1, -1), X).flatten()
    rec_idx = [i for i in sims.argsort()[::-1] if i != idx][:top_n]
    out = df.iloc[rec_idx].copy()
    out["similarity"] = sims[rec_idx]
    return out


def get_by_profile(profile, top_n=10):
    up = scaler.transform(pd.DataFrame([profile]))
    sims = cosine_similarity(up, X).flatten()
    rec_idx = sims.argsort()[::-1][:top_n]
    out = df.iloc[rec_idx].copy()
    out["similarity"] = sims[rec_idx]
    return out

# -------------------------------------------------
# Song Card
# -------------------------------------------------
def render_song_card(row):
    name = row["Track Name"]
    artist = row["Artists"]
    sim = row["similarity"]
    pop = row["Popularity"]
    tid = row["Track ID"]

    dance = row["Danceability"]
    energy = row["Energy"]
    valence = row["Valence"]

    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown(
            f"""
            <div class="song-card">
                <div class="song-header">
                    <div class="song-title">{name}</div>
                    <div class="similarity-pill">Similarity: {sim:.3f}</div>
                </div>
                <div class="song-artist">{artist}</div>
                <span class="metric-pill">Popularity: {pop}</span>
                <span class="metric-pill">Danceability: {dance:.2f}</span>
                <span class="metric-pill">Energy: {energy:.2f}</span>
                <span class="metric-pill">Valence: {valence:.2f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        components.iframe(f"https://open.spotify.com/embed/track/{tid}", height=80)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.markdown("### Controls")
    mode = st.radio("Recommendation mode", ["By Song", "By Mood / Preferences"])
    top_n = st.slider("Number of recommendations", 5, 20, 10)

# -------------------------------------------------
# Main Page
# -------------------------------------------------
st.markdown("## Music Recommender System")
st.markdown(
    '<p class="subtle-caption">'
    'Content-based recommendations using Spotify audio features + cosine similarity.'
    '</p>',
    unsafe_allow_html=True,
)

# -------------------------------------------------
# By Song
# -------------------------------------------------
if mode == "By Song":
    st.markdown('<div class="section-header">Find songs similar to a track</div>', unsafe_allow_html=True)

    pop_pick = st.selectbox("Popular songs", ["(None)"] + popular_labels)

    st.markdown("##### Search manually")
    q = st.text_input("Search by song or artist")

    if q:
        matches = [x for x in labels if q.lower() in x.lower()]
        match_pick = st.selectbox("Matches", matches) if matches else None
    else:
        match_pick = st.selectbox("Browse full list", labels)

    chosen = match_pick if pop_pick == "(None)" else pop_pick

    if chosen and st.button("Get similar songs"):
        recs = get_by_song(chosen, top_n)
        st.markdown('<div class="section-header">Recommended Tracks</div>', unsafe_allow_html=True)
        for _, row in recs.iterrows():
            render_song_card(row)

# -------------------------------------------------
# By Mood
# -------------------------------------------------
else:
    st.markdown('<div class="section-header">Generate playlist by mood</div>', unsafe_allow_html=True)

    mood = st.selectbox("Choose a mood", list(MOOD_PRESETS.keys()) + ["Custom"])

    if mood != "Custom":
        profile = MOOD_PRESETS[mood].copy()

        with st.expander("Adjust mood (optional)"):
            profile["Danceability"] = st.slider("Danceability", 0.0, 1.0, profile["Danceability"])
            profile["Energy"] = st.slider("Energy", 0.0, 1.0, profile["Energy"])
            profile["Valence"] = st.slider("Valence", 0.0, 1.0, profile["Valence"])

    else:
        c1, c2, c3 = st.columns(3)

        with c1:
            d = st.slider("Danceability", 0.0, 1.0, 0.7)
            e = st.slider("Energy", 0.0, 1.0, 0.7)
            v = st.slider("Valence", 0.0, 1.0, 0.8)

        with c2:
            t = st.slider("Tempo", 40.0, 220.0, 120.0)
            l = st.slider("Loudness", -60.0, 0.0, -8.0)
            s = st.slider("Speechiness", 0.0, 1.0, 0.1)

        with c3:
            a = st.slider("Acousticness", 0.0, 1.0, 0.2)
            i = st.slider("Instrumentalness", 0.0, 1.0, 0.5)
            lv = st.slider("Liveness", 0.0, 1.0, 0.2)

        profile = {
            "Danceability": d, "Energy": e, "Valence": v,
            "Tempo": t, "Loudness": l, "Speechiness": s,
            "Acousticness": a, "Instrumentalness": i, "Liveness": lv,
        }

    if st.button("Generate recommendations"):
        recs = get_by_profile(profile, top_n)
        st.markdown('<div class="section-header">Recommended Tracks</div>', unsafe_allow_html=True)
        for _, row in recs.iterrows():
            render_song_card(row)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Music Recommender • Built with Streamlit & scikit-learn
    </div>
    """,
    unsafe_allow_html=True,
)
