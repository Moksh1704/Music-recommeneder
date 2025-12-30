# Music Recommender System
   **Content-Based Recommendation using Spotify Audio Features**

   
## Overview
  This project is a content-based music recommender system that suggests songs based on intrinsic audio characteristics rather than user history. By analyzing features like danceability, energy, and valence, the system identifies tracks that are sonically similar or match a specific emotional mood.

  The system was developed as part of an AI & Machine Learning internship and is implemented as an interactive web application.


## Key Features
-Song-Based Recommendation: Find tracks similar to a selected song using mathematical similarity.

-Mood-Based Playlist Generation: Generate playlists for predefined moods (Happy, Chill, Sad, Workout, Party, Focus) or use custom feature sliders.

-Audio Feature Similarity: Uses normalized Spotify audio features for transparent and explainable recommendations.

-Spotify Player Integration: Embedded Spotify previews for instant listening within the app.

-Clean & Modern UI: Custom CSS-styled Streamlit interface with song cards and similarity scores.


## Recommendation Approach

-This system follows a Content-Based Filtering strategy, which works effectively even for new users without listening history (solving the "cold-start" problem).

-Feature Vectorization: Each song is represented as a numerical vector of its audio features.

-Normalization: Features are scaled using MinMaxScaler so that no single feature (like Tempo) outweighs others (like Danceability).

-Cosine Similarity: The system calculates the "distance" between songs.

-Song ↔ Song: Finds the nearest neighbors to a specific track.

-Mood ↔ Library: Matches a "mood profile" vector against the dataset.

## Tech Stack

-**Programming Language**: Python

-**Libraries**: * Pandas & NumPy (Data Manipulation)

-**Scikit-learn** (ML Algorithms & Preprocessing)

-**Streamlit** (Web Interface)


### ML Techniques:

-MinMax Scaling

-Cosine Similarity

-Random Forest (Used in the training phase for popularity prediction)


## Project Structure

Music-Recommender/

│

├── app.py                  # Main Streamlit application

├── Model_Training.ipynb    # ML training and EDA notebook

├── Spotify_data.csv        # Dataset

├── rf_model.pkl            # Trained Random Forest model

├── scaler.pkl              # Saved MinMaxScaler


##  Installation & Run

1️. Clone the repository

```Bash

git clone https://github.com/Moksh1704/Music-recommeneder.git
cd Music-recommeneder
```

2. Install dependencies
 
```Bash

pip install streamlit pandas numpy scikit-learn
```
3. Run the application

```Bash

streamlit run app.py
```

## Dataset Description

The system uses a public Spotify Audio Features dataset.

-Metadata: Track name, Artists, Popularity, Track ID.

-Audio Features: 9 key metrics (Acousticness, Danceability, Energy, Instrumentalness, Liveness, Loudness, Speechiness, Tempo, Valence).

-Preprocessing: Includes missing value removal, feature selection, and normalization.


## Future Enhancements
-Spotify Web API: Transition from a static CSV to real-time API integration.

-Hybrid Engine: Combine content-based filtering with collaborative filtering.

-NLP: Add genre tagging and lyric-level sentiment analysis.

-Scalability: Use ANN (Approximate Nearest Neighbors) for larger datasets.
