"""
recommender.py
--------------
Recommendation system logic based on clustering similarity.
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_recommender(df: pd.DataFrame, model, scaler):
    """
    Build recommender using cluster assignments and scaled features.
    """
    features = ['danceability', 'energy', 'valence', 'tempo', 'loudness',
                'speechiness', 'acousticness', 'instrumentalness', 'liveness']

    df_features = scaler.transform(df[features])
    df['features_scaled'] = list(df_features)
    return df


def recommend_songs(track_id: str, df: pd.DataFrame, recommender, top_n: int = 5) -> pd.DataFrame:
    """
    Recommend songs similar to the given track_id using cosine similarity.
    """
    if track_id not in df['track_id'].values:
        raise ValueError("Track ID not found in dataset.")

    # Convert the list of scaled features directly to a DataFrame
    features_matrix = pd.DataFrame(df['features_scaled'].tolist(), index=df.index)

    target_vector = features_matrix[df['track_id'] == track_id].values
    sim_scores = cosine_similarity(target_vector, features_matrix)[0]

    df['similarity'] = sim_scores
    recommendations = df.sort_values(by='similarity', ascending=False).head(top_n + 1)

    return recommendations[recommendations['track_id'] != track_id]