"""
preprocessing.py
----------------
Functions for loading, cleaning, and preprocessing the Spotify dataset.
"""

import pandas as pd


def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset, remove duplicates, handle missing values, and clean features.
    """
    print("📥 Loading dataset...")
    df = pd.read_csv(filepath)

    # Fix typo in column name if exists
    if 'acouticness' in df.columns:
        df = df.rename(columns={'acouticness': 'acousticness'})

    # Drop duplicates
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Removed {before - after} duplicate rows.")

    # Handle missing values (drop rows with NA in critical features)
    df = df.dropna(subset=['track_id', 'track_name', 'track_artist'])

    # Fill numeric NaNs with mean
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Convert release date to datetime
    if 'track_album_release_date' in df.columns:
        df['track_album_release_date'] = pd.to_datetime(
            df['track_album_release_date'], errors='coerce'
        )

    print("✅ Preprocessing complete.")
    return df