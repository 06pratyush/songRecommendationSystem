"""
train_and_save.py
-----------------
Train clustering model and save it into models/ folder.
Run this once if you want to generate models separately.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

# Load dataset
data_path = "../data/spotify_dataset.csv"
df = pd.read_csv(data_path)

# Features for clustering
features = ['danceability', 'energy', 'valence', 'tempo', 'loudness',
            'speechiness', 'acouticness', 'instrumentalness', 'liveness']

X = df[features].copy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeans
kmeans = KMeans(n_clusters=8, random_state=42)
kmeans.fit(X_scaled)

# Save models
os.makedirs(".", exist_ok=True)
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Models trained and saved in models/ folder.")
