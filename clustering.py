"""
clustering.py
-------------
Functions for running clustering algorithms and visualizing them.
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
import plotly.express as px


def run_kmeans_clustering(df, n_clusters=8, output_dir="outputs"):
    features = ['danceability', 'energy', 'valence', 'tempo', 'loudness',
                'speechiness', 'acousticness', 'instrumentalness', 'liveness']
    
    X = df[features]
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled)
    
    # Save models
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(kmeans, os.path.join(models_dir, "kmeans_model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    
    return df, kmeans, scaler


def plot_clusters_pca(clustered_df, output_dir="outputs/clusters"):
    os.makedirs(output_dir, exist_ok=True)
    
    features = ['danceability', 'energy', 'valence', 'tempo', 'loudness',
                'speechiness', 'acousticness', 'instrumentalness', 'liveness']
    
    pca = PCA(n_components=2)
    components = pca.fit_transform(clustered_df[features])
    clustered_df["PCA1"] = components[:,0]
    clustered_df["PCA2"] = components[:,1]
    
    fig = px.scatter(clustered_df, x="PCA1", y="PCA2",
                     color="Cluster",
                     hover_data=["track_name", "track_artist", "playlist_genre"])
    
    save_path = os.path.join(output_dir, "cluster_visualization.html")
    fig.write_html(save_path)
    print(f"🌀 Cluster visualization saved at {save_path}")