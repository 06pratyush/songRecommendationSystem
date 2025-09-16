"""
streamlit_app.py
----------------
Interactive Streamlit dashboard for Spotify Songs' Genre Segmentation project.
"""

import streamlit as st
import pandas as pd
import os

# from src import (
#     load_and_preprocess_data,
#     plot_basic_distributions,
#     plot_correlation_heatmap,
#     run_kmeans_clustering,
#     plot_clusters_pca,
#     build_recommender,
#     recommend_songs
# )
from ..src import (
    load_and_preprocess_data,
    plot_basic_distributions,
    plot_correlation_heatmap,
    run_kmeans_clustering,
    plot_clusters_pca,
    build_recommender,
    recommend_songs
)
st.title("Spotify Genre Segmentation")
st.write("Upload a CSV file to analyze.")
# Add file uploader or button to trigger functions
if st.button("Run Pipeline"):
    st.write("Processing...")
    # Example usage (add your logic)
# ===============================
# Streamlit App
# ===============================
def main():
    st.set_page_config(page_title="Spotify Genre Segmentation", layout="wide")
    st.title("🎵 Spotify Songs’ Genre Segmentation & Recommendation System")

    # Sidebar for inputs
    st.sidebar.header("Upload & Settings")
    uploaded_file = st.sidebar.file_uploader("Upload your Spotify dataset (CSV)", type=["csv"])
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    if uploaded_file is not None:
        # Load and preprocess
        df = load_and_preprocess_data(uploaded_file)
        st.success(f"✅ Data loaded successfully! Shape: {df.shape}")

        # Show preview
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(10))

        # EDA Section
        st.subheader("📊 Exploratory Data Analysis")
        if st.checkbox("Show Feature Distributions"):
            plot_basic_distributions(df, output_dir)
            st.image(os.path.join(output_dir, "feature_distributions.png"))

        if st.checkbox("Show Correlation Heatmap"):
            plot_correlation_heatmap(df, output_dir)
            st.image(os.path.join(output_dir, "correlation_heatmap.png"))

        # Clustering Section
        st.subheader("🎯 Clustering Songs")
        n_clusters = st.slider("Number of Clusters (K)", min_value=3, max_value=15, value=8, step=1)
        clustered_df, model, scaler = run_kmeans_clustering(df, n_clusters=n_clusters, output_dir=output_dir)
        st.write("✅ Clustering complete!")
        st.write(clustered_df["Cluster"].value_counts())

        # Cluster visualization
        st.subheader("🌀 Cluster Visualization (PCA)")
        plot_clusters_pca(clustered_df, output_dir)
        st.image(os.path.join(output_dir, "clusters_pca.png"))

        # Recommendation System
        st.subheader("🎶 Song Recommendation Engine")
        recommender = build_recommender(clustered_df, model, scaler)

        track_options = clustered_df[["track_id", "track_name", "track_artist"]].drop_duplicates()
        track_dict = {
            f"{row['track_name']} - {row['track_artist']}": row["track_id"]
            for _, row in track_options.iterrows()
        }

        selected_track = st.selectbox("Select a song to get recommendations:", list(track_dict.keys()))

        if selected_track:
            track_id = track_dict[selected_track]
            top_n = st.slider("Number of Recommendations", min_value=3, max_value=10, value=5)
            recs = recommend_songs(track_id, clustered_df, recommender, top_n=top_n)

            st.write("### Recommended Songs")
            st.dataframe(recs[["track_name", "track_artist", "playlist_genre", "Cluster"]])

    else:
        st.info("👈 Please upload a CSV file from the sidebar to start.")


if __name__ == "__main__":
    main()
