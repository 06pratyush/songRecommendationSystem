# """
# main.py
# --------
# Entry point for Spotify Songs' Genre Segmentation project.
# This script runs the full pipeline: preprocessing, visualization, clustering,
# and building the recommendation model.
# """

import argparse
import os
import pandas as pd

# Clean imports thanks to src/__init__.py
from src import (
    load_and_preprocess_data,
    plot_basic_distributions,
    plot_correlation_heatmap,
    run_kmeans_clustering,
    plot_clusters_pca,
    build_recommender,
    recommend_songs
)

# ===============================
# Main Pipeline
# ===============================
def main(input_file: str, output_dir: str = "outputs"):
    print("🎵 Starting Spotify Genre Segmentation Pipeline...")

    # Step 1: Load and preprocess
    df = load_and_preprocess_data(input_file)
    print(f"✅ Data loaded and preprocessed. Shape: {df.shape}")

    # Step 2: EDA & Visualizations
    os.makedirs(output_dir, exist_ok=True)
    plot_basic_distributions(df, output_dir)
    plot_correlation_heatmap(df, output_dir)

    # Step 3: Clustering
    clustered_df, model, scaler = run_kmeans_clustering(df, n_clusters=8, output_dir=output_dir)
    print("✅ Clustering complete. Example cluster counts:")
    print(clustered_df['Cluster'].value_counts())

    # Step 4: Plot clusters
    plot_clusters_pca(clustered_df, output_dir)

    # Step 5: Recommendation system
    recommender = build_recommender(clustered_df, model, scaler)

    # Example: Recommend songs for the first track in the dataset
    example_track_id = clustered_df.iloc[0]['track_id']
    recommendations = recommend_songs(example_track_id, clustered_df, recommender, top_n=5)

    print("\n🎶 Example Recommendations:")
    print(recommendations[['track_name', 'track_artist', 'playlist_genre', 'Cluster']])

    print("\n✅ Pipeline completed successfully!")


# ===============================
# Run with Arguments
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spotify Songs Genre Segmentation Project")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV dataset (e.g., data/spotify_dataset.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Directory to save outputs (plots, models, etc.)"
    )

    args = parser.parse_args()
    main(args.input, args.output)
