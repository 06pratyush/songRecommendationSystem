"""
visualization.py
----------------
Functions for visualizing distributions, correlations, and clusters.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_basic_distributions(df, output_dir="outputs/plots"):
    os.makedirs(output_dir, exist_ok=True)
    features_to_plot = ["danceability", "energy", "valence", "tempo", "duration_ms"]
    
    df[features_to_plot].hist(bins=30, figsize=(12,8), layout=(2,3))
    plt.suptitle("Feature Distributions")
    
    save_path = os.path.join(output_dir, "distributions.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"📊 Feature distributions saved at {save_path}")


def plot_correlation_heatmap(df, output_dir="outputs/plots"):
    os.makedirs(output_dir, exist_ok=True)
    numeric_features = df.select_dtypes(include=["float64", "int64"])
    
    plt.figure(figsize=(12,8))
    sns.heatmap(numeric_features.corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Feature Correlation Heatmap")
    
    save_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"📈 Correlation heatmap saved at {save_path}")