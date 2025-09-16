"""
src package for Spotify Genre Segmentation Project

This package contains:
- preprocessing utilities
- data visualization functions
- clustering algorithms
- recommendation system functions
- helper utilities
"""

# Re-export preprocessing functions
from .preprocessing import load_and_preprocess_data

# Re-export visualization functions

from .visualization import (
    plot_basic_distributions,
    plot_correlation_heatmap,
)

# Re-export clustering functions
from .clustering import (
    run_kmeans_clustering,
    plot_clusters_pca
    )

# Re-export recommender functions
from .recommender import (
    build_recommender,
    recommend_songs
)

# Re-export utils
from . import utils


__all__ = [
    "load_and_preprocess_data",
    "plot_basic_distributions",
    "plot_correlation_heatmap",
    "run_kmeans_clustering",
    "plot_clusters_pca",
    "build_recommender",
    "recommend_songs",
    "utils"
]