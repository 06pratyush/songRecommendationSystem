#!/usr/bin/env python3
"""
spotify_genre_segmentation_pipeline.py

Purpose:
  - Load the provided Spotify CSV dataset
  - Perform initial exploratory checks (shape, columns, head)
  - Check duplicates and missing values (summary + drop duplicates optionally)
  - Clean & preprocess:
      - fix column name typo 'acouticness' -> 'acousticness'
      - parse release date and extract release_year
      - fill missing audio features (median)
      - optional outlier capping (winsorize-like) for chosen columns
  - Scale numeric audio features using StandardScaler
  - (Optional) PCA dimensionality reduction
  - Save: processed CSV, scaler and PCA (joblib), and EDA plots in outputs/

Usage:
    python spotify_genre_segmentation_pipeline.py --input data/spotify_dataset.csv

Outputs:
    outputs/
      - processed_spotify_data.csv
      - scaler.joblib
      - pca.joblib (if enabled)
      - correlation_heatmap.png
      - feature_distributions.png
      - missing_values_bar.png
      - preprocessing_log.txt
"""

import os
import argparse
import logging
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# ---------------------------
# Configuration / Constants
# ---------------------------
PROJECT_NAME = "Spotify_Genre_Segmentation"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "duration_ms"
]
# If CSV has typo 'acouticness', we'll fix it automatically.
EXPECTED_COLUMNS = [
    "track_id", "track_name", "track_artist", "track_popularity", "track_album_id",
    "track_album_name", "track_album_release_date", "playlist_name", "playlist_id",
    "playlist_genre", "playlist_subgenre", "danceability", "energy", "key", "loudness",
    "mode", "speechiness", "acouticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_ms"
]

# ---------------------------
# Utility functions
# ---------------------------
def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "preprocessing_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="w")
        ]
    )
    logging.info(f"Logging initialized. Output directory: {output_dir}")


def safe_read_csv(path: str) -> pd.DataFrame:
    logging.info(f"Reading CSV from: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logging.exception("Failed to read CSV file.")
        raise
    logging.info(f"Loaded dataframe with shape: {df.shape}")
    return df


def summarize_dataframe(df: pd.DataFrame) -> Dict[str, any]:
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "num_duplicates": int(df.duplicated().sum()),
        "missing_per_column": df.isna().sum().to_dict(),
        "memory_usage_bytes": df.memory_usage(deep=True).sum()
    }
    logging.info(f"Data summary: rows={info['shape'][0]} cols={info['shape'][1]}; duplicates={info['num_duplicates']}")
    return info


def fix_column_typos(df: pd.DataFrame) -> pd.DataFrame:
    # Fix known typo 'acouticness' -> 'acousticness'
    if "acouticness" in df.columns:
        logging.info("Found 'acouticness' column — renaming to 'acousticness'")
        df = df.rename(columns={"acouticness": "acousticness"})
    return df


def extract_release_year(df: pd.DataFrame, date_col: str = "track_album_release_date", new_col: str = "release_year") -> pd.DataFrame:
    if date_col not in df.columns:
        logging.warning(f"Date column '{date_col}' not found — skipping release year extraction.")
        return df
    logging.info(f"Parsing release dates from column '{date_col}' and extracting year into '{new_col}'")
    # parse with pandas - tolerate different formats
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
    df[new_col] = df[date_col].dt.year
    num_parsed = df[new_col].notna().sum()
    logging.info(f"Parsed {num_parsed} release years (non-null).")
    return df


def drop_exact_duplicates(df: pd.DataFrame, subset: List[str] = None) -> Tuple[pd.DataFrame, int]:
    before = df.shape[0]
    if subset is None:
        subset = ["track_id"]
    df = df.drop_duplicates(subset=subset)
    after = df.shape[0]
    dropped = before - after
    logging.info(f"Dropped {dropped} duplicate rows based on subset={subset}. New shape: {df.shape}")
    return df, dropped


def numeric_impute_median(df: pd.DataFrame, columns: List[str]) -> Dict[str, int]:
    """
    Fill missing values in numeric columns using median. Returns dict {col: num_filled}.
    """
    fills = {}
    for col in columns:
        if col not in df.columns:
            logging.warning(f"Column {col} not in dataframe; skipping imputation.")
            continue
        na_count = int(df[col].isna().sum())
        if na_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            fills[col] = na_count
            logging.info(f"Imputed {na_count} missing values in '{col}' with median={median_val:.6g}")
        else:
            fills[col] = 0
    return fills


def cap_outliers(df: pd.DataFrame, columns: List[str], lower_q: float = 0.01, upper_q: float = 0.99) -> Dict[str, Tuple[float, float]]:
    """
    Caps values outside the given quantiles to the quantile values (simple winsorization).
    Returns dict of column -> (lower_bound, upper_bound) used.
    """
    bounds = {}
    for col in columns:
        if col not in df.columns:
            continue
        lower = df[col].quantile(lower_q)
        upper = df[col].quantile(upper_q)
        df[col] = np.clip(df[col], lower, upper)
        bounds[col] = (float(lower), float(upper))
        logging.info(f"Capped '{col}' to [{lower:.6g}, {upper:.6g}] (quantiles {lower_q}..{upper_q})")
    return bounds


def save_processed_data(df: pd.DataFrame, out_dir: str, filename: str = "processed_spotify_data.csv"):
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    logging.info(f"Saved processed dataframe to: {path}")


def plot_correlation_heatmap(df: pd.DataFrame, columns: List[str], out_dir: str, filename: str = "correlation_heatmap.png"):
    corr = df[columns].corr()
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Correlation matrix of audio features + popularity")
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"Saved correlation heatmap to {path}")


def plot_feature_distributions(df: pd.DataFrame, columns: List[str], out_dir: str, filename: str = "feature_distributions.png"):
    n = len(columns)
    cols = 3
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols * 5, rows * 3.5))
    for i, col in enumerate(columns, start=1):
        plt.subplot(rows, cols, i)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(col)
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"Saved feature distributions to {path}")


def plot_missing_value_bar(df: pd.DataFrame, out_dir: str, filename: str = "missing_values_bar.png"):
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=miss.values, y=miss.index)
    plt.xlabel("Number of missing values")
    plt.title("Columns with missing values")
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"Saved missing values bar chart to {path}")


def scale_and_pca(df: pd.DataFrame, feature_cols: List[str], out_dir: str, do_pca: bool = True, pca_variance: float = 0.95):
    """
    Scales features with StandardScaler and optionally fits PCA.
    Returns scaled array, scaler object, pca object (or None).
    Saves scaler and pca to out_dir.
    """
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaler_path = os.path.join(out_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    logging.info(f"Saved StandardScaler to {scaler_path}")

    pca = None
    X_transformed = X_scaled
    if do_pca:
        pca = PCA(n_components=pca_variance, svd_solver="full", random_state=42)
        X_transformed = pca.fit_transform(X_scaled)
        pca_path = os.path.join(out_dir, "pca.joblib")
        joblib.dump(pca, pca_path)
        logging.info(f"Saved PCA (var={pca_variance}) to {pca_path}; components={pca.n_components_}")
    return X_transformed, scaler, pca


# ---------------------------
# Main pipeline
# ---------------------------
def run_pipeline(input_csv: str,
                 output_dir: str = DEFAULT_OUTPUT_DIR,
                 audio_features: List[str] = DEFAULT_AUDIO_FEATURES,
                 drop_duplicates_subset: List[str] = None,
                 do_outlier_cap: bool = True,
                 pca_variance: float = 0.95,
                 do_pca: bool = True):
    setup_logging(output_dir)

    # 1) Load
    df = safe_read_csv(input_csv)

    # 2) Basic summary & head
    summary = summarize_dataframe(df)
    logging.info("Columns in dataset:\n" + ", ".join(summary['columns']))
    logging.info("Data types (first 10):\n" + "\n".join([f"{k}: {v}" for k, v in list(summary['dtypes'].items())[:20]]))
    logging.info("First 5 rows:\n" + df.head(5).to_string())

    # 3) Fix known column typos
    df = fix_column_typos(df)

    # ensure audio features exist, attempt to adapt if CSV had 'acouticness' only (handled above)
    available_audio_feats = [f for f in audio_features if f in df.columns]
    missing_audio_feats = [f for f in audio_features if f not in df.columns]
    if missing_audio_feats:
        logging.warning(f"The following expected audio features are missing from the dataset: {missing_audio_feats}")
    logging.info(f"Using audio feature columns: {available_audio_feats}")

    # 4) Extract release year
    df = extract_release_year(df, date_col="track_album_release_date", new_col="release_year")

    # 5) Duplicates
    df, dropped = drop_exact_duplicates(df, subset=drop_duplicates_subset or ["track_id"])
    logging.info(f"Duplicate removal complete. Rows dropped: {dropped}")

    # 6) Missing values summary + plot
    miss_counts = df.isna().sum().sort_values(ascending=False)
    logging.info("Missing values per column (top 20):\n" + miss_counts.head(20).to_string())
    plot_missing_value_bar(df, output_dir)

    # 7) Impute numerics (median) for audio features found
    fills = numeric_impute_median(df, available_audio_feats)

    # 8) If track_popularity exists, impute small missing with median too (useful)
    if "track_popularity" in df.columns:
        pop_na = int(df["track_popularity"].isna().sum())
        if pop_na > 0:
            pop_med = df["track_popularity"].median()
            df["track_popularity"] = df["track_popularity"].fillna(pop_med)
            logging.info(f"Imputed {pop_na} missing 'track_popularity' with median {pop_med}")

    # 9) Optional outlier capping for certain wide-range numeric cols
    if do_outlier_cap:
        cap_cols = [c for c in ["loudness", "tempo", "duration_ms"] if c in df.columns]
        if cap_cols:
            cap_outliers(df, cap_cols, lower_q=0.01, upper_q=0.99)

    # 10) Quick EDA plots (distributions + correlation)
    # Use the selected audio features plus popularity if present
    corr_cols = available_audio_feats.copy()
    if "track_popularity" in df.columns:
        corr_cols.append("track_popularity")

    try:
        plot_feature_distributions(df, available_audio_feats, output_dir)
        plot_correlation_heatmap(df, corr_cols, output_dir)
    except Exception as e:
        logging.exception("Failed during plotting EDA visuals. Continuing pipeline.")

    # 11) Scale and optional PCA
    # ensure no remaining NA in audio features (after imputation)
    for f in available_audio_feats:
        if df[f].isna().any():
            logging.warning(f"After imputation, column {f} still contains NA — filling with 0 as fallback.")
            df[f] = df[f].fillna(0)

    X_transformed, scaler, pca = scale_and_pca(df, available_audio_feats, output_dir, do_pca=do_pca, pca_variance=pca_variance)

    # If PCA used, attach PCA columns for inspection
    if pca is not None:
        n_comp = pca.n_components_
        logging.info(f"Attaching PCA columns (n={n_comp}) to dataframe for inspection.")
        for i in range(n_comp):
            df[f"pca_{i+1}"] = X_transformed[:, i]

    # 12) Save processed dataset and objects
    save_processed_data(df, output_dir, filename="processed_spotify_data.csv")
    # scaler and pca already saved in scale_and_pca
    logging.info("Pipeline finished successfully.")


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Spotify Genre Segmentation — preprocessing pipeline")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_dir", "-o", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for processed files/plots")
    parser.add_argument("--no_pca", action="store_true", help="If passed, PCA step is skipped")
    parser.add_argument("--no_outlier_cap", action="store_true", help="If passed, outlier capping is skipped")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        input_csv=args.input,
        output_dir=args.output_dir,
        do_outlier_cap=(not args.no_outlier_cap),
        do_pca=(not args.no_pca),
        pca_variance=0.95
    )
