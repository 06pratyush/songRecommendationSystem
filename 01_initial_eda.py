# 01_initial_eda.ipynb
# =====================
# Spotify Songs' Genre Segmentation
# Initial Exploratory Data Analysis (EDA)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data_path = "../data/spotify_dataset.csv"   # Adjust path if needed
df = pd.read_csv(data_path)

# -------------------------------
# 1. Basic overview
# -------------------------------
print("Dataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nFirst 5 rows:")
display(df.head())

# -------------------------------
# 2. Missing values
# -------------------------------
print("\nMissing values per column:")
print(df.isnull().sum())

# -------------------------------
# 3. Duplicates
# -------------------------------
print("\nNumber of duplicate rows:", df.duplicated().sum())

# -------------------------------
# 4. Descriptive statistics
# -------------------------------
print("\nStatistical summary of numeric features:")
display(df.describe())

# -------------------------------
# 5. Genre distribution
# -------------------------------
plt.figure(figsize=(10,5))
sns.countplot(data=df, x="playlist_genre", order=df["playlist_genre"].value_counts().index)
plt.xticks(rotation=45)
plt.title("Playlist Genre Distribution")
plt.show()

# -------------------------------
# 6. Correlation heatmap
# -------------------------------
numeric_features = df.select_dtypes(include=["float64", "int64"])
plt.figure(figsize=(12,8))
sns.heatmap(numeric_features.corr(), cmap="coolwarm", center=0, annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# -------------------------------
# 7. Distribution of key features
# -------------------------------
features_to_plot = ["danceability", "energy", "valence", "tempo", "duration_ms"]
df[features_to_plot].hist(bins=30, figsize=(12,8), layout=(2,3))
plt.suptitle("Feature Distributions")
plt.show()
