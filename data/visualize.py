"""Load the cleaned MovieLens CSV files into pandas DataFrames."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path(__file__).resolve().parent / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Main cleaned tables
ratings_df = pd.read_csv(DATA_DIR / "ratings_clean.csv")
users_df = pd.read_csv(DATA_DIR / "users_clean.csv")
movies_df = pd.read_csv(DATA_DIR / "movies_clean.csv", parse_dates=["release_date"])
ratings_enriched_df = pd.read_csv(
    DATA_DIR / "ratings_enriched.csv",
    parse_dates=["release_date", "rated_at_utc"],
)

# Official train/test partitions, keyed by names such as "u1_base" and
# "ub_test". Example: split_dfs["ub_test"].head()
split_dfs = {
    path.stem: pd.read_csv(path, parse_dates=["release_date", "rated_at_utc"])
    for path in sorted(SPLITS_DIR.glob("*.csv"))
}

def visualize_rates():
    rate_bar = plt.bar()