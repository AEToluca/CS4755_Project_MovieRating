

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent 
SPLITS_DIR = ROOT / "data" / "processed" / "splits"

#19 genre columns from dataset
GENRES = [
    "unknown", "action", "adventure", "animation", "children", "comedy",
    "crime", "documentary", "drama", "fantasy", "film_noir", "horror",
    "musical", "mystery", "romance", "sci_fi", "thriller", "war", "western",
]

#5 train/test folds we evaulate on
FOLDS = ["u1", "u2", "u3", "u4", "u5"]

#ffeature groups 
FEATURE_GROUPS = ("demographics", "history", "genre_taste")

#load one folds training/test tables
def load_fold(fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:

    train = pd.read_csv(SPLITS_DIR / f"{fold}_base.csv")
    test = pd.read_csv(SPLITS_DIR / f"{fold}_test.csv")
    return train, test


#basic features age,genre flags, one-hot gender/occupation
#test columns are forced to match training columns so both line up 
def demographic_features(df: pd.DataFrame, categories: pd.Index) -> pd.DataFrame:

    onehot = pd.get_dummies(df[["gender", "occupation"]], dtype=int)
    onehot = onehot.reindex(columns=categories, fill_value=0)
    return pd.concat([df[["age", *GENRES]], onehot], axis=1)


#rating history feautres for each user and each movie
#avg rating, num of ratings, and spread(std) of ratings
#computed from training data only
def history_features(train: pd.DataFrame, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
  
    global_mean = train["rating"].mean()
    out = pd.DataFrame(index=df.index)
    stats = train.assign(sq=train["rating"] ** 2)

    for key, prefix in (("user_id", "user"), ("movie_id", "movie")):
        agg = stats.groupby(key).agg(total=("rating", "sum"), sq=("sq", "sum"), n=("rating", "count"))
        total = df[key].map(agg["total"]).fillna(0)
        sq = df[key].map(agg["sq"]).fillna(0)
        n = df[key].map(agg["n"]).fillna(0)
        if is_train:  
            total, sq, n = total - df["rating"], sq - df["rating"] ** 2, n - 1

        safe_n = n.clip(lower=1)
        mean = np.where(n > 0, total / safe_n, global_mean)
        var = np.where(n > 1, sq / safe_n - (total / safe_n) ** 2, 0.0)
        out[f"{prefix}_mean"] = mean
        out[f"{prefix}_count"] = n
        out[f"{prefix}_std"] = np.sqrt(np.clip(var, 0, None))
    return out


#genre taste: users average rating on movies sharing this movies genre 
def genre_taste_features(train: pd.DataFrame, df: pd.DataFrame, is_train: bool,
                         user_mean: np.ndarray) -> pd.DataFrame:
  
    # Step 1: for every user and genre, add up stars given and ratings made.
    stars_by_genre = train[GENRES].mul(train["rating"], axis=0)
    user_genre_stars = stars_by_genre.groupby(train["user_id"]).sum()
    user_genre_ratings = train[GENRES].groupby(train["user_id"]).sum()

    # Step 2: for each row, keep only this movie's genres and add them up.
    movie_genres = df[GENRES].to_numpy()
    stars = user_genre_stars.reindex(df["user_id"]).fillna(0).to_numpy()
    ratings = user_genre_ratings.reindex(df["user_id"]).fillna(0).to_numpy()
    total = (stars * movie_genres).sum(axis=1)
    n = (ratings * movie_genres).sum(axis=1)

    # Step 3: leave this row's own rating out (it was counted once per genre).
    if is_train:
        n_genres = movie_genres.sum(axis=1)
        total = total - df["rating"].to_numpy() * n_genres
        n = n - n_genres

    taste = np.where(n > 0, total / np.clip(n, 1, None), user_mean)
    return pd.DataFrame({
        "user_genre_mean": taste,
        "user_genre_count": n,
        "user_genre_vs_overall": taste - user_mean,
    }, index=df.index)


#builds final feature tables for one fold: X_train, y_train, X_test, y_test
#`groups` chooses whcih feature groups to include 
def build_features(train: pd.DataFrame, test: pd.DataFrame,
                   groups: tuple[str, ...] = FEATURE_GROUPS):
 
    categories = pd.get_dummies(train[["gender", "occupation"]], dtype=int).columns

    def make(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        parts = []
        hist = history_features(train, df, is_train)
        if "demographics" in groups:
            parts.append(demographic_features(df, categories))
        if "history" in groups:
            parts.append(hist)
        if "genre_taste" in groups:
            parts.append(genre_taste_features(train, df, is_train, hist["user_mean"].to_numpy()))
        return pd.concat(parts, axis=1).reset_index(drop=True)

    return make(train, True), train["rating"].to_numpy(), make(test, False), test["rating"].to_numpy()

#quick self test to build fold u1 and check for missing values

if __name__ == "__main__":
    tr, te = load_fold("u1")
    X, y, Xt, yt = build_features(tr, te)
    print(f"u1: {X.shape[0]} train rows, {Xt.shape[0]} test rows, {X.shape[1]} features")
    print("Missing values:", int(X.isna().sum().sum() + Xt.isna().sum().sum()))