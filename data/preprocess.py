"""Clean the MovieLens 100K files and prepare model-ready tables.

Run from any directory with:
    python preprocess.py

Outputs are written to ``processed/`` beside this file. Raw data is never
modified. The official MovieLens train/test partitions are retained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ml-100k"
DEFAULT_OUTPUT_DIR = ROOT / "processed"

GENRES = [
    "unknown", "action", "adventure", "animation", "children", "comedy",
    "crime", "documentary", "drama", "fantasy", "film_noir", "horror",
    "musical", "mystery", "romance", "sci_fi", "thriller", "war",
    "western",
]
RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
CATEGORICAL_COLUMNS = ["gender", "occupation"]
NUMERIC_COLUMNS = ["age", *GENRES]
FEATURE_COLUMNS = [*CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS]


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three core MovieLens tables using their documented formats."""
    ratings = pd.read_csv(
        data_dir / "u.data", sep="\t", names=RATING_COLUMNS,
        dtype={"user_id": "int32", "movie_id": "int32", "rating": "int8", "timestamp": "int64"},
    )

    users = pd.read_csv(
        data_dir / "u.user", sep="|", header=None,
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        dtype={"user_id": "int32", "age": "int16", "gender": "string", "occupation": "string", "zip_code": "string"},
    )

    movies = pd.read_csv(
        data_dir / "u.item", sep="|", header=None, encoding="latin-1",
        names=["movie_id", "title", "release_date", "video_release_date", "imdb_url", *GENRES],
        dtype={"movie_id": "int32", "title": "string", "release_date": "string", "video_release_date": "string", "imdb_url": "string", **{genre: "int8" for genre in GENRES}},
    )
    movies["release_date"] = pd.to_datetime(movies["release_date"], format="%d-%b-%Y", errors="coerce")

    return ratings, users, movies


def validate_data(ratings: pd.DataFrame, users: pd.DataFrame, movies: pd.DataFrame) -> None:
    """Stop with a clear error when source data violates expected constraints."""
    if ratings[RATING_COLUMNS].isna().any().any():
        raise ValueError("Ratings contain missing values.")
    if not ratings["rating"].between(1, 5).all():
        raise ValueError("Ratings must be integers from 1 through 5.")
    if users["user_id"].duplicated().any() or movies["movie_id"].duplicated().any():
        raise ValueError("User and movie IDs must be unique in their metadata tables.")
    if ratings.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("Found duplicate user/movie ratings.")
    if not ratings["user_id"].isin(users["user_id"]).all():
        raise ValueError("Some ratings refer to user IDs missing from u.user.")
    if not ratings["movie_id"].isin(movies["movie_id"]).all():
        raise ValueError("Some ratings refer to movie IDs missing from u.item.")
    if not movies[GENRES].isin([0, 1]).all().all():
        raise ValueError("Movie genre indicators must be 0 or 1.")


def read_split(path: Path) -> pd.DataFrame:
    """Read a supplied base/test file (same format as u.data)."""
    return pd.read_csv(
        path, sep="\t", names=RATING_COLUMNS,
        dtype={"user_id": "int32", "movie_id": "int32", "rating": "int8", "timestamp": "int64"},
    )


def enrich(ratings: pd.DataFrame, users: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Join rating records to user and movie features and format timestamps."""
    enriched = ratings.merge(users, on="user_id", how="left", validate="many_to_one")
    enriched = enriched.merge(movies, on="movie_id", how="left", validate="many_to_one")
    enriched["rated_at_utc"] = pd.to_datetime(enriched["timestamp"], unit="s", utc=True)
    return enriched


def save_encoded_split(
    split: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
) -> None:
    """One-hot encode categories, fitting on train and reusing on test."""
    encoder = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", "passthrough", NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )

    train_matrix = encoder.fit_transform(train[FEATURE_COLUMNS])
    test_matrix = encoder.transform(test[FEATURE_COLUMNS])
    feature_names = encoder.get_feature_names_out()

    for partition, matrix, source in (
        ("base", train_matrix, train),
        ("test", test_matrix, test),
    ):
        if sparse.issparse(matrix):
            features = pd.DataFrame.sparse.from_spmatrix(
                matrix, columns=feature_names, index=source.index
            )
        else:
            features = pd.DataFrame(matrix, columns=feature_names, index=source.index)
        # Keep the target in the same file for convenience. IDs and other
        # metadata are deliberately excluded from model features.
        features["rating"] = source["rating"].to_numpy()
        features.to_csv(output_dir / f"{split}_{partition}_encoded.csv", index=False)

    (output_dir / f"{split}_feature_names.json").write_text(
        json.dumps(feature_names.tolist(), indent=2), encoding="utf-8"
    )


def preprocess(data_dir: Path = DATA_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, int]:
    """Validate and write cleaned core tables and official enriched splits."""
    ratings, users, movies = load_data(data_dir)
    validate_data(ratings, users, movies)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    encoded_dir = output_dir / "encoded_splits"
    encoded_dir.mkdir(parents=True, exist_ok=True)

    # Keep Unix timestamps in the canonical table; add a readable UTC time to
    # the joined table for analysis and plotting.
    ratings.to_csv(output_dir / "ratings_clean.csv", index=False)
    users.to_csv(output_dir / "users_clean.csv", index=False)
    movies.to_csv(output_dir / "movies_clean.csv", index=False)
    enrich(ratings, users, movies).to_csv(output_dir / "ratings_enriched.csv", index=False)

    split_counts: dict[str, int] = {}
    enriched_splits: dict[str, pd.DataFrame] = {}
    for split in ("u1", "u2", "u3", "u4", "u5", "ua", "ub"):
        for partition in ("base", "test"):
            source = data_dir / f"{split}.{partition}"
            if not source.exists():
                continue
            part = read_split(source)
            validate_data(part, users, movies)
            enriched_part = enrich(part, users, movies)
            enriched_part.to_csv(split_dir / f"{split}_{partition}.csv", index=False)
            enriched_splits[f"{split}_{partition}"] = enriched_part
            split_counts[f"{split}_{partition}"] = len(part)

    for split in ("u1", "u2", "u3", "u4", "u5", "ua", "ub"):
        train = enriched_splits.get(f"{split}_base")
        test = enriched_splits.get(f"{split}_test")
        if train is not None and test is not None:
            save_encoded_split(split, train, test, encoded_dir)

    summary = {
        "ratings": len(ratings),
        "users": len(users),
        "movies": len(movies),
        "ratings_with_missing_release_date": int(movies["release_date"].isna().sum()),
        **split_counts,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Folder containing the MovieLens files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder for processed CSV files.")
    args = parser.parse_args()
    summary = preprocess(args.data_dir, args.output_dir)
    print(f"Preprocessed MovieLens data into: {args.output_dir.resolve()}")
    for name, count in summary.items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
