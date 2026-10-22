from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent / 'ml-100k'

ratings = pd.read_csv(
    DATA_DIR / 'u.data',
    sep='\t',
    names=['user_id', 'movie_id', 'rating', 'timestamp'],
)
movies = pd.read_csv(
    DATA_DIR / 'u.item',
    sep='|',
    encoding='latin-1',
    header=None,
)
users = pd.read_csv(
    DATA_DIR / 'u.user',
    sep='|',
    names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
)
genre = pd.read_csv(
    DATA_DIR / 'u.genre',
    sep='|',
    names=['genre', 'count']
)

def main():
    print(movies.head(5))
    print(ratings.head(5))

if __name__ == "__main__":
    main()