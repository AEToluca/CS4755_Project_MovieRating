## This file describes the data we are working with

The ml-100k contains the original data from the movie-lens 100k dataset. Information about the original data can be found in the readme within that folder.



# The below files are a result of the preprocessing script

**ratings_clean.csv** - The original ratings in a consistently typed table: user ID, movie ID, rating, and Unix timestamp. Use this when you need ratings without demographic or movie features.

**users_clean.csv** - User ID, age, gender, occupation, and ZIP code. Use this as a separate user-feature table.

**movies_clean.csv** - Movie ID, title, release date, IMDb URL, and 19 genre indicator columns. Each genre column is 1 when that movie has the genre and 0 otherwise.

**ratings_enriched.csv** - Every rating joined with the matching user and movie details, plus rated_at_utc. This is convenient for analysis and models that use those features.

**summary.json** - Counts the data of all the cleaned csv files

**splits folder** - This folder contains the five official 80/20 train test splits provided by the original MovieLens dataset with movie and user details joined in. 

The encoded splits folder contains the csv files as a result of the preprocess function in preprocess.py. This one hot encodes categorical variables gender and occupation. For each split, the encoder is fitted on the training data and then applied to its matching test data to avoid data leakage. ID's, Zip codes, and timestamps have been left out of these splits because they seem useless in our task.


