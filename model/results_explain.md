# Model Results: Predicting MovieLens Ratings

This will explains how the model works, how it was tested, and what the results mean. All numbers come from running python model/train.py --ablation

---

## 1. The task

Given a user and a movie, predict the star rating (1–5) that user gave the movie.

Each test example is a real rating with the answer hidden. The model predicts a rating, and we compare it to the real one.

---

## 2. Data and testing setup

- Dataset: MovieLens 100K: 100,000 ratings from 943 users on 1,682 movies.
- Input files: the enriched splits from data/preprocess.py (data/processed/splits/). preprocessing was not changed.
- Train/test splits: the five official MovieLens folds (u1–u5). Each has 80,000 training and 20,000 test ratings. Every result below is the average over all five folds.
- How the folds work: each user's ratings are divided between training and test. So the question the model answers is "given this person's other ratings, what will they give a movie they haven't rated yet?"

---

## 3. Features (52 total)

Features are built in model/features.py, separately for each fold.

| Group | Features | What it captures |
|---|---|---|
| Demographics + genre | age, gender, occupation (one-hot), 19 genre flags | Who the user is and what kind of movie it is |
| Rating history | user's average rating, number of ratings and spread (std); the same three for the movie | How generous or harsh the user is; how well-liked the movie is |
| Genre taste | user's average rating on movies in this movie's genres; how many ratings that's based on; how it differs from their usual average | Whether this user likes this kind of movie more or less than usual |

Dropped: user/movie IDs, zip code and timestamps (not useful as direct inputs). also tested release year and movie age when rated, and removed them because they made no difference.

### Avoiding data leakage

Leakage means the model accidentally sees the answer, which makes results look better than they really are. Prevented  it in three ways:

1. Training data only. All averages, counts and spreads are calculated from each fold's training ratings, never the test ratings.
2. Leave-one-out. For training rows, the row's own rating is removed from its averages. Otherwise the model could learn to read the answer from the feature.
3. Safe fallback. Users or movies with no training history get the overall average rating.

---

## 4. The model

A random forest classifier (scikit-learn's RandomForestClassifier). It treats the rating as one of five categories.

- It builds many decision trees. Each asks yes/no questions about the features (e.g. "Is the movie's average above 3.6?") until it reaches a rating.
- Each tree is trained on a random sample of the data and considers a random subset of features at each question, so the trees make different mistakes.
- The forest predicts the rating that gets the most votes across all trees.

### Tuning (grid search)

Tried 8 combinations of settings. To keep the test sets unbiased, tuning used only fold u1's training data: 80% to train and 20% to validate. The best combination was chosen by macro-F1.

Best settings:

| Setting | Value | Meaning |
|---|---|---|
| n_estimators | 200 | Number of trees |
| max_depth | None | No limit on tree depth |
| min_samples_leaf | 5 | Each final prediction must be based on at least 5 training ratings (stops trees memorizing noise) |
| max_features | sqrt | Each question considers a random ~7 of the 52 features |

Tuning made only a small difference. Most of the improvement came from the features.

---

## 5. How we measure success

| Metric | Meaning | Better is |
|---|---|---|
| Accuracy | Share of ratings predicted exactly right | Higher |
| Macro-F1 | Average score across all five ratings, weighting rare ratings (1s, 2s) equally with common ones (3s, 4s). It catches a model that ignores rare ratings. | Higher (0–1) |
| MAE (mean absolute error) | Average distance from the true rating, in stars. Gives partial credit for close guesses. | Lower |

Compare against two simple baselines:
- Always predict 4: the most common rating.
- Round the user's average: predict each user's usual rating for every movie.

---

## 6. Results

### Main results (average over 5 folds)

| Model | Accuracy | Macro-F1 | MAE (stars) |
|---|---|---|---|
| Baseline: always predict 4 | 34.2% | 0.102 | 0.894 |
| Baseline: round user's average | 36.1% | 0.202 | 0.805 |
| Random forest | 44.2% | 0.369 | 0.713 |

Results were very consistent across folds (accuracy ±0.6%, MAE ±0.007).

### Which features matter (ablation)

Same model and settings, different feature sets:

| Feature set | Accuracy | Macro-F1 | MAE |
|---|---|---|---|
| Demographics + genre flags | 37.3% | 0.262 | 0.846 |
| Rating history only | 41.5% | 0.351 | 0.759 |
| Demographics + history | 43.9% | 0.364 | 0.721 |
| All features | 44.2% | 0.369 | 0.713 |

- Demographics alone are weak: only a few points better than guessing "4".
- Rating history is the biggest improvement.
- Combining both beats either one alone.
- Genre taste adds a small extra gain.

### Feature importance

The share of the model's decisions driven by each feature (averaged over the five folds):

| Feature | Importance |
|---|---|
| Movie's average rating | 14.8% |
| User's average rating | 13.6% |
| User's rating spread | 11.8% |
| User's average in this movie's genres | 11.2% |
| Movie's rating spread | 9.2% |
| User's genre average vs. usual average | 6.7% |
| Movie's number of ratings | 6.1% |
| User's genre rating count | 5.2% |
| User's number of ratings | 5.2% |
| Age | 4.3% |
| Each individual genre flag | under 1% |

As plain movie labels, genres barely matter. Combined with the user's own history ("genre taste"), they become some of the most useful features.

### Confusion matrix

Rows are the actual rating and columns are the predicted rating, summed over all 100,000 test ratings. See the confusion_matrix.png for the heatmap.

| Actual rating | Predicted correctly |
|---|---|
| 1 | 22.6% |
| 2 | 8.5% |
| 3 | 44.4% |
| 4 | 62.8% |
| 5 | 39.2% |

- 87.4% of predictions are within one star of the real rating. Most mistakes are near misses (3 vs. 4, 4 vs. 5).
- The model leans toward 3 and 4. These are the most common ratings, so they're the "safe" guess.
- 2s are the hardest: they're uncommon and are usually predicted as 3 or 4. This comes from the class imbalance in the data (only about 6% of ratings are 1s and 11% are 2s).

---

## 7. Is 44% accuracy good?

may look low, but it's actually good for this problem:

- Random guessing gets 20%, and always guessing "4" gets 34%.
- People's ratings are inconsistent. The same person might rate a movie 3 one day and 4 another, so no model can be close to 100%.
- A standard recommender method on the same data does about the same
- An MAE of 0.71 means predictions are off by about three-quarters of a star on average, and 87% are within one star.
- This is similar to the sucess rate popular sites like Netflix see with their models

---

## 8. Limitations and possible next steps

- Rare ratings (1s and 2s) are often missed. Trying class_weight="balanced" in the random forest could improve this, at some cost to overall accuracy.
- No "similar users" features. Collaborative filtering features (how users with similar taste rated this movie) would likely help most.
- Existing users only. The model relies on rating history, so brand-new users would only have demographics, which are weak predictors.

---

## 9. Output files

All files are written to model/results/ by train.py

| File | Contents |
|---|---|
| key_results.txt | Short summary of the key numbers (also printed at the end of each run) |
| tuning.csv | Every parameter combination tried, with validation scores, best first |
| best_params.json | The winning settings |
| metrics_per_fold.csv | Accuracy, macro-F1 and MAE for each fold, for the model and baselines |
| summary.csv | The same, averaged over the five folds (main results table) |
| ablation.csv | Results for each feature set (only with --ablation) |
| confusion_matrix.csv / .png | Actual vs. predicted ratings, as counts and as a heatmap |
| feature_importance.csv / .png | Importance of every feature, and a chart of the top 15 |


---

## 10. How to run

From the repository root:

```bash
python data/preprocess.py               # only needed if data/processed/ is missing
python model/train.py --ablation        # tune, evaluate, compare feature sets (a few minutes)
python model/train.py --quick           # faster test run with a smaller grid
```

Required packages: pandas, numpy, scikit-learn, matplotlib 