

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from sklearn.model_selection import ParameterGrid, train_test_split

from features import FOLDS, build_features, load_fold

#location of all the output files 
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEED = 42

#parameter combinations tried in the grid search (8 total)
FULL_GRID = {
    "n_estimators": [200],
    "max_depth": [None, 20],
    "min_samples_leaf": [5, 20],
    "max_features": ["sqrt", 0.5],
}
#tiny grid for a fast test run (--quick)
QUICK_GRID = {
    "n_estimators": [100],
    "max_depth": [None],
    "min_samples_leaf": [5, 20],
    "max_features": ["sqrt"],
}

#feature sets compared in the ablation (--ablation)
ABLATIONS = {
    "Demographics + genre flags": ("demographics",),
    "Rating history only": ("history",),
    "Demographics + history": ("demographics", "history"),
    "All features": ("demographics", "history", "genre_taste"),
}

#computes accuracy, macro-F1, and MAE for a set of predictions
def score(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "mae": mean_absolute_error(y_true, y_pred),
    }


#creates random forest with given settings
def make_model(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(**params, n_jobs=-1, random_state=SEED)


#grid search: split u1's training data 80/20 into train/validation
#try every parameter combination and pick the best by macro-F1
#official test sets are not used here to keep them unbiased
def tune(grid: dict) -> dict:

    train, _ = load_fold("u1")
    inner_train, val = train_test_split(train, test_size=0.2, random_state=SEED,
                                        stratify=train["rating"])
    X, y, Xv, yv = build_features(inner_train.reset_index(drop=True), val.reset_index(drop=True))

    rows = []
    combos = list(ParameterGrid(grid))
    for i, params in enumerate(combos, 1):
        start = time.time()
        pred = make_model(params).fit(X, y).predict(Xv)
        rows.append({**params, **score(yv, pred), "seconds": round(time.time() - start, 1)})
        print(f"  [{i}/{len(combos)}] {params} -> macro-F1 {rows[-1]['macro_f1']:.3f}", flush=True)

    pd.DataFrame(rows).sort_values("macro_f1", ascending=False).to_csv(RESULTS_DIR / "tuning.csv", index=False)
    best = max(range(len(combos)), key=lambda i: rows[i]["macro_f1"])
    return combos[best]


#trains model on ecah of the 5 folds and scores it on that folds test set
#also scores the two baselines (always predict4 and round the users average)
#saves the confusion matrix and feature importance
def evaluate(params: dict, groups=None, save_details: bool = True) -> pd.DataFrame:
   
    rows, cm_total, importances = [], np.zeros((5, 5), dtype=int), []
    for fold in FOLDS:
        train, test = load_fold(fold)
        kwargs = {"groups": groups} if groups else {}
        X, y, Xt, yt = build_features(train, test, **kwargs)

        if save_details:
            rows.append({"fold": fold, "model": "Always predict 4", **score(yt, np.full_like(yt, 4))})
            user_avg = np.clip(np.rint(Xt["user_mean"]), 1, 5)
            rows.append({"fold": fold, "model": "Round user's average", **score(yt, user_avg)})

        model = make_model(params).fit(X, y)
        pred = model.predict(Xt)
        rows.append({"fold": fold, "model": "Random forest", **score(yt, pred)})
        cm_total += confusion_matrix(yt, pred, labels=[1, 2, 3, 4, 5])
        importances.append(pd.Series(model.feature_importances_, index=X.columns))
        print(f"  {fold}: accuracy {rows[-1]['accuracy']:.3f}, macro-F1 {rows[-1]['macro_f1']:.3f}, "
              f"MAE {rows[-1]['mae']:.3f}", flush=True)

    per_fold = pd.DataFrame(rows)
    if save_details:
        per_fold.to_csv(RESULTS_DIR / "metrics_per_fold.csv", index=False)
        save_confusion_matrix(cm_total)
        save_importance(pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False))
    return per_fold

#averages each models metrics across the 5 folds (mean and std)
def summarize(per_fold: pd.DataFrame) -> pd.DataFrame:
    order = list(dict.fromkeys(per_fold["model"]))
    summary = per_fold.groupby("model")[["accuracy", "macro_f1", "mae"]].agg(["mean", "std"])
    return summary.reindex(order).round(3)

#saves confusion matrix as a csv and as a heatmap image (row percentages)
def save_confusion_matrix(cm: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [1, 2, 3, 4, 5]
    pd.DataFrame(cm, index=[f"true_{l}" for l in labels],
                 columns=[f"pred_{l}" for l in labels]).to_csv(RESULTS_DIR / "confusion_matrix.csv")
    share = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{share[i, j]:.0%}", ha="center", va="center",
                    color="white" if share[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(5), labels)
    ax.set_yticks(range(5), labels)
    ax.set_xlabel("Predicted rating")
    ax.set_ylabel("Actual rating")
    ax.set_title("Confusion matrix (all 5 folds, row %)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


#saves featre importance as a csv and a bar chart of the top 15 featres 
def save_importance(importance: pd.Series) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    importance.rename("importance").to_csv(RESULTS_DIR / "feature_importance.csv")
    top = importance.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top.index, top.values, color="#3b6ea8")
    ax.set_xlabel("Mean importance across folds")
    ax.set_title("Top 15 features")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


#builds short "KEY RESULTS" summary printed at end of the run 
def key_results(summary: pd.DataFrame, best: dict, ablation: pd.DataFrame | None = None) -> str:
   
    def line(df, name, label):
        r = df.loc[name]
        return (f"  {label:<28} accuracy {r[('accuracy', 'mean')]:6.1%}   "
                f"macro-F1 {r[('macro_f1', 'mean')]:.3f}   MAE {r[('mae', 'mean')]:.3f}")

    rf = summary.loc["Random forest"]
    lines = [
        "=" * 72,
        "KEY RESULTS (average over the 5 test folds)",
        "=" * 72,
        line(summary, "Random forest", "Random forest"),
        line(summary, "Always predict 4", "Baseline: always predict 4"),
        line(summary, "Round user's average", "Baseline: user's average"),
        "",
        f"  Fold-to-fold spread (std): accuracy +/-{rf[('accuracy', 'std')]:.1%}, "
        f"MAE +/-{rf[('mae', 'std')]:.3f}",
        f"  Best parameters: {best}",
    ]
    if ablation is not None:
        lines += ["", "  Feature sets:"]
        lines += [line(ablation, name, name) for name in ablation.index]
    lines += ["", "  Accuracy and macro-F1: higher is better.  MAE (stars off): lower is better.",
              "=" * 72]
    return "\n".join(lines)

#runs everything in order: tune -> evalulate -> (optional) ablation -> print summary
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="Use a small grid for a fast run.")
    parser.add_argument("--ablation", action="store_true", help="Also compare feature groups.")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("1) Tuning on fold u1 (inner train/validation split)")
    best = tune(QUICK_GRID if args.quick else FULL_GRID)
    (RESULTS_DIR / "best_params.json").write_text(json.dumps(best, indent=2))
    print(f"   Best parameters: {best}\n")

    print("2) Evaluating on the five official folds")
    summary = summarize(evaluate(best))
    summary.to_csv(RESULTS_DIR / "summary.csv")
    print("\n" + summary.to_string())

    ablation = None
    if args.ablation:
        print("\n3) Feature-group comparison (same parameters)")
        rows = []
        for name, groups in ABLATIONS.items():
            print(f" {name}")
            res = evaluate(best, groups=groups, save_details=False)
            rows.append(res.assign(model=name))
        ablation = summarize(pd.concat(rows))
        ablation.to_csv(RESULTS_DIR / "ablation.csv")
        print("\n" + ablation.to_string())

    recap = key_results(summary, best, ablation)
    (RESULTS_DIR / "key_results.txt").write_text(recap + "\n", encoding="utf-8")
    print("\n" + recap)
    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

