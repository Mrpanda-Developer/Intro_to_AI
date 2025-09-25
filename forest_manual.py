# Cross validation accuracy = (fill after running) %
import argparse
import math
import os
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

TARGET_COL = "Breast cancer type"   # must exist in both files
DEFAULT_LABELLED = "dataset_labelled.xlsx"
DEFAULT_UNLABELLED = "dataset_unlabelled.xlsx"
DEFAULT_OUT = "submission.xlsx"
RANDOM_SEED = 42


def kfold_indices(n: int, k: int, rng: np.random.Generator):
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate(folds[:i] + folds[i+1:])
        yield train_idx, val_idx


def build_forest(X: np.ndarray,
                 y: np.ndarray,
                 n_trees: int = 100,
                 max_depth: int | None = None,
                 min_samples_leaf: int = 1,
                 m_try: int | None = None,
                 rng: np.random.Generator | None = None):
    """
    Manual random forest:
      - bootstrap samples rows
      - random subset of features per tree (m_try)
    Returns list of tuples (fitted_tree, feature_idx)
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    n_samples, n_features = X.shape
    if m_try is None:
        m_try = int(math.sqrt(n_features)) or 1

    forest = []
    for _ in range(n_trees):
        # bootstrap
        boot_idx = rng.integers(0, n_samples, size=n_samples)
        feat_idx = np.sort(rng.choice(n_features, size=m_try, replace=False))
        Xb = X[boot_idx][:, feat_idx]
        yb = y[boot_idx]

        tree = DecisionTreeClassifier(
            criterion="gini",
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=rng.integers(0, 2**31 - 1),
        )
        tree.fit(Xb, yb)
        forest.append((tree, feat_idx))
    return forest


def forest_predict(forest, X: np.ndarray) -> np.ndarray:
    votes = []
    for tree, feat_idx in forest:
        pred = tree.predict(X[:, feat_idx])
        votes.append(pred)
    votes = np.vstack(votes)  # [n_trees, n_samples]
    # majority vote per column
    out = []
    for j in range(votes.shape[1]):
        counts = Counter(votes[:, j])
        # tie-break: smallest class id
        out.append(sorted(counts.items(), key=lambda t: (-t[1], t[0]))[0][0])
    return np.array(out)


def cv_score(X: np.ndarray,
             y: np.ndarray,
             k: int,
             forest_params: dict,
             rng: np.random.Generator):
    accs, f1s = [], []
    for tr_idx, va_idx in kfold_indices(len(X), k, rng):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]
        forest = build_forest(Xtr, ytr, rng=rng, **forest_params)
        yhat = forest_predict(forest, Xva)
        accs.append(accuracy_score(yva, yhat))
        f1s.append(f1_score(yva, yhat, average="macro"))
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s)),
        "f1_macro_std": float(np.std(f1s)),
        "acc_folds": accs,
        "f1_folds": f1s,
    }


def load_and_split(labelled_path: str, unlabelled_path: str):
    df_lab = pd.read_excel(labelled_path)
    df_unl = pd.read_excel(unlabelled_path)

    # Feature columns = all except target
    feature_cols = [c for c in df_lab.columns if c != TARGET_COL]

    # Ensure target is numeric 1..4
    y = df_lab[TARGET_COL].astype("Int64").to_numpy().astype(int)
    X = df_lab[feature_cols].to_numpy(dtype=float)

    X_unl = df_unl[feature_cols].to_numpy(dtype=float)

    return X, y, df_unl, feature_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", default=DEFAULT_LABELLED)
    ap.add_argument("--unlabelled", default=DEFAULT_UNLABELLED)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--min_leaf", type=int, default=2)
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--mtry", type=int, default=None,
                    help="Number of features per tree (default sqrt(p))")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load
    X, y, df_unl, feature_cols = load_and_split(args.labelled, args.unlabelled)
    n_features = X.shape[1]
    m_try = args.mtry if args.mtry is not None else int(math.sqrt(n_features)) or 1

    forest_params = dict(
        n_trees=args.trees,
        max_depth=args.depth,
        min_samples_leaf=args.min_leaf,
        m_try=m_try,
    )

    # Manual K-fold CV
    scores = cv_score(X, y, k=args.kfolds, forest_params=forest_params, rng=rng)
    print("=== Manual K-fold CV (no sklearn CV helpers) ===")
    print(f"Folds: {args.kfolds}")
    print(f"Accuracy: {scores['acc_mean']:.4f} ± {scores['acc_std']:.4f}")
    print(f"Macro F1 : {scores['f1_macro_mean']:.4f} ± {scores['f1_macro_std']:.4f}")
    # Reminder to update the first-line comment with your accuracy:
    print("\n>>> Copy the mean CV accuracy into the first line of the code file as required.")

    # Train forest on ALL labelled data
    forest = build_forest(X, y, rng=rng, **forest_params)

    # Predict unlabelled
    X_unl = df_unl[feature_cols].to_numpy(dtype=float)
    yhat_unl = forest_predict(forest, X_unl).astype(int)

    # Write Excel: same columns/order, fill target column with ints 1..4
    df_out = df_unl.copy()
    if TARGET_COL not in df_out.columns:
        # if missing, add it at the end (spec typically has it present, but just in case)
        df_out[TARGET_COL] = np.nan
    df_out[TARGET_COL] = yhat_unl

    # Ensure single sheet, keep original columns & order:
    # (pandas preserves column order; we write only the first/default sheet)
    with pd.ExcelWriter(args.out, engine="xlsxwriter") as writer:
        df_out.to_excel(writer, index=False, sheet_name="Sheet1")

    print(f"\nSaved predictions to: {os.path.abspath(args.out)}")
    print("Submission checklist:")
    print(" - File type: .xlsx")
    print(" - First sheet only")
    print(f" - Columns preserved; filled '{TARGET_COL}' with integers 1..4")
    print(" - Row order unchanged")


if __name__ == "__main__":
    main()
