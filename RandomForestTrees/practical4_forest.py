# Cross validation accuracy = XX %

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from openpyxl import load_workbook

# -----------------------------
# Parameters
# -----------------------------
N_TREES = 50         # number of trees in forest
MAX_DEPTH = None     # max depth of trees
K_FOLDS = 5          # manual k-fold cross-validation
RANDOM_STATE = 42    # reproducibility
# -----------------------------

# Load datasets
labelled = pd.read_excel("../dataset_labelled.xlsx")
unlabelled = pd.read_excel("../dataset_unlabelled.xlsx")


# Split into features and target
X = labelled.drop(columns=["Breast cancer type"]).values
y = labelled["Breast cancer type"].values

# -----------------------------
# Helper functions
# -----------------------------

def build_forest(X_train, y_train, n_trees=N_TREES, max_depth=MAX_DEPTH, random_state=None):
    """Train a forest of decision trees on bootstrapped samples with random feature subsets."""
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X_train.shape
    forest = []

    for i in range(n_trees):
        # bootstrap sample rows
        idx_samples = rng.choice(n_samples, size=n_samples, replace=True)
        # random subset of features (sqrt rule)
        n_sub_features = int(np.sqrt(n_features))
        idx_features = rng.choice(n_features, size=n_sub_features, replace=False)

        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=rng.randint(1e9))
        clf.fit(X_train[idx_samples][:, idx_features], y_train[idx_samples])
        forest.append((clf, idx_features))
    return forest


def forest_predict(forest, X):
    """Predict using majority vote from forest."""
    all_preds = []
    for clf, idx_features in forest:
        preds = clf.predict(X[:, idx_features])
        all_preds.append(preds)
    all_preds = np.array(all_preds).T  # shape: (n_samples, n_trees)
    
    final_preds = []
    for row in all_preds:
        vote = Counter(row).most_common(1)[0][0]
        final_preds.append(vote)
    return np.array(final_preds)


# -----------------------------
# Manual K-Fold Cross Validation
# -----------------------------
def cross_val_score_manual(X, y, k_folds=K_FOLDS, random_state=RANDOM_STATE):
    rng = np.random.RandomState(random_state)
    n_samples = len(y)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    fold_sizes = np.full(k_folds, n_samples // k_folds, dtype=int)
    fold_sizes[: n_samples % k_folds] += 1

    current = 0
    accuracies = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])

        forest = build_forest(X[train_idx], y[train_idx], random_state=random_state)
        y_pred = forest_predict(forest, X[test_idx])
        acc = accuracy_score(y[test_idx], y_pred)
        accuracies.append(acc)
        current = stop

    return np.mean(accuracies)


cv_acc = cross_val_score_manual(X, y)
print(f"Cross validation accuracy = {cv_acc*100:.2f}%")

# -----------------------------
# Train on full dataset & Predict Unlabelled
# -----------------------------
forest_final = build_forest(X, y, random_state=RANDOM_STATE)
X_unlab = unlabelled.drop(columns=["Breast cancer type"]).values
y_pred_unlab = forest_predict(forest_final, X_unlab)

# Save predictions into Excel (same structure, fill target column)
output = unlabelled.copy()
output["Breast cancer type"] = y_pred_unlab
output.to_excel("predictions.xlsx", index=False)

print("Predictions written to predictions.xlsx")
