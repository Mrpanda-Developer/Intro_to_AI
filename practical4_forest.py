import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from openpyxl import load_workbook

# -----------------------------
# Parameters
# -----------------------------
RANDOM_STATE = 100
CV_FOLDS = 5         # fewer folds for speed
N_ITER_SEARCH = 20   # fewer iterations for faster search
# -----------------------------

# Load datasets
labelled = pd.read_excel("dataset_labelled.xlsx")
unlabelled = pd.read_excel("dataset_unlabelled.xlsx")

# Split into features and target
X = labelled.drop(columns=["Breast cancer type"]).values
y = labelled["Breast cancer type"].values

# -----------------------------
# Smaller hyperparameter search space
# -----------------------------
param_dist = {
    "n_estimators": [300, 500, 1200, 2100],      # fewer trees → faster
    "max_depth": [None, 10, 20, 90],           # simpler depth range
    "min_samples_split": [2, 5],               # small split options
    "min_samples_leaf": [1, 3],                # small leaf sizes
    "max_features": ["sqrt", "log2"],          # standard feature subsampling
    "bootstrap": [True]                        # standard RF bootstrap only
}

# -----------------------------
# Stratified CV setup
# -----------------------------
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# -----------------------------
# RandomizedSearchCV for tuning
# -----------------------------
rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist,
    n_iter=N_ITER_SEARCH, cv=cv,
    scoring="accuracy", n_jobs=-1,
    random_state=RANDOM_STATE, verbose=1
)

random_search.fit(X, y)
best_model = random_search.best_estimator_

# -----------------------------
# Cross-validation accuracy
# -----------------------------
cv_acc = cross_val_score(best_model, X, y, cv=cv, scoring="accuracy").mean()
print(f"Best parameters: {random_search.best_params_}")
print(f"Cross validation accuracy = {cv_acc*100:.2f}%")

# -----------------------------
# Train final model & predict
# -----------------------------
best_model.fit(X, y)
X_unlab = unlabelled.drop(columns=["Breast cancer type"]).values
y_pred_unlab = best_model.predict(X_unlab)

# Save predictions into Excel
output = unlabelled.copy()
output["Breast cancer type"] = y_pred_unlab
output.to_excel("predictions.xlsx", index=False)

print("Predictions written to predictions.xlsx")
