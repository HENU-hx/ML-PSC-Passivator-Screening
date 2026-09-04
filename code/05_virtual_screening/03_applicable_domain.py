import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


# Expected location: <project_root>/code/05_virtual_screening/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ===== Input/output paths =====
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--feature-names", required=True)
parser.add_argument("--training", required=True)
parser.add_argument("--library", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

MODEL_PATH = resolve_project_path(args.model)
FEATURE_PATH = resolve_project_path(args.feature_names)
TRAIN_CSV = resolve_project_path(args.training)
LIBRARY_CSV = resolve_project_path(args.library)
OUTPUT_FILE = resolve_project_path(args.output)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ===== AD parameter =====
K_NEIGHBORS = 5
AD_BORDERLINE_Q = 0.95
AD_OUTSIDE_Q = 0.99

# ===== read =====
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)

train_df = pd.read_csv(TRAIN_CSV)
lib_df = pd.read_csv(LIBRARY_CSV)

# ===== feature preparation =====
def prepare_X(df):
    X = df[feature_names].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    return X

train_X = prepare_X(train_df)
lib_X = prepare_X(lib_df)

# ===== Standardization (using training set) =====
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
lib_X_scaled = scaler.transform(lib_X)

# ===== Calculate the AD threshold of the training set =====
dist_mat = pairwise_distances(train_X_scaled, train_X_scaled)
np.fill_diagonal(dist_mat, np.inf)

k = min(K_NEIGHBORS, len(train_X_scaled) - 1)
train_knn = np.partition(dist_mat, kth=k - 1, axis=1)[:, :k]
train_dist = train_knn.mean(axis=1)

th1 = np.quantile(train_dist, AD_BORDERLINE_Q)
th2 = np.quantile(train_dist, AD_OUTSIDE_Q)

# ===== Calculate the AD distance of the library molecules =====
dist_lib = pairwise_distances(lib_X_scaled, train_X_scaled)
lib_knn = np.partition(dist_lib, kth=k - 1, axis=1)[:, :k]
lib_dist = lib_knn.mean(axis=1)

# ===== AD Classification =====
def get_flag(d):
    if d <= th1:
        return "Inside"
    elif d <= th2:
        return "Borderline"
    else:
        return "Outside"

lib_df["AD_distance"] = lib_dist
lib_df["AD_flag"] = [get_flag(d) for d in lib_dist]

# ===== model scoring =====
if hasattr(model, "predict_proba"):
    scores = model.predict_proba(lib_X)[:, 1]
else:
    scores = model.predict(lib_X)

lib_df["Score"] = scores

# ===== Sorting (AD takes precedence) =====
priority = {"Inside": 0, "Borderline": 1, "Outside": 2}
lib_df["AD_priority"] = lib_df["AD_flag"].map(priority)

lib_df = lib_df.sort_values(
    by=["AD_priority", "Score"],
    ascending=[True, False]
).reset_index(drop=True)

lib_df["Rank"] = np.arange(1, len(lib_df) + 1)

lib_df.to_csv(OUTPUT_FILE, index=False)

# ===== output statistics =====
print("Total molecules:", len(lib_df))
print("AD distribution:")
print(lib_df["AD_flag"].value_counts())

print("\nTop 10:")
print(lib_df[["Rank", "Score", "AD_flag"]].head(10))
print(f"\nSaved: {OUTPUT_FILE}")