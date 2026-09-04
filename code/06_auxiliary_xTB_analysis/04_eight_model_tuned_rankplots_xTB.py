# -*- coding: utf-8 -*-
"""
Eight-model tuned comparison for binary classification of delta_PCE
with rank-oriented plots

Models:
1) SVM
2) RF
3) LR
4) MLP
5) XGBoost
6) CatBoost
7) LightGBM
8) LR_ElasticNet

Workflow:
1) Tune each model once on full dataset using 5-fold CV and PR-AUC
2) Repeated stratified hold-out evaluation (50 times)
3) Boundary down-weighting for samples near threshold
4) Unified metrics and screening-oriented comparison
5) Output rank-oriented plots:
   - Precision@3 / Precision@5 bar plot with std error bars
   - sorted horizontal point plot

Unified metrics:
- ROC-AUC
- PR-AUC
- Accuracy
- Precision
- Recall
- F1
- Precision@3
- Precision@5
- EF@3
- EF@5
"""

import argparse
import os
import time
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# =========================================================
# 1) Config
# =========================================================
# Expected location: <project_root>/code/03_model_construction/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value: str) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eight-model tuned comparison for binary delta_PCE classification."
    )
    parser.add_argument(
        "--input",
        default="data/training_14descriptors.csv",
        help="Input training CSV, relative to the project root.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/model_comparison/eight_model_tuned_rankplots_bar_SEM_01",
        help="Directory for all generated outputs, relative to the project root.",
    )
    return parser.parse_args()


args = parse_args()

INPUT_CSV = resolve_project_path(args.input)
OUT_DIR = resolve_project_path(args.output_dir)
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_COL = "delta_PCE"

POS_THRESHOLD = 2.0
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ITER = 50

BORDER_LOW = 1.8
BORDER_HIGH = 2.2
BORDER_WEIGHT = 0.5

K_LIST = [3, 5]

TUNE_ONCE = True

CV_FOLDS = 5

# =========================================================
# 2) Load data
# =========================================================
df = pd.read_csv(INPUT_CSV)
if TARGET_COL not in df.columns:
    raise ValueError(f"Missing target column '{TARGET_COL}'")

X_all = df.drop(columns=[TARGET_COL]).select_dtypes(include=["number"]).copy()
X_all = X_all.fillna(X_all.median(numeric_only=True))

y_all = (df[TARGET_COL] >= POS_THRESHOLD).astype(int)
delta_all = df[TARGET_COL].copy()

print("Dataset size:", len(df))
print("Class counts:", y_all.value_counts().to_dict())
print("Positive rate:", round(float(y_all.mean()), 3))
print("Feature dim:", X_all.shape[1])
print()

# =========================================================
# 3) Helper functions
# =========================================================
def make_border_weights(delta_series: pd.Series) -> np.ndarray:
    w = np.ones(len(delta_series), dtype=float)
    mask = delta_series.between(BORDER_LOW, BORDER_HIGH, inclusive="both").to_numpy()
    w[mask] = BORDER_WEIGHT
    return w

def precision_at_k(y_true: pd.Series, scores: np.ndarray, k: int) -> float:
    k = min(k, len(y_true))
    idx = np.argsort(scores)[::-1][:k]
    return float(y_true.iloc[idx].sum() / k)

def recall_at_k(y_true: pd.Series, scores: np.ndarray, k: int) -> float:
    pos = int(y_true.sum())
    if pos == 0:
        return np.nan
    k = min(k, len(y_true))
    idx = np.argsort(scores)[::-1][:k]
    hits = int(y_true.iloc[idx].sum())
    return float(hits / pos)

def ef_at_k(y_true: pd.Series, scores: np.ndarray, k: int) -> float:
    pos_rate = float(y_true.mean())
    if pos_rate == 0:
        return np.nan
    return precision_at_k(y_true, scores, k) / pos_rate

def strip_prefix(best_params: dict, prefix: str) -> dict:
    clean = {}
    for k, v in best_params.items():
        if k.startswith(prefix):
            clean[k.replace(prefix, "")] = v
    return clean

def get_last_step_name(pipe: Pipeline) -> str:
    return list(pipe.named_steps.keys())[-1]

def fit_with_sample_weight(model, X_train, y_train, w_train):
    """
    Try passing `sample_weight` to the last layer of the pipeline;
    If not compatible, it will automatically revert to a regular fit.
    """
    fitted = False
    step_name = get_last_step_name(model)

    candidate_keys = [
        f"{step_name}__sample_weight",
        "clf__sample_weight",
        "svc__sample_weight",
        "lgbm__sample_weight",
        "xgb__sample_weight",
        "cat__sample_weight",
        "rf__sample_weight",
        "mlp__sample_weight",
        "lr__sample_weight"
    ]

    for kw in candidate_keys:
        try:
            model.fit(X_train, y_train, **{kw: w_train})
            fitted = True
            break
        except Exception:
            pass

    if not fitted:
        model.fit(X_train, y_train)

    return model

# =========================================================
# 4) Model factory + param grids
# =========================================================
def build_pipeline_and_grid(model_name, random_state=42):
    if model_name == "SVM":
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svc", SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=random_state
            ))
        ])
        param_grid = {
            "svc__C": [0.01, 0.1, 1, 3, 10, 30],
            "svc__gamma": ["scale", 0.1, 0.03, 0.01, 0.003, 0.001],
        }

    elif model_name == "RF":
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            ))
        ])
        param_grid = {
            "rf__n_estimators": [100, 200, 300, 500],
            "rf__max_depth": [None, 3, 5, 8, 12],
            "rf__min_samples_split": [2, 5, 10],
            "rf__min_samples_leaf": [1, 2, 4],
            "rf__max_features": ["sqrt", "log2", None],
        }

    elif model_name == "LR":
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                penalty="l2",
                solver="liblinear",
                class_weight="balanced",
                random_state=random_state,
                max_iter=5000
            ))
        ])
        param_grid = {
            "lr__C": [0.01, 0.1, 1, 3, 10, 30]
        }

    elif model_name == "MLP":
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                random_state=random_state,
                max_iter=2000
            ))
        ])
        param_grid = {
            "mlp__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
            "mlp__alpha": [1e-4, 1e-3, 1e-2],
            "mlp__learning_rate_init": [1e-4, 5e-4, 1e-3, 5e-3],
        }

    elif model_name == "XGBoost":
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("xgb", XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1
            ))
        ])
        param_grid = {
            "xgb__n_estimators": [100, 200, 300],
            "xgb__max_depth": [2, 3, 4, 5],
            "xgb__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "xgb__subsample": [0.8, 0.9, 1.0],
            "xgb__colsample_bytree": [0.8, 0.9, 1.0],
            "xgb__reg_lambda": [0.5, 1.0, 2.0],
        }

    elif model_name == "CatBoost":
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("cat", CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="PRAUC",
                verbose=0,
                random_seed=random_state
            ))
        ])
        param_grid = {
            "cat__iterations": [100, 200, 300, 500],
            "cat__depth": [3, 4, 5, 6],
            "cat__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "cat__l2_leaf_reg": [1, 3, 5, 7],
        }

    elif model_name == "LightGBM":
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("lgbm", LGBMClassifier(
                objective="binary",
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1
            ))
        ])
        param_grid = {
            "lgbm__n_estimators": [50, 100, 200, 300, 500],
            "lgbm__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "lgbm__num_leaves": [7, 15, 31, 63],
            "lgbm__max_depth": [-1, 3, 5, 7, 10],
            "lgbm__min_child_samples": [5, 10, 20, 30],
        }

    elif model_name == "LR_ElasticNet":
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                class_weight="balanced",
                random_state=random_state,
                max_iter=8000
            ))
        ])
        param_grid = {
            "lr__C": [0.01, 0.1, 1, 3, 10, 30],
            "lr__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return pipe, param_grid

def rebuild_best_model(model_name, best_params_global, random_state=42):
    if model_name == "SVM":
        params = strip_prefix(best_params_global, "svc__")
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svc", SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=random_state,
                **params
            ))
        ])

    elif model_name == "RF":
        params = strip_prefix(best_params_global, "rf__")
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
                **params
            ))
        ])

    elif model_name == "LR":
        params = strip_prefix(best_params_global, "lr__")
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                penalty="l2",
                solver="liblinear",
                class_weight="balanced",
                random_state=random_state,
                max_iter=5000,
                **params
            ))
        ])

    elif model_name == "MLP":
        params = strip_prefix(best_params_global, "mlp__")
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                random_state=random_state,
                max_iter=2000,
                **params
            ))
        ])

    elif model_name == "XGBoost":
        params = strip_prefix(best_params_global, "xgb__")
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("xgb", XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
                **params
            ))
        ])

    elif model_name == "CatBoost":
        params = strip_prefix(best_params_global, "cat__")
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("cat", CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="PRAUC",
                verbose=0,
                random_seed=random_state,
                **params
            ))
        ])

    elif model_name == "LightGBM":
        params = strip_prefix(best_params_global, "lgbm__")
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("lgbm", LGBMClassifier(
                objective="binary",
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,
                **params
            ))
        ])

    elif model_name == "LR_ElasticNet":
        params = strip_prefix(best_params_global, "lr__")
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                class_weight="balanced",
                random_state=random_state,
                max_iter=8000,
                **params
            ))
        ])

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model

# =========================================================
# 5) Tune each model once
# =========================================================
MODEL_NAMES = [
    "SVM",
    "RF",
    "LR",
    "MLP",
    "XGBoost",
    "CatBoost",
    "LightGBM",
    "LR_ElasticNet"
]

cv5 = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
w_global = make_border_weights(delta_all)

best_params_by_model = {}
best_cv_ap_by_model = []

if TUNE_ONCE:
    print("Start TUNE_ONCE for all 8 models...\n")

    for model_name in MODEL_NAMES:
        t0 = time.time()
        print(f"[TUNE] {model_name}")

        pipe, param_grid = build_pipeline_and_grid(model_name, RANDOM_STATE)

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="average_precision",
            cv=cv5,
            n_jobs=-1,
            verbose=1
        )

        try:
            step_name = get_last_step_name(pipe)
            grid.fit(X_all, y_all, **{f"{step_name}__sample_weight": w_global})
        except Exception:
            grid.fit(X_all, y_all)

        best_params_by_model[model_name] = grid.best_params_
        best_cv_ap_by_model.append({
            "Model": model_name,
            "Best_CV_PR-AUC": grid.best_score_,
            "Best_Params_JSON": json.dumps(grid.best_params_, ensure_ascii=False)
        })

        dt = time.time() - t0
        print(f"    Best Params: {grid.best_params_}")
        print(f"    Best CV PR-AUC: {grid.best_score_:.4f}")
        print(f"    Time: {dt:.1f} s\n")

    best_cv_df = pd.DataFrame(best_cv_ap_by_model).sort_values("Best_CV_PR-AUC", ascending=False)
    best_cv_df.to_csv(
        os.path.join(OUT_DIR, "rankplots_tune_once_best_params_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    with open(os.path.join(OUT_DIR, "rankplots_tune_once_best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params_by_model, f, ensure_ascii=False, indent=2)

# =========================================================
# 6) Repeated evaluation
# =========================================================
all_metric_rows = []
all_screen_rows = []

print("Start repeated evaluation for tuned 8 models...\n")

for model_name in MODEL_NAMES:
    print(f"Running model: {model_name}")

    for it in range(N_ITER):
        rs = RANDOM_STATE + it

        X_train, X_test, y_train, y_test, d_train, d_test, idx_train, idx_test = train_test_split(
            X_all, y_all, delta_all, df.index,
            test_size=TEST_SIZE,
            stratify=y_all,
            random_state=rs
        )

        w_train = make_border_weights(d_train)

        if TUNE_ONCE:
            model = rebuild_best_model(model_name, best_params_by_model[model_name], RANDOM_STATE)
        else:
            pipe, _ = build_pipeline_and_grid(model_name, RANDOM_STATE)
            model = pipe

        model = fit_with_sample_weight(model, X_train, y_train, w_train)

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = y_pred.astype(float)

        roc_auc = roc_auc_score(y_test, y_score)
        pr_auc = average_precision_score(y_test, y_score)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        all_metric_rows.append({
            "Model": model_name,
            "iter": it + 1,
            "random_state": rs,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1
        })

        y_test_s = pd.Series(y_test.values, index=idx_test)

        for k in K_LIST:
            p_at = precision_at_k(y_test_s, y_score, k)
            r_at = recall_at_k(y_test_s, y_score, k)
            ef = ef_at_k(y_test_s, y_score, k)

            all_screen_rows.append({
                "Model": model_name,
                "iter": it + 1,
                "K": k,
                "Precision@K": p_at,
                "Recall@K": r_at,
                "EF@K": ef
            })

print("\nAll tuned models finished.\n")

# =========================================================
# 7) Save per-iteration outputs
# =========================================================
metrics_df = pd.DataFrame(all_metric_rows)
metrics_df.to_csv(
    os.path.join(OUT_DIR, "rankplots_all_models_repeated_metrics_per_iter.csv"),
    index=False,
    encoding="utf-8-sig"
)

screen_df = pd.DataFrame(all_screen_rows)
screen_df.to_csv(
    os.path.join(OUT_DIR, "rankplots_all_models_screening_metrics_per_iter.csv"),
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# 8) Build final summary table
# =========================================================
main_summary = metrics_df.groupby("Model")[[
    "ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall", "F1"
]].agg(["mean", "std"])

main_summary.columns = [f"{a}_{b}" for a, b in main_summary.columns]
main_summary = main_summary.reset_index()

def get_screen_mean_std(metric_name, k):
    sub = screen_df[screen_df["K"] == k].groupby("Model")[metric_name].agg(["mean", "std"]).reset_index()
    sub.columns = ["Model", f"{metric_name}{k}_mean", f"{metric_name}{k}_std"]
    return sub

p3 = get_screen_mean_std("Precision@K", 3)
p5 = get_screen_mean_std("Precision@K", 5)
ef3 = get_screen_mean_std("EF@K", 3)
ef5 = get_screen_mean_std("EF@K", 5)

summary_df = main_summary.merge(p3, on="Model", how="left")
summary_df = summary_df.merge(p5, on="Model", how="left")
summary_df = summary_df.merge(ef3, on="Model", how="left")
summary_df = summary_df.merge(ef5, on="Model", how="left")

final_cols = [
    "Model",
    "ROC-AUC_mean", "ROC-AUC_std",
    "PR-AUC_mean", "PR-AUC_std",
    "Accuracy_mean", "Accuracy_std",
    "Precision_mean", "Precision_std",
    "Recall_mean", "Recall_std",
    "F1_mean", "F1_std",
    "Precision@K3_mean", "Precision@K3_std",
    "Precision@K5_mean", "Precision@K5_std",
    "EF@K3_mean", "EF@K3_std",
    "EF@K5_mean", "EF@K5_std",
]

summary_df = summary_df[final_cols]
summary_df.to_csv(
    os.path.join(OUT_DIR, "rankplots_eight_models_summary_table_top3_top5_tuned.csv"),
    index=False,
    encoding="utf-8-sig"
)

compact_df = summary_df[[
    "Model",
    "ROC-AUC_mean",
    "PR-AUC_mean",
    "Accuracy_mean",
    "Precision_mean",
    "Recall_mean",
    "F1_mean",
    "Precision@K3_mean",
    "Precision@K5_mean",
    "EF@K3_mean",
    "EF@K5_mean",
]].copy()

compact_df.columns = [
    "Model",
    "ROC-AUC",
    "PR-AUC",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Precision@3",
    "Precision@5",
    "EF@3",
    "EF@5",
]

compact_df = compact_df.sort_values(by="PR-AUC", ascending=False)
compact_df.to_csv(
    os.path.join(OUT_DIR, "rankplots_eight_models_compact_table_top3_top5_tuned.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("===== Tuned eight-model compact table =====")
print(compact_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# # =========================================================
# # 9) Plot 1: PR-AUC boxplot
# # =========================================================
# plot_order = compact_df["Model"].tolist()
#
# plt.figure(figsize=(10, 6), dpi=300)
# box_data = [metrics_df.loc[metrics_df["Model"] == m, "PR-AUC"].values for m in plot_order]
# plt.boxplot(box_data, labels=plot_order, showfliers=False)
# plt.ylabel("PR-AUC")
# plt.title("PR-AUC Distribution Across 8 Tuned Models")
# plt.xticks(rotation=30)
# plt.grid(axis="y", linestyle="--", alpha=0.3)
# plt.tight_layout()
# plt.savefig(
#     os.path.join(OUT_DIR, "rankplots_eight_models_pr_auc_boxplot_top3_top5_tuned.png"),
#     bbox_inches="tight"
# )
# plt.close()
#
# # =========================================================
# # 10) Plot 2: Precision@3 / Precision@5 bar plot with std
# # =========================================================
# bar_df = summary_df[[
#     "Model",
#     "Precision@K3_mean", "Precision@K3_std",
#     "Precision@K5_mean", "Precision@K5_std"
# ]].copy()
#
# bar_df = bar_df.sort_values(by="Precision@K5_mean", ascending=False).reset_index(drop=True)
#
# x = np.arange(len(bar_df))
# width = 0.36
#
# plt.figure(figsize=(12, 6), dpi=300)
#
# plt.bar(
#     x - width/2,
#     bar_df["Precision@K3_mean"],
#     width=width,
#     yerr=bar_df["Precision@K3_std"],
#     capsize=4,
#     label="Precision@3"
# )
#
# plt.bar(
#     x + width/2,
#     bar_df["Precision@K5_mean"],
#     width=width,
#     yerr=bar_df["Precision@K5_std"],
#     capsize=4,
#     label="Precision@5"
# )
#
# plt.xticks(x, bar_df["Model"], rotation=35)
# plt.ylabel("Score")
# plt.title("Top-ranked Screening Precision with Standard Deviation")
# plt.ylim(0, 1.0)
# plt.legend()
# plt.grid(axis="y", linestyle="--", alpha=0.3)
# plt.tight_layout()
# plt.savefig(
#     os.path.join(OUT_DIR, "rankplots_eight_models_precision3_precision5_bar_std.png"),
#     dpi=300,
#     bbox_inches="tight"
# )
# plt.close()
#
# # =========================================================
# # 11) Plot 3: Sorted horizontal point plot
# # =========================================================
# point_df = summary_df[[
#     "Model",
#     "Precision@K3_mean", "Precision@K3_std",
#     "Precision@K5_mean", "Precision@K5_std"
# ]].copy()
#
# point_df = point_df.sort_values(by="Precision@K5_mean", ascending=True).reset_index(drop=True)
# y_pos = np.arange(len(point_df))
#
# plt.figure(figsize=(10, 6), dpi=300)
#
# plt.errorbar(
#     point_df["Precision@K3_mean"],
#     y_pos + 0.12,
#     xerr=point_df["Precision@K3_std"],
#     fmt="o",
#     capsize=4,
#     label="Precision@3"
# )
#
# plt.errorbar(
#     point_df["Precision@K5_mean"],
#     y_pos - 0.12,
#     xerr=point_df["Precision@K5_std"],
#     fmt="o",
#     capsize=4,
#     label="Precision@5"
# )
#
# plt.yticks(y_pos, point_df["Model"])
# plt.xlabel("Mean score")
# plt.title("Sorted Horizontal Point Plot of Top-ranked Precision")
# plt.xlim(0, 1.0)
# plt.legend()
# plt.grid(axis="x", linestyle="--", alpha=0.3)
# plt.tight_layout()
# plt.savefig(
#     os.path.join(OUT_DIR, "rankplots_eight_models_precision3_precision5_point_sorted.png"),
#     dpi=300,
#     bbox_inches="tight"
# )
# plt.close()

# =========================================================
# 9) Plot 1: PR-AUC boxplot (manuscript version)
# =========================================================
plot_order = compact_df["Model"].tolist()

plt.figure(figsize=(10, 6), dpi=300)
box_data = [metrics_df.loc[metrics_df["Model"] == m, "PR-AUC"].values for m in plot_order]
plt.boxplot(box_data, labels=plot_order, showfliers=False)
plt.ylabel("PR-AUC")
plt.title("PR-AUC distribution across 8 tuned models")
plt.xticks(rotation=30)
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "manuscript_pr_auc_boxplot.png"),
    bbox_inches="tight"
)
plt.close()

# =========================================================
# 10) Plot 2: Precision@3 / Precision@5 bar plot with std
#     optimized to avoid cramped upper space
# =========================================================
bar_df = summary_df[[
    "Model",
    "Precision@K3_mean", "Precision@K3_std",
    "Precision@K5_mean", "Precision@K5_std"
]].copy()


bar_df = bar_df.sort_values(by="Precision@K5_mean", ascending=False).reset_index(drop=True)

x = np.arange(len(bar_df))
width = 0.36


upper_p3 = (bar_df["Precision@K3_mean"] + bar_df["Precision@K3_std"]).max()
upper_p5 = (bar_df["Precision@K5_mean"] + bar_df["Precision@K5_std"]).max()
ymax = min(1.08, max(1.02, max(upper_p3, upper_p5) + 0.04))

plt.figure(figsize=(12, 6.8), dpi=300)

plt.bar(
    x - width/2,
    bar_df["Precision@K3_mean"],
    width=width,
    yerr=bar_df["Precision@K3_std"],
    capsize=4,
    label="Precision@3"
)

plt.bar(
    x + width/2,
    bar_df["Precision@K5_mean"],
    width=width,
    yerr=bar_df["Precision@K5_std"],
    capsize=4,
    label="Precision@5"
)

plt.xticks(x, bar_df["Model"], rotation=35)
plt.ylabel("Score")
plt.title("Top-ranked screening precision")
plt.ylim(0, ymax)
plt.legend(frameon=True)
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "manuscript_precision3_precision5_bar_std.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# =========================================================
# 11) Plot 3: Sorted horizontal point plot (manuscript version)
# =========================================================
point_df = summary_df[[
    "Model",
    "Precision@K3_mean", "Precision@K3_std",
    "Precision@K5_mean", "Precision@K5_std"
]].copy()

# # Sort by Precision@5 in ascending order
point_df = point_df.sort_values(by="Precision@K5_mean", ascending=True).reset_index(drop=True)
y_pos = np.arange(len(point_df))

plt.figure(figsize=(10, 6), dpi=300)

plt.errorbar(
    point_df["Precision@K3_mean"],
    y_pos + 0.12,
    xerr=point_df["Precision@K3_std"],
    fmt="o",
    capsize=4,
    label="Precision@3"
)

plt.errorbar(
    point_df["Precision@K5_mean"],
    y_pos - 0.12,
    xerr=point_df["Precision@K5_std"],
    fmt="o",
    capsize=4,
    label="Precision@5"
)

plt.yticks(y_pos, point_df["Model"])
plt.xlabel("Mean score")
plt.title("Top-ranked precision (sorted)")
plt.xlim(0, 1.0)
plt.legend(frameon=True)
plt.grid(axis="x", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "manuscript_precision3_precision5_point_sorted.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# print("\nSaved manuscript-style figures:")
# print(os.path.join(OUT_DIR, "manuscript_pr_auc_boxplot.png"))
# print(os.path.join(OUT_DIR, "manuscript_precision3_precision5_bar_std.png"))
# print(os.path.join(OUT_DIR, "manuscript_precision3_precision5_point_sorted.png"))
#
# print("\nSaved files:")
# print(os.path.join(OUT_DIR, "rankplots_tune_once_best_params_summary.csv"))
# print(os.path.join(OUT_DIR, "rankplots_tune_once_best_params.json"))
# print(os.path.join(OUT_DIR, "rankplots_all_models_repeated_metrics_per_iter.csv"))
# print(os.path.join(OUT_DIR, "rankplots_all_models_screening_metrics_per_iter.csv"))
# print(os.path.join(OUT_DIR, "rankplots_eight_models_summary_table_top3_top5_tuned.csv"))
# print(os.path.join(OUT_DIR, "rankplots_eight_models_compact_table_top3_top5_tuned.csv"))
# print(os.path.join(OUT_DIR, "rankplots_eight_models_pr_auc_boxplot_top3_top5_tuned.png"))
# print(os.path.join(OUT_DIR, "rankplots_eight_models_precision3_precision5_bar_std.png"))
# print(os.path.join(OUT_DIR, "rankplots_eight_models_precision3_precision5_point_sorted.png"))
#
# print("\nDone. Outputs saved in:", OUT_DIR)

print("\nSaved manuscript-style figures:")
print(os.path.join(OUT_DIR, "manuscript_pr_auc_boxplot.png"))
print(os.path.join(OUT_DIR, "manuscript_precision3_precision5_bar_std.png"))
print(os.path.join(OUT_DIR, "manuscript_precision3_precision5_point_sorted.png"))

print("\nSaved key result files:")
print(os.path.join(OUT_DIR, "rankplots_eight_models_summary_table_top3_top5_tuned.csv"))
print(os.path.join(OUT_DIR, "rankplots_eight_models_compact_table_top3_top5_tuned.csv"))
print(os.path.join(OUT_DIR, "rankplots_tune_once_best_params_summary.csv"))

print("\nDone. Outputs saved in:", OUT_DIR)
