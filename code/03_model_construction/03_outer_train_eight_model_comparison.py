#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Leakage-free repeated nested hold-out comparison of eight classifiers.

For every saved outer split, this script:
  1. Reads the L1-selected features obtained from that outer training set.
  2. Builds a preprocessing/model Pipeline (imputation/scaling are fitted
     inside each inner-CV fold by GridSearchCV).
  3. Tunes each of the eight models using five-fold CV and PR-AUC, using only
     the outer training rows.
  4. Fits the selected model on the complete outer training set and evaluates
     it once on the untouched outer test rows.

The script does not choose RF in advance. RF is selected only after the eight
models have been compared. The saved per-iteration predictions can be used to
recreate the manuscript performance figures.

Example:
  python code/03_model_construction/03_outer_train_eight_model_comparison.py ^
      --input data/training_25descriptors_labeled.csv ^
      --split-assignments outputs/outer_repeated_splits/outer_split_assignments.csv ^
      --l1-features outputs/outer_train_l1_selection/l1_selected_features_by_iteration.csv ^
      --output-dir outputs/outer_eight_model_comparison
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = [
    "SVM", "RF", "LR", "MLP", "XGBoost", "CatBoost", "LightGBM", "LR_ElasticNet"
]
WEIGHTED_MODELS = {"SVM", "RF", "LR", "XGBoost", "CatBoost", "LightGBM", "LR_ElasticNet"}


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            pass
    raise ValueError(f"Cannot decode CSV: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Labeled 25-descriptor CSV.")
    parser.add_argument("--split-assignments", required=True,
                        help="outer_split_assignments.csv")
    parser.add_argument("--l1-features", required=True,
                        help="l1_selected_features_by_iteration.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="delta_PCE")
    parser.add_argument("--positive-threshold", type=float, default=2.0)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--k-list", default="3,5",
                        help="Comma-separated K values for screening metrics.")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_l1_features(path: Path) -> dict[int, list[str]]:
    table = read_csv(path)
    required = {"iteration", "selected_feature"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"L1 feature file missing columns: {sorted(missing)}")
    result: dict[int, list[str]] = {}
    for iteration, group in table.groupby("iteration", sort=True):
        if "feature_order" in group.columns:
            group = group.sort_values("feature_order")
        features = group["selected_feature"].astype(str).tolist()
        if not features:
            raise ValueError(f"No L1 features for iteration {iteration}")
        result[int(iteration)] = features
    return result


def validate_assignments(assignments: pd.DataFrame, n_rows: int) -> None:
    required = {"iteration", "row_index", "split"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Split file missing columns: {sorted(missing)}")
    if not assignments["split"].isin(["train", "test"]).all():
        raise ValueError("split must contain only 'train' and 'test'.")
    assignments["row_index"] = assignments["row_index"].astype(int)
    if not assignments["row_index"].between(0, n_rows - 1).all():
        raise ValueError("row_index outside input table.")
    if assignments.duplicated(["iteration", "row_index"]).any():
        raise ValueError("Duplicate row assignment within an iteration.")
    for iteration, group in assignments.groupby("iteration"):
        if (group["split"] == "train").sum() == 0 or (group["split"] == "test").sum() == 0:
            raise ValueError(f"Iteration {iteration} has an empty train/test split.")


def boundary_weights(delta: pd.Series) -> np.ndarray:
    # Keep the weighting convention used in the original script/SI.
    return np.where(delta.between(1.8, 2.2, inclusive="both"), 0.5, 1.0).astype(float)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(int(k), len(y_true))
    if k <= 0:
        return np.nan
    top = np.argsort(scores)[::-1][:k]
    return float(np.mean(y_true[top]))


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    positives = int(np.sum(y_true))
    return np.nan if positives == 0 else float(np.sum(y_true[np.argsort(scores)[::-1][:k]]) / positives)


def ef_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    rate = float(np.mean(y_true))
    return np.nan if rate == 0 else precision_at_k(y_true, scores, k) / rate


def model_and_grid(name: str, seed: int):
    """Return the original model definitions and grids."""
    # sklearn.clone rejects some third-party estimators when a NumPy integer
    # is passed to their constructor (notably CatBoost random_seed).
    seed = int(seed)
    if name == "SVM":
        model = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)
        grid = {"clf__C": [0.1, 1, 10],
                "clf__gamma": ["scale", 0.03, 0.003]}
        scale = True
    elif name == "RF":
        model = RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1)
        grid = {"clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 5, 10],
                "clf__min_samples_split": [2, 5],
                "clf__min_samples_leaf": [1, 2],
                "clf__max_features": ["sqrt", "log2"]}
        scale = False
    elif name == "LR":
        model = LogisticRegression(penalty="l2", solver="liblinear", class_weight="balanced",
                                   random_state=seed, max_iter=5000)
        grid = {"clf__C": [0.1, 1, 10]}
        scale = True
    elif name == "MLP":
        model = MLPClassifier(random_state=seed, max_iter=2000)
        grid = {"clf__hidden_layer_sizes": [(32,), (64,), (64, 32)],
                "clf__alpha": [1e-3, 1e-2],
                "clf__learning_rate_init": [5e-4, 1e-3]}
        scale = True
    elif name == "XGBoost":
        model = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                              random_state=seed, n_jobs=-1)
        grid = {"clf__n_estimators": [150, 300],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.03, 0.1],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.8, 1.0],
                "clf__reg_lambda": [1.0, 2.0]}
        scale = False
    elif name == "CatBoost":
        model = CatBoostClassifier(loss_function="Logloss", eval_metric="PRAUC",
                                   verbose=0, random_seed=seed,
                                   allow_writing_files=False, thread_count=1)
        grid = {"clf__iterations": [200, 400],
                "clf__depth": [4, 6],
                "clf__learning_rate": [0.03, 0.1],
                "clf__l2_leaf_reg": [3, 7]}
        scale = False
    elif name == "LightGBM":
        model = LGBMClassifier(objective="binary", class_weight="balanced", random_state=seed,
                               n_jobs=-1, verbosity=-1)
        grid = {"clf__n_estimators": [100, 300],
                "clf__learning_rate": [0.03, 0.1],
                "clf__num_leaves": [15, 31],
                "clf__max_depth": [-1, 5],
                "clf__min_child_samples": [10, 20]}
        scale = False
    elif name == "LR_ElasticNet":
        model = LogisticRegression(penalty="elasticnet", solver="saga", class_weight="balanced",
                                   random_state=seed, max_iter=8000)
        grid = {"clf__C": [0.1, 1, 10],
                "clf__l1_ratio": [0.2, 0.5, 0.8]}
        scale = True
    else:
        raise ValueError(f"Unsupported model: {name}")

    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", model))
    return Pipeline(steps), grid


def fit_grid(grid: GridSearchCV, name: str, x: pd.DataFrame, y: pd.Series,
             weights: np.ndarray) -> None:
    # MLP has no sample_weight; all other models retain the SI weighting rule.
    if name in WEIGHTED_MODELS:
        grid.fit(x, y, clf__sample_weight=weights)
    else:
        grid.fit(x, y)


def get_scores(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return model.predict(x).astype(float)


def main() -> None:
    args = parse_args()
    if args.cv_folds < 2:
        raise ValueError("cv-folds must be at least 2")
    k_list = [int(k.strip()) for k in args.k_list.split(",") if k.strip()]
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv(resolve_path(args.input))
    assignments = read_csv(resolve_path(args.split_assignments))
    l1_features = load_l1_features(resolve_path(args.l1_features))
    if args.target not in df.columns:
        raise ValueError(f"Missing target column: {args.target}")
    validate_assignments(assignments, len(df))

    y = (df[args.target] >= args.positive_threshold).astype(int)
    if "label" in assignments.columns:
        expected = y.iloc[assignments["row_index"].to_numpy()].to_numpy()
        if not np.array_equal(expected, assignments["label"].astype(int).to_numpy()):
            raise ValueError("Assignment labels do not match the input target.")

    metadata = {args.target, "SMILES", "name", "label"}
    descriptor_columns = [c for c in df.columns if c not in metadata and pd.api.types.is_numeric_dtype(df[c])]
    if not descriptor_columns:
        raise ValueError("No numeric descriptor columns found.")

    metric_rows: list[dict] = []
    screen_rows: list[dict] = []
    prediction_rows: list[dict] = []
    parameter_rows: list[dict] = []

    for iteration in sorted(assignments["iteration"].astype(int).unique()):
        split = assignments[assignments["iteration"] == iteration]
        train_idx = split.loc[split["split"] == "train", "row_index"].astype(int).to_numpy()
        test_idx = split.loc[split["split"] == "test", "row_index"].astype(int).to_numpy()
        features = l1_features.get(iteration)
        if not features:
            raise ValueError(f"No L1 feature list for iteration {iteration}")
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing L1 descriptors in input: {missing}")

        x_train = df.iloc[train_idx][features].copy()
        x_test = df.iloc[test_idx][features].copy()
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        d_train = df.iloc[train_idx][args.target]
        if y_train.value_counts().min() < args.cv_folds:
            raise ValueError(f"Iteration {iteration}: too few samples for {args.cv_folds}-fold CV")
        weights = boundary_weights(d_train)
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                             random_state=args.random_state + iteration)

        for name in MODEL_NAMES:
            print(f"[iteration {iteration:02d}] tuning {name}")
            pipeline, grid_values = model_and_grid(name, args.random_state + iteration)
            search = GridSearchCV(pipeline, grid_values, scoring="average_precision",
                                  cv=cv, n_jobs=args.n_jobs, refit=True, error_score="raise")
            fit_grid(search, name, x_train, y_train, weights)
            model = search.best_estimator_
            scores = get_scores(model, x_test)
            pred = model.predict(x_test)
            yt = y_test.to_numpy()

            metric_rows.append({
                "Model": name, "iteration": iteration,
                "random_state": int(split["random_state"].iloc[0]) if "random_state" in split else np.nan,
                "train_n": len(train_idx), "test_n": len(test_idx),
                "selected_feature_n": len(features),
                "inner_cv_PR-AUC": float(search.best_score_),
                "ROC-AUC": roc_auc_score(yt, scores),
                "PR-AUC": average_precision_score(yt, scores),
                "Accuracy": accuracy_score(yt, pred),
                "Precision": precision_score(yt, pred, zero_division=0),
                "Recall": recall_score(yt, pred, zero_division=0),
                "F1": f1_score(yt, pred, zero_division=0),
            })
            parameter_rows.append({
                "Model": name, "iteration": iteration,
                "inner_cv_PR-AUC": float(search.best_score_),
                "best_params": json.dumps(search.best_params_, ensure_ascii=False, default=str),
            })
            for row_index, true, score, label_pred in zip(test_idx, yt, scores, pred):
                prediction_rows.append({
                    "Model": name, "iteration": iteration, "row_index": int(row_index),
                    "y_true": int(true), "y_score": float(score), "y_pred": int(label_pred),
                    "delta_PCE": float(df.iloc[row_index][args.target]),
                })
            for k in k_list:
                screen_rows.append({
                    "Model": name, "iteration": iteration, "K": k,
                    "Precision@K": precision_at_k(yt, scores, k),
                    "Recall@K": recall_at_k(yt, scores, k),
                    "EF@K": ef_at_k(yt, scores, k),
                })

    metrics = pd.DataFrame(metric_rows)
    screening = pd.DataFrame(screen_rows)
    predictions = pd.DataFrame(prediction_rows)
    parameters = pd.DataFrame(parameter_rows)
    metrics.to_csv(output_dir / "eight_models_metrics_per_iteration.csv", index=False, encoding="utf-8-sig")
    screening.to_csv(output_dir / "eight_models_screening_metrics_per_iteration.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output_dir / "eight_models_test_predictions.csv", index=False, encoding="utf-8-sig")
    parameters.to_csv(output_dir / "eight_models_best_parameters_per_iteration.csv", index=False, encoding="utf-8-sig")

    base_metrics = metrics.groupby("Model")[["ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall", "F1"]].agg(["mean", "std"])
    base_metrics.columns = [f"{a}_{b}" for a, b in base_metrics.columns]
    summary = base_metrics.reset_index()
    for metric in ["Precision@K", "Recall@K", "EF@K"]:
        part = screening.pivot_table(index="Model", columns="K", values=metric, aggfunc=["mean", "std"])
        part.columns = [f"{metric}{k}_{stat}" for stat, k in part.columns]
        summary = summary.merge(part.reset_index(), on="Model", how="left")
    summary = summary.sort_values("PR-AUC_mean", ascending=False)

    summary.to_csv(output_dir / "eight_models_summary.csv", index=False, encoding="utf-8-sig")

    order = summary["Model"].tolist()
    plt.figure(figsize=(10, 6), dpi=300)
    plt.boxplot([metrics.loc[metrics["Model"] == m, "PR-AUC"] for m in order], labels=order, showfliers=False)
    plt.ylabel("PR-AUC")
    plt.xticks(rotation=30)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "eight_models_pr_auc_boxplot.png", bbox_inches="tight")
    plt.close()

    p3 = summary.set_index("Model").reindex(order)["Precision@K3_mean"]
    p5 = summary.set_index("Model").reindex(order)["Precision@K5_mean"]
    # Show the complete mean +/- SD to communicate variation across outer
    # splits. Mean +/- SD is descriptive and may extend beyond the physical
    # [0, 1] bounds even though every observed Precision value is bounded.
    s3 = summary.set_index("Model").reindex(order)["Precision@K3_std"].to_numpy()
    s5 = summary.set_index("Model").reindex(order)["Precision@K5_std"].to_numpy()
    p3 = p3.to_numpy()
    p5 = p5.to_numpy()
    x = np.arange(len(order))
    width = 0.36
    plt.figure(figsize=(10, 6), dpi=300)
    plt.bar(x - width / 2, p3, width, yerr=s3, capsize=3, label="Precision@3")
    plt.bar(x + width / 2, p5, width, yerr=s5, capsize=3, label="Precision@5")
    plt.xticks(x, order, rotation=30)
    plt.ylabel("Precision (mean +/- SD)")
    max_whisker = float(np.nanmax(np.concatenate([p3 + s3, p5 + s5])))
    y_upper = max(1.10, np.ceil((max_whisker + 0.03) * 20) / 20)
    plt.ylim(0, y_upper)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
               ncol=2, borderaxespad=0.0, frameon=False)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "eight_models_precision_at_k_bar.png", bbox_inches="tight")
    plt.close()

    print("Completed leakage-free eight-model comparison.")
    print(summary[["Model", "PR-AUC_mean", "Precision@K3_mean", "Precision@K5_mean"]].to_string(index=False))
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
