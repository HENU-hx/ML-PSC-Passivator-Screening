#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run L1 feature selection within each outer training split.

Inputs:
  1. The labeled 25-descriptor training table.
  2. The saved outer train/test assignments.
  3. Pearson-selected features for each outer iteration.

The outer test rows are never used for imputation, scaling, L1 fitting, or
feature selection. This script does not train the final RF model.

Example:
  python code/03_model_construction/02_outer_train_l1_feature_selection.py ^
      --input data/training_25descriptors_labeled.csv ^
      --split-assignments outputs/outer_repeated_splits/outer_split_assignments.csv ^
      --pearson-features outputs/outer_train_pearson_selection/pearson_selected_features_by_iteration.csv ^
      --output-dir outputs/outer_train_l1_selection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_COLUMNS = {"name", "SMILES", "label"}


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read CSV encoding: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run L1 feature selection within outer training splits."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--split-assignments", required=True)
    parser.add_argument("--pearson-features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="delta_PCE")
    parser.add_argument("--positive-threshold", type=float, default=2.0)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def validate_assignments(assignments: pd.DataFrame, n_rows: int) -> None:
    required = {"iteration", "row_index", "split"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")

    assignments["row_index"] = assignments["row_index"].astype(int)
    if not assignments["split"].isin(["train", "test"]).all():
        raise ValueError("split must contain only train or test.")
    if not assignments["row_index"].between(0, n_rows - 1).all():
        raise ValueError("Split assignments contain invalid row_index values.")
    if assignments.duplicated(["iteration", "row_index"]).any():
        raise ValueError("A sample was assigned more than once in an iteration.")


def load_pearson_features(path: Path) -> dict[int, list[str]]:
    table = read_csv_with_fallback(path)
    required = {"iteration", "selected_feature"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing Pearson columns: {sorted(missing)}")

    result = {}
    for iteration, group in table.groupby("iteration", sort=True):
        ordered = group.sort_values("feature_order") if "feature_order" in group else group
        features = ordered["selected_feature"].astype(str).tolist()
        if not features:
            raise ValueError(f"No Pearson-selected features for iteration {iteration}")
        result[int(iteration)] = features
    return result


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input)
    assignments_path = resolve_project_path(args.split_assignments)
    pearson_path = resolve_project_path(args.pearson_features)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cv_folds < 2:
        raise ValueError("cv-folds must be at least 2")

    df = read_csv_with_fallback(input_path)
    assignments = read_csv_with_fallback(assignments_path)
    pearson_features = load_pearson_features(pearson_path)

    if args.target not in df.columns:
        raise ValueError(f"Missing target column: {args.target}")
    validate_assignments(assignments, len(df))

    labels = (df[args.target] >= args.positive_threshold).astype(int)
    if "label" in assignments.columns:
        expected = labels.iloc[assignments["row_index"].to_numpy()].to_numpy()
        actual = assignments["label"].astype(int).to_numpy()
        if not np.array_equal(expected, actual):
            raise ValueError("Assignment labels do not match input delta_PCE labels.")

    selected_rows = []
    coefficient_rows = []
    median_rows = []
    scale_rows = []
    summary_rows = []
    selected_by_iteration = {}

    for iteration in sorted(assignments["iteration"].unique()):
        iteration = int(iteration)
        if iteration not in pearson_features:
            raise ValueError(f"Missing Pearson features for iteration {iteration}")

        split_rows = assignments[assignments["iteration"] == iteration]
        train_indices = split_rows.loc[
            split_rows["split"] == "train", "row_index"
        ].to_numpy(dtype=int)
        features = pearson_features[iteration]

        missing_features = [feature for feature in features if feature not in df.columns]
        if missing_features:
            raise ValueError(
                f"Pearson features missing from input for iteration {iteration}: "
                f"{missing_features}"
            )

        x_train = df.iloc[train_indices][features].copy()
        y_train = labels.iloc[train_indices]

        if y_train.nunique() < 2:
            raise ValueError(f"Iteration {iteration} has only one training class.")
        if y_train.value_counts().min() < args.cv_folds:
            raise ValueError(
                f"cv-folds={args.cv_folds} is too large for iteration {iteration}."
            )

        # Fit imputation and scaling using outer training rows only.
        medians = x_train.median(numeric_only=True)
        if medians.isna().any():
            raise ValueError("At least one descriptor is entirely missing in training data.")
        x_imputed = x_train.fillna(medians)

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_imputed)

        model = LogisticRegressionCV(
            penalty="l1",
            solver="liblinear",
            Cs=np.logspace(-3, 1, 20),
            cv=args.cv_folds,
            scoring="average_precision",
            random_state=args.random_state,
            max_iter=5000,
        )
        model.fit(x_scaled, y_train)

        coefficients = model.coef_[0]
        selected = [
            feature
            for feature, coefficient in zip(features, coefficients)
            if coefficient != 0
        ]
        selected_by_iteration[str(iteration)] = selected

        for order, (feature, coefficient) in enumerate(zip(features, coefficients), start=1):
            coefficient_rows.append(
                {
                    "iteration": iteration,
                    "feature_order": order,
                    "feature": feature,
                    "coefficient": float(coefficient),
                    "abs_coefficient": float(abs(coefficient)),
                    "selected": bool(coefficient != 0),
                }
            )

        selected_rows.extend(
            {
                "iteration": iteration,
                "feature_order": order,
                "selected_feature": feature,
            }
            for order, feature in enumerate(selected, start=1)
        )
        median_rows.extend(
            {
                "iteration": iteration,
                "feature": feature,
                "training_median": float(value),
            }
            for feature, value in medians.items()
        )
        scale_rows.extend(
            {
                "iteration": iteration,
                "feature": feature,
                "training_mean": float(mean),
                "training_scale": float(scale),
            }
            for feature, mean, scale in zip(features, scaler.mean_, scaler.scale_)
        )
        summary_rows.append(
            {
                "iteration": iteration,
                "train_n": len(train_indices),
                "pearson_feature_n": len(features),
                "l1_selected_feature_n": len(selected),
                "best_C": float(model.C_[0]),
                "selected_features": "; ".join(selected),
            }
        )

    pd.DataFrame(selected_rows).to_csv(
        output_dir / "l1_selected_features_by_iteration.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(coefficient_rows).to_csv(
        output_dir / "l1_coefficients_by_iteration.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(median_rows).to_csv(
        output_dir / "l1_training_medians_by_iteration.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(scale_rows).to_csv(
        output_dir / "l1_training_scaling_by_iteration.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "l1_selection_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with open(output_dir / "l1_selected_features_by_iteration.json", "w", encoding="utf-8") as file:
        json.dump(selected_by_iteration, file, ensure_ascii=False, indent=2)

    counts = [len(features) for features in selected_by_iteration.values()]
    print(f"Completed L1 selection for {len(counts)} outer splits.")
    print(
        f"Selected feature count: min={min(counts)}, "
        f"max={max(counts)}, mean={np.mean(counts):.2f}"
    )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
