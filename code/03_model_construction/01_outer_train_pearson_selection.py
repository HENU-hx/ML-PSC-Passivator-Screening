#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run Pearson descriptor filtering within each outer training split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


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
        description="Run Pearson filtering within outer training splits."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--split-assignments", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="delta_PCE")
    parser.add_argument("--positive-threshold", type=float, default=2.0)
    parser.add_argument("--correlation-threshold", type=float, default=0.80)
    return parser.parse_args()


def descriptor_columns(df: pd.DataFrame, target: str) -> list[str]:
    excluded = METADATA_COLUMNS | {target}
    columns = [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    if not columns:
        raise ValueError("No numeric descriptor columns were found.")
    return columns


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
        raise ValueError("A sample was assigned more than once within an iteration.")


def pearson_filter(
    x_train: pd.DataFrame,
    threshold: float,
) -> tuple[list[str], pd.DataFrame, pd.Series]:
    # Median values are fitted from this outer training set only.
    medians = x_train.median(numeric_only=True)
    if medians.isna().any():
        missing = medians.index[medians.isna()].tolist()
        raise ValueError(f"All values are missing for: {missing}")

    x_train = x_train.fillna(medians)
    corr_abs = x_train.corr(method="pearson").abs()
    upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))

    dropped_rows = []
    for column in upper.columns:
        correlated_with = upper.index[upper[column] > threshold].tolist()
        if correlated_with:
            dropped_rows.append(
                {
                    "dropped_feature": column,
                    "correlated_with": "; ".join(correlated_with),
                    "max_abs_pearson_r": float(upper[column].max()),
                }
            )

    dropped = [row["dropped_feature"] for row in dropped_rows]
    selected = [column for column in x_train.columns if column not in dropped]

    return selected, pd.DataFrame(dropped_rows), medians


def main() -> None:
    args = parse_args()

    input_path = resolve_project_path(args.input)
    assignments_path = resolve_project_path(args.split_assignments)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not 0.0 < args.correlation_threshold < 1.0:
        raise ValueError("correlation-threshold must be between 0 and 1")

    df = read_csv_with_fallback(input_path)
    assignments = read_csv_with_fallback(assignments_path)

    if args.target not in df.columns:
        raise ValueError(f"Missing target column: {args.target}")

    validate_assignments(assignments, len(df))

    # Verify that the saved split labels match delta_PCE in the input file.
    if "label" in assignments.columns:
        expected_labels = (
            df[args.target] >= args.positive_threshold
        ).astype(int).iloc[assignments["row_index"].to_numpy()]

        if not np.array_equal(
            assignments["label"].astype(int).to_numpy(),
            expected_labels.to_numpy(),
        ):
            raise ValueError(
                "Split-assignment labels do not match input delta_PCE labels."
            )

    features = descriptor_columns(df, args.target)

    selected_rows = []
    dropped_rows = []
    median_rows = []
    summary_rows = []
    selected_by_iteration = {}

    for iteration in sorted(assignments["iteration"].unique()):
        split_rows = assignments[assignments["iteration"] == iteration]

        train_indices = split_rows.loc[
            split_rows["split"] == "train", "row_index"
        ].to_numpy(dtype=int)

        test_indices = split_rows.loc[
            split_rows["split"] == "test", "row_index"
        ].to_numpy(dtype=int)

        if len(train_indices) == 0 or len(test_indices) == 0:
            raise ValueError(f"Iteration {iteration} has an empty split.")

        # Pearson uses only the 162 outer-training rows.
        x_train = df.iloc[train_indices][features].copy()

        selected, dropped, medians = pearson_filter(
            x_train,
            args.correlation_threshold,
        )

        selected_by_iteration[str(int(iteration))] = selected

        selected_rows.extend(
            {
                "iteration": int(iteration),
                "feature_order": order,
                "selected_feature": feature,
            }
            for order, feature in enumerate(selected, start=1)
        )

        if not dropped.empty:
            dropped.insert(0, "iteration", int(iteration))
            dropped_rows.extend(dropped.to_dict(orient="records"))

        median_rows.extend(
            {
                "iteration": int(iteration),
                "feature": feature,
                "training_median": float(value),
            }
            for feature, value in medians.items()
        )

        summary_rows.append(
            {
                "iteration": int(iteration),
                "train_n": len(train_indices),
                "test_n": len(test_indices),
                "input_feature_n": len(features),
                "selected_feature_n": len(selected),
                "dropped_feature_n": len(features) - len(selected),
                "correlation_threshold": args.correlation_threshold,
                "selected_features": "; ".join(selected),
            }
        )

    pd.DataFrame(selected_rows).to_csv(
        output_dir / "pearson_selected_features_by_iteration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(dropped_rows).to_csv(
        output_dir / "pearson_dropped_features_by_iteration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(median_rows).to_csv(
        output_dir / "pearson_training_medians_by_iteration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "pearson_selection_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        output_dir / "pearson_selected_features_by_iteration.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(selected_by_iteration, file, ensure_ascii=False, indent=2)

    counts = [len(features) for features in selected_by_iteration.values()]
    print(f"Completed Pearson filtering for {len(counts)} outer splits.")
    print(f"Input descriptor count: {len(features)}")
    print(
        f"Selected descriptor count: min={min(counts)}, "
        f"max={max(counts)}, mean={np.mean(counts):.2f}"
    )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()