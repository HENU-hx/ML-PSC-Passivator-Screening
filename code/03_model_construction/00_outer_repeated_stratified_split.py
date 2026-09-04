#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate repeated outer stratified train/test splits.

This script performs only the outer data split. It does not calculate
descriptors, select features, tune hyperparameters, or train a model.
Later preprocessing and model-selection steps must be fitted using the
training rows of each saved split only.

Example:
  python code/03_model_construction/00_outer_repeated_stratified_split.py ^
      --input data/processed/training_25descriptors_clean.csv ^
      --output-dir outputs/outer_repeated_splits
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


# Expected location: <project_root>/code/03_model_construction/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate repeated outer stratified train/test splits."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Cleaned training CSV containing descriptors and delta_PCE.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for split assignments and summary files.",
    )
    parser.add_argument("--target", default="delta_PCE")
    parser.add_argument("--positive-threshold", type=float, default=2.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if args.target not in df.columns:
        raise ValueError(f"Missing target column: {args.target}")
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("test-size must be between 0 and 1")

    labels = (df[args.target] >= args.positive_threshold).astype(int)
    splitter = StratifiedShuffleSplit(
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    assignment_rows = []
    summary_rows = []
    all_indices = df.index.to_numpy()

    for iteration, (train_pos, test_pos) in enumerate(
        splitter.split(all_indices, labels),
        start=1,
    ):
        train_indices = all_indices[train_pos]
        test_indices = all_indices[test_pos]

        for row_index in train_indices:
            assignment_rows.append(
                {
                    "iteration": iteration,
                    "random_state": args.random_state,
                    "row_index": int(row_index),
                    "split": "train",
                    "label": int(labels.iloc[row_index]),
                }
            )
        for row_index in test_indices:
            assignment_rows.append(
                {
                    "iteration": iteration,
                    "random_state": args.random_state,
                    "row_index": int(row_index),
                    "split": "test",
                    "label": int(labels.iloc[row_index]),
                }
            )

        summary_rows.append(
            {
                "iteration": iteration,
                "random_state": args.random_state,
                "train_n": len(train_indices),
                "test_n": len(test_indices),
                "train_positive_n": int(labels.iloc[train_indices].sum()),
                "test_positive_n": int(labels.iloc[test_indices].sum()),
                "train_positive_rate": float(labels.iloc[train_indices].mean()),
                "test_positive_rate": float(labels.iloc[test_indices].mean()),
            }
        )

    assignments = pd.DataFrame(assignment_rows)
    summary = pd.DataFrame(summary_rows)

    assignments.to_csv(
        output_dir / "outer_split_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        output_dir / "outer_split_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Generated {args.n_splits} outer stratified splits.")
    print(f"Assignments: {output_dir / 'outer_split_assignments.csv'}")
    print(f"Summary: {output_dir / 'outer_split_summary.csv'}")


if __name__ == "__main__":
    main()
