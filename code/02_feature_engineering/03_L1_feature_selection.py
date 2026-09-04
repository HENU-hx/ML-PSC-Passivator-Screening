#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L1 feature selection for the final model, using all available data.

The input must be the output of the final all-data Pearson script. This script
does not perform outer evaluation and must not be used to report test metrics.

Example:
  python code/02_feature_engineering/03_L1_feature_selection.py ^
      --input outputs/final_pearson_selection/training_all203_pearson_selected.csv ^
      --output-dir outputs/final_l1_selection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IDENTIFIER_COLUMNS = {"name", "Name", "SMILES", "smiles"}
NON_DESCRIPTOR_COLUMNS = IDENTIFIER_COLUMNS | {"delta_PCE", "label"}


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read CSV: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="All-data Pearson-selected CSV.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="delta_PCE")
    parser.add_argument("--positive-threshold", type=float, default=2.0)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def descriptor_columns(df: pd.DataFrame, target: str) -> list[str]:
    excluded = set(NON_DESCRIPTOR_COLUMNS) | {target}
    features = [
        column for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not features:
        raise ValueError("No numeric descriptors found in Pearson output.")
    return features


def main() -> None:
    args = parse_args()
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2.")
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = read_csv(input_path)
    df.columns = df.columns.astype(str).str.strip()
    if args.target not in df.columns:
        raise ValueError(f"Missing target column: {args.target}")

    features = descriptor_columns(df, args.target)
    x = df[features].copy()
    y = (pd.to_numeric(df[args.target], errors="raise") >= args.positive_threshold).astype(int)
    if y.nunique() < 2:
        raise ValueError("The target produces only one class.")
    if y.value_counts().min() < args.cv_folds:
        raise ValueError("The minority class is too small for the requested CV folds.")

    imputer = SimpleImputer(strategy="median")
    x_imputed = imputer.fit_transform(x)
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
    model.fit(x_scaled, y)
    coefficients = model.coef_[0]
    selected = [feature for feature, coef in zip(features, coefficients) if coef != 0]
    if not selected:
        raise ValueError("L1 selected zero descriptors; inspect regularization or data.")

    # Preserve metadata, target, label, and only the selected descriptor columns.
    keep = [column for column in df.columns if column not in features] + selected
    output_df = df[keep].copy()
    output_df.to_csv(
        output_dir / "training_all203_l1_selected.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {
            "feature_order": range(1, len(features) + 1),
            "feature": features,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
            "selected": [coef != 0 for coef in coefficients],
        }
    ).to_csv(output_dir / "final_l1_coefficients.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {"feature_order": range(1, len(selected) + 1), "selected_feature": selected}
    ).to_csv(output_dir / "final_l1_selected_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {"feature": features, "training_median": imputer.statistics_, "training_scale": scaler.scale_}
    ).to_csv(output_dir / "final_l1_preprocessing_parameters.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "final_l1_selected_features.json").open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, ensure_ascii=False, indent=2)
    pd.DataFrame(
        {
            "input_rows": [len(df)],
            "input_descriptor_n": [len(features)],
            "selected_descriptor_n": [len(selected)],
            "best_C": [float(model.C_[0])],
            "positive_n": [int(y.sum())],
            "negative_n": [int((1 - y).sum())],
        }
    ).to_csv(output_dir / "final_l1_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2C7FB8" if value >= 0 else "#D95F0E" for value in coefficients]
    ax.bar(features, coefficients, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("L1-logistic-regression coefficient")
    ax.set_xlabel("Pearson-retained descriptor")
    ax.tick_params(axis="x", rotation=75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "final_l1_coefficients.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "final_l1_coefficients.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Rows used: {len(df)}")
    print(f"Pearson input descriptors: {len(features)}")
    print(f"Final L1 descriptors: {len(selected)}")
    print(f"Best C: {model.C_[0]:.6g}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
