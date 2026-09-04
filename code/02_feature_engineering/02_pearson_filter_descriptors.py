#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pearson filtering for the final model, using all available labeled data.

This script belongs to the final application-model line. It is intentionally
different from the outer-train Pearson script: it uses all rows only after
the leakage-controlled performance evaluation has been completed.

Example (run from the project root):
  python code/02_feature_engineering/02_pearson_filter_descriptors.py ^
      --input data/training_25descriptors_labeled.csv ^
      --output-dir outputs/final_pearson_selection
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
    parser.add_argument("--input", required=True, help="Labeled 25-descriptor CSV.")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for final Pearson results."
    )
    parser.add_argument("--target", default="delta_PCE")
    parser.add_argument("--threshold", type=float, default=0.80)
    return parser.parse_args()


def descriptor_columns(df: pd.DataFrame, target: str) -> list[str]:
    excluded = set(NON_DESCRIPTOR_COLUMNS) | {target}
    candidates = [column for column in df.columns if column not in excluded]
    numeric = [column for column in candidates if pd.api.types.is_numeric_dtype(df[column])]
    if not numeric:
        raise ValueError("No numeric descriptor columns were found.")
    return numeric


def save_heatmap(correlation: pd.DataFrame, path: Path, title: str) -> None:
    # Match the original publication style: lower-triangular matrix with
    # numeric Pearson coefficients in every displayed cell.
    size = (12.0, 10.0) if len(correlation) > 20 else (10.0, 8.0)
    fig, ax = plt.subplots(figsize=size)
    mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)
    sns.heatmap(
        correlation,
        mask=mask,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        ax=ax,
    )
    ax.set_title(title, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = read_csv(input_path)
    df.columns = df.columns.astype(str).str.strip()
    features = descriptor_columns(df, args.target)
    x = df[features].copy()

    # Fit imputation values on all 203 rows because this is the final model line.
    medians = x.median(numeric_only=True)
    if medians.isna().any():
        missing = medians.index[medians.isna()].tolist()
        raise ValueError(f"Descriptors entirely missing from the input: {missing}")
    x_imputed = x.fillna(medians)
    correlation = x_imputed.corr(method="pearson")
    upper = correlation.abs().where(
        np.triu(np.ones(correlation.shape), k=1).astype(bool)
    )

    dropped_rows = []
    dropped = []
    for column in upper.columns:
        correlated = upper.index[upper[column] > args.threshold].tolist()
        if correlated:
            dropped.append(column)
            dropped_rows.append(
                {
                    "dropped_feature": column,
                    "correlated_with": "; ".join(correlated),
                    "max_abs_pearson_r": float(upper[column].max()),
                }
            )
    selected = [feature for feature in features if feature not in dropped]

    output_df = df.copy()
    output_df[selected] = x_imputed[selected]
    output_df = output_df[
        [column for column in df.columns if column in output_df.columns and
         (column not in features or column in selected)]
    ]
    output_df.to_csv(
        output_dir / "training_all203_pearson_selected.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(dropped_rows).to_csv(
        output_dir / "final_pearson_dropped_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {"feature_order": range(1, len(selected) + 1), "selected_feature": selected}
    ).to_csv(
        output_dir / "final_pearson_selected_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {"feature": medians.index, "training_median": medians.to_numpy()}
    ).to_csv(
        output_dir / "final_pearson_training_medians.csv",
        index=False,
        encoding="utf-8-sig",
    )
    correlation.to_csv(output_dir / "final_pearson_correlation_matrix.csv", encoding="utf-8-sig")
    pd.DataFrame(
        {
            "input_rows": [len(df)],
            "input_descriptor_n": [len(features)],
            "selected_descriptor_n": [len(selected)],
            "dropped_descriptor_n": [len(dropped)],
            "correlation_threshold": [args.threshold],
        }
    ).to_csv(output_dir / "final_pearson_summary.csv", index=False, encoding="utf-8-sig")

    save_heatmap(correlation, output_dir / "final_pearson_heatmap_before.png", "Initial descriptors")
    save_heatmap(
        correlation.loc[selected, selected],
        output_dir / "final_pearson_heatmap_after.png",
        "Pearson-retained descriptors",
    )
    print(f"Rows used: {len(df)}")
    print(f"Initial descriptors: {len(features)}")
    print(f"Pearson-retained descriptors: {len(selected)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
