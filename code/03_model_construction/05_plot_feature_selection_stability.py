#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot feature-selection stability across repeated outer training splits.

This script does not rerun Pearson filtering or L1 selection. It summarizes
the existing iteration-level selection outputs and creates two Supporting
Information figures:
  1. Retention frequency of all initial descriptors after Pearson filtering.
  2. Retention frequency of Pearson-retained descriptors after L1 selection.

Example:
  python code/03_model_construction/05_plot_feature_selection_stability.py ^
      --training-data data/training_25descriptors_labeled.csv ^
      --pearson-selected outputs/outer_train_pearson_selection/pearson_selected_features_by_iteration.csv ^
      --l1-selected outputs/outer_train_l1_selection/l1_selected_features_by_iteration.csv ^
      --output-dir outputs/feature_selection_stability
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_COLUMNS = {"name", "SMILES", "label"}


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read CSV encoding: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-data",
        required=True,
        help="Labeled table containing the original 25 descriptor columns.",
    )
    parser.add_argument(
        "--pearson-selected",
        required=True,
        help="pearson_selected_features_by_iteration.csv",
    )
    parser.add_argument(
        "--l1-selected",
        required=True,
        help="l1_selected_features_by_iteration.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for SI figures and frequency tables.",
    )
    parser.add_argument("--target", default="delta_PCE")
    return parser.parse_args()


def descriptor_columns(data: pd.DataFrame, target: str) -> list[str]:
    excluded = DEFAULT_METADATA_COLUMNS | {target}
    columns = [column for column in data.columns if column not in excluded]
    if not columns:
        raise ValueError("No descriptor columns were found in the training table.")
    return columns


def validate_selection_table(table: pd.DataFrame, label: str) -> None:
    required = {"iteration", "selected_feature"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{label} selection file is missing columns: {sorted(missing)}")
    if table[["iteration", "selected_feature"]].isna().any().any():
        raise ValueError(f"{label} selection file contains missing iteration or feature values.")
    if table.duplicated(["iteration", "selected_feature"]).any():
        raise ValueError(f"{label} selection file repeats a feature within an iteration.")


def frequency_table(
    all_features: list[str], selected: pd.DataFrame, n_iterations: int
) -> pd.DataFrame:
    counts = selected.groupby("selected_feature")["iteration"].nunique()
    return pd.DataFrame(
        {
            "descriptor": all_features,
            "retained_iterations": [int(counts.get(feature, 0)) for feature in all_features],
        }
    ).assign(
        total_iterations=n_iterations,
        retention_frequency=lambda frame: frame["retained_iterations"] / n_iterations,
        retention_percentage=lambda frame: 100.0 * frame["retention_frequency"],
    )


def plot_frequency(
    summary: pd.DataFrame,
    color: str,
    output_path: Path,
) -> None:
    # Descending order makes the stability pattern readable without changing
    # the underlying descriptor order in the exported CSV table.
    plot_data = summary.sort_values(
        ["retention_frequency", "descriptor"], ascending=[True, True]
    ).reset_index(drop=True)

    height = max(5.0, 0.31 * len(plot_data) + 1.4)
    fig, ax = plt.subplots(figsize=(7.2, height))
    bars = ax.barh(
        plot_data["descriptor"],
        plot_data["retention_frequency"],
        color=color,
        edgecolor="white",
        linewidth=0.7,
    )

    ax.set_xlim(0.0, 1.08)
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xticklabels([f"{int(value * 100)}%" for value in np.arange(0.0, 1.01, 0.2)])
    ax.set_xlabel("Descriptor retention frequency across 50 outer training partitions")
    ax.set_ylabel("Molecular descriptor")
    ax.xaxis.grid(True, color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, count in zip(bars, plot_data["retained_iterations"]):
        ax.text(
            min(bar.get_width() + 0.018, 1.025),
            bar.get_y() + bar.get_height() / 2,
            f"{count}/50",
            va="center",
            ha="left",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    training_path = resolve_path(args.training_data)
    pearson_path = resolve_path(args.pearson_selected)
    l1_path = resolve_path(args.l1_selected)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (training_path, pearson_path, l1_path):
        if not path.exists():
            raise FileNotFoundError(path)

    training_data = read_csv(training_path)
    pearson_selected = read_csv(pearson_path)
    l1_selected = read_csv(l1_path)
    validate_selection_table(pearson_selected, "Pearson")
    validate_selection_table(l1_selected, "L1")

    initial_features = descriptor_columns(training_data, args.target)
    pearson_iterations = set(pearson_selected["iteration"].astype(int))
    l1_iterations = set(l1_selected["iteration"].astype(int))
    if pearson_iterations != l1_iterations:
        raise ValueError("Pearson and L1 results do not contain the same iterations.")
    n_iterations = len(pearson_iterations)
    if n_iterations < 2:
        raise ValueError("At least two iterations are required to calculate stability.")

    unknown_pearson = set(pearson_selected["selected_feature"]) - set(initial_features)
    unknown_l1 = set(l1_selected["selected_feature"]) - set(initial_features)
    if unknown_pearson or unknown_l1:
        unknown = sorted(unknown_pearson | unknown_l1)
        raise ValueError(f"Selection results contain unknown descriptors: {unknown}")

    pearson_summary = frequency_table(initial_features, pearson_selected, n_iterations)
    pearson_features = pearson_summary.loc[
        pearson_summary["retained_iterations"] > 0, "descriptor"
    ].tolist()
    l1_summary = frequency_table(pearson_features, l1_selected, n_iterations)

    pearson_summary.to_csv(
        output_dir / "pearson_descriptor_retention_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )
    l1_summary.to_csv(
        output_dir / "l1_descriptor_retention_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_frequency(
        pearson_summary,
        "#4C78A8",
        output_dir / "Figure_Sx_Pearson_descriptor_retention_frequency",
    )
    plot_frequency(
        l1_summary,
        "#F58518",
        output_dir / "Figure_Sy_L1_descriptor_retention_frequency",
    )

    print(f"Iterations summarized: {n_iterations}")
    print(f"Initial descriptors: {len(initial_features)}")
    print(f"Pearson-retained descriptors: {len(pearson_features)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
