#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract RF results and generate RF publication/supplementary figures.

This script does not retrain the model. It reads the RF rows from the saved
leakage-free eight-model outer-test results and generates:
  1. RF PR-AUC, ROC-AUC, and F1 bar plot (mean +/- SD).
  2. RF Precision@K curve (mean +/- SD) with the random positive-rate baseline.
  3. RF EF@K curve (mean +/- SD) with the random baseline EF=1.
  4. Combined Precision@K, Recall@K, and EF@K curve for the
     Supporting Information (mean +/- SD; no baseline lines).

Example:
  python code/03_model_construction/04_plot_rf_repeated_evaluation.py ^
      --input-dir outputs/outer_eight_model_comparison ^
      --output-dir outputs/rf_paper_figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RF_NAME = "RF"


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing the eight-model result CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for RF figures and RF summary tables.",
    )
    return parser.parse_args()


def summarize(values: pd.Series) -> tuple[float, float, int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan, np.nan, 0
    return float(numeric.mean()), float(numeric.std(ddof=1)), len(numeric)


def upper_axis_limit(values: np.ndarray, minimum: float, step: float) -> float:
    maximum = float(np.nanmax(values))
    return max(minimum, np.ceil((maximum + 0.03) / step) * step)


def main() -> None:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = input_dir / "eight_models_metrics_per_iteration.csv"
    screening_path = input_dir / "eight_models_screening_metrics_per_iteration.csv"
    predictions_path = input_dir / "eight_models_test_predictions.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not screening_path.exists():
        raise FileNotFoundError(screening_path)
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)

    metrics = pd.read_csv(metrics_path)
    screening = pd.read_csv(screening_path)
    predictions = pd.read_csv(predictions_path)
    for table_name, table in [("metrics", metrics), ("screening", screening)]:
        if "Model" not in table.columns:
            raise ValueError(f"The {table_name} file has no 'Model' column.")

    rf_metrics = metrics.loc[metrics["Model"].eq(RF_NAME)].copy()
    rf_screening = screening.loc[screening["Model"].eq(RF_NAME)].copy()
    rf_predictions = predictions.loc[predictions["Model"].eq(RF_NAME)].copy()
    if rf_metrics.empty:
        raise ValueError("No RF rows found in eight_models_metrics_per_iteration.csv")
    if rf_screening.empty:
        raise ValueError("No RF rows found in eight_models_screening_metrics_per_iteration.csv")
    if rf_predictions.empty:
        raise ValueError("No RF rows found in eight_models_test_predictions.csv")
    if "y_true" not in rf_predictions.columns:
        raise ValueError("eight_models_test_predictions.csv has no 'y_true' column")

    # Each outer split has its own test-set positive rate. Their mean is the
    # expected Precision@K under random ranking and the denominator of EF@K.
    random_precision = float(
        rf_predictions.groupby("iteration")["y_true"].mean().mean()
    )

    main_metric_names = ["PR-AUC", "ROC-AUC", "F1"]
    missing = [name for name in main_metric_names if name not in rf_metrics.columns]
    if missing:
        raise ValueError(f"Missing RF metrics: {missing}")

    main_rows = []
    for name in main_metric_names:
        mean, std, n = summarize(rf_metrics[name])
        main_rows.append({"metric": name, "mean": mean, "std": std, "n": n})
    main_summary = pd.DataFrame(main_rows)
    main_summary.to_csv(
        output_dir / "rf_main_metrics_mean_sd.csv",
        index=False,
        encoding="utf-8-sig",
    )

    required_screening = {"K", "Precision@K", "Recall@K", "EF@K"}
    missing = required_screening - set(rf_screening.columns)
    if missing:
        raise ValueError(f"Missing RF screening columns: {sorted(missing)}")

    screen_rows = []
    for k, group in rf_screening.groupby("K", sort=True):
        precision_mean, precision_std, precision_n = summarize(group["Precision@K"])
        recall_mean, recall_std, recall_n = summarize(group["Recall@K"])
        ef_mean, ef_std, ef_n = summarize(group["EF@K"])
        screen_rows.append(
            {
                "K": int(k),
                "Precision@K_mean": precision_mean,
                "Precision@K_std": precision_std,
                "Precision@K_n": precision_n,
                "Recall@K_mean": recall_mean,
                "Recall@K_std": recall_std,
                "Recall@K_n": recall_n,
                "EF@K_mean": ef_mean,
                "EF@K_std": ef_std,
                "EF@K_n": ef_n,
            }
        )
    screen_summary = pd.DataFrame(screen_rows).sort_values("K")
    screen_summary.to_csv(
        output_dir / "rf_screening_metrics_mean_sd.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            {
                "random_precision_baseline": random_precision,
                "random_ef_baseline": 1.0,
                "theoretical_ef_max": 1.0 / random_precision,
            }
        ]
    ).to_csv(
        output_dir / "rf_random_baselines.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    # Figure 1: RF main performance metrics.
    labels = main_summary["metric"].tolist()
    means = main_summary["mean"].to_numpy()
    stds = main_summary["std"].to_numpy()
    x = np.arange(len(labels))
    y_upper = upper_axis_limit(means + stds, minimum=0.90, step=0.05)

    plt.figure(figsize=(5.2, 4.6), dpi=300)
    bars = plt.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        width=0.58,
        color="#2878b5",
        edgecolor="black",
        linewidth=0.6,
        error_kw={"ecolor": "black", "lw": 1.1},
    )
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Metric value (mean +/- SD)")
    plt.ylim(0.0, y_upper)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, mean, std in zip(bars, means, stds):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + y_upper * 0.015,
            f"{mean:.3f}+/-{std:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(output_dir / "rf_main_metrics_mean_sd.png", bbox_inches="tight")
    plt.close()

    # Figure 2: RF Precision@K.
    k_values = screen_summary["K"].to_numpy()
    precision_mean = screen_summary["Precision@K_mean"].to_numpy()
    precision_std = screen_summary["Precision@K_std"].to_numpy()
    precision_upper = upper_axis_limit(
        precision_mean + precision_std, minimum=1.10, step=0.05
    )

    plt.figure(figsize=(6.0, 4.8), dpi=300)
    plt.errorbar(
        k_values,
        precision_mean,
        yerr=precision_std,
        fmt="-o",
        color="#2878b5",
        capsize=4,
        linewidth=1.6,
        markersize=5,
        label="RF",
    )
    plt.axhline(
        random_precision,
        color="#666666",
        linestyle="--",
        linewidth=1.2,
        label=f"Random baseline ({random_precision:.3f})",
    )
    plt.xticks(k_values, [str(int(value)) for value in k_values])
    plt.xlabel("K")
    plt.ylabel("Precision@K (mean +/- SD)")
    plt.ylim(0.0, precision_upper)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / "rf_precision_at_k_mean_sd.png", bbox_inches="tight")
    plt.close()

    # Figure 3: RF EF@K.
    ef_mean = screen_summary["EF@K_mean"].to_numpy()
    ef_std = screen_summary["EF@K_std"].to_numpy()
    ef_upper = upper_axis_limit(ef_mean + ef_std, minimum=2.20, step=0.10)

    plt.figure(figsize=(6.0, 4.8), dpi=300)
    plt.errorbar(
        k_values,
        ef_mean,
        yerr=ef_std,
        fmt="-o",
        color="#2878b5",
        capsize=4,
        linewidth=1.6,
        markersize=5,
        label="RF",
    )
    plt.axhline(
        1.0,
        color="#666666",
        linestyle="--",
        linewidth=1.2,
        label="Random baseline (EF=1)",
    )
    plt.xticks(k_values, [str(int(value)) for value in k_values])
    plt.xlabel("K")
    plt.ylabel("EF@K (mean +/- SD)")
    plt.ylim(0.0, ef_upper)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / "rf_ef_at_k_mean_sd.png", bbox_inches="tight")
    plt.close()

    # Supplementary figure: combined screening behavior without baselines.
    # Precision/Recall are proportions, whereas EF is a relative enrichment
    # factor; the shared axis is intended to show the K-dependent trends,
    # not to compare the absolute magnitudes of the three metrics.
    recall_mean = screen_summary["Recall@K_mean"].to_numpy()
    recall_std = screen_summary["Recall@K_std"].to_numpy()

    plt.figure(figsize=(6.0, 4.8), dpi=300)
    plt.errorbar(
        k_values,
        precision_mean,
        yerr=precision_std,
        fmt="-o",
        color="#2878b5",
        capsize=4,
        linewidth=1.6,
        markersize=5,
        label="Precision@K",
    )
    plt.errorbar(
        k_values,
        recall_mean,
        yerr=recall_std,
        fmt="-s",
        color="#ff7f0e",
        capsize=4,
        linewidth=1.6,
        markersize=5,
        label="Recall@K",
    )
    plt.errorbar(
        k_values,
        ef_mean,
        yerr=ef_std,
        fmt="-^",
        color="#2ca02c",
        capsize=4,
        linewidth=1.6,
        markersize=6,
        label="EF@K",
    )
    plt.xticks(k_values, [str(int(value)) for value in k_values])
    plt.xlabel("K")
    plt.ylabel("Metric value")
    combined_upper = upper_axis_limit(
        np.concatenate(
            [
                precision_mean + precision_std,
                recall_mean + recall_std,
                ef_mean + ef_std,
            ]
        ),
        minimum=2.20,
        step=0.10,
    )
    plt.ylim(0.0, combined_upper)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(
        output_dir / "rf_screening_overview_mean_sd.png", bbox_inches="tight"
    )
    plt.close()

    print("RF figures generated without retraining:")
    print(output_dir / "rf_main_metrics_mean_sd.png")
    print(output_dir / "rf_precision_at_k_mean_sd.png")
    print(output_dir / "rf_ef_at_k_mean_sd.png")
    print(output_dir / "rf_screening_overview_mean_sd.png")


if __name__ == "__main__":
    main()
