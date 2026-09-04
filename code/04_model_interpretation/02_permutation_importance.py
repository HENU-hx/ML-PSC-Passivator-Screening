#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Leakage-controlled permutation importance for the outer RF models.

For each saved outer split, this script rebuilds the RF model using only the
corresponding outer training set, its iteration-specific L1 features, and the
RF hyperparameters already selected by inner cross-validation. Permutation
importance is then calculated exclusively on the untouched outer test set
using PR-AUC. No hyperparameter tuning is repeated here.

Example (Windows cmd.exe, run from the project root):
  python code/04_model_interpretation/02_permutation_importance.py ^
      --input data/training_25descriptors_labeled.csv ^
      --split-assignments outputs/outer_repeated_splits/outer_split_assignments.csv ^
      --l1-features outputs/outer_train_l1_selection/l1_selected_features_by_iteration.csv ^
      --rf-best-parameters outputs/outer_eight_model_comparison/eight_models_best_parameters_per_iteration.csv ^
      --final-features outputs/final_rf_model/rf_final_feature_order.csv ^
      --output-dir outputs/outer_rf_permutation_importance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

# The analysis runs non-interactively, including on systems without Tk.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode CSV: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Labeled 25-descriptor CSV.")
    parser.add_argument("--split-assignments", required=True,
                        help="outer_split_assignments.csv")
    parser.add_argument("--l1-features", required=True,
                        help="l1_selected_features_by_iteration.csv")
    parser.add_argument("--rf-best-parameters", required=True,
                        help="eight_models_best_parameters_per_iteration.csv")
    parser.add_argument("--final-features", required=True,
                        help="rf_final_feature_order.csv; controls the SI plot.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="delta_PCE")
    parser.add_argument("--positive-threshold", type=float, default=2.0)
    parser.add_argument("--n-repeats", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--permutation-n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for permutation repeats; 1 avoids nested parallelism.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--border-low", type=float, default=1.8)
    parser.add_argument("--border-high", type=float, default=2.2)
    parser.add_argument("--border-weight", type=float, default=0.5)
    return parser.parse_args()


def validate_assignments(assignments: pd.DataFrame, n_rows: int) -> None:
    required = {"iteration", "row_index", "split"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Split file missing columns: {sorted(missing)}")
    if not assignments["split"].isin(["train", "test"]).all():
        raise ValueError("split must contain only 'train' and 'test'.")
    assignments["iteration"] = assignments["iteration"].astype(int)
    assignments["row_index"] = assignments["row_index"].astype(int)
    if not assignments["row_index"].between(0, n_rows - 1).all():
        raise ValueError("row_index is outside the input table.")
    if assignments.duplicated(["iteration", "row_index"]).any():
        raise ValueError("Duplicate row assignment within an iteration.")


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
        result[int(iteration)] = group["selected_feature"].astype(str).tolist()
    return result


def load_rf_parameters(path: Path) -> dict[int, dict]:
    table = read_csv(path)
    required = {"Model", "iteration", "best_params"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"RF parameter file missing columns: {sorted(missing)}")
    table = table.loc[table["Model"].eq("RF")].copy()
    if table.empty:
        raise ValueError("No RF rows were found in the parameter file.")

    result: dict[int, dict] = {}
    for row in table.itertuples(index=False):
        raw = json.loads(row.best_params)
        result[int(row.iteration)] = {
            key.removeprefix("clf__"): value for key, value in raw.items()
        }
    return result


def load_final_features(path: Path) -> list[str]:
    table = read_csv(path)
    if "feature" not in table.columns:
        raise ValueError("Final feature file must contain a 'feature' column.")
    if "feature_order" in table.columns:
        table = table.sort_values("feature_order")
    features = table["feature"].dropna().astype(str).tolist()
    if not features:
        raise ValueError("Final feature list is empty.")
    return features


def boundary_weights(delta: pd.Series, low: float, high: float,
                     boundary_weight: float) -> np.ndarray:
    weights = np.ones(len(delta), dtype=float)
    weights[delta.between(low, high, inclusive="both").to_numpy()] = boundary_weight
    return weights


def build_rf_pipeline(parameters: dict, seed: int, n_jobs: int) -> Pipeline:
    parameters = dict(parameters)
    parameters.update(
        class_weight="balanced",
        random_state=int(seed),
        n_jobs=n_jobs,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(**parameters)),
        ]
    )


def summarize_importance(per_iteration: pd.DataFrame,
                         all_iterations: list[int]) -> pd.DataFrame:
    summary = (
        per_iteration.groupby("feature", as_index=False)
        .agg(
            selected_iteration_n=("iteration", "nunique"),
            importance_mean=("importance_mean", "mean"),
            importance_sd=("importance_mean", "std"),
            median_importance=("importance_mean", "median"),
            positive_importance_fraction=("importance_mean", lambda x: float((x > 0).mean())),
        )
    )
    summary["importance_sd"] = summary["importance_sd"].fillna(0.0)
    summary["selection_frequency"] = (
        summary["selected_iteration_n"] / len(all_iterations)
    )
    summary["importance_rank"] = (
        summary["importance_mean"].rank(method="min", ascending=False).astype(int)
    )
    return summary.sort_values("importance_mean", ascending=False, kind="stable")


def save_plot(summary: pd.DataFrame, final_features: list[str],
              output_dir: Path) -> None:
    plot_data = (
        pd.DataFrame({"feature": final_features})
        .merge(summary, on="feature", how="left")
    )
    if plot_data["importance_mean"].isna().any():
        missing = plot_data.loc[plot_data["importance_mean"].isna(), "feature"].tolist()
        raise ValueError(
            "Final features were never selected in the outer models: " + ", ".join(missing)
        )
    # The SI figure is deliberately compact: show only the ten most important
    # final descriptors. SD remains available in the accompanying CSV table.
    plot_data = plot_data.nlargest(10, "importance_mean").sort_values(
        "importance_mean", ascending=True
    )

    y = np.arange(len(plot_data))
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=300)
    ax.barh(
        y,
        plot_data["importance_mean"],
        color="#1F77B4",
        edgecolor="none",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_data["feature"])
    ax.set_xlabel("Decrease in test PR-AUC after permutation")
    ax.tick_params(axis="both", labelsize=10)
    ax.xaxis.label.set_size(11)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"rf_outer_permutation_importance.{suffix}",
            dpi=600 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)

    plot_data.sort_values("importance_mean", ascending=False).to_csv(
        output_dir / "rf_outer_permutation_importance_final_features.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    args = parse_args()
    if args.n_repeats < 1:
        raise ValueError("--n-repeats must be at least 1.")
    if not 0.0 < args.border_weight <= 1.0:
        raise ValueError("--border-weight must be in (0, 1].")
    if args.border_low > args.border_high:
        raise ValueError("--border-low cannot exceed --border-high.")

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = read_csv(resolve_path(args.input))
    df.columns = df.columns.astype(str).str.strip()
    if args.target not in df.columns:
        raise ValueError(f"Missing target column: {args.target}")

    assignments = read_csv(resolve_path(args.split_assignments))
    validate_assignments(assignments, len(df))
    l1_features = load_l1_features(resolve_path(args.l1_features))
    rf_parameters = load_rf_parameters(resolve_path(args.rf_best_parameters))
    final_features = load_final_features(resolve_path(args.final_features))

    delta = pd.to_numeric(df[args.target], errors="raise")
    y = (delta >= args.positive_threshold).astype(int)
    if "label" in assignments.columns:
        expected = y.iloc[assignments["row_index"].to_numpy()].to_numpy()
        observed = assignments["label"].astype(int).to_numpy()
        if not np.array_equal(expected, observed):
            raise ValueError("Assignment labels do not match the input target.")

    iterations = sorted(assignments["iteration"].unique().astype(int).tolist())
    missing_l1 = sorted(set(iterations) - set(l1_features))
    missing_params = sorted(set(iterations) - set(rf_parameters))
    if missing_l1:
        raise ValueError(f"Missing L1 features for iterations: {missing_l1}")
    if missing_params:
        raise ValueError(f"Missing RF parameters for iterations: {missing_params}")

    iteration_rows: list[dict] = []
    repeat_rows: list[dict] = []
    model_rows: list[dict] = []

    for iteration in iterations:
        split = assignments.loc[assignments["iteration"].eq(iteration)]
        train_idx = split.loc[split["split"].eq("train"), "row_index"].to_numpy(dtype=int)
        test_idx = split.loc[split["split"].eq("test"), "row_index"].to_numpy(dtype=int)
        features = l1_features[iteration]
        missing = [feature for feature in features if feature not in df.columns]
        if missing:
            raise ValueError(f"Iteration {iteration}: missing descriptors {missing}")

        x_train = df.iloc[train_idx][features].copy()
        x_test = df.iloc[test_idx][features].copy()
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        weights = boundary_weights(
            delta.iloc[train_idx],
            args.border_low,
            args.border_high,
            args.border_weight,
        )
        model = build_rf_pipeline(
            rf_parameters[iteration],
            args.random_state + iteration,
            args.n_jobs,
        )
        model.fit(x_train, y_train, clf__sample_weight=weights)
        baseline = average_precision_score(y_test, model.predict_proba(x_test)[:, 1])

        result = permutation_importance(
            model,
            x_test,
            y_test,
            scoring="average_precision",
            n_repeats=args.n_repeats,
            random_state=args.random_state + iteration,
            n_jobs=args.permutation_n_jobs,
        )
        model_rows.append(
            {
                "iteration": iteration,
                "train_n": len(train_idx),
                "test_n": len(test_idx),
                "selected_feature_n": len(features),
                "test_PR-AUC": baseline,
                "rf_parameters": json.dumps(rf_parameters[iteration], sort_keys=True),
            }
        )
        for feature_index, feature in enumerate(features):
            values = result.importances[feature_index]
            iteration_rows.append(
                {
                    "iteration": iteration,
                    "feature": feature,
                    "test_PR-AUC": baseline,
                    "importance_mean": float(values.mean()),
                    "importance_sd_within_iteration": float(values.std(ddof=1))
                    if len(values) > 1 else 0.0,
                }
            )
            repeat_rows.extend(
                {
                    "iteration": iteration,
                    "feature": feature,
                    "permutation_repeat": repeat + 1,
                    "PR-AUC_decrease": float(value),
                }
                for repeat, value in enumerate(values)
            )
        print(
            f"[iteration {iteration:02d}] test PR-AUC={baseline:.3f}; "
            f"features={len(features)}"
        )

    per_iteration = pd.DataFrame(iteration_rows)
    repeats = pd.DataFrame(repeat_rows)
    model_summary = pd.DataFrame(model_rows)
    summary = summarize_importance(per_iteration, iterations)

    per_iteration.to_csv(
        output_dir / "rf_outer_permutation_importance_per_iteration.csv",
        index=False,
        encoding="utf-8-sig",
    )
    repeats.to_csv(
        output_dir / "rf_outer_permutation_importance_repeats.csv",
        index=False,
        encoding="utf-8-sig",
    )
    model_summary.to_csv(
        output_dir / "rf_outer_permutation_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        output_dir / "rf_outer_permutation_importance_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_plot(summary, final_features, output_dir)

    print("\nPermutation importance completed.")
    print(f"Outer iterations: {len(iterations)}")
    print(f"Permutation repeats per feature and iteration: {args.n_repeats}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
