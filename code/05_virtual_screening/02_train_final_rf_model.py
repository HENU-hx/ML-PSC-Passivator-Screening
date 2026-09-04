#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tune and train the final Random Forest application model.

This script is run only after leakage-controlled outer evaluation and final
all-data feature selection have been completed. It performs two operations:

1. Tune RF hyperparameters by five-fold stratified CV on all 203 samples,
   using PR-AUC as the scoring metric and the same RF search space used in
   the leakage-controlled eight-model comparison.
2. Refit the best RF pipeline on all 203 samples and save it for SHAP,
   permutation importance, and molecular-library screening.

The optional outer-best-parameters file is summarized for stability/audit
purposes only. It does not directly determine the final parameter values.

Example (Windows cmd.exe, run from the project root):
  python code/05_virtual_screening/02_train_final_rf_model.py ^
      --input data/training_14descriptors.csv ^
      --outer-best-parameters outputs/outer_eight_model_comparison/eight_models_best_parameters_per_iteration.csv ^
      --output-dir outputs/final_rf_model
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IDENTIFIER_COLUMNS = {"name", "Name", "SMILES", "smiles", "label"}


def resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
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
        "--input",
        required=True,
        help="CSV containing all 203 samples, final descriptors, and delta_PCE.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--outer-best-parameters",
        default=None,
        help="Optional eight_models_best_parameters_per_iteration.csv.",
    )
    parser.add_argument("--target", default="delta_PCE")
    parser.add_argument("--positive-threshold", type=float, default=2.0)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--border-low", type=float, default=1.8)
    parser.add_argument("--border-high", type=float, default=2.2)
    parser.add_argument("--border-weight", type=float, default=0.5)
    return parser.parse_args()


def descriptor_columns(df: pd.DataFrame, target: str) -> list[str]:
    excluded = IDENTIFIER_COLUMNS | {target}
    features = [
        column
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not features:
        raise ValueError("No numeric descriptor columns were found.")
    return features


def boundary_weights(
    delta: pd.Series,
    low: float,
    high: float,
    boundary_weight: float,
) -> np.ndarray:
    weights = np.ones(len(delta), dtype=float)
    mask = delta.between(low, high, inclusive="both").to_numpy()
    weights[mask] = boundary_weight
    return weights


def clean_parameter_names(parameters: dict) -> dict:
    return {
        key.removeprefix("clf__"): value
        for key, value in parameters.items()
    }


def save_outer_parameter_summary(path: Path, output_dir: Path) -> None:
    table = read_csv(path)
    required = {"Model", "iteration", "inner_cv_PR-AUC", "best_params"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"Outer parameter file is missing columns: {sorted(missing)}"
        )

    rf = table.loc[table["Model"].eq("RF")].copy()
    if rf.empty:
        raise ValueError("No RF rows were found in the outer parameter file.")

    normalized = []
    for raw in rf["best_params"]:
        parameters = clean_parameter_names(json.loads(raw))
        normalized.append(json.dumps(parameters, sort_keys=True))

    counts = Counter(normalized)
    rows = []
    for rank, (parameter_json, count) in enumerate(counts.most_common(), start=1):
        rows.append(
            {
                "frequency_rank": rank,
                "count": count,
                "frequency": count / len(rf),
                "parameters": parameter_json,
            }
        )
    pd.DataFrame(rows).to_csv(
        output_dir / "rf_outer_parameter_frequency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rf[["iteration", "inner_cv_PR-AUC", "best_params"]].to_csv(
        output_dir / "rf_outer_best_parameters_per_iteration.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    outer_parameters_path = resolve_path(args.outer_best_parameters)
    assert input_path is not None and output_dir is not None

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2.")
    if not 0.0 < args.border_weight <= 1.0:
        raise ValueError("--border-weight must be in (0, 1].")
    if args.border_low > args.border_high:
        raise ValueError("--border-low cannot exceed --border-high.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if outer_parameters_path is not None:
        if not outer_parameters_path.exists():
            raise FileNotFoundError(outer_parameters_path)
        save_outer_parameter_summary(outer_parameters_path, output_dir)

    df = read_csv(input_path)
    df.columns = df.columns.astype(str).str.strip()
    if args.target not in df.columns:
        raise ValueError(f"Missing target column: {args.target}")

    features = descriptor_columns(df, args.target)
    x = df[features].copy()
    delta = pd.to_numeric(df[args.target], errors="raise")
    y = (delta >= args.positive_threshold).astype(int)
    if y.nunique() != 2:
        raise ValueError("The target must contain both classes.")
    if y.value_counts().min() < args.cv_folds:
        raise ValueError("The minority class is too small for the requested CV folds.")

    weights = boundary_weights(
        delta,
        args.border_low,
        args.border_high,
        args.border_weight,
    )

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=args.random_state,
                    n_jobs=args.n_jobs,
                ),
            ),
        ]
    )

    # Same reduced RF grid used in the leakage-controlled eight-model comparison.
    parameter_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf": [1, 2],
        "clf__max_features": ["sqrt", "log2"],
    }
    cv = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state,
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=args.n_jobs,
        refit=True,
        return_train_score=True,
        error_score="raise",
    )

    print("Tuning final RF on all available samples...")
    search.fit(x, y, clf__sample_weight=weights)

    # GridSearchCV(refit=True) has already refitted this pipeline on all rows.
    final_model = search.best_estimator_
    best_parameters = clean_parameter_names(search.best_params_)

    joblib.dump(final_model, output_dir / "rf_final_model.pkl")
    joblib.dump(features, output_dir / "feature_names.pkl")

    with (output_dir / "rf_final_best_parameters.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(best_parameters, handle, ensure_ascii=False, indent=2)

    pd.DataFrame(
        [
            {
                "model": "RF",
                "samples": len(df),
                "descriptor_n": len(features),
                "positive_n": int(y.sum()),
                "negative_n": int((1 - y).sum()),
                "cv_folds": args.cv_folds,
                "scoring": "PR-AUC",
                "best_cv_PR-AUC": float(search.best_score_),
                **best_parameters,
            }
        ]
    ).to_csv(
        output_dir / "rf_final_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cv_results = pd.DataFrame(search.cv_results_).sort_values(
        "rank_test_score", kind="stable"
    )
    cv_results.to_csv(
        output_dir / "rf_final_grid_search_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    imputer = final_model.named_steps["imputer"]
    pd.DataFrame(
        {"feature": features, "training_median": imputer.statistics_}
    ).to_csv(
        output_dir / "rf_final_imputation_parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {"feature_order": range(1, len(features) + 1), "feature": features}
    ).to_csv(
        output_dir / "rf_final_feature_order.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Samples: {len(df)}")
    print(f"Descriptors: {len(features)}")
    print(f"Best five-fold CV PR-AUC: {search.best_score_:.6f}")
    print(f"Best parameters: {best_parameters}")
    print(f"Final model saved to: {output_dir / 'rf_final_model.pkl'}")


if __name__ == "__main__":
    main()
