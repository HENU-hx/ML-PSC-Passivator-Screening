#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SHAP analysis for the saved final RF Pipeline (no retraining)."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET = "delta_PCE"
THRESHOLD = 2.0


def resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    raise ValueError(f"Cannot decode {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--top-n-dependence", type=int, default=7)
    return p.parse_args()


def normalize_shap(values: object, n: int, m: int) -> np.ndarray:
    a = np.asarray(values[1] if isinstance(values, list) and len(values) == 2 else values)
    if a.ndim == 3:
        if a.shape == (n, m, 2):
            a = a[:, :, 1]
        elif a.shape == (2, n, m):
            a = a[1]
    if a.ndim == 2 and a.shape == (m, n):
        a = a.T
    if a.shape != (n, m):
        raise ValueError(f"Unexpected SHAP shape {a.shape}, expected {(n, m)}")
    return a


def expected_positive(explainer: shap.TreeExplainer) -> float:
    a = np.asarray(explainer.expected_value).reshape(-1)
    return float(a[1] if len(a) == 2 else a[0])


def save(fig: plt.Figure, stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.top_n_dependence < 1:
        raise ValueError("top-n-dependence must be positive")
    input_path = resolve(args.input)
    model_dir = resolve(args.model_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "rf_final_model.pkl"
    feature_path = model_dir / "feature_names.pkl"
    for path in (input_path, model_path, feature_path):
        if not path.exists():
            raise FileNotFoundError(path)

    df = read_csv(input_path)
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    model = joblib.load(model_path)
    features = list(joblib.load(feature_path))
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing final descriptors: {missing}")
    if not hasattr(model, "named_steps"):
        raise TypeError("Final model must be a Pipeline")
    steps = model.named_steps
    imputer = steps.get("imputer")
    classifier = steps.get("clf") or steps.get("rf")
    if imputer is None or classifier is None:
        raise ValueError("Pipeline must contain imputer and clf/rf steps")

    x_raw = df[features].copy()
    x = pd.DataFrame(imputer.transform(x_raw), columns=features, index=df.index)
    y = (pd.to_numeric(df[TARGET], errors="raise") >= THRESHOLD).astype(int)
    explainer = shap.TreeExplainer(classifier)
    shap_values = normalize_shap(explainer.shap_values(x), len(x), len(features))
    base_value = expected_positive(explainer)

    pd.DataFrame(shap_values, columns=features, index=df.index).to_csv(
        output_dir / "shap_values.csv", index_label="row_index", encoding="utf-8-sig"
    )
    importance = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "mean_shap": shap_values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance.to_csv(output_dir / "shap_feature_importance.csv", index=False, encoding="utf-8-sig")

    shap.summary_plot(shap_values, x, feature_names=features, max_display=len(features), show=False, plot_size=(8, 6.5))
    plt.gca().set_xlabel("SHAP value (impact on model output)")
    save(plt.gcf(), output_dir / "shap_summary_beeswarm")

    shap.summary_plot(shap_values, x, feature_names=features, plot_type="bar", max_display=len(features), show=False, plot_size=(8, 6.5))
    plt.gca().set_xlabel("mean(|SHAP value|)")
    save(plt.gcf(), output_dir / "shap_summary_bar")

    for feature in importance.feature.head(args.top_n_dependence):
        shap.dependence_plot(feature, shap_values, x, feature_names=features, interaction_index="auto", show=False)
        plt.gca().set_xlabel(feature)
        plt.gca().set_ylabel("SHAP value (impact on model output)")
        save(plt.gcf(), output_dir / f"shap_dependence_{feature}")

    probability = model.predict_proba(x_raw)[:, 1]
    position = int(np.argmax(probability))
    explanation = shap.Explanation(
        values=shap_values[position], base_values=base_value,
        data=x.iloc[position].to_numpy(), feature_names=features,
    )
    shap.plots.waterfall(explanation, max_display=len(features), show=False)
    save(plt.gcf(), output_dir / "shap_waterfall_highest_probability")

    pd.DataFrame({
        "row_index": df.index, "delta_PCE": df[TARGET], "label": y,
        "predicted_probability_positive": probability,
    }).to_csv(output_dir / "final_rf_predictions_for_shap.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "shap_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"model": str(model_path), "sample_n": len(df), "feature_n": len(features),
                   "expected_value_positive_class": base_value, "representative_row_index": int(df.index[position]),
                   "representative_positive_probability": float(probability[position]),
                   "top_features": importance.head(10).to_dict("records")}, f, ensure_ascii=False, indent=2)
    print(f"SHAP matrix: {shap_values.shape}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
