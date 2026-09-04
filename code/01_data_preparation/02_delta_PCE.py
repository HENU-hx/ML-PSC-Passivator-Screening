# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# Expected script location:
# <project_root>/code/03_model_construction/01_delta_PCE.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value):
    """Resolve relative paths from the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ==============================
# Input and output paths
# ==============================
parser = argparse.ArgumentParser(
    description="Plot the distribution of delta_PCE values."
)
parser.add_argument(
    "--input",
    required=True,
    help="Input CSV path, relative to the project root or absolute.",
)
parser.add_argument(
    "--output-dir",
    required=True,
    help="Output directory, relative to the project root or absolute.",
)
args = parser.parse_args()

INPUT_CSV = resolve_project_path(args.input)
OUTPUT_DIR = resolve_project_path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUTPUT_DIR / "Figure_Sx_deltaPCE_distribution.png"
OUTPUT_TIFF = OUTPUT_DIR / "Figure_Sx_deltaPCE_distribution.tiff"


# ==============================
# Journal-style figure settings
# ==============================
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)


# ==============================
# Load data
# ==============================
df = pd.read_csv(INPUT_CSV)

if "delta_PCE" not in df.columns:
    raise ValueError(
        f"Column 'delta_PCE' was not found in: {INPUT_CSV}"
    )

delta_pce = pd.to_numeric(
    df["delta_PCE"],
    errors="coerce",
).dropna()

threshold = 2.0


# ==============================
# Prepare histogram data
# ==============================
# A 0.5-percentage-point bin width makes the 2.0 threshold
# a bin boundary.
bin_edges = np.arange(-5.5, 5.51, 0.5)

below_threshold = delta_pce[delta_pce < threshold]
at_or_above_threshold = delta_pce[delta_pce >= threshold]


# ==============================
# Plot
# ==============================
fig, ax = plt.subplots(figsize=(3.6, 2.8))

ax.hist(
    below_threshold,
    bins=bin_edges,
    color="#A6A6A6",
    edgecolor="white",
    linewidth=0.6,
)

ax.hist(
    at_or_above_threshold,
    bins=bin_edges,
    color="#1F77B4",
    edgecolor="white",
    linewidth=0.6,
)

ax.axvline(
    threshold,
    color="#D62728",
    linestyle="--",
    linewidth=1.2,
)

ax.set_xlabel(r"$\Delta$PCE (percentage points)")
ax.set_ylabel("Count")
ax.set_xlim(-5.5, 5.0)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_handles = [
    Patch(
        facecolor="#A6A6A6",
        edgecolor="none",
        label=rf"$\Delta$PCE < 2.0 (n = {len(below_threshold)})",
    ),
    Patch(
        facecolor="#1F77B4",
        edgecolor="none",
        label=rf"$\Delta$PCE $\geq$ 2.0 "
        rf"(n = {len(at_or_above_threshold)})",
    ),
    Line2D(
        [0],
        [0],
        color="#D62728",
        linestyle="--",
        linewidth=1.2,
        label="Threshold = 2.0",
    ),
]

ax.legend(
    handles=legend_handles,
    frameon=False,
    loc="upper left",
    fontsize=8,
    handlelength=1.5,
)

fig.tight_layout(pad=0.4)

fig.savefig(
    OUTPUT_PNG,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    OUTPUT_TIFF,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print(f"Saved: {OUTPUT_PNG}")
print(f"Saved: {OUTPUT_TIFF}")