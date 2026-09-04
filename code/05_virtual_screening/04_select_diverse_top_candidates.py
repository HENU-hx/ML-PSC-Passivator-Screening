import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


# Expected location: <project_root>/code/05_virtual_screening/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def diversity_selection(
    df,
    smiles_col="SMILES",
    score_col="Score",
    threshold=0.6,
    max_select=5
):
    # Sort in descending order by score
    df_sorted = df.sort_values(
        by=score_col,
        ascending=False
    ).reset_index(drop=True)

    good_mols = []
    good_fps = []
    selected_indices = []

    for idx, row in df_sorted.iterrows():
        smiles = row[smiles_col]
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=1024
        )

        # Check similarity with already selected molecules
        keep = True

        for existing_fp in good_fps:
            similarity = DataStructs.TanimotoSimilarity(fp, existing_fp)

            if similarity > threshold:
                keep = False
                break

        if keep:
            selected_indices.append(idx)
            good_mols.append(mol)
            good_fps.append(fp)

            if len(selected_indices) >= max_select:
                break

    final_df = df_sorted.iloc[selected_indices]
    return final_df


# =========================
# Input/output paths
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--top20-output", required=True)
parser.add_argument("--diverse-output", required=True)
args = parser.parse_args()

input_path = resolve_project_path(args.input)
top20_output_path = resolve_project_path(args.top20_output)
diverse_output_path = resolve_project_path(args.diverse_output)

top20_output_path.parent.mkdir(parents=True, exist_ok=True)
diverse_output_path.parent.mkdir(parents=True, exist_ok=True)

# =========================
# 1) Read data including Score and AD
# =========================
df = pd.read_csv(input_path)

# =========================
# 2) AD priority
# =========================
df["AD_priority"] = df["AD_flag"].map({
    "Inside": 0,
    "Borderline": 1,
    "Outside": 2
})

# =========================
# 3) Sort by AD priority and Score
# =========================
df_sorted = df.sort_values(
    by=["AD_priority", "Score"],
    ascending=[True, False]
).reset_index(drop=True)

# =========================
# 4) Retain Inside candidates
# =========================
df_inside = df_sorted[df_sorted["AD_flag"] == "Inside"]

# =========================
# 5) Top 20 candidate pool
# =========================
top20 = df_inside.head(20)

top20.to_csv(top20_output_path, index=False)

# =========================
# 6) Diversity selection from Top 20
# =========================
final_candidates = diversity_selection(
    top20,
    smiles_col="SMILES",
    score_col="Score",
    threshold=0.6,
    max_select=5
)

final_candidates.to_csv(diverse_output_path, index=False)

print("Top 20 candidates:")
print(top20[["Compound_CID", "Score"]])

print("\nDiverse final candidates:")
print(final_candidates[["Compound_CID", "Score"]])

print(f"\nSaved Top 20 candidate pool: {top20_output_path}")
print(f"Saved diverse final candidates: {diverse_output_path}")