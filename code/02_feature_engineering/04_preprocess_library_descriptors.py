import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from rdkit import Chem


# Expected location: <project_root>/code/02_feature_engineering/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ======================
# Input/output paths
# ======================
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument(
    "--output",
    required=True,
    help="Output CSV after the original library preprocessing.",
)
parser.add_argument(
    "--model-output",
    required=True,
    help="Output CSV containing Compound_CID, SMILES, and the 14 final descriptors.",
)
args = parser.parse_args()

input_path = resolve_project_path(args.input)
output_path = resolve_project_path(args.output)
model_output_path = resolve_project_path(args.model_output)

output_path.parent.mkdir(parents=True, exist_ok=True)
model_output_path.parent.mkdir(parents=True, exist_ok=True)

# Read data
df = pd.read_csv(input_path)

deleted = []


def has_NO(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return any(atom.GetSymbol() in ["N", "O"] for atom in mol.GetAtoms())


rows_to_drop = []

# The descriptor currently in use
desc_cols = [
    "N_O_MinCharge",
    "Frac_NegativeAtoms",
    "PEOE_VSA1",
    "PEOE_VSA14",
    "N_coord",
    "MolecularWeight",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "LogP",
    "RingCount",
    "HeteroRing",
    "FSP3"
]

for i, row in df.iterrows():
    smiles = row["SMILES"]

    # SMILES itself has issues -> delete directly
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        rows_to_drop.append(i)
        deleted.append(smiles)
        continue

    # ---- N/O missing logic ----
    if not has_NO(smiles):
        df.at[i, "N_O_MinCharge"] = 0
        df.at[i, "Frac_NegativeAtoms"] = 0

    # ---- Missing value handling ----
    for col in desc_cols:
        if pd.isna(row[col]):
            # Classified processing
            if col in [
                "N_coord",
                "HBondDonorCount",
                "HBondAcceptorCount",
                "RotatableBondCount",
                "RingCount",
                "HeteroRing"
            ]:
                df.at[i, col] = 0
            else:
                # Fill continuous variables with median
                df[col].fillna(df[col].median(), inplace=True)

# Eliminate bad elements
df_clean = df.drop(index=rows_to_drop)

# Save the original preprocessing output
df_clean.to_csv(output_path, index=False)

# Extract the final screening input in the exact training-model feature order
model_input_columns = [
    "Compound_CID",
    "SMILES",
    "N_O_MinCharge",
    "Frac_NegativeAtoms",
    "PEOE_VSA1",
    "PEOE_VSA14",
    "N_coord",
    "coord_types",
    "MolecularWeight",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "LogP",
    "RingCount",
    "HeteroRing",
    "FSP3",
]

missing_columns = [
    column for column in model_input_columns
    if column not in df_clean.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing from the cleaned library: {missing_columns}"
    )

df_model_input = df_clean[model_input_columns].copy()
df_model_input.to_csv(model_output_path, index=False)

print(f"Number of deleted molecules: {len(deleted)}")
print(f"Final retention: {len(df_clean)}")
print(f"Saved cleaned library: {output_path}")
print(f"Saved screening input: {model_output_path}")