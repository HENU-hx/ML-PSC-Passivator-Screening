import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem


# Expected location: <project_root>/code/05_virtual_screening/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ===== Input/output paths =====
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

input_file = resolve_project_path(args.input)
output_file = resolve_project_path(args.output)
output_file.parent.mkdir(parents=True, exist_ok=True)

MW_MAX = 300
MW_MIN = 80

# ===== read =====
df = pd.read_csv(input_file)

rows_to_drop = []
reasons = []

for i, row in df.iterrows():
    smiles = row["SMILES"]

    mol = Chem.MolFromSmiles(smiles)

    # 1. Invalid SMILES
    if mol is None:
        rows_to_drop.append(i)
        reasons.append("bad_smiles")
        continue

    # 2. Multicomponent
    if "." in smiles:
        rows_to_drop.append(i)
        reasons.append("multi_component")
        continue

    mw = row["MolecularWeight"]
    if mw > MW_MAX or mw < MW_MIN:
        rows_to_drop.append(i)
        reasons.append("MW_out_of_range")
        continue

# ===== delete =====
df_clean = df.drop(index=rows_to_drop)

df_clean.to_csv(output_file, index=False)

print(f"Original number of molecules: {len(df)}")
print(f"Number of molecules after screening: {len(df_clean)}")
print(f"Delete quantity: {len(rows_to_drop)}")

from collections import Counter

print("Deletion reason statistics:")
print(Counter(reasons))
print(f"Saved: {output_file}")