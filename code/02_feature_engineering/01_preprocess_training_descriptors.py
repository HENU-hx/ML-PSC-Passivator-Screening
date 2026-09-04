#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Filter incomplete charge-descriptor records from the training dataset.

This script implements the original 204-to-203 preprocessing rule:
1. Set N_O_MinCharge to 0 for molecules without N or O atoms.
2. Remove records with missing values in the seven required charge descriptors.

Example:
  python clean_training_descriptors.py ^
      --input passivation_layer_screening-data/processed/training_25descriptors.csv ^
      --output passivation_layer_screening-data/processed/training_25descriptors_clean.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem


# Expected location: <project_root>/code/02_feature_engineering/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
POSSIBLE_SMILES_COLS = [
    "SMILES",
    "smiles",
    "canonical_smiles",
    "IsomericSMILES",
    "PUBCHEM_SMILES",
]
REQUIRED_CHARGE_COLUMNS = [
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "StdPartialCharge",
    "ChargeRange",
    "Frac_NegativeAtoms",
    "PEOE_VSA1",
    "PEOE_VSA14",
]


def resolve_project_path(value: str) -> Path:
    """Resolve relative paths from the project root, not the working directory."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def find_smiles_column(df: pd.DataFrame) -> str:
    for column in POSSIBLE_SMILES_COLS:
        if column in df.columns:
            return column
    raise ValueError(
        "SMILES column not found. Expected one of: "
        f"{POSSIBLE_SMILES_COLS}; found: {list(df.columns)}"
    )


def has_n_or_o(smiles: object) -> bool:
    """Return whether a valid molecule contains at least one N or O atom."""
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return False
    return any(atom.GetSymbol() in {"N", "O"} for atom in mol.GetAtoms())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter training records with missing required charge descriptors."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input descriptor CSV path, relative to the project root or absolute.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output cleaned CSV path, relative to the project root or absolute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input)
    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    smiles_column = find_smiles_column(df)

    missing_columns = [
        column
        for column in ["N_O_MinCharge", *REQUIRED_CHARGE_COLUMNS]
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Required descriptor columns are missing: {missing_columns}")

    deleted_smiles = []
    rows_to_drop = []

    # Preserve the original preprocessing logic and its order of operations.
    for index, row in df.iterrows():
        smiles = row[smiles_column]

        if not has_n_or_o(smiles):
            df.at[index, "N_O_MinCharge"] = 0.0

        if row[REQUIRED_CHARGE_COLUMNS].isna().any():
            rows_to_drop.append(index)
            deleted_smiles.append(smiles)

    df_clean = df.drop(index=rows_to_drop).reset_index(drop=True)

    # Status columns are retained only in the raw descriptor output, not in model input.
    df_clean = df_clean.drop(columns=["rdkit_ok", "fail_reason"], errors="ignore")

    df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Input records: {len(df)}")
    print(f"Number of molecules deleted: {len(deleted_smiles)}")
    print(f"Output records: {len(df_clean)}")
    print(f"Saved: {output_path}")
    if deleted_smiles:
        print("Deleted SMILES:")
        for smiles in deleted_smiles:
            print(smiles)


if __name__ == "__main__":
    main()
