#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calculate the same 25 initial RDKit descriptors used in the original scripts.

This script combines the calculations previously performed by:
  1) Calc_rdkit3_N_O_type_coord.py
  2) Calc_rdkit8_for_library.py
  3) Calc_rdkit14_for_library.py

It preserves every column in the input CSV and appends the descriptor columns
in this fixed order: 3 coordination-related, 8 charge-related, and 14 standard
RDKit descriptors.  The descriptor formulae are unchanged from the three
source scripts.

Example:
  python calculate_rdkit_descriptors.py ^
      --input data/smiles_training.csv ^
      --output data/processed/training_25descriptors.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges


# This file is expected at: <project_root>/code/01_data_preparation/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

POSSIBLE_SMILES_COLS = [
    "SMILES",
    "smiles",
    "canonical_smiles",
    "IsomericSMILES",
    "PUBCHEM_SMILES",
]

CHARGE_COLUMNS = [
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "StdPartialCharge",
    "ChargeRange",
    "N_O_MinCharge",
    "Frac_NegativeAtoms",
    "PEOE_VSA1",
    "PEOE_VSA14",
]
COORD_COLUMNS = ["N_coord", "O_coord", "coord_types"]
RDKIT14_COLUMNS = [
    "MolecularWeight",
    "TPSA",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "HeavyAtomCount",
    "FormalCharge",
    "Complexity",
    "LogP",
    "MolarRefractivity",
    "RingCount",
    "AromaticRing",
    "HeteroRing",
    "FSP3",
]
# Fixed manuscript/data-package order: charge-related descriptors, then
# coordination descriptors, then the standard RDKit descriptors.
DESCRIPTOR_COLUMNS = CHARGE_COLUMNS + COORD_COLUMNS + RDKIT14_COLUMNS

# SMARTS definitions copied without modification from Calc_rdkit3_N_O_type_coord.py.
SMARTS = {
    "pyridine_N": "[n;H0]",
    "amine_N": "[N;!a;!$(N-C(=O));!$(N-S(=O)(=O))]",
    "amide_O": "[O]=[C][N]",
    "carbonyl_O_all": "[O]=[C]",
}
PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in SMARTS.items()}


def resolve_project_path(value: str) -> Path:
    """Interpret relative paths from the project root, not the working directory."""
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


def count_substructure(mol: Chem.Mol, pattern: Chem.Mol) -> int:
    return len(mol.GetSubstructMatches(pattern))


def calculate_coordination_descriptors(mol: Chem.Mol) -> dict[str, float]:
    """Return the same three descriptors as Calc_rdkit3_N_O_type_coord.py."""
    pyridine_n = count_substructure(mol, PATTERNS["pyridine_N"])
    amine_n = count_substructure(mol, PATTERNS["amine_N"])
    amide_o = count_substructure(mol, PATTERNS["amide_O"])
    carbonyl_o_all = count_substructure(mol, PATTERNS["carbonyl_O_all"])

    carbonyl_o_non_amide = max(carbonyl_o_all - amide_o, 0)
    n_coord = pyridine_n + amine_n
    o_coord = amide_o + carbonyl_o_non_amide
    coord_types = int((n_coord > 0) + (o_coord > 0))

    return {"N_coord": n_coord, "O_coord": o_coord, "coord_types": coord_types}


def calculate_charge_descriptors(mol: Chem.Mol) -> dict[str, float]:
    """Return the same eight descriptors as Calc_rdkit8_for_library.py."""
    ComputeGasteigerCharges(mol)

    charges = []
    n_charges = []
    o_charges = []
    for atom in mol.GetAtoms():
        try:
            charge = float(atom.GetProp("_GasteigerCharge"))
        except Exception:
            charge = 0.0
        charges.append(charge)
        if atom.GetSymbol() == "N":
            n_charges.append(charge)
        if atom.GetSymbol() == "O":
            o_charges.append(charge)

    charges = np.asarray(charges)
    peoe_vsa = rdMolDescriptors.PEOE_VSA_(mol)
    no_charges = n_charges + o_charges

    return {
        "MinPartialCharge": np.min(charges),
        "MaxAbsPartialCharge": np.max(np.abs(charges)),
        "StdPartialCharge": np.std(charges),
        "ChargeRange": np.max(charges) - np.min(charges),
        "N_O_MinCharge": np.min(no_charges) if no_charges else np.nan,
        "Frac_NegativeAtoms": np.sum(charges < 0) / len(charges),
        "PEOE_VSA1": peoe_vsa[0],
        "PEOE_VSA14": peoe_vsa[13],
    }


def calculate_rdkit14_descriptors(mol: Chem.Mol) -> dict[str, float]:
    """Return the same fourteen descriptors as Calc_rdkit14_for_library.py."""
    return {
        "MolecularWeight": Descriptors.MolWt(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBondDonorCount": rdMolDescriptors.CalcNumHBD(mol),
        "HBondAcceptorCount": rdMolDescriptors.CalcNumHBA(mol),
        "RotatableBondCount": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        "FormalCharge": Chem.GetFormalCharge(mol),
        "Complexity": Descriptors.BertzCT(mol),
        "LogP": Descriptors.MolLogP(mol),
        "MolarRefractivity": Descriptors.MolMR(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "AromaticRing": rdMolDescriptors.CalcNumAromaticRings(mol),
        "HeteroRing": rdMolDescriptors.CalcNumHeterocycles(mol),
        "FSP3": rdMolDescriptors.CalcFractionCSP3(mol),
    }


def empty_descriptor_record() -> dict[str, float]:
    return {column: np.nan for column in DESCRIPTOR_COLUMNS}


def calculate_all_descriptors(smiles: object) -> dict[str, float | int | str]:
    """Calculate all 25 descriptors for one SMILES string."""
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        record = empty_descriptor_record()
        record.update({"rdkit_ok": 0, "fail_reason": "bad_smiles"})
        return record

    try:
        record = calculate_coordination_descriptors(mol)
        record.update(calculate_charge_descriptors(mol))
        record.update(calculate_rdkit14_descriptors(mol))
        record.update({"rdkit_ok": 1, "fail_reason": ""})
        return record
    except Exception as exc:
        record = empty_descriptor_record()
        record.update({"rdkit_ok": 0, "fail_reason": f"rdkit_error:{exc}"})
        return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate the 25 initial RDKit descriptors for an input SMILES CSV."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV path, relative to the project root or absolute.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path, relative to the project root or absolute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input)
    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    smiles_column = find_smiles_column(df)
    print(f"Reading completed: {len(df)} records; SMILES column: {smiles_column}")

    records = [calculate_all_descriptors(smiles) for smiles in df[smiles_column]]
    descriptor_df = pd.DataFrame(records)
    descriptor_df = descriptor_df[DESCRIPTOR_COLUMNS + ["rdkit_ok", "fail_reason"]]

    # Keep all original input columns, followed by the fixed descriptor order.
    output_df = pd.concat([df.reset_index(drop=True), descriptor_df], axis=1)
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {output_path}")
    print(f"Total: {len(output_df)} | rdkit_ok: {int(output_df['rdkit_ok'].sum())}")
    failures = output_df.loc[output_df["rdkit_ok"] == 0, "fail_reason"].value_counts()
    if not failures.empty:
        print("Top fail reasons:\n", failures.head(10))


if __name__ == "__main__":
    main()
