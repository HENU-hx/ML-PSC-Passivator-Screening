# -*- coding: utf-8 -*-
"""Generate initial 3D XYZ geometries from training-set SMILES for xTB analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


# Expected location: <project_root>/code/06_auxiliary_xTB_analysis/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RANDOM_SEED = 42


def resolve_project_path(value: str) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate initial XYZ geometries from SMILES for xTB analysis."
    )
    parser.add_argument(
        "--input",
        default="data/smiles_training.csv",
        help="Input CSV containing a SMILES column, relative to the project root.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/xtb/xyz_initial",
        help="Directory for generated XYZ files, relative to the project root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = resolve_project_path(args.input)
    output_folder = resolve_project_path(args.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()

    if "SMILES" not in df.columns:
        raise ValueError("The CSV file must contain a SMILES column.")

    success = 0
    fail = 0

    for idx, row in df.iterrows():
        smiles = str(row["SMILES"]).strip()
        name = f"Mol_{idx}"

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            print(f"SMILES error: {smiles}")
            fail += 1
            continue

        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = RANDOM_SEED

        result = AllChem.EmbedMolecule(mol, params)

        if result != 0:
            print(f"3D generation failed: {smiles}")
            fail += 1
            continue

        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            print(f"MMFF optimization failed: {smiles}")
            fail += 1
            continue

        xyz_path = output_folder / f"{name}.xyz"

        with xyz_path.open("w", encoding="utf-8") as file:
            conf = mol.GetConformer()
            atoms = mol.GetAtoms()

            file.write(f"{mol.GetNumAtoms()}\n")
            file.write(f"{name}\n")

            for atom in atoms:
                pos = conf.GetAtomPosition(atom.GetIdx())
                file.write(
                    f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n"
                )

        success += 1

    print("\nDone!")
    print(f"Successfully generated XYZ files: {success}")
    print(f"Failed molecules: {fail}")
    print(f"Saved to: {output_folder}")


if __name__ == "__main__":
    main()