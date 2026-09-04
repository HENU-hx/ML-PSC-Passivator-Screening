# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd


# Expected location: <project_root>/code/06_auxiliary_xTB_analysis/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value: str) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract electronic descriptors from xTB output files."
    )
    parser.add_argument(
        "--input-dir",
        default="data/xtb/xtb_sp",
        help="Directory containing molecule subdirectories with xtb.out files.",
    )
    parser.add_argument(
        "--output",
        default="data/xtb/xtb_electronic_descriptors.csv",
        help="Output CSV path for extracted electronic descriptors.",
    )
    return parser.parse_args()


def parse_xtb_output(xtb_out_file):
    dipole = None
    polar = None
    homo = None
    lumo = None
    gap = None

    if not os.path.exists(xtb_out_file):
        return None

    with open(xtb_out_file, "r", encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if "molecular dipole:" in line:
            for j in range(i, min(i + 6, len(lines))):
                if "full:" in lines[j]:
                    try:
                        dipole = float(lines[j].split()[-1])
                    except Exception:
                        pass
                    break

        if "Mol. 伪(0) /au" in line:
            try:
                polar = float(line.split()[-1])
            except Exception:
                pass

        if "(HOMO)" in line:
            parts = line.split()
            try:
                homo = float(parts[-2])
            except Exception:
                pass

        if "(LUMO)" in line:
            parts = line.split()
            try:
                lumo = float(parts[-2])
            except Exception:
                pass

        if "HOMO-LUMO GAP" in line:
            match = re.search(r"([-+]?\d*\.\d+|\d+)\s*eV", line)
            if match:
                try:
                    gap = float(match.group(1))
                except Exception:
                    pass

    return {
        "Dipole_Debye": dipole,
        "Polarizability_au": polar,
        "HOMO_eV": homo,
        "LUMO_eV": lumo,
        "Gap_eV": gap,
    }


def main() -> None:
    args = parse_args()
    xtb_folder = resolve_project_path(args.input_dir)
    output_csv = resolve_project_path(args.output)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for folder in sorted(os.listdir(xtb_folder)):
        work_dir = xtb_folder / folder

        if work_dir.is_dir():
            xtb_out_file = work_dir / "xtb.out"

            print(f"Parsing {folder} ...")

            data = parse_xtb_output(xtb_out_file)

            if data is None:
                continue

            data["Molecule"] = folder
            results.append(data)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("Electronic descriptors extracted successfully.")
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()