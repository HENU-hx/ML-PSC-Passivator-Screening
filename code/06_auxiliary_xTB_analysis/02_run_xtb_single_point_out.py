# -*- coding: utf-8 -*-
"""Run xTB single-point calculations for initial XYZ geometries."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


# Expected location: <project_root>/code/06_auxiliary_xTB_analysis/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value: str) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run xTB single-point calculations for XYZ geometries."
    )
    parser.add_argument(
        "--input-dir",
        default="data/xtb/xyz_initial",
        help="Directory containing input XYZ files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/xtb/xtb_sp",
        help="Directory for xTB single-point calculation outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_folder = resolve_project_path(args.input_dir)
    output_folder = resolve_project_path(args.output_dir)

    output_folder.mkdir(parents=True, exist_ok=True)

    xyz_files = sorted(input_folder.glob("*.xyz"))

    for input_path in xyz_files:
        file_name = input_path.name
        work_dir = output_folder / input_path.stem
        work_dir.mkdir(parents=True, exist_ok=True)

        new_xyz_path = work_dir / file_name
        shutil.copy(input_path, new_xyz_path)

        print(f"Performing single-point calculation: {file_name}")

        cmd = ["xtb", file_name, "--gfn", "2", "--sp", "--alpb", "acetonitrile"]

        result = subprocess.run(
            cmd,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        with (work_dir / "xtb.out").open("wb") as file:
            file.write(result.stdout)

    print("All single-point calculations have been completed.")
    print(f"Saved to: {output_folder}")


if __name__ == "__main__":
    main()