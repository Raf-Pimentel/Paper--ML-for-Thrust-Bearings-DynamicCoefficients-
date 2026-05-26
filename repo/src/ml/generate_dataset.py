"""
generate_dataset.py
===================
Sweeps the 3-D parameter space {H₀, V, Λ} and, for every operating point,
computes the dimensionless load capacity Wz, stiffness K, and damping C using
the 2-D Reynolds solver in `src/solver/reynolds_solver.py`.

Parameter grid
--------------
    H₀ ∈ [0.05, 3.0]   — 30 points
    V  ∈ [−2.0, 2.0]   — 15 points
    Λ  ∈ [0.25, 4.0]   — 10 points
    Total: 30 × 15 × 10 = 4,500 operating points

Output
------
    data/dataset_thrust_2D.csv   columns: H0, V, Lambda, Wz, K, C

Resume / checkpoint behaviour
------------------------------
If the output CSV already exists, the script counts its rows, skips those
operating points, and appends the remaining results.  A checkpoint is
flushed to disk every CHECKPOINT_EVERY rows.

Usage
-----
    python src/ml/generate_dataset.py

    # Override output path:
    python src/ml/generate_dataset.py --output data/my_dataset.csv

Authors
-------
Rafael R. P. de Melo, Thales F. Peixoto — LAMAR / FEM-UNICAMP — MECSOL 2026
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure project root is on sys.path when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.solver.reynolds_solver import ThrustBearingSolver

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_CSV       = Path("data/dataset_thrust_2D.csv")
CHECKPOINT_EVERY = 500          # flush to disk every N completed rows
N_MESH           = 100          # solver mesh (100×100)
EPSILON_FD       = 1e-4         # finite-difference step for K and C
SOLVER_TOL       = 1e-6         # Gauss–Seidel convergence tolerance

# Parameter grids — must match those reported in the paper (Table 2)
H0_VALUES     = np.linspace(0.05, 3.0, 30)
V_VALUES      = np.linspace(-2.0, 2.0, 15)
LAMBDA_VALUES = np.linspace(0.25, 4.0, 10)

COLUMNS = ["H0", "V", "Lambda", "Wz", "K", "C"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_existing_rows(csv_path: Path) -> int:
    """Return number of data rows already written to *csv_path* (0 if absent)."""
    if not csv_path.exists():
        return 0
    try:
        return len(pd.read_csv(csv_path))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run(output: Path = OUTPUT_CSV) -> None:
    """Execute the parameter sweep and write results to *output*."""
    output.parent.mkdir(parents=True, exist_ok=True)
    solver = ThrustBearingSolver(
        nx=N_MESH, ny=N_MESH,
        tolerance=SOLVER_TOL,
        epsilon_fd=EPSILON_FD,
    )

    all_points = list(itertools.product(H0_VALUES, V_VALUES, LAMBDA_VALUES))
    total = len(all_points)

    n_done = _count_existing_rows(output)
    remaining = all_points[n_done:]

    if n_done:
        print(
            f"[resume] {n_done} rows already saved in {output}. "
            f"Continuing from row {n_done + 1}."
        )

    write_header = n_done == 0
    csv_fh = open(output, "a", newline="")
    if write_header:
        csv_fh.write(",".join(COLUMNS) + "\n")

    buffer: list[str] = []
    rows_written = n_done

    print(
        f"\nParameter sweep: {total} points total ({len(remaining)} remaining)\n"
        f"  H₀     : {len(H0_VALUES)} pts  [{H0_VALUES[0]:.3f} … {H0_VALUES[-1]:.3f}]\n"
        f"  V      : {len(V_VALUES)} pts  [{V_VALUES[0]:.2f} … {V_VALUES[-1]:.2f}]\n"
        f"  Λ      : {len(LAMBDA_VALUES)} pts  [{LAMBDA_VALUES[0]:.2f} … {LAMBDA_VALUES[-1]:.2f}]\n"
        f"  Mesh   : {N_MESH}×{N_MESH}   ε = {EPSILON_FD}   "
        f"checkpoint every {CHECKPOINT_EVERY} rows\n"
        f"  Output : {output}\n"
    )

    pbar = tqdm(total=len(remaining), unit="pt", desc="Sweep", dynamic_ncols=True)

    for H0, V, Lambda in remaining:
        result = solver.solve(H0=H0, V=V, Lambda=Lambda)

        buffer.append(
            f"{H0:.6f},{V:.6f},{Lambda:.6f},"
            f"{result.Wz:.8f},{result.K:.8f},{result.C:.8f}\n"
        )
        rows_written += 1

        if rows_written % CHECKPOINT_EVERY == 0:
            csv_fh.writelines(buffer)
            csv_fh.flush()
            buffer.clear()
            pbar.set_postfix({"saved": rows_written})

        pbar.update(1)

    if buffer:
        csv_fh.writelines(buffer)
        csv_fh.flush()

    csv_fh.close()
    pbar.close()

    df = pd.read_csv(output)
    print(f"\nDataset complete: {len(df)} rows → {output}")
    print(df.describe().to_string())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help="Path for the output CSV file (default: data/dataset_thrust_2D.csv)",
    )
    args = parser.parse_args()
    run(output=args.output)
