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

Parallelisation
---------------
Each operating point is independent; the sweep is distributed across all
available CPU cores using joblib (backend='loky').  Pass --workers N to
limit the number of parallel workers (default: all cores).

Resume / checkpoint behaviour
------------------------------
If the output CSV already exists, the script counts its rows, skips those
operating points, and appends the remaining results.  A checkpoint is
flushed to disk every CHECKPOINT_EVERY rows.

Usage
-----
    python src/ml/generate_dataset.py
    python src/ml/generate_dataset.py --workers 4
    python src/ml/generate_dataset.py --output data/my_dataset.csv

Authors
-------
Rafael R. P. de Melo, Thales F. Peixoto — LAMAR / FEM-UNICAMP — MECSOL 2026
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
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
SOLVER_TOL       = 1e-6         # Gauss-Seidel convergence tolerance
N_WORKERS        = -1           # -1 = all available CPU cores

# Parameter grids — must match those reported in the paper (Table 2)
H0_VALUES     = np.linspace(0.05, 3.0, 30)
V_VALUES      = np.linspace(-2.0, 2.0, 15)
LAMBDA_VALUES = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

COLUMNS = ["H0", "V", "Lambda", "Wz", "K", "C"]


# ---------------------------------------------------------------------------
# Per-point worker (runs in a separate process — no shared state)
# ---------------------------------------------------------------------------

def _compute_point(
    H0: float, V: float, Lambda: float,
    nx: int, ny: int, tol: float, eps_fd: float,
) -> tuple[float, float, float, float, float, float]:
    """Solve one operating point and return (H0, V, Lambda, Wz, K, C)."""
    solver = ThrustBearingSolver(nx=nx, ny=ny, tolerance=tol, epsilon_fd=eps_fd)
    r = solver.solve(H0=H0, V=V, Lambda=Lambda)
    return (H0, V, Lambda, r.Wz, r.K, r.C)


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

def run(output: Path = OUTPUT_CSV, n_workers: int = N_WORKERS) -> None:
    """Execute the parallel parameter sweep and write results to *output*."""
    import os
    n_cpu = os.cpu_count() or 1
    workers_used = n_cpu if n_workers == -1 else min(n_workers, n_cpu)

    output.parent.mkdir(parents=True, exist_ok=True)

    all_points = list(itertools.product(H0_VALUES, V_VALUES, LAMBDA_VALUES))
    total = len(all_points)

    n_done = _count_existing_rows(output)
    remaining = all_points[n_done:]

    if n_done:
        print(
            f"[resume] {n_done} rows already saved in {output}. "
            f"Continuing from row {n_done + 1}."
        )

    print(
        f"\nParameter sweep: {total} points total ({len(remaining)} remaining)\n"
        f"  H0     : {len(H0_VALUES)} pts  [{H0_VALUES[0]:.3f} ... {H0_VALUES[-1]:.3f}]\n"
        f"  V      : {len(V_VALUES)} pts  [{V_VALUES[0]:.2f} ... {V_VALUES[-1]:.2f}]\n"
        f"  Lambda : {len(LAMBDA_VALUES)} pts  [{LAMBDA_VALUES[0]:.2f} ... {LAMBDA_VALUES[-1]:.2f}]\n"
        f"  Mesh   : {N_MESH}x{N_MESH}   eps = {EPSILON_FD}\n"
        f"  Workers: {workers_used} / {n_cpu} cores\n"
        f"  Output : {output}\n"
    )

    if not remaining:
        print("Nothing to do — dataset already complete.")
        return

    write_header = n_done == 0
    csv_fh = open(output, "a", newline="")
    if write_header:
        csv_fh.write(",".join(COLUMNS) + "\n")

    rows_written = n_done

    # Process in batches so we can checkpoint periodically
    batch_size = CHECKPOINT_EVERY
    pbar = tqdm(total=len(remaining), unit="pt", desc="Sweep", dynamic_ncols=True)

    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start : batch_start + batch_size]

        results = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(_compute_point)(
                H0, V, Lambda, N_MESH, N_MESH, SOLVER_TOL, EPSILON_FD
            )
            for H0, V, Lambda in batch
        )

        for row in results:
            H0, V, Lambda, Wz, K, C = row
            csv_fh.write(
                f"{H0:.6f},{V:.6f},{Lambda:.6f},{Wz:.8f},{K:.8f},{C:.8f}\n"
            )
        csv_fh.flush()
        rows_written += len(results)
        pbar.update(len(results))
        pbar.set_postfix({"saved": rows_written})

    csv_fh.close()
    pbar.close()

    df = pd.read_csv(output)
    print(f"\nDataset complete: {len(df)} rows -> {output}")
    print(df.describe().to_string())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate bearing dataset (parallel).")
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_CSV,
        help="Path for the output CSV (default: data/dataset_thrust_2D.csv)",
    )
    parser.add_argument(
        "--workers", type=int, default=N_WORKERS,
        help="Number of parallel workers (-1 = all cores, default: -1)",
    )
    args = parser.parse_args()
    run(output=args.output, n_workers=args.workers)
