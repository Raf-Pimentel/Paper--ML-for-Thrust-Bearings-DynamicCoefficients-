"""
generate_dataset_thrust_2D.py
=============================================================================
Generates the training dataset for the ML surrogate model by sweeping the
parameter space of the 2-D hydrodynamic thrust bearing solver.

Parameters sampled:
    H0     : dimensionless minimum film-thickness ratio   [0.05, 3.0]  x30
    V      : dimensionless squeeze velocity               [-2.0, 2.0]  x15
    Lambda : pad aspect ratio  lx / ly                   [0.25, 4.0]  x10

    Total points : 30 x 15 x 10 = 4,500

For each point the script computes:
    Wz  -- dimensionless load capacity  (Riemann integration over the 2-D mesh)
    K   -- stiffness coefficient        (central finite difference in H0)
    C   -- damping coefficient          (central finite difference in V)

Output:
    dataset_thrust_2D.csv   columns: H0, V, Lambda, Wz, K, C

Resume behaviour:
    If dataset_thrust_2D.csv already exists, the script reads its last row,
    skips all points that were already computed, and appends new rows.
    A checkpoint is also written every CHECKPOINT_EVERY rows to avoid
    losing progress.

Usage:
    python generate_dataset_thrust_2D.py

Dependencies:
    numpy, pandas, tqdm
    grafico26.py  (must be in the same directory or on PYTHONPATH)
=============================================================================
"""

import sys
import os
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Make sure grafico26.py is importable.
# Add the directory that contains grafico26.py to sys.path if needed.
# Adjust this path to match your local repository layout, e.g.:
#   GRAFICO26_DIR = r"C:\Users\rafae\...\src\1S_2026"
# ---------------------------------------------------------------------------
GRAFICO26_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "src", "1S_2026")
if os.path.isdir(GRAFICO26_DIR):
    sys.path.insert(0, GRAFICO26_DIR)

try:
    from grafico26 import (
        calcular_pressao_adimensional,
        calcular_carga_suportada,
        calcular_coeficiente_K,
        calcular_coeficiente_C,
    )
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "grafico26.py not found. "
        "Set GRAFICO26_DIR at the top of this script to the folder "
        "that contains grafico26.py."
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_CSV       = "dataset_thrust_2D.csv"
CHECKPOINT_EVERY = 250          # write CSV after every N completed rows
EPSILON          = 1e-4         # finite-difference step for K and C
SOLVER_ALPHA     = 1e-6         # Gauss-Seidel convergence tolerance
N_MESH           = 50          # mesh points in each direction (50x50)

# Parameter grids (must match what is described in the paper / Table 3)
H0_VALUES     = np.linspace(0.05, 3.0, 30)
V_VALUES      = np.linspace(-2.0, 2.0, 15)
LAMBDA_VALUES = np.linspace(0.25, 4.0, 10)

COLUMNS = ["H0", "V", "Lambda", "Wz", "K", "C"]


# ---------------------------------------------------------------------------
# Helper: build the 2-D film-thickness matrix for a given H0
# ---------------------------------------------------------------------------
def build_mesh_and_H(n_mesh: int):
    """
    Returns the 1-D coordinate arrays X, Y and the meshgrid arrays X_grid,
    Y_grid.  The H matrix is rebuilt inside the loop for each H0.
    """
    X = np.linspace(0, 1, n_mesh)
    Y = np.linspace(0, 1, n_mesh)
    X_grid, Y_grid = np.meshgrid(X, Y, indexing="ij")
    return X, Y, X_grid, Y_grid


def make_H(H0: float, X_grid: np.ndarray) -> np.ndarray:
    """Fixed-incline pad: H(X) = H0 + (1 - X)."""
    return H0 + 1.0 - X_grid


# ---------------------------------------------------------------------------
# Resume logic: find how many rows already exist
# ---------------------------------------------------------------------------
def count_existing_rows(csv_path: str) -> int:
    if not os.path.exists(csv_path):
        return 0
    try:
        df = pd.read_csv(csv_path)
        return len(df)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    X, Y, X_grid, Y_grid = build_mesh_and_H(N_MESH)

    # Full ordered list of all parameter combinations
    all_points = list(itertools.product(H0_VALUES, V_VALUES, LAMBDA_VALUES))
    total = len(all_points)

    # Resume: skip rows that are already saved
    n_done = count_existing_rows(OUTPUT_CSV)
    if n_done > 0:
        print(f"[resume] Found {n_done} existing rows in {OUTPUT_CSV}. "
              f"Skipping those and continuing from row {n_done + 1}.")
    remaining = all_points[n_done:]

    # Open CSV in append mode (write header only if starting fresh)
    write_header = (n_done == 0)
    csv_file = open(OUTPUT_CSV, "a", newline="")
    if write_header:
        csv_file.write(",".join(COLUMNS) + "\n")

    buffer = []          # accumulate rows before flushing
    rows_written = n_done

    print(f"\nParameter sweep: {total} points total  "
          f"({len(remaining)} remaining)\n"
          f"  H0     : {len(H0_VALUES)} values  [{H0_VALUES[0]:.3f} … {H0_VALUES[-1]:.3f}]\n"
          f"  V      : {len(V_VALUES)} values  [{V_VALUES[0]:.2f} … {V_VALUES[-1]:.2f}]\n"
          f"  Lambda : {len(LAMBDA_VALUES)} values  [{LAMBDA_VALUES[0]:.2f} … {LAMBDA_VALUES[-1]:.2f}]\n"
          f"  Mesh   : {N_MESH}x{N_MESH}  |  eps={EPSILON}  |  "
          f"checkpoint every {CHECKPOINT_EVERY} rows\n"
          f"  Output : {OUTPUT_CSV}\n")

    pbar = tqdm(total=len(remaining), unit="pt",
                desc="Sweep", dynamic_ncols=True)

    for H0, V, Lambda in remaining:

        # ----------------------------------------------------------------
        # Build H matrix for this H0
        # ----------------------------------------------------------------
        H = make_H(H0, X_grid)
        construir_H = lambda h0_val: make_H(h0_val, X_grid)

        # ----------------------------------------------------------------
        # Baseline Wz (at the exact operating point)
        # ----------------------------------------------------------------
        P_base, _ = calcular_pressao_adimensional(
            X, Y, H, Lambda, dHdt=12.0 * V, alpha=SOLVER_ALPHA
        )
        Wz = calcular_carga_suportada(P_base, X, Y)

        # ----------------------------------------------------------------
        # Stiffness K = dWz / dH0   (central FD, V held at 0 for K)
        # ----------------------------------------------------------------
        K = calcular_coeficiente_K(
            X, Y, H0, construir_H, Lambda,
            dHdt=0.0,           # V=0 for the stiffness perturbation
            epsilon=EPSILON,
            alpha=SOLVER_ALPHA,
        )

        # ----------------------------------------------------------------
        # Damping C = dWz / dV      (central FD, H0 fixed)
        # ----------------------------------------------------------------
        C = calcular_coeficiente_C(
            X, Y, H, Lambda,
            dHdt=12.0 * V,      # grafico26 convention: dHdt = 12·V_paper
            epsilon=EPSILON,
            alpha=SOLVER_ALPHA,
        )

        buffer.append(f"{H0:.6f},{V:.6f},{Lambda:.6f},"
                      f"{Wz:.8f},{K:.8f},{C:.8f}\n")
        rows_written += 1

        # ----------------------------------------------------------------
        # Checkpoint: flush buffer to disk
        # ----------------------------------------------------------------
        if rows_written % CHECKPOINT_EVERY == 0:
            csv_file.writelines(buffer)
            csv_file.flush()
            buffer.clear()
            pbar.set_postfix({"saved": rows_written, "H0": f"{H0:.3f}",
                              "Lambda": f"{Lambda:.2f}"})

        pbar.update(1)

    # Flush any remaining rows
    if buffer:
        csv_file.writelines(buffer)
        csv_file.flush()
        buffer.clear()

    csv_file.close()
    pbar.close()

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    df = pd.read_csv(OUTPUT_CSV)
    print(f"\n{'='*60}")
    print(f"Dataset complete: {len(df)} rows  →  {OUTPUT_CSV}")
    print(f"{'='*60}")
    print(df.describe().to_string())
    print(f"{'='*60}\n")
    print("NaN check:")
    print(df.isnull().sum())


if __name__ == "__main__":
    main()
