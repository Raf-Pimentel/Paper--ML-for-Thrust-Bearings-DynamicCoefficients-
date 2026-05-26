"""
plot_coefficients.py
====================
Generates Figures 5 and 6 of the MECSOL 2026 paper:
    Fig. 5 — Dimensionless stiffness K vs H₀ for three aspect ratios
    Fig. 6 — Dimensionless damping   C vs H₀ for three aspect ratios

Both figures overlay the 1-D analytical reference curves.

Usage
-----
    python src/visualization/plot_coefficients.py

    # Save to a specific directory:
    python src/visualization/plot_coefficients.py --outdir figures/paper

Authors
-------
Rafael R. P. de Melo, Thales F. Peixoto — LAMAR / FEM-UNICAMP — MECSOL 2026
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.analytical import stiffness_1d, damping_1d

DATA_CSV  = Path("data/dataset_thrust_2D.csv")
TARGET_LAMBDAS = [0.5, 1.0, 2.0]
COLORS  = ["#1f77b4", "#d62728", "#2ca02c"]
MARKERS = ["o", "s", "^"]


def _closest_lambda(unique_lambdas: np.ndarray, target: float) -> float:
    return unique_lambdas[np.argmin(np.abs(unique_lambdas - target))]


def make_figure(target: str, y_label: str, ref_fn, outdir: Path) -> None:
    """Create and save one coefficient figure."""
    df = pd.read_csv(DATA_CSV)
    v_vals = df["V"].unique()
    v0 = v_vals[np.argmin(np.abs(v_vals))]
    df_v0 = df[df["V"] == v0].copy()
    unique_lam = df["Lambda"].unique()

    H0_ref = np.linspace(0.05, 3.0, 300)
    ref_curve = ref_fn(H0_ref)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(H0_ref, ref_curve, "k--", linewidth=2.0,
            label=r"1-D analytical ($\Lambda \to 0$)", zorder=5)

    for tgt, col, mrk in zip(TARGET_LAMBDAS, COLORS, MARKERS):
        lam = _closest_lambda(unique_lam, tgt)
        sub = df_v0[np.isclose(df_v0["Lambda"], lam, atol=1e-3)].sort_values("H0")
        ax.plot(sub["H0"], sub[target], color=col, marker=mrk,
                markersize=5, linewidth=1.6,
                label=rf"2-D numerical  $\Lambda \approx {lam:.2f}$")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel(r"Film-thickness ratio,  $H_0 = h_0/s_h$", fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_xlim(0.0, 3.05)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.set_title(
        rf"Dynamic {target} coefficient vs $H_0$  (V = 0)"
        "\n(solid: 2-D numerical  |  dashed: 1-D reference)",
        fontsize=12,
    )
    fig.tight_layout()
    outpath = outdir / f"{target}_vs_H0.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {outpath}")


def main(outdir: Path = Path("figures/paper")) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    make_figure("K", r"Dimensionless stiffness,  $K$",
                lambda H0: stiffness_1d(H0, V=0.0), outdir)
    make_figure("C", r"Dimensionless damping,  $C$",
                lambda H0: damping_1d(H0), outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot K and C vs H₀.")
    parser.add_argument("--outdir", type=Path, default=Path("figures/paper"))
    args = parser.parse_args()
    main(outdir=args.outdir)
