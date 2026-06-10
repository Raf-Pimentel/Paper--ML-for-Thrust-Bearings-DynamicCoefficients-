"""
plot_B_effect.py
================
Generates figures showing the effect of the dimensionless taper length B = β/lₓ
on the dynamic stiffness K and damping C coefficients, at fixed Λ = 1.0 and V = 0.

Four values of B are shown: 0.25, 0.50, 0.75, 1.00 (full taper).
The 1-D analytical reference (Λ → 0, B = 1) is overlaid as a dashed line.

Usage
-----
    python src/visualization/plot_B_effect.py
    python src/visualization/plot_B_effect.py --outdir figures/paper

Authors
-------
Rafael R. P. de Melo, Thales F. Peixoto — LAMAR / FEM-UNICAMP — MECSOL 2026
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.analytical import stiffness_1d, damping_1d

DATA_CSV       = Path("data/dataset_thrust_2D.csv")
TARGET_LAMBDA  = 1.0
TARGET_B_VALS  = [0.25, 0.50, 0.75, 1.00]
COLORS         = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
MARKERS        = ["^", "s", "D", "o"]
LINESTYLES     = ["--", "-.", ":", "-"]

FS_LABEL  = 16
FS_TICK   = 13
FS_LEGEND = 12
FS_TITLE  = 14


def _closest(arr: np.ndarray, v: float) -> float:
    return arr[np.argmin(np.abs(arr - v))]


def make_figure(target: str, y_label: str, ref_fn, outdir: Path) -> None:
    df = pd.read_csv(DATA_CSV)
    if "B" not in df.columns:
        raise ValueError("Dataset does not contain column 'B'. Regenerate with B_VALUES.")

    v_vals = df["V"].unique()
    v0 = v_vals[np.argmin(np.abs(v_vals))]

    unique_lam = df["Lambda"].unique()
    lam = _closest(unique_lam, TARGET_LAMBDA)

    unique_b = df["B"].unique()

    H0_ref = np.linspace(0.05, 3.0, 300)
    ref_curve = ref_fn(H0_ref)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(H0_ref, ref_curve, "k--", linewidth=2.0,
            label=r"1-D analytical ($\Lambda \to 0$,  $B = 1$)", zorder=5)

    for b_tgt, col, mrk, ls in zip(TARGET_B_VALS, COLORS, MARKERS, LINESTYLES):
        b = _closest(unique_b, b_tgt)
        mask = (np.isclose(df["Lambda"], lam, atol=1e-3) &
                np.isclose(df["B"],      b,   atol=1e-3) &
                np.isclose(df["V"],      v0,  atol=1e-6))
        sub = df[mask].sort_values("H0")
        ax.plot(sub["H0"], sub[target], color=col, marker=mrk,
                markersize=5, linewidth=1.8, linestyle=ls,
                label=rf"2-D numerical  $B = {b:.2f}$")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel(r"Film-thickness ratio,  $H_0 = h_0/s_h$", fontsize=FS_LABEL)
    ax.set_ylabel(y_label, fontsize=FS_LABEL)
    ax.set_xlim(0.0, 3.05)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.set_title(
        rf"Effect of taper length $B$ on {target}  ($\Lambda = {lam:.1f}$,  $V = 0$)"
        "\n(solid/dashed: 2-D numerical  |  black dashed: 1-D reference)",
        fontsize=FS_TITLE,
    )
    fig.tight_layout()
    outpath = outdir / f"{target}_vs_H0_B_effect.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {outpath}")


def main(outdir: Path = Path("figures/paper")) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    make_figure("K", r"Dimensionless stiffness,  $K$",
                lambda H0: stiffness_1d(H0, V=0.0), outdir)
    make_figure("C", r"Dimensionless damping,  $C$",
                lambda H0: damping_1d(H0), outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot B taper-length effect on K and C.")
    parser.add_argument("--outdir", type=Path, default=Path("figures/paper"))
    args = parser.parse_args()
    main(outdir=args.outdir)
