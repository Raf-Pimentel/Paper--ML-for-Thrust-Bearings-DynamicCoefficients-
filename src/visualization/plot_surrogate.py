"""
plot_surrogate.py
=================
Generates Figures 7 and 8 of the MECSOL 2026 paper:
    Fig. 7 — Surrogate validation for K (solid: numerical, dots: RF prediction)
    Fig. 8 — Surrogate validation for C (same convention)
Both include a zoom inset for the critical region H₀ < 0.1.

FIXES applied (per professor's review):
    - Increased font sizes throughout (axis labels, ticks, legend, title)
    - Inset now shows BOTH the numerical solid line AND surrogate scatter dots
      (previously only the surrogate scatter was plotted in the inset)

Usage
-----
    python src/visualization/plot_surrogate.py

Authors
-------
Rafael R. P. de Melo, Thales F. Peixoto — LAMAR / FEM-UNICAMP — MECSOL 2026
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_CSV  = Path("data/dataset_thrust_2D.csv")
MODEL_DIR = Path("models")
TARGET_LAMBDAS = [0.5, 1.0, 2.0]
H0_DENSE = np.linspace(0.05, 3.0, 300)
COLORS   = ["#1f77b4", "#d62728", "#2ca02c"]
MARKERS  = ["o", "s", "^"]

# Font sizes
FS_LABEL  = 16   # axis labels
FS_TICK   = 13   # tick labels
FS_LEGEND = 12   # legend text
FS_TITLE  = 14   # figure title
FS_INSET_LABEL  = 12
FS_INSET_TICK   = 10
FS_INSET_TITLE  = 11


def _closest(arr: np.ndarray, v: float) -> float:
    return arr[np.argmin(np.abs(arr - v))]


def make_figure(target: str, outdir: Path) -> None:
    df = pd.read_csv(DATA_CSV)
    model = joblib.load(MODEL_DIR / f"best_model_{target}.pkl")

    # Surrogate validation uses full-taper pads (B=1)
    if "B" in df.columns:
        df = df[np.isclose(df["B"], 1.0, atol=1e-3)]
    v_vals = df["V"].unique()
    v0 = v_vals[np.argmin(np.abs(v_vals))]
    df_v0 = df[df["V"] == v0].copy()
    unique_lam = df["Lambda"].unique()

    fig, ax = plt.subplots(figsize=(10, 6))
    axins = ax.inset_axes([0.42, 0.32, 0.55, 0.62])

    for tgt, col, mrk in zip(TARGET_LAMBDAS, COLORS, MARKERS):
        lam = _closest(unique_lam, tgt)
        sub = df_v0[np.isclose(df_v0["Lambda"], lam, atol=1e-3)].sort_values("H0")

        # ── Main axes: numerical solid line ──────────────────────────────
        ax.plot(sub["H0"], sub[target], color=col, linewidth=2.0,
                label=rf"Numerical  $\Lambda ={lam:.2f}$")

        # ── Main axes: surrogate scatter ──────────────────────────────────
        feat = np.column_stack([H0_DENSE, np.full_like(H0_DENSE, v0),
                                np.full_like(H0_DENSE, lam),
                                np.ones_like(H0_DENSE)])   # B=1.0
        pred = model.predict(feat)
        ax.scatter(H0_DENSE[::6], pred[::6], color=col, marker=mrk,
                   s=22, zorder=4, label=rf"Surrogate  $\Lambda ={lam:.2f}$")

        # ── Inset: numerical solid line (FIX: was missing before) ────────
        sub_zoom = sub[sub["H0"] < 0.55]
        axins.plot(sub_zoom["H0"], sub_zoom[target],
                   color=col, linewidth=2.0)

        # ── Inset: surrogate scatter ──────────────────────────────────────
        H0_zoom = H0_DENSE[H0_DENSE < 0.50]
        feat_z = np.column_stack([H0_zoom, np.full_like(H0_zoom, v0),
                                  np.full_like(H0_zoom, lam),
                                  np.ones_like(H0_zoom)])   # B=1.0
        pred_z = model.predict(feat_z)
        axins.scatter(H0_zoom[::6], pred_z[::6],
                      color=col, marker=mrk, s=18, zorder=4)

    # ── Main axes formatting ──────────────────────────────────────────────
    ax.set_xlabel(r"Film-thickness ratio,  $H_0 = h_0/s_h$",
                  fontsize=FS_LABEL)
    ax.set_ylabel(rf"Dimensionless {target} coefficient,  ${target}$",
                  fontsize=FS_LABEL)
    ax.set_title(
        rf"Surrogate validation — ${target}$ vs $H_0$  (V = 0)"
        "\n(solid: 2-D numerical  |  dots: Random Forest surrogate)",
        fontsize=FS_TITLE,
    )
    ax.set_xlim(0.0, 3.05)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, ncol=2, framealpha=0.9, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.45)

    # ── Inset formatting ─────────────────────────────────────────────────
    axins.set_xlim(0.04, 0.50)
    axins.set_xlabel(r"$H_0$", fontsize=FS_INSET_LABEL)
    axins.set_ylabel(rf"${target}$", fontsize=FS_INSET_LABEL)
    axins.set_title(r"Zoom: $H_0 < 0.5$", fontsize=FS_INSET_TITLE)
    axins.tick_params(labelsize=FS_INSET_TICK)
    axins.grid(True, linestyle=":", alpha=0.4)
    ax.indicate_inset_zoom(axins, edgecolor="gray", alpha=0.6)

    fig.tight_layout()
    outpath = outdir / f"surrogate_{target}_validation.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {outpath}")


def main(outdir: Path = Path("figures/paper")) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for t in ["K", "C"]:
        make_figure(t, outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("figures/paper"))
    main(outdir=parser.parse_args().outdir)