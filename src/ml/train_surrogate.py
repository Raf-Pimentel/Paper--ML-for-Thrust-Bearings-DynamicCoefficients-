"""
train_surrogate.py
==================
Trains and benchmarks three surrogate regression models for the dynamic
coefficients K and C of a finite-length hydrodynamic thrust bearing.

Models trained
--------------
    Random Forest   — 200 trees, scikit-learn RandomForestRegressor
    XGBoost         — 500 estimators, lr=0.05, max_depth=6
    ANN (MLP)       — hidden layers (128, 64), early stopping, StandardScaler

Train / test split
------------------
An 80/20 stratified split over the Λ column is used to guarantee that every
aspect-ratio level appears in both training and test sets.

Evaluation metrics
------------------
RMSE and R² are reported on the held-out test set both for the full H₀ range
and for the critical asymptotic sub-range H₀ < 0.1.

Output
------
    models/best_model_K.pkl   — Random Forest for K (retrained on full dataset)
    models/best_model_C.pkl   — Random Forest for C (retrained on full dataset)

Usage
-----
    python src/ml/train_surrogate.py

    # Use a custom dataset:
    python src/ml/train_surrogate.py --data data/my_dataset.csv

Authors
-------
Rafael R. P. de Melo, Thales F. Peixoto — LAMAR / FEM-UNICAMP — MECSOL 2026
"""

from __future__ import annotations

import argparse
import copy
import warnings
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_CSV    = Path("data/dataset_thrust_2D.csv")
MODELS_DIR  = Path("models")
FEATURES    = ["H0", "V", "Lambda"]
TARGETS     = ["K", "C"]

MODEL_DEFS = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200, n_jobs=-1, random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        n_jobs=-1, random_state=42, verbosity=0,
    ),
    "ANN (MLP)": Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=1000,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=42,
        )),
    ]),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    lambda_col: np.ndarray,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """80/20 stratified split over Λ quantile bins."""
    lambda_bins = pd.qcut(lambda_col, q=5, labels=False)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_idx, test_idx = next(skf.split(X, lambda_bins))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def _score(
    model, X_tr, y_tr, X_te, y_te, crit_mask: np.ndarray
) -> dict:
    """Fit, predict, and return accuracy metrics."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    rmse_full = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    r2_full   = float(r2_score(y_te, y_pred))

    if crit_mask.sum() >= 2:
        rmse_c = float(np.sqrt(mean_squared_error(y_te[crit_mask], y_pred[crit_mask])))
        r2_c   = float(r2_score(y_te[crit_mask], y_pred[crit_mask]))
    else:
        rmse_c, r2_c = float("nan"), float("nan")

    return {
        "model_obj": model,
        "rmse_full": rmse_full, "r2_full": r2_full,
        "rmse_crit": rmse_c,   "r2_crit": r2_c,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(data_csv: Path = DATA_CSV) -> None:
    """Train, evaluate, and export surrogate models."""
    # -- Load data -----------------------------------------------------------
    df = pd.read_csv(data_csv)
    print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} cols")

    X = df[FEATURES].values
    lambda_col = df["Lambda"].values

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[dict]] = {t: [] for t in TARGETS}

    for target in TARGETS:
        y = df[target].values
        X_tr, X_te, y_tr, y_te = _stratified_split(X, y, lambda_col)
        crit_mask = X_te[:, 0] < 0.1   # H₀ < 0.1 sub-range

        print(f"\n{'─'*60}")
        print(f"  Target: {target}   train={len(X_tr)}   test={len(X_te)}")
        print(f"  Test points with H₀ < 0.1 : {crit_mask.sum()}")
        print(f"{'─'*60}")

        for name, mdl in MODEL_DEFS.items():
            print(f"  Training {name} ...", end=" ", flush=True)
            res = _score(copy.deepcopy(mdl), X_tr, y_tr, X_te, y_te, crit_mask)
            res["name"] = name
            results[target].append(res)
            print(
                f"R²(full)={res['r2_full']:.6f}  "
                f"RMSE(full)={res['rmse_full']:.2e}  "
                f"R²(H₀<0.1)={res['r2_crit']:.6f}"
            )

    # -- Summary table -------------------------------------------------------
    SEP = "─" * 84
    HDR = (
        f"{'Model':<18} │ {'Target':<4} │ {'RMSE (full)':>11} │ "
        f"{'R² (full)':>10} │ {'RMSE (H₀<0.1)':>13} │ {'R² (H₀<0.1)':>12}"
    )
    print(f"\n{'═'*84}")
    print("  SURROGATE ACCURACY SUMMARY")
    print(f"{'═'*84}")
    print(HDR)
    for target in TARGETS:
        print(SEP)
        for r in results[target]:
            print(
                f"{r['name']:<18} │ {target:<4} │ "
                f"{r['rmse_full']:>11.2e} │ {r['r2_full']:>10.6f} │ "
                f"{r['rmse_crit']:>13.2e} │ {r['r2_crit']:>12.6f}"
            )
    print(f"{'═'*84}\n")

    # -- Export best models (retrain on full dataset) ------------------------
    for target in TARGETS:
        best = max(results[target], key=lambda r: r["r2_full"])
        print(
            f"Best model for {target}: {best['name']}  "
            f"R²={best['r2_full']:.6f}  RMSE={best['rmse_full']:.2e}"
        )
        final = copy.deepcopy(best["model_obj"])
        final.fit(X, df[target].values)
        out_path = MODELS_DIR / f"best_model_{target}.pkl"
        joblib.dump(final, out_path)
        print(f"  → Saved {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML surrogate models.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_CSV,
        help=f"Path to the input CSV dataset (default: {DATA_CSV})",
    )
    args = parser.parse_args()
    run(data_csv=args.data)
