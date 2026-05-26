# ML Surrogate for Dynamic Coefficients of Finite-Length Hydrodynamic Thrust Bearings

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MECSOL 2026](https://img.shields.io/badge/Conference-MECSOL%202026-green.svg)](https://mecsol2026.com)
[![LAMAR/UNICAMP](https://img.shields.io/badge/Lab-LAMAR%2FUNICAMP-orange.svg)](https://www.fem.unicamp.br)

> **Paper:** *Machine Learning–Based Surrogate Modeling for Dynamic Coefficients of
> Finite-Length Hydrodynamic Thrust Bearings*  
> **Authors:** Rafael R. P. de Melo, Thales F. Peixoto  
> **Conference:** MECSOL 2026 — São José dos Campos, Brazil, October 19–21, 2026  
> **Lab:** LAMAR / FEM-UNICAMP

---

## Overview

Hydrodynamic thrust bearings support axial loads in rotating machinery through a
pressurised lubricant film. Accurately characterising their **dynamic stiffness (K)**
and **damping (C)** coefficients is critical for rotordynamic stability analysis.
Classical 1-D analytical models ignore side leakage, which can reduce K and C by up
to **92 %** and **57 %** respectively for realistic pad aspect ratios.

This repository implements a complete end-to-end pipeline:

1. **2-D Reynolds solver** — Finite Difference Method with Gauss–Seidel/SOR on a
   100 × 100 mesh, accounting for side leakage through the aspect ratio Λ.
2. **Structured dataset generation** — 4,500 operating points over
   {H₀, V, Λ} with checkpoint-based resumability.
3. **Surrogate training and comparison** — Random Forest, XGBoost, and MLP
   regressors benchmarked on a Λ-stratified 80/20 split.
4. **Validated surrogate** — Random Forest achieves R² = 1.000000 on both K and C,
   with a **69× speed-up** per prediction over the full solver.

---

## Key Results

| Metric | Value |
|--------|-------|
| Side-leakage reduction in \|K\| (Λ ≈ 1.08, H₀ ≈ 0.46) | **92 %** |
| Side-leakage reduction in \|C\| (Λ ≈ 1.08, H₀ ≈ 0.46) | **57 %** |
| Best surrogate R² — K (Random Forest) | **1.000000** |
| Best surrogate R² — C (Random Forest) | **1.000000** |
| Speed-up vs 2-D solver (single call) | **69×** |
| Speed-up vs 2-D solver (batch mode) | **> 10,000×** |

---

## Repository Structure

```
.
├── src/
│   ├── solver/
│   │   └── reynolds_solver.py       # 2-D FDM Gauss–Seidel/SOR pressure solver
│   ├── ml/
│   │   ├── generate_dataset.py      # Dataset generation with checkpoint support
│   │   └── train_surrogate.py       # Model training, evaluation, and export
│   ├── visualization/
│   │   ├── plot_pressure.py         # 1-D pressure distribution (Fig. 3)
│   │   ├── plot_pressure_2d.py      # 3-D pressure surface (Fig. 4)
│   │   ├── plot_coefficients.py     # K and C vs H₀ (Figs. 5–6)
│   │   └── plot_surrogate.py        # Surrogate validation plots (Figs. 7–8)
│   └── utils/
│       └── analytical.py            # Closed-form 1-D reference formulas
├── data/
│   └── dataset_thrust_2D.csv        # Pre-computed 4,500-point dataset
├── models/
│   ├── best_model_K.pkl             # Trained Random Forest for K
│   └── best_model_C.pkl             # Trained Random Forest for C
├── figures/
│   ├── paper/                       # Publication-ready figures (300 dpi)
│   └── exploratory/                 # Development and validation plots
├── paper/
│   └── MSL-2026-RafaelMelo.tex      # MECSOL 2026 LaTeX manuscript
├── tests/
│   └── test_solver.py               # Unit tests for solver and analytics
├── docs/
│   └── equations.md                 # Mathematical derivations reference
├── environment.yml                  # Conda environment specification
├── requirements.txt                 # Pip requirements
└── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Raf-Pimentel/Paper--ML-for-Thrust-Bearings-DynamicCoefficients-.git
cd Paper--ML-for-Thrust-Bearings-DynamicCoefficients-

# Option A — conda (recommended)
conda env create -f environment.yml
conda activate thrust-bearings-ml

# Option B — pip
pip install -r requirements.txt
```

### 2. Run a single bearing evaluation

```python
from src.solver.reynolds_solver import ThrustBearingSolver

solver = ThrustBearingSolver(nx=100, ny=100)
result = solver.solve(H0=0.5, V=0.0, Lambda=1.0)
print(f"Wz = {result.Wz:.4f}   K = {result.K:.4f}   C = {result.C:.4f}")
```

### 3. Predict with the trained surrogate (69× faster)

```python
import joblib, numpy as np

model_K = joblib.load("models/best_model_K.pkl")
model_C = joblib.load("models/best_model_C.pkl")

# Predict K and C at a single operating point
X = np.array([[0.5, 0.0, 1.0]])   # [H0, V, Lambda]
print(f"K = {model_K.predict(X)[0]:.6f}")
print(f"C = {model_C.predict(X)[0]:.6f}")
```

### 4. Regenerate the dataset (≈ several hours on a single CPU core)

```bash
python src/ml/generate_dataset.py
```

Progress is saved every 500 rows and the run can be resumed after interruption.

### 5. Retrain the surrogate models

```bash
python src/ml/train_surrogate.py
```

---

## Physical Background

The dimensionless 2-D Reynolds equation for a fixed-incline thrust pad is:

$$\frac{\partial}{\partial X}\!\left(H^3\frac{\partial P}{\partial X}\right)
+\Lambda^2\frac{\partial}{\partial Y}\!\left(H^3\frac{\partial P}{\partial Y}\right)
=\frac{\partial H}{\partial X}+12V$$

with pressure scale $P = ps_h^2/(\mu u_b l_x)$, film profile $H = H_0 + (1-X)$,
and Dirichlet boundary conditions $P = 0$ on all four edges.

The dynamic coefficients are defined as:

$$K = -\frac{\partial W_z}{\partial H_0}, \qquad C = -\frac{\partial W_z}{\partial V}$$

and their 1-D analytical reference formulas (valid for $\Lambda \to 0$) are:

$$W_z = (1-12V)\!\left[\ln\!\frac{H_0+1}{H_0}-\frac{2}{2H_0+1}\right]$$

$$K_{1D} = (1-12V)\!\left[\frac{1}{H_0}-\frac{1}{H_0+1}-\frac{4}{(1+2H_0)^2}\right]$$

$$C_{1D} = 12\!\left[\ln\!\frac{H_0+1}{H_0}-\frac{2}{2H_0+1}\right]$$

See [`docs/equations.md`](docs/equations.md) for full derivations.

---

## Parameter Space

| Parameter | Symbol | Range | Grid points |
|-----------|--------|-------|-------------|
| Film-thickness ratio | H₀ | [0.05, 3.0] | 30 |
| Squeeze velocity | V | [−2.0, 2.0] | 15 |
| Pad aspect ratio | Λ | [0.25, 4.0] | 10 |
| **Total combinations** | | | **4,500** |

---

## Surrogate Accuracy

| Target | Model | RMSE (full) | R² (full) | RMSE (H₀ < 0.1) | R² (H₀ < 0.1) |
|--------|-------|-------------|-----------|-----------------|----------------|
| K | **Random Forest** ★ | 7 × 10⁻⁶ | **1.000000** | < 10⁻⁶ | **1.000000** |
| K | XGBoost | 7 × 10⁻⁴ | 1.000000 | 1.7 × 10⁻⁴ | 1.000000 |
| K | ANN (MLP) | 5.9 × 10⁻² | 0.998895 | 3.0 × 10⁻¹ | 0.990113 |
| C | **Random Forest** ★ | 7.6 × 10⁻⁵ | **1.000000** | 3 × 10⁻⁶ | **1.000000** |
| C | XGBoost | 1.7 × 10⁻⁴ | 0.999998 | 1.7 × 10⁻⁴ | 1.000000 |
| C | ANN (MLP) | 7.1 × 10⁻³ | 0.996622 | 3.2 × 10⁻² | 0.987961 |

★ Selected model — retrained on the full 4,500-point dataset.

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{deMelo2026,
  author    = {de Melo, Rafael R. P. and Peixoto, Thales F.},
  title     = {Machine Learning--Based Surrogate Modeling for Dynamic Coefficients
               of Finite-Length Hydrodynamic Thrust Bearings},
  booktitle = {Proceedings of the 10th International Symposium on Solid Mechanics
               (MECSOL 2026)},
  year      = {2026},
  address   = {S{\~a}o Jos{\'e} dos Campos, Brazil},
  month     = {October}
}
```

---

## References

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Chen, T. & Guestrin, C. (2016). XGBoost. *KDD '16*, 785–794.
- Childs, D. (1993). *Turbomachinery Rotordynamics*. Wiley.
- Frêne, J. et al. (1997). *Hydrodynamic Lubrication*. Elsevier.
- Hamrock, B. J. et al. (2004). *Fundamentals of Fluid Film Lubrication*, 2nd ed. Marcel Dekker.
- Patankar, S. V. (1980). *Numerical Heat Transfer and Fluid Flow*. Hemisphere.
- Pedregosa, F. et al. (2011). Scikit-learn. *JMLR*, 12, 2825–2830.
- Szeri, A. Z. (2010). *Fluid Film Lubrication*, 2nd ed. Cambridge.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

The first author thanks UNICAMP for the undergraduate research scholarship
(Iniciação Científica). The authors thank LAMAR/FEM-UNICAMP for computational
infrastructure and a stimulating research environment.
