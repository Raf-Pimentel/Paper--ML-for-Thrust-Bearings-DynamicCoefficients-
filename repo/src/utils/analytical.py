"""
analytical.py
=============
Closed-form 1-D reference formulas for the infinite-width (no side leakage)
thrust bearing limit (Λ → 0).

All formulas use the standard pressure scale  P = p s_h² / (μ u_b l_x),
which is consistent with the 2-D FDM solver in `reynolds_solver.py`.
This scale was verified by SymPy integration of the corrected 1-D Reynolds
ODE with RHS = dH/dX + 12V and boundary conditions P(0) = P(1) = 0.

Summary of formulas
-------------------
    Film profile    :  H(X) = H₀ + (1 − X)
    Pressure        :  P_1D = 6 X(1−X) / [(H₀+1−X)²(1+2H₀)]
    Load capacity   :  Wz   = (1 − 12V) [ln((H₀+1)/H₀) − 2/(2H₀+1)]
    Stiffness       :  K    = (1 − 12V) [1/H₀ − 1/(H₀+1) − 4/(1+2H₀)²]
    Damping         :  C    = 12 [ln((H₀+1)/H₀) − 2/(2H₀+1)]

Note: C is independent of V (as expected for a linear Wz–V relationship).

Authors
-------
Rafael R. P. de Melo, Thales F. Peixoto — LAMAR / FEM-UNICAMP — MECSOL 2026
"""

from __future__ import annotations

import numpy as np


def pressure_1d(X: np.ndarray, H0: float) -> np.ndarray:
    """Dimensionless 1-D pressure distribution P₁D(X; H₀).

    Parameters
    ----------
    X : array-like
        Dimensionless coordinate in [0, 1].
    H0 : float
        Dimensionless outlet film-thickness ratio  h₀/sₕ > 0.

    Returns
    -------
    np.ndarray
        Pressure values. Returns zeros where the denominator vanishes.
    """
    X = np.asarray(X, dtype=float)
    num = 6.0 * X * (1.0 - X)
    den = (H0 + 1.0 - X) ** 2 * (1.0 + 2.0 * H0)
    return np.divide(num, den, out=np.zeros_like(num), where=den != 0)


def load_capacity_1d(H0: float | np.ndarray, V: float | np.ndarray = 0.0) -> np.ndarray:
    """Dimensionless load capacity Wz₁D(H₀, V).

    Formula (SymPy-verified, corrected pressure scale):
        Wz = (1 − 12V) [ln((H₀+1)/H₀) − 2/(2H₀+1)]

    Parameters
    ----------
    H0 : float or array-like
        Film-thickness ratio. Must be > 0.
    V : float or array-like
        Squeeze velocity (default 0).

    Returns
    -------
    np.ndarray or float
    """
    H0 = np.asarray(H0, dtype=float)
    V = np.asarray(V, dtype=float)
    term = np.log((H0 + 1.0) / H0) - 2.0 / (2.0 * H0 + 1.0)
    return (1.0 - 12.0 * V) * term


def stiffness_1d(H0: float | np.ndarray, V: float | np.ndarray = 0.0) -> np.ndarray:
    """Dimensionless stiffness coefficient K₁D(H₀, V) = −∂Wz₁D/∂H₀.

    Formula:
        K = (1 − 12V) [1/H₀ − 1/(H₀+1) − 4/(1+2H₀)²]

    Parameters
    ----------
    H0, V : float or array-like
    """
    H0 = np.asarray(H0, dtype=float)
    V = np.asarray(V, dtype=float)
    bracket = 1.0 / H0 - 1.0 / (H0 + 1.0) - 4.0 / (1.0 + 2.0 * H0) ** 2
    return (1.0 - 12.0 * V) * bracket


def damping_1d(H0: float | np.ndarray) -> np.ndarray:
    """Dimensionless damping coefficient C₁D(H₀) = −∂Wz₁D/∂V.

    Formula (independent of V):
        C = 12 [ln((H₀+1)/H₀) − 2/(2H₀+1)]

    Parameters
    ----------
    H0 : float or array-like
        Must be > 0.
    """
    H0 = np.asarray(H0, dtype=float)
    term = np.log((H0 + 1.0) / H0) - 2.0 / (2.0 * H0 + 1.0)
    return 12.0 * term
