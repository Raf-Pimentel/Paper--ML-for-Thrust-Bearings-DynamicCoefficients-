"""
reynolds_solver.py
==================
2-D finite-difference solver for the dimensionless Reynolds lubrication equation
applied to a fixed-incline hydrodynamic thrust bearing.

Governing equation (Eq. 6 — MECSOL 2026 paper)
------------------------------------------------
    d/dX(H³ dP/dX) + Λ² d/dY(H³ dP/dY) = dH/dX + 12V

where the pressure scale is  P = p s_h² / (μ u_b l_x),  the film geometry is
H(X) = H₀ + (1 − X),  and homogeneous Dirichlet conditions P = 0 hold on all
four edges of the unit square [0,1]×[0,1].

Numerical method
----------------
Second-order central finite differences with Gauss–Seidel/SOR iteration.
Near-optimal SOR factor:  ω ≈ 2 − √2 π √(1/nₓ² + 1/nᵧ²).
Convergence: max|P_new − P_old| < tolerance.

Dynamic coefficients
--------------------
K = −∂Wz/∂H₀  (stiffness)   — central FD, ε = 1e-4, V = 0
C = −∂Wz/∂V   (damping)     — central FD, ε = 1e-4, H₀ fixed

References
----------
Hamrock, B. J., Schmid, S. R., Jacobson, B. O. (2004).
    Fundamentals of Fluid Film Lubrication (2nd ed.). Marcel Dekker.
Frêne, J. et al. (1997). Hydrodynamic Lubrication. Elsevier.

Authors
-------
Rafael R. P. de Melo, Thales F. Peixoto — LAMAR / FEM-UNICAMP — MECSOL 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi, sqrt

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BearingResult:
    """Output of a single bearing operating-point evaluation.

    Attributes
    ----------
    H0 : float
        Dimensionless outlet film-thickness ratio h₀/sₕ.
    V : float
        Dimensionless squeeze velocity.
    Lambda : float
        Pad aspect ratio lₓ/lᵧ.
    P : np.ndarray, shape (nx, ny)
        Converged dimensionless pressure field.
    Wz : float
        Dimensionless load capacity  ∬ P dX dY.
    K : float
        Dimensionless stiffness coefficient  −∂Wz/∂H₀.
    C : float
        Dimensionless damping coefficient    −∂Wz/∂V.
    omega : float
        SOR relaxation factor used.
    converged : bool
        True if the Gauss–Seidel loop converged within max_iter.
    """

    H0: float
    V: float
    Lambda: float
    P: np.ndarray = field(repr=False)
    Wz: float = 0.0
    K: float = 0.0
    C: float = 0.0
    omega: float = 0.0
    converged: bool = False


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class ThrustBearingSolver:
    """2-D Reynolds equation solver for a fixed-incline thrust pad.

    Parameters
    ----------
    nx, ny : int
        Number of grid points in X and Y directions (default 100 each).
    tolerance : float
        Gauss–Seidel convergence criterion (default 1e-6).
    max_iter : int
        Maximum iterations per solve call (default 2000).
    epsilon_fd : float
        Finite-difference step for K and C computation (default 1e-4).

    Examples
    --------
    >>> from src.solver.reynolds_solver import ThrustBearingSolver
    >>> solver = ThrustBearingSolver()
    >>> result = solver.solve(H0=0.5, V=0.0, Lambda=1.0)
    >>> print(f"Wz={result.Wz:.4f}  K={result.K:.4f}  C={result.C:.4f}")
    """

    def __init__(
        self,
        nx: int = 100,
        ny: int = 100,
        tolerance: float = 1e-6,
        max_iter: int = 2000,
        epsilon_fd: float = 1e-4,
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.epsilon_fd = epsilon_fd

        # Coordinate arrays (built once, reused for every solve call)
        self.X = np.linspace(0.0, 1.0, nx)
        self.Y = np.linspace(0.0, 1.0, ny)
        self.dx = self.X[1] - self.X[0]
        self.dy = self.Y[1] - self.Y[0]
        self.X_grid, self.Y_grid = np.meshgrid(self.X, self.Y, indexing="ij")

        # Near-optimal SOR relaxation factor (Frêne et al., 1997)
        self.omega = 2.0 - sqrt(2) * pi * sqrt(1.0 / nx**2 + 1.0 / ny**2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, H0: float, V: float, Lambda: float) -> BearingResult:
        """Evaluate all bearing quantities at one operating point.

        Parameters
        ----------
        H0 : float
            Dimensionless outlet film-thickness ratio.  Must be > 0.
        V : float
            Dimensionless squeeze velocity (positive = surfaces separating).
        Lambda : float
            Pad aspect ratio lₓ/lᵧ.  Must be > 0.

        Returns
        -------
        BearingResult
        """
        H = self._film(H0)
        P, converged = self._solve_pressure(H, Lambda, V)
        Wz = self._integrate(P)
        K = self._stiffness(H0, Lambda)
        C = self._damping(H, Lambda, V)
        return BearingResult(
            H0=H0, V=V, Lambda=Lambda,
            P=P, Wz=Wz, K=K, C=C,
            omega=self.omega, converged=converged,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _film(self, H0: float) -> np.ndarray:
        """Film-thickness matrix  H(X) = H₀ + (1 − X)."""
        return H0 + 1.0 - self.X_grid

    def _solve_pressure(
        self, H: np.ndarray, Lambda: float, dHdt: float
    ) -> tuple[np.ndarray, bool]:
        """Gauss–Seidel/SOR iteration for the dimensionless pressure field."""
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        omega = self.omega
        P = np.zeros((nx, ny))

        for _ in range(self.max_iter):
            P_old = P.copy()
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    Hp = H[i, j]
                    Pe, Pw = P[i + 1, j], P[i - 1, j]
                    Pn, Ps = P[i, j + 1], P[i, j - 1]

                    dHdX = (H[i + 1, j] - H[i - 1, j]) / (2.0 * dx)
                    dHdY = (H[i, j + 1] - H[i, j - 1]) / (2.0 * dy)

                    # Source term consistent with pressure scale P = p s²/(μ u l)
                    Bp = dHdX + dHdt

                    tXd = Hp**3 * (Pe + Pw) / dx**2
                    tXc = 3.0 * Hp**2 * dHdX * (Pe - Pw) / (2.0 * dx)
                    tYd = Lambda**2 * Hp**3 * (Pn + Ps) / dy**2
                    tYc = 3.0 * Lambda**2 * Hp**2 * dHdY * (Pn - Ps) / (2.0 * dy)
                    denom = 2.0 * Hp**3 * (1.0 / dx**2 + Lambda**2 / dy**2)

                    pij = (tXd + tXc + tYd + tYc - Bp) / denom
                    P[i, j] += omega * (pij - P[i, j])

            if np.max(np.abs(P - P_old)) < self.tolerance:
                return P, True

        return P, False  # max_iter reached without convergence

    def _integrate(self, P: np.ndarray) -> float:
        """Dimensionless load capacity via Riemann sum  Wz ≈ Σ P ΔX ΔY."""
        return float(np.sum(P) * self.dx * self.dy)

    def _stiffness(self, H0: float, Lambda: float) -> float:
        """K = −∂Wz/∂H₀  (central FD, V = 0)."""
        eps = self.epsilon_fd
        Hp, _ = self._solve_pressure(self._film(H0 + eps), Lambda, 0.0)
        Hm, _ = self._solve_pressure(self._film(H0 - eps), Lambda, 0.0)
        return -(self._integrate(Hp) - self._integrate(Hm)) / (2.0 * eps)

    def _damping(self, H: np.ndarray, Lambda: float, V: float) -> float:
        """C = −∂Wz/∂V  (central FD, H₀ fixed)."""
        eps = self.epsilon_fd
        Pp, _ = self._solve_pressure(H, Lambda, V + eps)
        Pm, _ = self._solve_pressure(H, Lambda, V - eps)
        return -(self._integrate(Pp) - self._integrate(Pm)) / (2.0 * eps)
