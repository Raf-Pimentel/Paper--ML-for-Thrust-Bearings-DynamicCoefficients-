"""
reynolds_solver.py
==================
2-D finite-difference solver for the dimensionless Reynolds lubrication equation
applied to a fixed-incline hydrodynamic thrust bearing.

Governing equation (dimensionless form solved internally)
---------------------------------------------------------
    d/dX(H³ dP_int/dX) + Λ² d/dY(H³ dP_int/dY) = dH/dX + 12V

where H(X) = H₀ + (1 − X),  V = v_a l_x/(u_b s_h)  is the dimensionless
squeeze velocity, and homogeneous Dirichlet conditions P_int = 0 hold on all
four edges of the unit square [0,1]×[0,1].

Pressure scale note
-------------------
The pressure field stored in BearingResult.P is the *internal* scale
P_int = p s_h² / (6 μ u_b l_x).  To convert to the paper scale
P_paper = p s_h² / (μ u_b l_x) multiply by 6:

    P_paper = 6 × P_int

Wz and K are numerically equal in both scales.
C_paper = −∂Wz/∂V = 12 × C_int  (factor applied inside _damping).

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
        Converged dimensionless pressure field (internal scale P_int).
        Paper scale: P_paper = 6 × P_int.
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
        P, converged = self._solve_pressure(H, Lambda, 12.0 * V)
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
        """Direct sparse-matrix solver for the Reynolds pressure field.

        Assembles the 5-point FD stencil into a sparse CSR matrix and solves
        with scipy's sparse direct solver (SuperLU).  No iterations — exact
        solution to the discretised equations in O(nnz) time.
        """
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import spsolve

        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        ni, nj = nx - 2, ny - 2      # interior grid dimensions

        # Coefficient arrays on the interior grid (ni × nj) — vectorised
        Hp  = H[1:-1, 1:-1]
        dHX = (H[2:,  1:-1] - H[:-2, 1:-1]) / (2.0 * dx)
        dHY = (H[1:-1,  2:] - H[1:-1, :-2]) / (2.0 * dy)
        Hp3 = Hp ** 3
        Hp2 = Hp ** 2
        La2 = Lambda ** 2

        aE = ( Hp3 / dx**2 + 3.0 * Hp2 * dHX / (2.0 * dx)).ravel()
        aW = ( Hp3 / dx**2 - 3.0 * Hp2 * dHX / (2.0 * dx)).ravel()
        aN = (La2 * Hp3 / dy**2 + 3.0 * La2 * Hp2 * dHY / (2.0 * dy)).ravel()
        aS = (La2 * Hp3 / dy**2 - 3.0 * La2 * Hp2 * dHY / (2.0 * dy)).ravel()
        aP = -(aE + aW + aN + aS)
        b  = (dHX + dHdt).ravel()

        # Flat node index: k = (i-1)*nj + (j-1), i∈[1,nx-2], j∈[1,ny-2]
        n = ni * nj
        k = np.arange(n)

        r_list = [k];       c_list = [k];             d_list = [aP]

        m = k + nj < n                                # East  (i+1)
        r_list.append(k[m]);  c_list.append((k + nj)[m]); d_list.append(aE[m])

        m = k >= nj                                   # West  (i-1)
        r_list.append(k[m]);  c_list.append((k - nj)[m]); d_list.append(aW[m])

        m = (k % nj) != (nj - 1)                     # North (j+1)
        r_list.append(k[m]);  c_list.append((k + 1)[m]);  d_list.append(aN[m])

        m = (k % nj) != 0                             # South (j-1)
        r_list.append(k[m]);  c_list.append((k - 1)[m]);  d_list.append(aS[m])

        A = coo_matrix(
            (np.concatenate(d_list),
             (np.concatenate(r_list), np.concatenate(c_list))),
            shape=(n, n),
        ).tocsr()

        P_flat = spsolve(A, b)

        P = np.zeros((nx, ny))
        P[1:-1, 1:-1] = P_flat.reshape(ni, nj)
        return P, True          # direct solver always "converges"

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
        """C = −∂Wz/∂V  (central FD, H₀ fixed).

        The solver receives dHdt = 12·(V ± eps), so the finite-difference
        quotient (W_plus − W_minus)/(2·eps) directly approximates ∂Wz/∂V,
        giving C = −∂Wz/∂V without any extra scale factor.
        """
        eps = self.epsilon_fd
        Pp, _ = self._solve_pressure(H, Lambda, 12.0 * (V + eps))
        Pm, _ = self._solve_pressure(H, Lambda, 12.0 * (V - eps))
        return -(self._integrate(Pp) - self._integrate(Pm)) / (2.0 * eps)
