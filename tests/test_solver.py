"""
test_solver.py
==============
Unit tests for the Reynolds solver and the 1-D analytical reference formulas.

Run with:
    pytest tests/test_solver.py -v

Authors
-------
Rafael R. P. de Melo, Thales F. Peixoto — LAMAR / FEM-UNICAMP — MECSOL 2026
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.analytical import (
    pressure_1d, load_capacity_1d, stiffness_1d, damping_1d
)

# ---------------------------------------------------------------------------
# Analytical formula tests
# ---------------------------------------------------------------------------

class TestAnalytical1D:
    """Verify the closed-form 1-D formulas against known values and identities."""

    def test_pressure_boundary_conditions(self):
        """P₁D must vanish at X = 0 and X = 1."""
        X = np.array([0.0, 1.0])
        P = pressure_1d(X, H0=1.0)
        np.testing.assert_allclose(P, 0.0, atol=1e-12)

    def test_pressure_peak_positive(self):
        """Pressure must be positive in the interior for H₀ > 0."""
        X = np.linspace(0.0, 1.0, 200)
        P = pressure_1d(X, H0=0.5)
        assert np.all(P[1:-1] >= 0.0), "Interior pressure should be non-negative."

    def test_wz_finite_nonzero(self):
        """Wz₁D should be finite and non-zero for typical operating points."""
        Wz = load_capacity_1d(H0=1.0, V=0.0)
        assert np.isfinite(Wz)
        assert abs(Wz) > 1e-9

    def test_wz_linear_in_V(self):
        """Wz = (1-12V)*f(H₀) → must be linear in V."""
        H0 = 1.0
        V_vals = np.linspace(-2, 2, 11)
        Wz_vals = load_capacity_1d(H0, V_vals)
        # Fit a line to Wz(V) — R² should be 1.0
        coeffs = np.polyfit(V_vals, Wz_vals, 1)
        residuals = Wz_vals - np.polyval(coeffs, V_vals)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-12)

    def test_K_finite_differentiation(self):
        """K₁D must agree with numerical d/dH₀ of Wz₁D."""
        H0 = 1.0
        eps = 1e-5
        K_num = -(load_capacity_1d(H0 + eps) - load_capacity_1d(H0 - eps)) / (2 * eps)
        K_ana = stiffness_1d(H0, V=0.0)
        np.testing.assert_allclose(K_ana, K_num, rtol=1e-5)

    def test_C_finite_differentiation(self):
        """C₁D must agree with numerical d/dV of Wz₁D."""
        H0 = 1.0
        V = 0.0
        eps = 1e-5
        C_num = -(load_capacity_1d(H0, V + eps) - load_capacity_1d(H0, V - eps)) / (2 * eps)
        C_ana = damping_1d(H0)
        np.testing.assert_allclose(float(C_ana), C_num, rtol=1e-5)

    def test_C_independent_of_V(self):
        """C₁D = −∂Wz/∂V must not depend on V."""
        H0 = 1.5
        V_vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        eps = 1e-5
        C_numerics = np.array([
            -(load_capacity_1d(H0, V + eps) - load_capacity_1d(H0, V - eps)) / (2 * eps)
            for V in V_vals
        ])
        np.testing.assert_allclose(C_numerics, C_numerics[0], rtol=1e-8,
                                   err_msg="C should be independent of V.")

    @pytest.mark.parametrize("H0", [0.1, 0.5, 1.0, 2.0, 3.0])
    def test_formulas_vectorized(self, H0):
        """Scalar and array inputs should give consistent results."""
        V = 0.5
        H0_arr = np.array([H0])
        np.testing.assert_allclose(
            float(load_capacity_1d(H0, V)),
            float(load_capacity_1d(H0_arr, V).item()),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            float(stiffness_1d(H0, V)),
            float(stiffness_1d(H0_arr, V).item()),
            rtol=1e-12,
        )

    def test_wz_correct_value(self):
        """Wz at H₀=1, V=0 must equal ln(2) − 2/3 (SymPy-verified)."""
        expected = np.log(2.0) - 2.0 / 3.0   # ≈ 0.02648
        result = float(load_capacity_1d(1.0, 0.0))
        np.testing.assert_allclose(result, expected, rtol=1e-10)
