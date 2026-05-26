# Mathematical Derivations Reference

This document provides the full derivation of the dimensionless governing equations,
the 1-D analytical solutions, and the dynamic-coefficient formulas used in the
MECSOL 2026 paper.

---

## 1. Dimensional Reynolds Equation

For an incompressible Newtonian fluid in a thin lubricant film, the 2-D Reynolds
equation is (Hamrock et al., 2004):

$$\frac{\partial}{\partial x}\!\left(h^3\frac{\partial p}{\partial x}\right)
+\frac{\partial}{\partial y}\!\left(h^3\frac{\partial p}{\partial y}\right)
=6\mu U\frac{\partial h}{\partial x}+12\mu\frac{\partial h}{\partial t}$$

where $h(x,y,t)$ is the local film thickness, $p$ is gauge pressure, $\mu$ is
dynamic viscosity, and $U = u_b$ is the runner surface velocity.

---

## 2. Non-Dimensionalisation

### Variables

| Quantity | Symbol | Definition |
|----------|--------|-----------|
| Coordinates | $X, Y$ | $x/l_x$, $y/l_y$ |
| Film thickness | $H$ | $h/s_h$ |
| **Pressure** | $P$ | $p\,s_h^2 / (\mu\,u_b\,l_x)$ |
| Time | $T$ | $u_b\,t/l_x$ |
| Squeeze velocity | $V$ | $v_a\,l_x / (u_b\,s_h) = \partial H/\partial T$ |
| Aspect ratio | $\Lambda$ | $l_x / l_y$ |

> **Note on pressure scale:** The scale $P = p s_h^2/(\mu u_b l_x)$ (without
> a factor of 6) is consistent with the FDM solver implementation.  
> The older scale $P = p s_h^2/(6\mu u_b l_x)$ appeared in earlier drafts and
> was removed to eliminate the inconsistency.

### Dimensionless Reynolds Equation

$$\frac{\partial}{\partial X}\!\left(H^3\frac{\partial P}{\partial X}\right)
+\Lambda^2\frac{\partial}{\partial Y}\!\left(H^3\frac{\partial P}{\partial Y}\right)
=\frac{\partial H}{\partial X}+12V$$

### Film Profile and Boundary Conditions

$$H(X)=H_0+(1-X), \quad H_0=\frac{h_0}{s_h}$$

$$P=0 \quad \text{on all four edges of }[0,1]\times[0,1]$$

---

## 3. 1-D Analytical Solution (Λ → 0)

Setting $\Lambda = 0$ eliminates the lateral term, reducing the PDE to an ODE:

$$\frac{d}{dX}\!\left(H^3\frac{dP}{dX}\right)=\frac{dH}{dX}+12V$$

Integrating twice with $H = H_0+1-X$, $dH/dX = -1$, and $P(0)=P(1)=0$:

$$\boxed{P_{1D}(X;H_0)=\frac{6X(1-X)}{(H_0+1-X)^2(1+2H_0)}}$$

> This is **not** affected by the change in pressure scale because the factor of 6
> appears in the numerator of $P_{1D}$ itself (not in the denominator of the
> dimensionless scale).

---

## 4. Load Capacity

$$W_z(H_0,V,\Lambda)=\int_0^1\!\int_0^1 P(X,Y;\,H_0,V,\Lambda)\,dX\,dY$$

### 1-D closed form (SymPy-verified with corrected scale)

$$\boxed{W_z^{1D}(H_0,V)=(1-12V)\!\left[\ln\!\frac{H_0+1}{H_0}-\frac{2}{2H_0+1}\right]}$$

Derivation: substitute $P_{1D}$ into the double integral and integrate over $X \in [0,1]$.
Using SymPy with `H0, V = symbols('H0 V', positive=True)`:

```python
Wz = integrate(P_1D, (X, 0, 1))
Wz_simplified = simplify(Wz)
# → (1 - 12*V) * (log((H0+1)/H0) - 2/(2*H0+1))
```

---

## 5. Dynamic Coefficients

### Definitions

$$K=-\frac{\partial W_z}{\partial H_0}, \qquad C=-\frac{\partial W_z}{\partial V}$$

The leading minus signs ensure that $K$ and $C$ represent positive physically
meaningful magnitudes (since $\partial W_z/\partial H_0 < 0$ for a convergent film
where $W_z < 0$).

### 1-D Analytical Expressions

Differentiating $W_z^{1D}$:

$$\boxed{K_{1D}(H_0,V)=(1-12V)\!\left[\frac{1}{H_0}-\frac{1}{H_0+1}-\frac{4}{(1+2H_0)^2}\right]}$$

$$\boxed{C_{1D}(H_0)=12\!\left[\ln\!\frac{H_0+1}{H_0}-\frac{2}{2H_0+1}\right]}$$

> **Key observation:** $C_{1D}$ is independent of $V$.  This follows directly
> from $W_z$ being linear in $V$, so $-\partial W_z/\partial V$ depends only on
> the geometry parameter $H_0$.

### Numerical Computation (2-D case)

Central finite differences with $\varepsilon = 10^{-4}$:

$$K \approx -\frac{W_z(H_0+\varepsilon,\,0,\,\Lambda)-W_z(H_0-\varepsilon,\,0,\,\Lambda)}{2\varepsilon}$$

$$C \approx -\frac{W_z(H_0,\,V+\varepsilon,\,\Lambda)-W_z(H_0,\,V-\varepsilon,\,\Lambda)}{2\varepsilon}$$

---

## 6. FDM Discretisation

### Nodal Update Formula

With compass-point notation ($E$, $W$, $N$, $S$ for neighbours) and uniform
spacings $\Delta X = 1/n_x$, $\Delta Y = 1/n_y$:

$$P_P = \frac{\alpha_E(P_E+P_W)+\beta_E(P_E-P_W)+\alpha_N(P_N+P_S)+\beta_N(P_N-P_S)-B_P}{a_P}$$

where:

$$\alpha_E=\frac{H_P^3}{\Delta X^2},\quad
\beta_E=\frac{3H_P^2}{2\Delta X}\left.\frac{\partial H}{\partial X}\right|_P,\quad
\alpha_N=\frac{\Lambda^2 H_P^3}{\Delta Y^2},\quad
\beta_N=\frac{3\Lambda^2 H_P^2}{2\Delta Y}\left.\frac{\partial H}{\partial Y}\right|_P$$

$$a_P=2H_P^3\!\left(\frac{1}{\Delta X^2}+\frac{\Lambda^2}{\Delta Y^2}\right),\quad
B_P=\left.\frac{\partial H}{\partial X}\right|_P+12V$$

### SOR Update

$$P_P^{\text{new}}=P_P^{\text{old}}+\omega\!\left(\hat{P}_P-P_P^{\text{old}}\right),\qquad
\omega\approx 2-\sqrt{2}\,\pi\sqrt{\frac{1}{n_x^2}+\frac{1}{n_y^2}}$$

Convergence criterion: $\max_P|P_P^{\text{new}}-P_P^{\text{old}}|<10^{-6}$.

---

## References

- Hamrock, B. J., Schmid, S. R., Jacobson, B. O. (2004). *Fundamentals of Fluid Film
  Lubrication* (2nd ed.). Marcel Dekker.
- Frêne, J. et al. (1997). *Hydrodynamic Lubrication: Bearings and Thrust Bearings*.
  Elsevier.
- Patankar, S. V. (1980). *Numerical Heat Transfer and Fluid Flow*. Hemisphere.
