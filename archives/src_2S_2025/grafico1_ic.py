import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi

# =============================================================================
# SOLUÇÃO ANALÍTICA 1D  (Eq. 6 do paper, V = 0)
# Escala de pressão: P = p * s_h^2 / (mu * u_b * l_x)   [SEM o 6 no denominador]
# O fator 6 no numerador é parte da solução analítica da EDO de Reynolds,
# não da escala de adimensionalização.
# =============================================================================
def P_analytical(X, H0):
    """
    Solução fechada da equação de Reynolds 1D para perfil linear H = H0 + 1 - X,
    com condições de contorno P(0) = P(1) = 0 e V = 0.
    Referência: Eq. (6) do paper.
    """
    num = 6.0 * X * (1.0 - X)
    den = (H0 + 1.0 - X)**2 * (1.0 + 2.0 * H0)
    return np.divide(num, den, out=np.zeros_like(num), where=den != 0)


# =============================================================================
# SOLVER 1D NUMÉRICO POR DIFERENÇAS FINITAS + SOR
# Resolve d/dX(H^3 * dP/dX) = dH/dX   com P(0) = P(1) = 0
# Usado para validação: pontos discretos devem coincidir com a curva analítica.
# =============================================================================
def P_numerical_1D(n_pts, H0, alpha=1e-9):
    """
    Resolve a equação de Reynolds 1D por diferenças finitas centradas + SOR.

    Parameters
    ----------
    n_pts : int   — número de pontos da grade (incluindo bordas)
    H0    : float — razão de espessura mínima do filme
    alpha : float — tolerância de convergência

    Returns
    -------
    X_inner : np.array — coordenadas dos pontos internos
    P_inner : np.array — pressão adimensional nos pontos internos
    """
    X = np.linspace(0.0, 1.0, n_pts)
    dx = X[1] - X[0]
    H = H0 + 1.0 - X

    # Fator SOR ótimo para domínio 1D
    omega = 2.0 - sqrt(2.0) * pi / n_pts

    P = np.zeros(n_pts)

    for _ in range(10_000):
        P_old = P.copy()
        for i in range(1, n_pts - 1):
            Hp   = H[i]
            dHdX = (H[i + 1] - H[i - 1]) / (2.0 * dx)   # dH/dX = -1 para perfil linear

            # Coeficientes (sem termo em Y pois Λ = 0)
            aE   =  Hp**3 / dx**2
            bE   =  3.0 * Hp**2 * dHdX / (2.0 * dx)
            aP   =  2.0 * Hp**3 / dx**2
            Bp   =  6.0 * dHdX                            # termo fonte (V = 0)
            # Fator 6 obrigatório: com a escala P = p·s_h²/(μ·u_b·l_x) do paper,
            # a integração da equação de Reynolds dimensional gera
            # d/dX(H³·dP/dX) = 6·dH/dX  (verificado: LHS da solução analítica = -6)

            P_gs = (aE * (P[i + 1] + P[i - 1])
                    + bE * (P[i + 1] - P[i - 1])
                    - Bp) / aP

            P[i] = P[i] + omega * (P_gs - P[i])

        if np.max(np.abs(P - P_old)) < alpha:
            break

    # Retorna apenas os pontos internos (bordas são zero por condição de contorno)
    return X[1:-1], P[1:-1]


# =============================================================================
# PARÂMETROS DO GRÁFICO
# =============================================================================
X_fine  = np.linspace(0.0, 1.0, 600)          # grade fina para a curva analítica
H0_list = [2, 1, 0.5, 0.25]
colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']   # azul, laranja, verde, vermelho
n_numerical = 18                               # pontos da grade numérica esparsa

FONTSIZE_TITLE  = 16
FONTSIZE_LABEL  = 14
FONTSIZE_LEGEND = 13
FONTSIZE_TICKS  = 12

# =============================================================================
# PLOTAGEM
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 6))

for H0, color in zip(H0_list, colors):

    # — Curva analítica (linha contínua) —
    P_ana = P_analytical(X_fine, H0)

    if H0 >= 1:
        label_ana = rf'$H_0 = {int(H0)}$'
    else:
        label_ana = rf'$H_0 = 1/{int(round(1/H0))}$'

    ax.plot(X_fine, P_ana, color=color, linewidth=2.0, label=label_ana)

    # — Solução numérica 1D (marcadores dispersos) —
    X_num, P_num = P_numerical_1D(n_numerical, H0)
    ax.plot(X_num, P_num,
            marker='o', linestyle='none',
            color=color, markersize=6,
            markeredgecolor='black', markeredgewidth=0.6,
            zorder=5)

# Entrada manual na legenda para explicar os marcadores
ax.plot([], [], marker='o', linestyle='none',
        color='gray', markersize=6,
        markeredgecolor='black', markeredgewidth=0.6,
        label='Numerical (FDM, 1-D)')

# =============================================================================
# FORMATAÇÃO
# =============================================================================
ax.set_xlabel(r'Dimensionless coordinate, $X = x/l_x$',
              fontsize=FONTSIZE_LABEL)
ax.set_ylabel(r'Dimensionless pressure, $P = p\,s_h^2/(\mu\,u_b\,l_x)$',
              fontsize=FONTSIZE_LABEL)
ax.set_title(r'1-D pressure distribution for fixed-incline thrust bearing ($\Lambda = 0$)',
             fontsize=FONTSIZE_TITLE)

ax.legend(title=r'Film-thickness ratio, $H_0 = h_0/s_h$',
          fontsize=FONTSIZE_LEGEND,
          title_fontsize=FONTSIZE_LEGEND)

ax.tick_params(axis='both', labelsize=FONTSIZE_TICKS)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 3.5)
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('Distribuicao_Pressao_H0.png', dpi=300, bbox_inches='tight')
plt.show()