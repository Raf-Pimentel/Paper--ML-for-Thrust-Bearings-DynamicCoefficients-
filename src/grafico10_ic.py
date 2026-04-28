# Gráfico que plota a vazão volumétrica Q em função de Ho

import numpy as np
import matplotlib.pyplot as plt

def calcular_vazao_adimensional(Ho):
    """
    Calcula a Vazão Volumétrica Adimensional (Q) para um mancal de inclinação fixa.
    Baseado na Equação 8.36 do livro.

    Args:
        Ho (np.array): Array com valores da razão de espessura do filme (h₀ / sₕ).

    Returns:
        np.array: Array com os valores de vazão adimensional Q.
    """
    numerador = 2 * Ho * (1 + Ho)
    denominador = 1 + 2 * Ho
    
    # A função é segura para Ho=0, mas começamos de um valor pequeno por consistência.
    return numerador / denominador

# --- Configuração dos Dados para o Gráfico ---

# 1. Criar um array de pontos para o eixo X (H₀).
Ho_coords = np.linspace(0, 2.0, 500)

# 2. Calcular os valores de Y (Q) correspondentes
Q_valores = calcular_vazao_adimensional(Ho_coords)

# --- Criação do Gráfico ---

# 1. Configurar a figura e os eixos
fig, ax = plt.subplots(figsize=(10, 7))

# 2. Plotar a curva
ax.plot(Ho_coords, Q_valores, lw=2.5, color='black')

# --- Customização e Legendas (para replicar a Figura 8.11) ---

ax.set_xlabel(r'Razão de Espessura do Filme, $H_0 = h_0/s_h$', fontsize=12)
ax.set_ylabel(r'Vazão Volumétrica Adimensional, $Q = \frac{2q_x^{\prime}}{u_b s_h}$', fontsize=12)
ax.set_title('Figura 8.11: Efeito da Razão de Espessura na Vazão Volumétrica', fontsize=14)

# Configurar os limites dos eixos para ficarem idênticos ao original
ax.set_xlim(0, 2.0)
ax.set_ylim(0, 3.0)

# Adicionar uma grade para facilitar a leitura
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Mostrar o gráfico final
plt.show()