# Gráfico da figura 8.8 do livro "Hydrodynamic Lubrication" de J. Hamrock
# Versão aprimorada com melhor visualização dos eixos

import numpy as np
import matplotlib.pyplot as plt

def calcular_carga_normal_adimensional(Ho):
    """
    Calcula a Carga Normal Adimensional (Wz) para um mancal de inclinação fixa.
    Baseado na Equação 8.30. 

    Args:
        Ho (np.array): Array com valores da razão de espessura do filme (h₀ / sₕ).

    Returns:
        np.array: Array com os valores de carga adimensional Wz.
    """
    # Termo do logaritmo natural (ln) na equação
    termo_log = np.log((Ho + 1) / Ho)
    
    # Termo da fração na equação
    termo_fracao = 12 / (1 + 2 * Ho)
    
    # Cálculo final
    Wz = 6 * termo_log - termo_fracao
    return Wz

# --- Configuração dos Dados para o Gráfico ---

# 1. Criar um array de pontos para o eixo X (H₀).
#    Começamos de um valor muito pequeno (0.01) para evitar erro no log(0).
Ho_coords = np.linspace(0.01, 2.0, 500)

# 2. Calcular os valores de Y (Wz) correspondentes
Wz_valores = calcular_carga_normal_adimensional(Ho_coords)

# --- Criação do Gráfico ---

# 1. Configurar a figura e os eixos
fig, ax = plt.subplots(figsize=(8, 6))

# 2. Plotar a curva
ax.plot(Ho_coords, Wz_valores, lw=2.5, color='black')

# --- Customização e Legendas (para replicar a Figura 8.8) ---

ax.set_xlabel(r'Razão de Espessura do Filme, $H_0 = h_0/s_h$', fontsize=12)
ax.set_ylabel(r'Capacidade de Carga Normal Adimensional', fontsize=12)
ax.set_title('Efeito da Razão de Espessura do Filme na Capacidade de Carga', fontsize=14)

# Configurar os limites dos eixos para ficarem idênticos ao original
ax.set_xlim(0, 2.0)
ax.set_ylim(0, 4.0)

# Adicionar uma grade para facilitar a leitura
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()