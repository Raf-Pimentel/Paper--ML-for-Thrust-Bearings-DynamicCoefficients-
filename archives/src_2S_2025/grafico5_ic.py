# Gráfico da figura 8.9 do livro "Hydrodynamic Lubrication" de J. Hamrock
# Versão aprimorada com melhor visualização dos eixos e zoom automático

import numpy as np
import matplotlib.pyplot as plt

# --- Funções de Cálculo (as mesmas, corrigidas) ---

def calcular_carga_normal(Ho):
    """Calcula a Carga Normal Adimensional (Wz) - Eq. 8.30."""
    termo_log = np.log((Ho + 1) / Ho)
    termo_fracao = 12 / (1 + 2 * Ho)
    return 6 * termo_log - termo_fracao

def calcular_atrito_fixo(Ho):
    """Calcula a Força de Atrito Adimensional na superfície fixa (Fb) - Eq. 8.32, com sinal corrigido."""
    termo_log = np.log((Ho + 1) / Ho)
    termo_fracao = 6 / (1 + 2 * Ho)
    return -(4 * termo_log + termo_fracao)

def calcular_atrito_movel(Ho):
    """Calcula a Força de Atrito Adimensional na superfície móvel (Fa) - Eq. 8.33."""
    termo_log = np.log((Ho + 1) / Ho)
    termo_fracao = 6 / (1 + 2 * Ho)
    return 2 * termo_log + termo_fracao
    
# --- Configuração dos Dados ---
Ho_coords = np.linspace(0.05, 2.0, 400) # Começamos um pouco depois de zero para evitar log(0)

Wz_valores = calcular_carga_normal(Ho_coords)
Fa_valores = calcular_atrito_movel(Ho_coords)
Fb_valores = calcular_atrito_fixo(Ho_coords)

# --- Criação do Gráfico Aprimorado ---

fig, ax = plt.subplots(figsize=(12, 8))

# Plotar as curvas com linhas mais espessas
ax.plot(Ho_coords, Wz_valores, lw=2.5, label=r'$W_{za}$ (Carga Normal)')
ax.plot(Ho_coords, Fa_valores, lw=2.5, label=r'$F_a$ (Atrito na Sup. Móvel)')
ax.plot(Ho_coords, Fb_valores, lw=2.5, label=r'$F_b$ (Atrito na Sup. Fixa)')

# --- MELHORIAS NA VISUALIZAÇÃO ---

# 1. Destaque dos eixos X e Y para melhor visualização da origem
ax.axhline(0, color='black', linestyle='-', linewidth=1.2, zorder=0)
ax.axvline(0, color='black', linestyle='-', linewidth=1.2, zorder=0)

# 2. Zoom automático e inteligente com margem de 10%
# Encontra o valor mínimo entre todas as forças de atrito
min_y = np.min(Fb_valores)
# Encontra o valor máximo da curva de carga
max_y = np.max(Wz_valores)
# Define os limites do eixo Y com uma margem de 10% para não cortar as curvas
ax.set_ylim(min_y * 1.1, max_y * 1.1)
ax.set_xlim(left=0)


# --- Customização e Legendas ---
ax.set_xlabel(r'Razão de Espessura do Filme, $H_0 = h_0/s_h$', fontsize=14)
ax.set_ylabel(r'Carga e Forças de Cisalhamento Adimensionais', fontsize=14)
ax.set_title('Desempenho do Mancal de Inclinação Fixa', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()