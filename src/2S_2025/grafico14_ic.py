# Baseado no documento Coeficientes de Rigidez e Amortecimento do Mancal de Inclinação Fixa analíticos
# Utilizei o Canalitico corrigido para o amortecimento, que é fisicamente consistente.
# A função de rigidez permanece a mesma, embora seja inconsistente para Ho muito pequeno.
# O gráfico mostra claramente a inconsistência do coeficiente de rigidez (valores negativos)

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# MÓDULO DE CÁLCULO
# Funções para os coeficientes de rigidez e amortecimento.
# =============================================================================

def calcular_rigidez_analitica(Ho):
    """
    Calcula o Coeficiente de Rigidez Adimensional (K_analitico).
    Baseado na Equação (2.13) do documento, que é inconsistente para Ho pequeno.
    """
    termo1 = 6 * ((1 / (Ho + 1)) - (1 / Ho))
    termo2 = 24 / ((1 + 2 * Ho)**2)
    return termo1 + termo2

def calcular_amortecimento_analitico(Ho):
    """
    Calcula o Coeficiente de Amortecimento Adimensional (C_analitico).
    Baseado na nossa versão corrigida e fisicamente consistente da Eq. (2.35).
    """
    # A fórmula corrigida é ln((Ho+1)/Ho) - 2/(2Ho+1)
    termo_log = np.log((Ho + 1) / Ho)
    termo_fracao = 2 / (2 * Ho + 1)
    return termo_log - termo_fracao

# --- Configuração dos Dados para o Gráfico ---

# Criar um array de pontos para o eixo X (H₀).
# Começamos de um valor muito pequeno (0.01) para visualizar o comportamento perto de zero.
Ho_coords = np.linspace(0.01, 3.0, 500)

# Calcular os valores de Y para cada coeficiente
K_valores = calcular_rigidez_analitica(Ho_coords)
C_valores = calcular_amortecimento_analitico(Ho_coords)

# --- Criação do Gráfico ---

# Criar uma figura com 2 subplots (2 linhas, 1 coluna), compartilhando o eixo X
fig, (ax_K, ax_C) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
fig.suptitle('Análise dos Coeficientes Dinâmicos do Mancal de Inclinação Fixa', fontsize=16)

# --- PLOT SUPERIOR: COEFICIENTE DE RIGIDEZ (K) ---

ax_K.plot(Ho_coords, K_valores, lw=2.5, color='darkred')
ax_K.set_title('Coeficiente de Rigidez ($K_{analitico}$)', fontsize=14)
ax_K.set_ylabel(r'Rigidez Adimensional, $K_{analitico}$', fontsize=12)

# Adiciona uma linha de referência em y=0 para destacar a região negativa
ax_K.axhline(0, color='black', linestyle='--', linewidth=1.0)

# Ajusta os limites para mostrar claramente a "inconsistência" (valores negativos)
ax_K.set_ylim(-100, 25)
ax_K.grid(True, linestyle=':', alpha=0.7)

# --- PLOT INFERIOR: COEFICIENTE DE AMORTECIMENTO (C) ---

ax_C.plot(Ho_coords, C_valores, lw=2.5, color='darkblue')
ax_C.set_title('Coeficiente de Amortecimento ($C_{analitico}$)', fontsize=14)
ax_C.set_ylabel(r'Amortecimento Adimensional, $C_{analitico}$', fontsize=12)
ax_C.set_xlabel(r'Razão de Espessura do Filme, $H_0 = h_0/s_h$', fontsize=12)

# Ajusta os limites para uma boa visualização
ax_C.set_ylim(bottom=0)
ax_C.grid(True, linestyle=':', alpha=0.7)

# --- Finalização ---
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajusta o layout para o supertítulo
plt.show()