# Gráfico que plota o aumento de temperatura adimensional DeltaT em função de Ho e a perda de potência
# Baseado nas Equações 8.37 e 8.38 do livro

import numpy as np
import matplotlib.pyplot as plt

# --- Funções de Cálculo (Baseadas nas equações do livro) ---

def calcular_perda_potencia_adimensional(Ho):
    """
    Calcula a Perda de Potência Adimensional (Hp).
    Baseado na magnitude da Equação 8.37 para ter um valor físico positivo.
    """
    termo_log = np.log((Ho + 1) / Ho)
    termo_fracao = 6 / (1 + 2 * Ho)
    Hp = 4 * termo_log + termo_fracao
    return Hp

def calcular_vazao_adimensional(Ho):
    """
    Calcula a Vazão Volumétrica Adimensional (Q).
    Baseado na Equação 8.36.
    """
    numerador = 2 * Ho * (1 + Ho)
    denominador = 1 + 2 * Ho
    return numerador / denominador

def calcular_aumento_temperatura_adimensional(Ho):
    """
    Calcula o Aumento de Temperatura Adimensional (H_T).
    Baseado na Equação 8.38 (H_T = Hp / Q).
    """
    Hp = calcular_perda_potencia_adimensional(Ho)
    Q = calcular_vazao_adimensional(Ho)
    # Evita divisão por zero para Q, retornando infinito como no limite teórico
    return np.divide(Hp, Q, out=np.full_like(Hp, np.inf), where=Q!=0)

# --- Configuração dos Dados para os Gráficos ---
# Criamos um array de pontos para o eixo X (H₀)
# Começamos um pouco depois de zero para evitar erros matemáticos no limite
Ho_coords = np.linspace(0.01, 2.0, 500)

# Calcular os valores de Y para cada gráfico
Hp_valores = calcular_perda_potencia_adimensional(Ho_coords)
Ht_valores = calcular_aumento_temperatura_adimensional(Ho_coords)


# --- GRÁFICO 1: PERDA DE POTÊNCIA ---

fig1, ax1 = plt.subplots(figsize=(10, 7))
ax1.plot(Ho_coords, Hp_valores, lw=2.5, color='black')

# Customização e Legendas
ax1.set_xlabel(r'Razão de Espessura do Filme, $H_0 = h_0/s_h$', fontsize=12)
ax1.set_ylabel(r'Perda de Potência Adimensional, $H_p$', fontsize=12)
ax1.set_title('Efeito da Razão de Espessura na Perda de Potência', fontsize=14)
ax1.set_xlim(0, 2.0)
ax1.set_ylim(bottom=0) # Perda de potência não pode ser negativa
ax1.grid(True, linestyle=':', alpha=0.7)
fig1.tight_layout()


# --- GRÁFICO 2: AUMENTO DE TEMPERATURA ---

fig2, ax2 = plt.subplots(figsize=(10, 7))
ax2.plot(Ho_coords, Ht_valores, lw=2.5, color='black')

# Customização e Legendas
ax2.set_xlabel(r'Razão de Espessura do Filme, $H_0 = h_0/s_h$', fontsize=12)
ax2.set_ylabel(r'Aumento de Temp. Adimensional, $H_T$', fontsize=12)
ax2.set_title('Efeito da Razão de Espessura no Aumento de Temperatura (Fig. 8.12)', fontsize=14)
ax2.set_xlim(0, 2.0)
ax2.set_ylim(0, 30) # Limite para melhor visualização, já que a curva tende ao infinito
ax2.grid(True, linestyle=':', alpha=0.7)
fig2.tight_layout()


# --- Mostrar os gráficos ---
plt.show()