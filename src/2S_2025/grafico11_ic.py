# Gráfico que plota o aumento de temperatura adimensional DeltaT em função de Ho
# Baseado nas Equações 8.37 e 8.38 do livro

import numpy as np
import matplotlib.pyplot as plt

def calcular_perda_potencia_adimensional(Ho):
    """Calcula a Perda de Potência Adimensional (Hp)."""
    termo_log = np.log((Ho + 1) / Ho)
    termo_fracao = 6 / (1 + 2 * Ho)
    return 4 * termo_log + termo_fracao

def calcular_vazao_adimensional(Ho):
    """Calcula a Vazão Volumétrica Adimensional (Q)."""
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
    return np.divide(Hp, Q, out=np.full_like(Hp, np.inf), where=Q!=0)

# --- Configuração dos Dados ---
Ho_coords = np.linspace(0.01, 2.0, 500)
Ht_valores = calcular_aumento_temperatura_adimensional(Ho_coords)

# --- Criação do Gráfico (semelhante à Fig. 8.12) ---
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(Ho_coords, Ht_valores, lw=2.5, color='black')

# --- Customização e Legendas ---
ax.set_xlabel(r'Razão de Espessura do Filme, $H_0 = h_0/s_h$', fontsize=12)
ax.set_ylabel(r'Aumento de Temp. Adimensional, $H_T$', fontsize=12)
ax.set_title('Efeito da Razão de Espessura no Aumento de Temperatura (Fig. 8.12)', fontsize=14)
ax.set_xlim(0, 2.0)
ax.set_ylim(0, 30) # Limite para melhor visualização, já que a curva tende ao infinito
ax.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()