# Grafico que tenta plotar o gráfico da figura 8.10 do Hamrock
# Calcula o parâmetro de atrito (μl/sₕ) usando a definição: Fa / Wz.
# Gráfico do parâmetro do coeficiente de atrito (μl/sh) em função da razão de espessura do filme (H0)
# Tive problemas em plotar o gráfico de forma correta, então fiz o cálculo de Fa e Wz e dividi
# Nota: Acredito que a fórmula de Fa estava errada no Hamrock, no entanto, deixei assim.
# A fórmula de Fa é dada na equação 8.33
# A fórmula de Wz é dada na equação 8.30
# O gráfico resultante corresponde ao gráfico plotado nesse código.

import numpy as np
import matplotlib.pyplot as plt


def calcular_carga_normal(Ho):
    """
    Calcula a Carga Normal Adimensional (Wz).
    Baseado na Equação 8.30.
    """
    termo_log = np.log((Ho + 1) / Ho)
    termo_fracao = 12 / (1 + 2 * Ho)
    return 6 * termo_log - termo_fracao

def calcular_atrito_movel(Ho):
    """
    Calcula a Força de Atrito Adimensional na superfície móvel (Fa).
    Baseado na Equação 8.33.
    """
    termo_log = np.log(Ho / (Ho + 1))
    termo_fracao = 6 / (1 + 2 * Ho)
    return 4 * termo_log + termo_fracao

def calcular_parametro_atrito_corrigido(Ho):
    """
    Calcula o parâmetro de atrito (μl/sₕ) usando a definição: Fa / Wz.
    """
    # Calcula os valores de Fa e Wz usando as funções corretas
    Fa_valores = calcular_atrito_movel(Ho)
    Wz_valores = calcular_carga_normal(Ho)
    
    # Retorna a razão, que é o parâmetro de atrito
    # Evita divisão por zero para Wz muito pequeno
    return np.divide(Fa_valores, Wz_valores, out=np.full_like(Fa_valores, np.inf), where=Wz_valores!=0)

# --- Configuração dos Dados para o Gráfico ---

# Criar um array de pontos para o eixo X (H₀).
Ho_coords = np.linspace(0.05, 2.0, 500)

# Calcular os valores de Y (parâmetro de atrito) com a função corrigida
parametro_atrito_valores = calcular_parametro_atrito_corrigido(Ho_coords)

# --- Criação do Gráfico ---

fig, ax = plt.subplots(figsize=(10, 7))

# Plotar a curva
ax.plot(Ho_coords, parametro_atrito_valores, lw=2.5, color='black')

# --- Customização e Legendas (para replicar a Figura 8.10) ---

ax.set_xlabel(r'Razão de Espessura do Filme, $H_0 = h_0/s_h$', fontsize=12)
ax.set_ylabel(r'Parâmetro do Coeficiente de Atrito, $\mu\ell/s_h$', fontsize=12)
ax.set_title('Figura 8.10: Efeito da Razão de Espessura no Coeficiente de Atrito (Corrigido)', fontsize=14)

# Configurar os limites dos eixos para ficarem idênticos ao original
ax.set_xlim(0, 2.0)
ax.set_ylim(0, 20)

ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()