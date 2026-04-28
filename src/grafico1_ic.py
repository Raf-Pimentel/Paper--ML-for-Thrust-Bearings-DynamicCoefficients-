# Código que gera um gráfico da pressão adimensional P em um mancal de inclinação fixa.
# Baseado na Equação 8.24. do livro do Hamrock

import numpy as np
import matplotlib.pyplot as plt

def calcular_pressao_adimensional(X, Ho):
    """
    Calcula a pressão adimensional P em um mancal de inclinação fixa.
    Baseado na Equação 8.24.

    Args:
        X (np.array): Array de coordenadas adimensionais (de 0 a 1).
        Ho (float): Razão de espessura do filme (h₀ / sₕ).

    Returns:
        np.array: Array com os valores de pressão adimensional P.
    """
    # Numerador da Equação 8.24
    numerador = 6 * X * (1 - X)
    
    # Denominador da Equação 8.24
    denominador = ((Ho + 1 - X)**2) * (1 + 2 * Ho)
    
    # Evita divisão por zero, embora seja improvável neste caso
    # Retorna 0 onde o denominador for zero para evitar erros
    return np.divide(numerador, denominador, out=np.zeros_like(numerador), where=denominador!=0)

# --- Configuração dos Dados para o Gráfico ---

# 1. Criar um array de pontos para o eixo X, de 0 a 1, para ter uma curva suave.
X_coords = np.linspace(0, 1, 500)

# 2. Definir os valores de Ho (razão de espessura do filme) que queremos plotar.
#    Estes são os mesmos valores mostrados na imagem.
Ho_valores = [2, 1, 1/2, 1/4]

# --- Criação do Gráfico ---

# 1. Configurar a figura e os eixos do gráfico
fig, ax = plt.subplots(figsize=(10, 6))

# 2. Loop para plotar uma curva para cada valor de Ho
for Ho in Ho_valores:
    # Calcular os valores de P para o Ho atual
    P_valores = calcular_pressao_adimensional(X_coords, Ho)
    
    # Formatar a legenda para frações, se aplicável
    if Ho < 1:
        label = f'H₀ = 1/{int(1/Ho)}'
    else:
        label = f'H₀ = {int(Ho)}'
        
    # Plotar a curva
    ax.plot(X_coords, P_valores, label=label)

# --- Customização e Legendas ---

# 1. Legenda do eixo X
ax.set_xlabel(r'Coordenada Cartesiana Adimensional, $X = x/l$', fontsize=12)

# 2. Legenda do eixo Y
ax.set_ylabel(r'Pressão Adimensional, $P = \frac{p s_h^2}{\eta_0 u_b l}$', fontsize=12)

# 3. Título do Gráfico
ax.set_title('Distribuição de Pressão em Mancal de Deslizamento de Inclinação Fixa', fontsize=14)

# 4. Legenda das Curvas
ax.legend(title="Razão de Espessura\ndo Filme, $H_0=h_0/s_h$")

# 5. Configurar os limites dos eixos para ficarem parecidos com o original
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 3.5)

# 6. Adicionar uma grade para facilitar a leitura
ax.grid(True, linestyle='--', alpha=0.6)

# 7. Ajustar o layout para garantir que nada fique cortado
plt.tight_layout()

# 8. Mostrar o gráfico
plt.show()