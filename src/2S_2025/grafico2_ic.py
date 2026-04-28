# Codigo que calcula a pressão adimensional P em um mancal de inclinação fixa.
# Com animação e deslizantes
# Baseado na Equação 8.24.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- Função de Cálculo (a mesma de antes) ---
def calcular_pressao_adimensional(X, Ho):
    """
    Calcula a pressão adimensional P em um mancal de inclinação fixa.
    Baseado na Equação 8.24.
    """
    numerador = 6 * X * (1 - X)
    denominador = ((Ho + 1 - X)**2) * (1 + 2 * Ho)
    # Evita divisão por zero para valores de Ho muito pequenos ou negativos
    # Retorna um array de zeros se o denominador for problemático
    if (1 + 2 * Ho) <= 0:
        return np.zeros_like(X)
    return np.divide(numerador, denominador, out=np.zeros_like(numerador), where=denominador!=0)

# --- Função para encontrar o pico de pressão ---
def encontrar_pico_pressao(Ho):
    """
    Calcula a localização (Xm) e o valor (Pm) da pressão máxima.
    """
    # Localização do pico de pressão (Eq. 8.25)
    Xm = (1 + Ho) / (1 + 2 * Ho)
    # Calcula o valor da pressão nesse ponto
    Pm = calcular_pressao_adimensional(Xm, Ho)
    return Xm, Pm

# --- Configuração Inicial do Gráfico ---

# Definir os dados para o eixo X
X_coords = np.linspace(0, 1, 500)
# Definir o valor inicial de Ho para o slider
Ho_inicial = 1.0

# Criar a figura e o eixo principal do gráfico
# Deixamos um espaço na parte de baixo para o slider
fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(bottom=0.25)

# Plotar a curva inicial e guardar a referência da linha para poder atualizá-la
P_inicial = calcular_pressao_adimensional(X_coords, Ho_inicial)
linha, = ax.plot(X_coords, P_inicial, lw=2, color='royalblue')

# Adicionar um marcador e texto para o pico de pressão inicial
Xm_inicial, Pm_inicial = encontrar_pico_pressao(Ho_inicial)
ponto_pico, = ax.plot(Xm_inicial, Pm_inicial, 'ro', label=f'Pico de Pressão (P_max)') # 'ro' = red circle
texto_pico = ax.text(Xm_inicial, Pm_inicial + 0.1, f'P_max = {Pm_inicial:.3f}', ha='center', fontsize=10)

# --- Customização do Gráfico Principal ---

# Legendas dos eixos (usando LaTeX para uma aparência profissional)
ax.set_xlabel(r'Coordenada Cartesiana Adimensional, $X = x/l$', fontsize=12)
ax.set_ylabel(r'Pressão Adimensional, $P = \frac{p s_h^2}{\eta_0 u_b l}$', fontsize=12)
ax.set_title('Distribuição de Pressão Interativa em Mancal de Inclinação Fixa', fontsize=14)
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 3.5)
ax.grid(True, linestyle='--', alpha=0.6)

# Adicionar um texto que mostrará o valor atual de H₀
texto_Ho = ax.text(0.05, 0.9, f'$H_0 = {Ho_inicial:.2f}$', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))


# --- Criação do Slider Interativo ---

# Definir a posição do eixo do slider [esquerda, baixo, largura, altura]
ax_slider = plt.axes([0.20, 0.1, 0.65, 0.03])

# Criar o slider
slider_Ho = Slider(
    ax=ax_slider,
    label='Controle de $H_0$', # Legenda do slider
    valmin=0.05,              # Valor mínimo
    valmax=4.0,               # Valor máximo
    valinit=Ho_inicial,       # Valor inicial
    valstep=0.01              # Incremento
)

# --- Função de Atualização (O Coração da Interatividade) ---
def update(val):
    # Obter o novo valor de H₀ a partir do slider
    Ho_atual = slider_Ho.val
    
    # Recalcular os valores de Pressão com o novo H₀
    novos_P = calcular_pressao_adimensional(X_coords, Ho_atual)
    
    # Atualizar os dados da linha do gráfico
    linha.set_ydata(novos_P)
    
    # Recalcular e atualizar a posição do marcador de pico e seu texto
    novo_Xm, novo_Pm = encontrar_pico_pressao(Ho_atual)
    ponto_pico.set_data(novo_Xm, novo_Pm)
    texto_pico.set_position((novo_Xm, novo_Pm + 0.1))
    texto_pico.set_text(f'P_max = {novo_Pm:.3f}')
    
    # Atualizar o texto que mostra o valor de H₀
    texto_Ho.set_text(f'$H_0 = {Ho_atual:.2f}$')
    
    # Redesenhar o gráfico para mostrar as mudanças
    fig.canvas.draw_idle()

# --- Conectar o Slider à Função de Atualização ---
slider_Ho.on_changed(update)

# Mostrar o gráfico interativo
plt.show()