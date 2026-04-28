# Baseado no documento Coeficientes de Rigidez e Amortecimento do Mancal de Inclinação Fixa analíticos
# Inclui uma interface interativa para ajustar os parâmetros geométricos e ver o efeito nos coeficientes dinâmicos.
# Utilizei o Canalitico corrigido para o amortecimento, que é fisicamente consistente.
# A cor da sapata muda conforme a velocidade de squeeze (Vz), e setas indicam a direção do movimento.
# O gráfico é atualizado em tempo real conforme os sliders são ajustados.
# Veja que Vz positivo indica (seta para cima) e negativo (seta para baixo).
# Veja que a perturbação de deslocamento (delta_H) afeta a posição vertical da sapata.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Polygon
import matplotlib.cm as cm

# =============================================================================
# MÓDULO DE CÁLCULO
# Inclui as funções para os coeficientes dinâmicos que analisamos.
# =============================================================================

def calcular_rigidez_analitica(Ho):
    """
    Calcula o Coeficiente de Rigidez Adimensional (K_analitico).
    Baseado na Equação (2.13) do documento.
    """
    if Ho <= 1e-9:
        return -np.inf
    termo1 = 6 * ((1 / (Ho + 1)) - (1 / Ho))
    termo2 = 24 / ((1 + 2 * Ho)**2)
    return termo1 + termo2

def calcular_amortecimento_analitico(Ho):
    """
    Calcula o Coeficiente de Amortecimento Adimensional (C_analitico).
    Baseado na nossa versão corrigida e fisicamente consistente da Eq. (2.35).
    """
    if Ho <= 1e-9:
        return np.inf
    termo_log = np.log((Ho + 1) / Ho)
    termo_fracao = 2 / (2 * Ho + 1)
    return termo_log - termo_fracao

def criar_vertices_sapata(l, sh, h0, delta_H):
    """
    Calcula os vértices do polígono da sapata, incluindo a perturbação de deslocamento.
    """
    altura_bloco = (sh + h0) * 0.4
    vertices_base = np.array([
        [0, h0 + sh], [l, h0], [l, h0 + sh + altura_bloco], [0, h0 + sh + altura_bloco]
    ])
    # Adiciona a perturbação de deslocamento delta_H a todas as coordenadas Y
    vertices_perturbados = vertices_base + [0, delta_H]
    return vertices_perturbados

# =============================================================================
# SCRIPT PRINCIPAL (CONTROLADOR DA APLICAÇÃO)
# =============================================================================

# --- Configuração Inicial ---
l_inicial, sh_inicial, h0_inicial = 100.0, 14.1, 10.0
delta_H_inicial = 0.0
Vz_inicial = 0.0

# --- Criação da Janela e do Painel Principal ---
fig, ax = plt.subplots(figsize=(12, 9))
fig.suptitle('Animação Interativa de Perturbações em Mancal', fontsize=18)
plt.subplots_adjust(left=0.1, bottom=0.4) # Deixa bastante espaço para os sliders

# --- Desenho Inicial da Geometria e Anotações ---
ax.set_title('Geometria do Mancal e Resposta Dinâmica', fontsize=14)
sapata = Polygon(criar_vertices_sapata(l_inicial, sh_inicial, h0_inicial, delta_H_inicial), 
                 facecolor='lightgray', edgecolor='black', lw=1.5)
ax.add_patch(sapata)
pista, = ax.plot([0, l_inicial], [0, 0], color='black', lw=2)
ax.set_aspect('equal', adjustable='box')
ax.tick_params(labelbottom=False, labelleft=False)

# Textos dinâmicos para os coeficientes
texto_K = ax.text(0.05, 0.95, '', transform=ax.transAxes, ha='left', va='top', fontsize=12,
                  bbox=dict(boxstyle='round,pad=0.3', fc='lightblue', alpha=0.7))
texto_C = ax.text(0.05, 0.85, '', transform=ax.transAxes, ha='left', va='top', fontsize=12,
                  bbox=dict(boxstyle='round,pad=0.3', fc='lightblue', alpha=0.7))

# Setas para visualizar a velocidade de squeeze (inicialmente invisíveis)
arrow_up = ax.arrow(l_inicial/2, h0_inicial + sh_inicial, 0, 10, color='green', width=2, head_width=5, visible=False)
arrow_down = ax.arrow(l_inicial/2, h0_inicial + sh_inicial, 0, -10, color='red', width=2, head_width=5, visible=False)


# --- Criação dos Sliders ---
# Posições dos sliders [esquerda, baixo, largura, altura]
ax_slider_l = plt.axes([0.2, 0.25, 0.65, 0.03])
ax_slider_sh = plt.axes([0.2, 0.20, 0.65, 0.03])
ax_slider_h0 = plt.axes([0.2, 0.15, 0.65, 0.03])
ax_slider_dH = plt.axes([0.2, 0.10, 0.65, 0.03])
ax_slider_Vz = plt.axes([0.2, 0.05, 0.65, 0.03])

# Instanciando os sliders
slider_l = Slider(ax=ax_slider_l, label='Comprimento (l)', valmin=10, valmax=200, valinit=l_inicial)
slider_sh = Slider(ax=ax_slider_sh, label='Altura da Cunha (sₕ)', valmin=0.1, valmax=50, valinit=sh_inicial)
slider_h0 = Slider(ax=ax_slider_h0, label='Folga de Saída (h₀)', valmin=0.1, valmax=50, valinit=h0_inicial)
slider_dH = Slider(ax=ax_slider_dH, label='Pert. Deslocamento (ΔH)', valmin=-10, valmax=10, valinit=delta_H_inicial)
slider_Vz = Slider(ax=ax_slider_Vz, label='Pert. Velocidade (Vz)', valmin=-10, valmax=10, valinit=Vz_inicial)

# Mapa de cores para a velocidade: azul (negativo) -> branco (zero) -> vermelho (positivo)
cmap = cm.get_cmap('coolwarm')

# --- Função de Atualização Mestra ---
def update(val):
    # 1. Pega os valores atuais de todos os sliders
    l, sh, h0, delta_H, Vz = slider_l.val, slider_sh.val, slider_h0.val, slider_dH.val, slider_Vz.val
    
    # 2. Atualiza a geometria (posição da sapata)
    novos_vertices = criar_vertices_sapata(l, sh, h0, delta_H)
    sapata.set_xy(novos_vertices)
    pista.set_data([-l*0.1, l*1.1], [0, 0])

    # 3. Atualiza a visualização da velocidade (cor e setas)
    # Normaliza Vz para o intervalo [0, 1] para o mapa de cores
    Vz_normalizado = (Vz - slider_Vz.valmin) / (slider_Vz.valmax - slider_Vz.valmin)
    sapata.set_facecolor(cmap(Vz_normalizado))
    
    # Atualiza as setas de velocidade
    arrow_up.set_data(x=l/2, y=novos_vertices[0,1] + 15, dx=0, dy=10)
    arrow_down.set_data(x=l/2, y=novos_vertices[0,1] + 5, dx=0, dy=-10)
    arrow_up.set_visible(Vz > 0.5)
    arrow_down.set_visible(Vz < -0.5)

    # 4. Calcula e atualiza os coeficientes dinâmicos (baseados na geometria SEM perturbação)
    Ho = h0 / sh
    K_analitico = calcular_rigidez_analitica(Ho)
    C_analitico = calcular_amortecimento_analitico(Ho)
    
    texto_K.set_text(f'$K_{{analitico}} = {K_analitico:.2f}$')
    texto_C.set_text(f'$C_{{analitico}} = {C_analitico:.2f}$')
    
    # 5. Reajusta os limites do gráfico dinamicamente
    ax.set_xlim(-l*0.2, l*1.2)
    ax.set_ylim(-l*0.15, (sh+h0)*2.5 + delta_H)
    
    # 6. Redesenha a figura
    fig.canvas.draw_idle()

# --- Conectar Sliders e Inicializar ---
slider_l.on_changed(update)
slider_sh.on_changed(update)
slider_h0.on_changed(update)
slider_dH.on_changed(update)
slider_Vz.on_changed(update)

update(0) # Chama a função uma vez para desenhar o estado inicial
plt.show()