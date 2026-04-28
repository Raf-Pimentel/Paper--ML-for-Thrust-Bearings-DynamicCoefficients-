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
import matplotlib.gridspec as gridspec

# =============================================================================
# MÓDULO DE CÁLCULO
# Funções para os coeficientes dinâmicos e geometria.
# =============================================================================

def calcular_rigidez_analitica(Ho):
    """
    [FUNÇÃO CORRIGIDA]
    Calcula o Coeficiente de Rigidez Adimensional (K_analitico).
    A função agora é vetorizada para aceitar tanto escalares quanto arrays.
    """
    # Suprime avisos de divisão por zero, que serão tratados pelo np.where
    with np.errstate(divide='ignore', invalid='ignore'):
        # Realiza os cálculos para todos os elementos do array
        termo1 = 6 * ((1 / (Ho + 1)) - (1 / Ho))
        termo2 = 24 / ((1 + 2 * Ho)**2)
        resultado = termo1 + termo2
    
    # Usa np.where para aplicar a condição: se Ho <= 1e-9, o resultado é -inf,
    # caso contrário, usa o valor calculado. Funciona para escalares e arrays.
    return np.where(Ho <= 1e-9, -np.inf, resultado)

def calcular_amortecimento_analitico(Ho):
    """
    [FUNÇÃO CORRIGIDA]
    Calcula o Coeficiente de Amortecimento Adimensional (C_analitico).
    A função agora é vetorizada para aceitar tanto escalares quanto arrays.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        # Realiza os cálculos para todos os elementos
        termo_log = np.log((Ho + 1) / Ho)
        termo_fracao = 2 / (2 * Ho + 1)
        resultado = termo_log - termo_fracao

    # Usa np.where para aplicar a condição: se Ho <= 1e-9, o resultado é inf,
    # caso contrário, usa o valor calculado.
    return np.where(Ho <= 1e-9, np.inf, resultado)

def criar_vertices_sapata(l, sh, h0, delta_H):
    """Calcula os vértices da sapata, incluindo a perturbação de deslocamento."""
    altura_bloco = (sh + h0) * 0.4
    vertices_base = np.array([[0, h0 + sh], [l, h0], [l, h0 + sh + altura_bloco], [0, h0 + sh + altura_bloco]])
    return vertices_base + [0, delta_H]

# =============================================================================
# SCRIPT PRINCIPAL (CONTROLADOR DA APLICAÇÃO)
# O restante do código permanece o mesmo, pois agora as funções são robustas.
# =============================================================================

# --- Configuração Inicial ---
l_inicial, sh_inicial, h0_inicial = 100.0, 14.1, 10.0
delta_H_inicial, Vz_inicial = 0.0, 0.0

# --- Layout da Janela ---
fig = plt.figure(figsize=(16, 9))
fig.suptitle('Plataforma de Análise Dinâmica de Mancal de Escora', fontsize=20)
gs = gridspec.GridSpec(2, 2, height_ratios=[3, 2])

ax_geom = fig.add_subplot(gs[0, :])
ax_K = fig.add_subplot(gs[1, 0])
ax_C = fig.add_subplot(gs[1, 1])

plt.subplots_adjust(bottom=0.3, hspace=0.4)

# --- PAINEL SUPERIOR: GEOMETRIA INTERATIVA ---
ax_geom.set_title('Animação da Geometria e Perturbações', fontsize=14)
sapata = Polygon(criar_vertices_sapata(l_inicial, sh_inicial, h0_inicial, delta_H_inicial), fc='lightgray', ec='black', lw=1.5)
ax_geom.add_patch(sapata)
pista, = ax_geom.plot([], [], color='black', lw=2)
ax_geom.set_aspect('equal', adjustable='box')
ax_geom.tick_params(labelbottom=False, labelleft=False)
texto_K = ax_geom.text(0.02, 0.95, '', transform=ax_geom.transAxes, ha='left', va='top', fontsize=12, bbox=dict(boxstyle='round', fc='lightblue', alpha=0.8))
texto_C = ax_geom.text(0.02, 0.85, '', transform=ax_geom.transAxes, ha='left', va='top', fontsize=12, bbox=dict(boxstyle='round', fc='lightblue', alpha=0.8))
arrow_up = ax_geom.arrow(0, 0, 0, 0, color='green', width=2, head_width=5, visible=False)
arrow_down = ax_geom.arrow(0, 0, 0, 0, color='red', width=2, head_width=5, visible=False)

# --- PAINÉIS INFERIORES: GRÁFICOS DE DESEMPENHO ---
Ho_coords = np.linspace(0.01, 3.0, 500)
# As chamadas agora funcionam corretamente com as funções vetorizadas
K_valores = calcular_rigidez_analitica(Ho_coords)
C_valores = calcular_amortecimento_analitico(Ho_coords)

# Gráfico de Rigidez (K)
ax_K.plot(Ho_coords, K_valores, lw=2, color='darkred')
ax_K.set_title('Coeficiente de Rigidez ($K_{analitico}$)', fontsize=12)
ax_K.set_xlabel(r'$H_0 = h_0/s_h$'); ax_K.set_ylabel(r'$K_{analitico}$')
ax_K.axhline(0, color='black', linestyle='--', lw=1)
ax_K.set_ylim(-100, 25); ax_K.grid(True, linestyle=':')
vline_K = ax_K.axvline(x=0, color='blue', linestyle='--', lw=1.5)
marker_K, = ax_K.plot([], [], 'o', color='blue', markersize=8)
texto_K_val = ax_K.text(0, 0, '', ha='left', va='bottom', fontsize=10, bbox=dict(fc='white', alpha=0.7))

# Gráfico de Amortecimento (C)
ax_C.plot(Ho_coords, C_valores, lw=2, color='darkblue')
ax_C.set_title('Coeficiente de Amortecimento ($C_{analitico}$)', fontsize=12)
ax_C.set_xlabel(r'$H_0 = h_0/s_h$'); ax_C.set_ylabel(r'$C_{analitico}$')
ax_C.set_ylim(0, 5); ax_C.grid(True, linestyle=':')
vline_C = ax_C.axvline(x=0, color='blue', linestyle='--', lw=1.5)
marker_C, = ax_C.plot([], [], 'o', color='blue', markersize=8)
texto_C_val = ax_C.text(0, 0, '', ha='left', va='bottom', fontsize=10, bbox=dict(fc='white', alpha=0.7))

# --- SLIDERS DE CONTROLE ---
slider_ax_l = fig.add_axes([0.25, 0.18, 0.5, 0.02])
slider_ax_sh = fig.add_axes([0.25, 0.14, 0.5, 0.02])
slider_ax_h0 = fig.add_axes([0.25, 0.10, 0.5, 0.02])
slider_ax_dH = fig.add_axes([0.25, 0.06, 0.5, 0.02])
slider_ax_Vz = fig.add_axes([0.25, 0.02, 0.5, 0.02])

slider_l = Slider(ax=slider_ax_l, label='Comp. (l)', valmin=10, valmax=200, valinit=l_inicial)
slider_sh = Slider(ax=slider_ax_sh, label='Cunha (sₕ)', valmin=0.1, valmax=50, valinit=sh_inicial)
slider_h0 = Slider(ax=slider_ax_h0, label='Folga (h₀)', valmin=0.1, valmax=50, valinit=h0_inicial)
slider_dH = Slider(ax=slider_ax_dH, label='Pert. Desloc. (ΔH)', valmin=-10, valmax=10, valinit=delta_H_inicial)
slider_Vz = Slider(ax=slider_ax_Vz, label='Pert. Veloc. (Vz)', valmin=-10, valmax=10, valinit=Vz_inicial)

cmap = cm.get_cmap('coolwarm')

# --- FUNÇÃO DE ATUALIZAÇÃO MESTRA ---
def update(val):
    l, sh, h0, delta_H, Vz = slider_l.val, slider_sh.val, slider_h0.val, slider_dH.val, slider_Vz.val
    
    # Atualiza Geometria
    novos_vertices = criar_vertices_sapata(l, sh, h0, delta_H)
    sapata.set_xy(novos_vertices)
    pista.set_data([-l*0.1, l*1.1], [0, 0])
    Vz_norm = (Vz - slider_Vz.valmin) / (slider_Vz.valmax - slider_Vz.valmin)
    sapata.set_facecolor(cmap(Vz_norm))
    arrow_up.set_data(x=l/2, y=novos_vertices[0,1] + 15, dx=0, dy=10)
    arrow_down.set_data(x=l/2, y=novos_vertices[0,1] + 5, dx=0, dy=-10)
    arrow_up.set_visible(Vz > 0.5); arrow_down.set_visible(Vz < -0.5)
    ax_geom.set_xlim(-l*0.2, l*1.2); ax_geom.set_ylim(-l*0.15, (sh+h0)*2.5 + delta_H)
    
    # Calcula e exibe coeficientes dinâmicos
    Ho = h0 / sh
    K_analitico = calcular_rigidez_analitica(Ho)
    C_analitico = calcular_amortecimento_analitico(Ho)
    texto_K.set_text(f'$K_{{analitico}} = {K_analitico:.2f}$')
    texto_C.set_text(f'$C_{{analitico}} = {C_analitico:.2f}$')
    
    # Atualiza os marcadores nos gráficos de desempenho
    vline_K.set_xdata([Ho, Ho]); vline_C.set_xdata([Ho, Ho])
    marker_K.set_data(Ho, K_analitico); texto_K_val.set_text(f'{K_analitico:.2f}'); texto_K_val.set_position((Ho, K_analitico))
    marker_C.set_data(Ho, C_analitico); texto_C_val.set_text(f'{C_analitico:.2f}'); texto_C_val.set_position((Ho, C_analitico))
    
    fig.canvas.draw_idle()

# --- Conectar Sliders e Inicializar ---
slider_l.on_changed(update); slider_sh.on_changed(update); slider_h0.on_changed(update)
slider_dH.on_changed(update); slider_Vz.on_changed(update)
update(0)
plt.show()