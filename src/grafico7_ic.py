# Grafico que integra Geometria da cunha, Wz, Fa, Fb e a Pressão Maxima
# Baseado nos códigos grafico2, grafico3, grafico4, grafico5 e grafico6
# Baseado nas figuras 8.8 e 8.9 do Hamrock

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Polygon

# --- FUNÇÕES DE CÁLCULO (A base de toda a análise) ---

def calcular_pressao_adimensional(X, Ho):
    """Calcula a pressão adimensional P (Eq. 8.24)."""
    numerador = 6 * X * (1 - X)
    denominador = ((Ho + 1 - X)**2) * (1 + 2 * Ho)
    if (1 + 2 * Ho) <= 1e-9: return np.zeros_like(X)
    return np.divide(numerador, denominador, out=np.zeros_like(numerador), where=denominador!=0)

def encontrar_pico_pressao(Ho):
    """Calcula a localização (Xm) e o valor (Pm) da pressão máxima (Eq. 8.25)."""
    if (1 + 2 * Ho) <= 1e-9: return 1.0, 0.0
    Xm = (1 + Ho) / (1 + 2 * Ho)
    Pm = calcular_pressao_adimensional(Xm, Ho)
    return Xm, Pm

def criar_vertices_sapata(l, sh, h0):
    """Calcula os vértices do polígono da sapata superior."""
    altura_bloco = (sh + h0) * 0.4
    return np.array([[0, h0 + sh], [l, h0], [l, h0 + sh + altura_bloco], [0, h0 + sh + altura_bloco]])

def calcular_carga_normal(Ho):
    """Calcula a Carga Normal Adimensional (Wz) (Eq. 8.30)."""
    return 6 * np.log((Ho + 1) / Ho) - (12 / (1 + 2 * Ho))

def calcular_atrito_fixo(Ho):
    """Calcula a Força de Atrito Adimensional na superfície fixa (Fb) (Eq. 8.32, com sinal corrigido)."""
    return -(4 * np.log((Ho + 1) / Ho) + (6 / (1 + 2 * Ho)))

def calcular_atrito_movel(Ho):
    """Calcula a Força de Atrito Adimensional na superfície móvel (Fa) (Eq. 8.33)."""
    return 2 * np.log((Ho + 1) / Ho) + (6 / (1 + 2 * Ho))

# --- CONFIGURAÇÃO INICIAL E DADOS GERAIS ---
l_inicial, sh_inicial, h0_inicial = 100.0, 14.1, 10.0
X_coords = np.linspace(0, 1, 500)
Ho_coords = np.linspace(0.05, 3.0, 400)

# --- CRIAÇÃO DA INTERFACE GRÁFICA (GRID 2x2) ---
fig, ((ax_geom, ax_pressao), (ax_wz, ax_forces)) = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle('Dashboard Interativo para Análise de Mancal de Escora', fontsize=20, fontweight='bold')
plt.subplots_adjust(left=0.1, bottom=0.25, hspace=0.4, wspace=0.3)

# --- PAINEL 1: GEOMETRIA (SUPERIOR ESQUERDO) ---
ax_geom.set_title('1. Geometria do Mancal (Entrada)', fontsize=14)
sapata = Polygon(criar_vertices_sapata(l_inicial, sh_inicial, h0_inicial), fc='lightgray', ec='black', lw=1.5)
ax_geom.add_patch(sapata)
pista, = ax_geom.plot([0, l_inicial], [0, 0], color='black', lw=2)
ax_geom.set_aspect('equal', adjustable='box')
ax_geom.tick_params(labelbottom=False, labelleft=False)
texto_Ho_geom = ax_geom.text(0.5, 0.85, '', transform=ax_geom.transAxes, ha='center', fontsize=14, fontweight='bold')

# --- PAINEL 2: PRESSÃO (SUPERIOR DIREITO) ---
ax_pressao.set_title('2. Distribuição de Pressão Resultante', fontsize=14)
Ho_inicial_calc = h0_inicial / sh_inicial
linha_pressao, = ax_pressao.plot(X_coords, calcular_pressao_adimensional(X_coords, Ho_inicial_calc), lw=2.5)
ponto_pico, = ax_pressao.plot([], [], 'ro', markersize=8)
texto_pico = ax_pressao.text(0, 0, '', ha='center', fontsize=10)
ax_pressao.set_xlabel(r'$X = x/l$', fontsize=12)
ax_pressao.set_ylabel(r'Pressão Adimensional, $P$', fontsize=12)
ax_pressao.grid(True, linestyle=':', alpha=0.7)

# --- PAINEL 3: CAPACIDADE DE CARGA (INFERIOR ESQUERDO) ---
ax_wz.set_title('3. Capacidade de Carga vs. Geometria (Fig. 8.8)', fontsize=14)
ax_wz.plot(Ho_coords, calcular_carga_normal(Ho_coords), lw=2, color='green', label='Curva de Desempenho de Carga')
vline_wz = ax_wz.axvline(x=Ho_inicial_calc, color='r', linestyle='--', lw=1.5)
marker_wz, = ax_wz.plot([], [], 'ro', markersize=8)
texto_wz = ax_wz.text(0, 0, '', ha='left', va='bottom', fontsize=10, bbox=dict(fc='white', alpha=0.7))
ax_wz.set_xlabel(r'Razão de Espessura do Filme, $H_0 = h_0/s_h$', fontsize=12)
ax_wz.set_ylabel(r'Carga Normal Adimensional, $W_z$', fontsize=12)
ax_wz.set_xlim(0, 3.0); ax_wz.set_ylim(0, 10); ax_wz.grid(True, linestyle=':', alpha=0.7)

# --- PAINEL 4: FORÇAS DE ATRITO (INFERIOR DIREITO) ---
ax_forces.set_title('4. Forças de Atrito vs. Geometria (Fig. 8.9)', fontsize=14)
ax_forces.plot(Ho_coords, calcular_atrito_movel(Ho_coords), lw=2, color='orange', label=r'$F_a$ (Atrito Sup. Móvel)')
ax_forces.plot(Ho_coords, calcular_atrito_fixo(Ho_coords), lw=2, color='purple', label=r'$F_b$ (Atrito Sup. Fixa)')
vline_forces = ax_forces.axvline(x=Ho_inicial_calc, color='r', linestyle='--', lw=1.5)
marker_fa, = ax_forces.plot([], [], 'o', color='orange', markersize=8)
marker_fb, = ax_forces.plot([], [], 'o', color='purple', markersize=8)
texto_fa = ax_forces.text(0, 0, '', ha='left', va='bottom', fontsize=10, bbox=dict(fc='white', alpha=0.7))
texto_fb = ax_forces.text(0, 0, '', ha='left', va='top', fontsize=10, bbox=dict(fc='white', alpha=0.7))
ax_forces.set_xlabel(r'Razão de Espessura do Filme, $H_0 = h_0/s_h$', fontsize=12)
ax_forces.set_ylabel(r'Forças de Atrito Adimensionais', fontsize=12)
ax_forces.set_xlim(0, 3.0); ax_forces.set_ylim(-10, 10); ax_forces.axhline(0, color='black', lw=1); ax_forces.grid(True, linestyle=':', alpha=0.7)
ax_forces.legend()

# --- SLIDERS DE CONTROLE ---
ax_slider_l = plt.axes([0.25, 0.12, 0.5, 0.02]); ax_slider_sh = plt.axes([0.25, 0.08, 0.5, 0.02]); ax_slider_h0 = plt.axes([0.25, 0.04, 0.5, 0.02])
slider_l = Slider(ax=ax_slider_l, label='Comprimento (l)', valmin=10, valmax=200, valinit=l_inicial)
slider_sh = Slider(ax=ax_slider_sh, label='Altura da Cunha (sₕ)', valmin=0.1, valmax=50, valinit=sh_inicial)
slider_h0 = Slider(ax=ax_slider_h0, label='Folga de Saída (h₀)', valmin=0.1, valmax=50, valinit=h0_inicial)

# --- FUNÇÃO DE ATUALIZAÇÃO GLOBAL ---
def update(val):
    l, sh, h0 = slider_l.val, slider_sh.val, slider_h0.val
    Ho = h0 / sh

    # Atualiza Geometria (Painel 1)
    sapata.set_xy(criar_vertices_sapata(l, sh, h0))
    pista.set_data([-l*0.1, l*1.1], [0, 0])
    ax_geom.set_xlim(-l*0.2, l*1.2); ax_geom.set_ylim(-l*0.15, (sh+h0)*1.8)
    texto_Ho_geom.set_text(f'$H_0 = {Ho:.3f}$')
    Ho_otimo = 1 / np.sqrt(2)
    texto_Ho_geom.set_color('green' if abs(Ho - Ho_otimo) < 0.05 else 'black')

    # Atualiza Pressão (Painel 2)
    novos_P = calcular_pressao_adimensional(X_coords, Ho)
    linha_pressao.set_ydata(novos_P)
    novo_Xm, novo_Pm = encontrar_pico_pressao(Ho)
    ponto_pico.set_data(novo_Xm, novo_Pm)
    texto_pico.set_text(f'P_max = {novo_Pm:.3f}'); texto_pico.set_position((novo_Xm, novo_Pm * 1.05))
    ax_pressao.set_ylim(0, max(0.5, novo_Pm * 1.2))

    # Atualiza Marcadores (Painéis 3 e 4)
    vline_wz.set_xdata([Ho, Ho]); vline_forces.set_xdata([Ho, Ho])
    
    Wz_atual = calcular_carga_normal(Ho)
    marker_wz.set_data(Ho, Wz_atual)
    texto_wz.set_text(f'$W_z = {Wz_atual:.2f}$'); texto_wz.set_position((Ho, Wz_atual))
    
    Fa_atual = calcular_atrito_movel(Ho)
    Fb_atual = calcular_atrito_fixo(Ho)
    marker_fa.set_data(Ho, Fa_atual); marker_fb.set_data(Ho, Fb_atual)
    texto_fa.set_text(f'$F_a = {Fa_atual:.2f}$'); texto_fa.set_position((Ho, Fa_atual))
    texto_fb.set_text(f'$F_b = {Fb_atual:.2f}$'); texto_fb.set_position((Ho, Fb_atual))

    fig.canvas.draw_idle()

# --- Conectar e Inicializar ---
slider_l.on_changed(update); slider_sh.on_changed(update); slider_h0.on_changed(update)
update(0)
plt.show()