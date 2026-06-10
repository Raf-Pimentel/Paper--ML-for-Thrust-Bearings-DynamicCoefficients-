import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.patches import Polygon

# =============================================================================
# MÓDULO DE CÁLCULO
# =============================================================================

def calcular_pressao_adimensional(X, Ho):
    numerador = 6 * X * (1 - X)
    denominador = ((Ho + 1 - X)**2) * (1 + 2 * Ho)
    if np.isscalar(Ho) and (1 + 2 * Ho) <= 1e-9: return np.zeros_like(X)
    return np.divide(numerador, denominador, out=np.zeros_like(numerador), where=denominador!=0)

def encontrar_pico_pressao(Ho):
    if np.isscalar(Ho) and (1 + 2 * Ho) <= 1e-9: return 1.0, 0.0
    Xm = (1 + Ho) / (1 + 2 * Ho)
    Pm = calcular_pressao_adimensional(Xm, Ho)
    return Xm, Pm

def criar_vertices_sapata(l, sh, h0):
    altura_bloco = (sh + h0) * 0.4
    return np.array([[0, h0 + sh], [l, h0], [l, h0 + sh + altura_bloco], [0, h0 + sh + altura_bloco]])

def calcular_desempenho(Ho):
    with np.errstate(divide='ignore', invalid='ignore'):
        K = 1 / Ho
        log_K = np.log(1 + K)
        frac_K = 2 * K / (2 + K)
        Wz = (6 / K**2) * (log_K - frac_K)
        Fa = (1 / K) * (4 * log_K - 3 * frac_K)
        Fb = (1/K) * (-4*log_K + (12*K)/(2+K))
        param_atrito = np.divide(Fa, Wz)
        param_atrito = np.where(np.abs(Wz) > 1e-9, param_atrito, np.inf)

    if not np.isscalar(Ho):
        Wz = np.nan_to_num(Wz, nan=0.0)
        Fa = np.nan_to_num(Fa, nan=0.0)
        Fb = np.nan_to_num(Fb, nan=0.0)
        param_atrito = np.nan_to_num(param_atrito, nan=np.inf)
        
    return Wz, Fa, Fb, param_atrito

# =============================================================================
# MÓDULO DE PLOTAGEM
# =============================================================================

def plotar_geometria(ax, params):
    l, sh, h0, Ho = params['l'], params['sh'], params['h0'], params['Ho']
    ax.set_title('Visualização da Geometria da Cunha', fontsize=16)
    sapata = Polygon(criar_vertices_sapata(l, sh, h0), fc='lightgray', ec='black', lw=1.5)
    ax.add_patch(sapata)
    ax.plot([-l*0.1, l*1.1], [0, 0], color='black', lw=2)
    ax.text(0.5, 0.85, f'$H_0 = h_0/s_h = {Ho:.3f}$', transform=ax.transAxes, ha='center', fontsize=14, fontweight='bold',
            color='green' if abs(Ho - 1/np.sqrt(2)) < 0.05 else 'black')
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(labelbottom=False, labelleft=False)

def plotar_pressao(ax, params):
    Ho, Xm, Pm = params['Ho'], params['Xm'], params['Pm']
    X_coords = np.linspace(0, 1, 500)
    P_valores = calcular_pressao_adimensional(X_coords, Ho)
    ax.set_title('Distribuição de Pressão Adimensional ($P$ vs. $X$)', fontsize=16)
    ax.plot(X_coords, P_valores, lw=2.5, color='royalblue')
    ax.plot(Xm, Pm, 'ro', markersize=8, label=f'P_max = {Pm:.3f}')
    ax.set_xlabel(r'Coordenada Adimensional, $X = x/l$', fontsize=12)
    ax.set_ylabel(r'Pressão Adimensional, $P$', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend()

def plotar_desempenho(ax, params, tipo_plot, curvas_staticas):
    Ho = params['Ho']
    Ho_coords = curvas_staticas['Ho_coords']
    ax.set_xlabel(r'Razão de Espessura do Filme, $H_0 = h_0/s_h$', fontsize=12)
    ax.axvline(x=Ho, color='r', linestyle='--', lw=1.5, label=f'Posição Atual ($H_0={Ho:.2f}$)')
    
    if tipo_plot == 'carga':
        Wz = params['Wz']
        ax.set_title('Capacidade de Carga vs. Geometria (Fig. 8.8)', fontsize=16)
        ax.plot(Ho_coords, curvas_staticas['Wz'], lw=2, color='green')
        ax.plot(Ho, Wz, 'ro', markersize=8, label=f'$W_z = {Wz:.2f}$')
        ax.set_ylabel(r'Carga Normal Adimensional, $W_z$', fontsize=12)
        
    elif tipo_plot == 'atrito':
        Fa, Fb = params['Fa'], params['Fb']
        ax.set_title('Forças de Atrito vs. Geometria (Fig. 8.9)', fontsize=16)
        ax.plot(Ho_coords, curvas_staticas['Fa'], lw=2, color='orange', label=r'$F_a$ (Móvel)')
        ax.plot(Ho_coords, curvas_staticas['Fb'], lw=2, color='purple', label=r'$F_b$ (Fixa)')
        ax.plot(Ho, Fa, 'o', color='orange', markersize=8, label=f'$F_a = {Fa:.2f}$')
        ax.plot(Ho, Fb, 'o', color='purple', markersize=8, label=f'$F_b = {Fb:.2f}$')
        ax.set_ylabel(r'Forças de Atrito Adimensionais', fontsize=12)
        ax.axhline(0, color='black', lw=1)
        
    elif tipo_plot == 'coef_atrito':
        mu_param = params['mu_param']
        ax.set_title('Coeficiente de Atrito vs. Geometria (Fig. 8.10)', fontsize=16)
        ax.plot(Ho_coords, curvas_staticas['mu_param'], lw=2, color='brown')
        ax.plot(Ho, mu_param, 'o', color='brown', markersize=8, label=f'$\\mu\\ell/s_h = {mu_param:.2f}$')
        ax.set_ylabel(r'Parâmetro de Atrito, $\mu\ell/s_h$', fontsize=12)

    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right')

# =============================================================================
# SCRIPT PRINCIPAL
# =============================================================================

fig = plt.figure(figsize=(14, 8))
plt.suptitle('Plataforma de Análise de Mancal de Escora (Final V2)', fontsize=20, fontweight='bold')
ax_main = fig.add_axes([0.3, 0.3, 0.65, 0.6])
ax_radio = fig.add_axes([0.05, 0.5, 0.15, 0.25])
ax_slider_l = fig.add_axes([0.3, 0.15, 0.55, 0.03])
ax_slider_sh = fig.add_axes([0.3, 0.10, 0.55, 0.03])
ax_slider_h0 = fig.add_axes([0.3, 0.05, 0.55, 0.03])

Ho_coords = np.linspace(0.01, 3.0, 400)
Wz_static, Fa_static, Fb_static, mu_param_static = calcular_desempenho(Ho_coords)
curvas_staticas = {'Ho_coords': Ho_coords, 'Wz': Wz_static, 'Fa': Fa_static, 'Fb': Fb_static, 'mu_param': mu_param_static}

l_inicial, sh_inicial, h0_inicial = 100.0, 14.1, 10.0
slider_l = Slider(ax=ax_slider_l, label='Comprimento (l)', valmin=10, valmax=200, valinit=l_inicial)
slider_sh = Slider(ax=ax_slider_sh, label='Altura da Cunha (sₕ)', valmin=0.1, valmax=50, valinit=sh_inicial)
slider_h0 = Slider(ax=ax_slider_h0, label='Folga de Saída (h₀)', valmin=0.1, valmax=50, valinit=h0_inicial)

opcoes_grafico = ['Geometria', 'Pressão (P)', 'Carga (Wz)', 'Atrito (Fa, Fb)', 'Coef. Atrito (μ)']
radio_botoes = RadioButtons(ax_radio, opcoes_grafico)

funcoes_de_plotagem = {
    'Geometria': plotar_geometria,
    'Pressão (P)': plotar_pressao,
    'Carga (Wz)': lambda ax, p: plotar_desempenho(ax, p, 'carga', curvas_staticas),
    'Atrito (Fa, Fb)': lambda ax, p: plotar_desempenho(ax, p, 'atrito', curvas_staticas),
    'Coef. Atrito (μ)': lambda ax, p: plotar_desempenho(ax, p, 'coef_atrito', curvas_staticas)
}

# Limites fixos por tipo de gráfico
limites_plotagem = {
    'Geometria': {'xlim': (-20, 220), 'ylim': (-10, 120)},
    'Pressão (P)': {'xlim': (0, 1), 'ylim': (0, 4.0)},
    'Carga (Wz)': {'xlim': (0, 3.0), 'ylim': (0, 15)},
    'Atrito (Fa, Fb)': {'xlim': (0, 3.0), 'ylim': (-15, 15)},
    'Coef. Atrito (μ)': {'xlim': (0, 3.0), 'ylim': (0, 15)}
}

def update(val):
    ax_main.clear()
    l, sh, h0 = slider_l.val, slider_sh.val, slider_h0.val
    grafico_selecionado = radio_botoes.value_selected
    Ho = h0 / sh
    Wz, Fa, Fb, mu_param = calcular_desempenho(Ho)
    Xm, Pm = encontrar_pico_pressao(Ho)
    params = {'l': l, 'sh': sh, 'h0': h0, 'Ho': Ho, 'Wz': Wz, 'Fa': Fa, 
              'Fb': Fb, 'mu_param': mu_param, 'Xm': Xm, 'Pm': Pm}
    funcao_selecionada = funcoes_de_plotagem[grafico_selecionado]
    funcao_selecionada(ax_main, params)

    # Aplica limites fixos
    limites = limites_plotagem[grafico_selecionado]
    ax_main.set_xlim(*limites['xlim'])
    ax_main.set_ylim(*limites['ylim'])

    fig.canvas.draw_idle()

slider_l.on_changed(update)
slider_sh.on_changed(update)
slider_h0.on_changed(update)
radio_botoes.on_clicked(update)

update(None)
plt.show()
