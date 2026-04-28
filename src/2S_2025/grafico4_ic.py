# Código que integra os codigos grafico2_ic.py e grafico3_ic.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Polygon

# --- FUNÇÕES DE CÁLCULO (Baseadas nas equações que analisamos) ---

def calcular_pressao_adimensional(X, Ho):
    """Calcula a pressão adimensional P (Eq. 8.24)."""
    numerador = 6 * X * (1 - X)
    denominador = ((Ho + 1 - X)**2) * (1 + 2 * Ho)
    if (1 + 2 * Ho) <= 1e-9: # Evita divisão por zero
        return np.zeros_like(X)
    return np.divide(numerador, denominador, out=np.zeros_like(numerador), where=denominador!=0)

def encontrar_pico_pressao(Ho):
    """Calcula a localização (Xm) e o valor (Pm) da pressão máxima (Eq. 8.25)."""
    if (1 + 2 * Ho) <= 1e-9:
        return 1.0, 0.0 # Define um padrão para evitar erros
    Xm = (1 + Ho) / (1 + 2 * Ho)
    Pm = calcular_pressao_adimensional(Xm, Ho)
    return Xm, Pm

def criar_vertices_sapata(l, sh, h0):
    """Calcula os vértices do polígono da sapata superior."""
    altura_bloco = (sh + h0) * 0.4
    vertices = [
        [0, h0 + sh], [l, h0], [l, h0 + sh + altura_bloco], [0, h0 + sh + altura_bloco]
    ]
    return np.array(vertices)

# --- CONFIGURAÇÃO INICIAL ---
l_inicial = 100.0
sh_inicial = 14.1
h0_inicial = 10.0

# --- CRIAÇÃO DA INTERFACE GRÁFICA ---
# Criamos uma figura com dois subplots (gráficos) lado a lado
fig, (ax_geom, ax_pressao) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Análise Interativa de Mancal de Escora de Inclinação Fixa', fontsize=18, fontweight='bold')
# Ajusta o espaçamento para dar lugar aos sliders e títulos
plt.subplots_adjust(bottom=0.3, wspace=0.3)

# --- GRÁFICO 1: GEOMETRIA INTERATIVA (ESQUERDA) ---
ax_geom.set_title('Geometria do Mancal', fontsize=14)
sapata = Polygon(criar_vertices_sapata(l_inicial, sh_inicial, h0_inicial), facecolor='lightgray', edgecolor='black', lw=1.5)
ax_geom.add_patch(sapata)
pista, = ax_geom.plot([0, l_inicial], [0, 0], color='black', lw=2)
ax_geom.set_aspect('equal', adjustable='box')
ax_geom.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

# Anotações dinâmicas para a geometria
texto_l = ax_geom.text(l_inicial / 2, -l_inicial*0.1, f'l = {l_inicial:.1f}', ha='center', va='center', fontsize=12)
texto_h0 = ax_geom.text(l_inicial * 1.02, h0_inicial / 2, f'h₀ = {h0_inicial:.1f}', ha='left', va='center', fontsize=12)
texto_sh = ax_geom.text(-l_inicial * 0.02, h0_inicial + sh_inicial / 2, f'sₕ = {sh_inicial:.1f}', ha='right', va='center', fontsize=12)
texto_angulo = ax_geom.text(0.5, 0.95, '', transform=ax_geom.transAxes, ha='center', va='center', fontsize=12)
texto_Ho_geom = ax_geom.text(0.5, 0.85, '', transform=ax_geom.transAxes, ha='center', va='center', fontsize=14, fontweight='bold')


# --- GRÁFICO 2: DISTRIBUIÇÃO DE PRESSÃO (DIREITA) ---
ax_pressao.set_title('Distribuição de Pressão Adimensional', fontsize=14)
X_coords = np.linspace(0, 1, 500)
Ho_inicial = h0_inicial / sh_inicial
P_inicial = calcular_pressao_adimensional(X_coords, Ho_inicial)
linha_pressao, = ax_pressao.plot(X_coords, P_inicial, lw=2.5, color='royalblue')

# Marcador de pico de pressão
Xm_inicial, Pm_inicial = encontrar_pico_pressao(Ho_inicial)
ponto_pico, = ax_pressao.plot(Xm_inicial, Pm_inicial, 'ro', markersize=8)
texto_pico = ax_pressao.text(Xm_inicial, Pm_inicial + 0.1, f'P_max = {Pm_inicial:.3f}', ha='center', fontsize=10)

# Customização do gráfico de pressão
ax_pressao.set_xlabel(r'Coordenada Adimensional, $X = x/l$', fontsize=12)
ax_pressao.set_ylabel(r'Pressão Adimensional, $P = \frac{p s_h^2}{\eta_0 u_b l}$', fontsize=12)
ax_pressao.set_xlim(0, 1.0)
ax_pressao.set_ylim(0, 1.0) # Começa com um limite baixo, será ajustado
ax_pressao.grid(True, linestyle='--', alpha=0.7)


# --- SLIDERS DE CONTROLE (NA PARTE INFERIOR) ---
ax_slider_l = plt.axes([0.25, 0.15, 0.5, 0.03])
ax_slider_sh = plt.axes([0.25, 0.10, 0.5, 0.03])
ax_slider_h0 = plt.axes([0.25, 0.05, 0.5, 0.03])

slider_l = Slider(ax=ax_slider_l, label='Comprimento (l)', valmin=10, valmax=200, valinit=l_inicial)
slider_sh = Slider(ax=ax_slider_sh, label='Altura da Cunha (sₕ)', valmin=0.1, valmax=50, valinit=sh_inicial)
slider_h0 = Slider(ax=ax_slider_h0, label='Folga de Saída (h₀)', valmin=0.1, valmax=50, valinit=h0_inicial)

# --- FUNÇÃO DE ATUALIZAÇÃO GLOBAL ---
def update(val):
    # 1. Pega os valores atuais dos sliders
    l, sh, h0 = slider_l.val, slider_sh.val, slider_h0.val
    
    # 2. Atualiza o gráfico de GEOMETRIA (esquerda)
    sapata.set_xy(criar_vertices_sapata(l, sh, h0))
    pista.set_data([-l*0.1, l*1.1], [0, 0])
    texto_l.set_position((l/2, -l*0.1)); texto_l.set_text(f'l = {l:.1f}')
    texto_h0.set_position((l*1.02, h0/2)); texto_h0.set_text(f'h₀ = {h0:.1f}')
    texto_sh.set_position((-l*0.02, h0 + sh/2)); texto_sh.set_text(f'sₕ = {sh:.1f}')
    ax_geom.set_xlim(-l*0.2, l*1.2)
    ax_geom.set_ylim(-l*0.15, (sh+h0)*1.8)
    
    # 3. Calcula os parâmetros derivados
    Ho = h0 / sh
    angulo_graus = np.degrees(np.arctan(sh / l))
    
    # Atualiza textos de parâmetros derivados no gráfico de geometria
    texto_angulo.set_text(f'Ângulo (θ) = {angulo_graus:.2f}°')
    texto_Ho_geom.set_text(f'$H_0 = h_0/s_h = {Ho:.3f}$')
    
    # "Surpreendentemente bom": Muda a cor do texto de H₀ se estiver perto do ótimo
    Ho_otimo = 1 / np.sqrt(2) # ~0.707
    if abs(Ho - Ho_otimo) < 0.05:
        texto_Ho_geom.set_color('green')
    else:
        texto_Ho_geom.set_color('black')
        
    # 4. Atualiza o gráfico de PRESSÃO (direita)
    novos_P = calcular_pressao_adimensional(X_coords, Ho)
    linha_pressao.set_ydata(novos_P)
    
    novo_Xm, novo_Pm = encontrar_pico_pressao(Ho)
    ponto_pico.set_data(novo_Xm, novo_Pm)
    texto_pico.set_position((novo_Xm, novo_Pm * 1.05))
    texto_pico.set_text(f'P_max = {novo_Pm:.3f}')
    
    # Ajusta o limite Y do gráfico de pressão dinamicamente
    ax_pressao.set_ylim(0, max(0.5, novo_Pm * 1.2))
    
    # 5. Redesenha a figura inteira
    fig.canvas.draw_idle()

# --- Conectar os sliders à função e chamar uma vez para inicializar ---
slider_l.on_changed(update)
slider_sh.on_changed(update)
slider_h0.on_changed(update)
update(0) # Chame para ajustar os eixos na primeira execução

plt.show()