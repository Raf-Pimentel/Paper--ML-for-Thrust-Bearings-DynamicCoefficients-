# Código que realiza uma animação e ilustração geométrica da cunha física de um mancal hidrodinâmico
# Baseado no livro do Hamrock

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Polygon

# --- Função para criar a geometria da sapata (o polígono) ---
def criar_vertices_sapata(l, sh, h0):
    """
    Calcula as coordenadas dos vértices do polígono que representa a sapata superior.
    A altura total da sapata é mantida constante para um visual mais limpo.
    """
    # Define uma altura fixa para a parte retangular da sapata
    altura_bloco = (sh + h0) * 0.4 
    
    # Vértices do polígono no sentido horário, começando do canto inferior esquerdo
    # [canto_inf_esq, canto_inf_dir, canto_sup_dir, canto_sup_esq]
    vertices = [
        [0, h0 + sh],          # Canto inferior esquerdo
        [l, h0],               # Canto inferior direito
        [l, h0 + sh + altura_bloco], # Canto superior direito
        [0, h0 + sh + altura_bloco]  # Canto superior esquerdo
    ]
    return np.array(vertices)

# --- Configuração Inicial ---
l_inicial = 100.0
sh_inicial = 10.0
h0_inicial = 5.0

# --- Criação da Figura e dos Eixos ---
# Criamos a figura e o eixo principal, deixando espaço para os sliders
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.35)

# --- Desenho Inicial da Geometria ---

# 1. Sapata Superior (o polígono)
vertices_iniciais = criar_vertices_sapata(l_inicial, sh_inicial, h0_inicial)
sapata = Polygon(vertices_iniciais, facecolor='lightgray', edgecolor='black')
ax.add_patch(sapata)

# 2. Pista Inferior (linha horizontal)
pista, = ax.plot([0, l_inicial], [0, 0], color='black', lw=2)

# --- Anotações e Legendas Dinâmicas ---
# Estas anotações vão se mover junto com a geometria

# Anotação para 'l'
texto_l = ax.text(l_inicial / 2, -l_inicial*0.08, f'l = {l_inicial:.1f}', ha='center', va='center', fontsize=12)
# Anotação para 'h₀'
texto_h0 = ax.text(l_inicial * 1.02, h0_inicial / 2, f'h₀ = {h0_inicial:.1f}', ha='left', va='center', fontsize=12)
# Anotação para 'sₕ'
texto_sh = ax.text(-l_inicial * 0.02, h0_inicial + sh_inicial / 2, f'sₕ = {sh_inicial:.1f}', ha='right', va='center', fontsize=12)
# Anotação para o ângulo theta
angulo_rad = np.arctan(sh_inicial / l_inicial)
angulo_graus = np.degrees(angulo_rad)
texto_angulo = ax.text(0.5, 0.9, f'Ângulo da Cunha (θ) = {angulo_graus:.2f}°', 
                       transform=ax.transAxes, ha='center', fontsize=12,
                       bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

# --- Configuração dos Eixos ---
ax.set_title('Geometria Interativa do Mancal de Inclinação Fixa', fontsize=16)
ax.set_aspect('equal', adjustable='box') # Mantém a proporção visual correta
ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
               labelbottom=False, labelleft=False) # Esconde os eixos numéricos

# --- Criação dos Sliders ---

# Posições dos sliders [esquerda, baixo, largura, altura]
ax_slider_l = plt.axes([0.2, 0.20, 0.65, 0.03])
ax_slider_sh = plt.axes([0.2, 0.15, 0.65, 0.03])
ax_slider_h0 = plt.axes([0.2, 0.10, 0.65, 0.03])

# Instanciando os sliders
slider_l = Slider(ax=ax_slider_l, label='Comprimento (l)', valmin=10, valmax=200, valinit=l_inicial)
slider_sh = Slider(ax=ax_slider_sh, label='Altura da Cunha (sₕ)', valmin=1, valmax=50, valinit=sh_inicial)
slider_h0 = Slider(ax=ax_slider_h0, label='Folga de Saída (h₀)', valmin=1, valmax=50, valinit=h0_inicial)


# --- Função de Atualização da Animação ---
def update(val):
    # Pega os valores atuais de todos os sliders
    l = slider_l.val
    sh = slider_sh.val
    h0 = slider_h0.val
    
    # Recalcula e atualiza os vértices da sapata
    novos_vertices = criar_vertices_sapata(l, sh, h0)
    sapata.set_xy(novos_vertices)
    
    # Atualiza a linha da pista inferior
    pista.set_data([0 - l*0.1, l + l*0.1], [0, 0]) # Adiciona margem
    
    # Atualiza a posição das anotações
    texto_l.set_position((l / 2, -l*0.08))
    texto_l.set_text(f'l = {l:.1f}')
    
    texto_h0.set_position((l * 1.02, h0 / 2))
    texto_h0.set_text(f'h₀ = {h0:.1f}')
    
    texto_sh.set_position((-l * 0.02, h0 + sh / 2))
    texto_sh.set_text(f'sₕ = {sh:.1f}')
    
    # Recalcula e atualiza o ângulo
    angulo_rad = np.arctan(sh / l)
    angulo_graus = np.degrees(angulo_rad)
    texto_angulo.set_text(f'Ângulo da Cunha (θ) = {angulo_graus:.2f}°')
    
    # Reajusta os limites do gráfico para a animação ficar centralizada e visível
    ax.set_xlim(-l * 0.15, l * 1.15)
    ax.set_ylim(-l*0.1, (sh + h0) * 2) # Garante espaço em cima
    
    # Redesenha a figura
    fig.canvas.draw_idle()

# --- Conectar os Sliders à Função de Atualização ---
slider_l.on_changed(update)
slider_sh.on_changed(update)
slider_h0.on_changed(update)

# Chamar a função de atualização uma vez no início para ajustar os limites
update(0)

# Mostrar a interface
plt.show()