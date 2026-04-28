"""
================================================================================
ANÁLISE DE CONVERGÊNCIA NUMÉRICA DA INTEGRAL Wz(V, H₀)
================================================================================

Este código compara a solução numérica (soma de Riemann - retângulos à esquerda)
com a solução analítica da integral:

    Wz(V, H₀) = ∫[H₀ to H₀+1] P(V,H,H₀) dH

onde P(V,H,H₀) é uma função definida no contexto do problema.

OBJETIVO: Demonstrar a convergência da solução numérica e visualizar como
o erro diminui quando aumentamos o número de subdivisões (n).

Autor: Rafael Rodrigues Pimentel de Melo
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec

# ==============================================================================
# SEÇÃO 1: DEFINIÇÃO DAS FUNÇÕES MATEMÁTICAS
# ==============================================================================

def P(V, H, H0):
    """
    Calcula a função P(V,H,H₀) que é o integrando de Wz.
    
    Fórmula:
        P(V,H,H₀) = (V-6) * [1/(2H₀+1) - 1/H + H₀(H₀+1)/(H²(2H₀+1))]
    
    Args:
        V (float ou array): Parâmetro V
        H (float ou array): Variável de integração H
        H0 (float): Parâmetro H₀ (constante durante cada integração)
    
    Returns:
        float ou array: Valor de P(V,H,H₀)
    """
    term1 = 1 / (2*H0 + 1)
    term2 = -1 / H
    term3 = H0 * (H0 + 1) / (H**2 * (2*H0 + 1))
    return (V - 6) * (term1 + term2 + term3)


def Wz_numerica(V, H0, n):
    """
    Calcula Wz numericamente usando a **SOMA DE RIEMANN (retângulos à esquerda)**.
    
    A soma de Riemann aproxima a integral dividindo o intervalo [H₀, H₀+1]
    em n subintervalos de largura ΔH, e somando as áreas dos retângulos:
    
        ∫[a to b] f(x)dx ≈ Δx * Σ f(x_i)
    
    onde usamos amostragem à esquerda:
        ΔH = (1 / n)
        H_i = H0 + i*ΔH,  i = 0..n-1
    
    Args:
        V (float): Parâmetro V
        H0 (float): Limite inferior de integração
        n (int): Número de subdivisões (quanto maior, mais preciso)
    
    Returns:
        float: Valor aproximado de Wz(V, H₀)
    
    Complexidade: O(n)
    Erro teórico: O(1/n) - o erro diminui linearmente com n (soma de Riemann esquerda)
    """
    # largura do subintervalo
    dH = 1.0 / n
    # pontos de amostragem (esquerda)
    H = H0 + np.arange(n) * dH  # H0, H0 + dH, ..., H0 + (n-1)*dH
    
    # Avalia o integrando P em cada ponto (vetorial)
    P_vals = P(V, H, H0)
    
    # Soma de Riemann (retângulos à esquerda)
    integral = np.sum(P_vals) * dH
    
    return integral


def Wz_analitica(V, H0):
    """
    Calcula Wz usando a SOLUÇÃO ANALÍTICA (exata).
    
    Esta é a solução obtida resolvendo a integral simbolicamente:
        Wz(V,H₀) = (6-V) * [ln((H₀+1)/H₀) - 2/(2H₀+1)]
    
    Args:
        V (float): Parâmetro V
        H0 (float): Parâmetro H₀
    
    Returns:
        float: Valor exato de Wz(V, H₀)
    """
    termo_log = np.log((H0 + 1) / H0)  # ln((H₀+1)/H₀)
    termo_frac = 2 / (2*H0 + 1)        # 2/(2H₀+1)
    return (6 - V) * (termo_log - termo_frac)


# ==============================================================================
# SEÇÃO 2: FUNÇÕES AUXILIARES PARA CÁLCULO DE SUPERFÍCIES
# ==============================================================================

def compute_surface_numerica(V_vals, H0_vals, n):
    """
    Calcula a superfície Wz numérica para uma grade de valores (V, H₀).
    """
    W = np.zeros((len(H0_vals), len(V_vals)))
    for i, H0 in enumerate(H0_vals):
        for j, V in enumerate(V_vals):
            W[i, j] = Wz_numerica(V, H0, n)
    return W


def compute_surface_analitica(V_vals, H0_vals):
    """
    Calcula a superfície Wz analítica para uma grade de valores (V, H₀).
    """
    W = np.zeros((len(H0_vals), len(V_vals)))
    for i, H0 in enumerate(H0_vals):
        for j, V in enumerate(V_vals):
            W[i, j] = Wz_analitica(V, H0)
    return W


# ==============================================================================
# SEÇÃO 3: ANÁLISE DE CONVERGÊNCIA
# ==============================================================================

def analyze_convergence(V_test, H0_test, n_values):
    """
    Analisa como o erro varia com diferentes valores de n.
    
    EXPLICAÇÃO DO ERRO:
    -------------------
    O erro absoluto é definido como:
        Erro(n) = |Wz_numérico(n) - Wz_exato|
    
    Para a soma de Riemann (retângulos à esquerda), esperamos que:
        Erro(n) ≈ C / n
    
    onde C é uma constante. Isso significa que ao DOBRAR n,
    o erro cai aproximadamente 2x (pois 2^1 = 2).
    """
    erros = []
    wz_exact = Wz_analitica(V_test, H0_test)
    
    for n in n_values:
        wz_num = Wz_numerica(V_test, H0_test, n)
        erro = abs(wz_num - wz_exact)
        erros.append(erro)
    
    return np.array(erros)


# ==============================================================================
# SEÇÃO 4: TESTE E DIAGNÓSTICO NO CONSOLE
# ==============================================================================

print("\n" + "="*70)
print("ANÁLISE DE CONVERGÊNCIA")
print("="*70)

# Escolhe um ponto de teste representativo
V_test, H0_test = 10.0, 2.0
wz_exact = Wz_analitica(V_test, H0_test)

print(f"Ponto teste: V={V_test}, H0={H0_test}")
print(f"Wz exato (analítico) = {wz_exact:.10f}\n")
print(f"{'n':>6} | {'Erro absoluto':>15} | {'Redução':>10}")
print("-"*70)

# Testa convergência para diferentes valores de n
n_prev_erro = None
for n in [10, 20, 50, 100, 200, 500]:
    wz_num = Wz_numerica(V_test, H0_test, n)
    erro = abs(wz_num - wz_exact)
    
    if n_prev_erro is not None:
        razao = n_prev_erro / erro if erro > 0 else np.inf
        print(f"{n:>6} | {erro:>15.2e} | {razao:>9.1f}x")
    else:
        print(f"{n:>6} | {erro:>15.2e} | {'---':>10}")
    
    n_prev_erro = erro

print("\nInterpretação: A 'Redução' mostra quantas vezes o erro diminuiu")
print("em relação ao n anterior. Para O(1/n), esperamos ~2x ao dobrar n.")
print("="*70 + "\n")


# ==============================================================================
# SEÇÃO 5: CONFIGURAÇÃO INICIAL E PREPARAÇÃO DE DADOS
# ==============================================================================

# Parâmetros iniciais para visualização
Vmax_init = 12      # Valor máximo de V a ser plotado
H0max_init = 5      # Valor máximo de H₀ a ser plotado
n_init = 100        # Número inicial de subdivisões

# Cria grades de valores para plotar superfícies
V_vals = np.linspace(6.1, Vmax_init, 50)   # 50 pontos em V (começando em 6.1)
H0_vals = np.linspace(0.5, H0max_init, 50) # 50 pontos em H₀

# Calcula curva de convergência (para o gráfico log-log)
n_convergence = np.logspace(np.log10(10), np.log10(500), 40, dtype=int)
n_convergence = np.unique(n_convergence)  # Remove possíveis duplicatas e ordena
erros_convergence = analyze_convergence(V_test, H0_test, n_convergence)

# Calcula superfícies iniciais
W_num = compute_surface_numerica(V_vals, H0_vals, n_init)
W_ana = compute_surface_analitica(V_vals, H0_vals)

# Cria meshgrids para plotagem 3D
V_mesh, H0_mesh = np.meshgrid(V_vals, H0_vals)


# ==============================================================================
# SEÇÃO 6: CRIAÇÃO DA FIGURA E SUBPLOTS
# ==============================================================================

# Cria figura com 4 subplots em grade 2x2
fig = plt.figure(figsize=(16, 9))
gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.3, 
              left=0.08, right=0.95, top=0.92, bottom=0.15)

# ------------------------------------------------------------------------------ 
# SUBPLOT 1: Superfície Wz Numérica
# ------------------------------------------------------------------------------ 
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
surf_num = ax1.plot_surface(V_mesh, H0_mesh, W_num, cmap='viridis', alpha=0.9, 
                             edgecolor='none', antialiased=True)
ax1.set_title(f"Wz Numérica (n={n_init})", fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel("V", fontsize=11, labelpad=8)
ax1.set_ylabel("H₀", fontsize=11, labelpad=8)
ax1.set_zlabel("Wz", fontsize=11, labelpad=8)
ax1.view_init(elev=25, azim=130)

# ------------------------------------------------------------------------------ 
# SUBPLOT 2: Superfície Wz Analítica (Referência Exata)
# ------------------------------------------------------------------------------ 
ax2 = fig.add_subplot(gs[0, 1], projection='3d')
surf_ana = ax2.plot_surface(V_mesh, H0_mesh, W_ana, cmap='plasma', alpha=0.9,
                             edgecolor='none', antialiased=True)
ax2.set_title("Wz Analítica (Exata)", fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel("V", fontsize=11, labelpad=8)
ax2.set_ylabel("H₀", fontsize=11, labelpad=8)
ax2.set_zlabel("Wz", fontsize=11, labelpad=8)
ax2.view_init(elev=25, azim=130)

# ------------------------------------------------------------------------------ 
# SUBPLOT 3: Gráfico de Convergência (Log-Log)
# ------------------------------------------------------------------------------ 
ax3 = fig.add_subplot(gs[1, 0])

# Plota curva erro vs n em escala log-log
line_convergence, = ax3.loglog(n_convergence, erros_convergence, 'b-', 
                                linewidth=3, alpha=0.6, label='Erro numérico')

# Marca o ponto atual com destaque
erro_atual = np.interp(n_init, n_convergence, erros_convergence)
point_current, = ax3.loglog([n_init], [erro_atual], 
                             'ro', markersize=12, label=f'n = {n_init}', zorder=10,
                             markeredgecolor='darkred', markeredgewidth=2)

# Adiciona linha de referência teórica O(1/n)
n_ref = np.array([10, 500])
erro_ref = erros_convergence[0] * (10 / n_ref)  # Escala proporcional a 1/n
ax3.loglog(n_ref, erro_ref, 'k--', linewidth=2, alpha=0.6, label='Teórico: O(1/n)')

# Anotação
annotation = ax3.annotate(f'Erro: {erro_atual:.2e}', 
                         xy=(n_init, erro_atual), xytext=(30, 30),
                         textcoords='offset points', fontsize=11, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', 
                                       color='red', lw=2))

ax3.set_xlabel('n (subdivisões)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Erro Absoluto', fontsize=12, fontweight='bold')
ax3.set_title('Convergência: Como o erro diminui com n', fontsize=13, fontweight='bold', pad=12)
ax3.grid(True, alpha=0.3, linestyle=':', which='both')
ax3.legend(fontsize=11, loc='upper right', framealpha=0.9)
ax3.set_xlim(8, 600)
ax3.set_ylim(erros_convergence.min()*0.5, erros_convergence.max()*2)

# ------------------------------------------------------------------------------ 
# SUBPLOT 4: Superfície de Erro Absoluto
# ------------------------------------------------------------------------------ 
ax4 = fig.add_subplot(gs[1, 1], projection='3d')

# Calcula erro absoluto ponto a ponto
W_erro = np.abs(W_num - W_ana)

# Plota superfície de erro
surf_erro = ax4.plot_surface(V_mesh, H0_mesh, W_erro, cmap='hot_r', alpha=0.95,
                              edgecolor='none', antialiased=True)

# Calcula estatísticas do erro
erro_max_init = np.max(W_erro)      # Erro máximo em toda a superfície
erro_medio_init = np.mean(W_erro)   # Erro médio (integral do erro / área)

ax4.set_title(f"Erro: máx={erro_max_init:.2e}, médio={erro_medio_init:.2e}", 
              fontsize=13, fontweight='bold', color='darkred', pad=12)
ax4.set_xlabel("V", fontsize=11, labelpad=8)
ax4.set_ylabel("H₀", fontsize=11, labelpad=8)
ax4.set_zlabel("Erro", fontsize=11, labelpad=8)
ax4.view_init(elev=25, azim=130)


# ============================================================================== 
# SEÇÃO 7: SLIDERS INTERATIVOS
# ============================================================================== 

# Slider para n (número de subdivisões) - PRINCIPAL
ax_n = plt.axes([0.2, 0.06, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_n = Slider(ax_n, 'n (subdivisões)', 10, 500, valinit=n_init, valstep=5,
                  color='steelblue')

# Slider para V máximo
ax_Vmax = plt.axes([0.15, 0.02, 0.35, 0.02], facecolor='lightgray')
slider_Vmax = Slider(ax_Vmax, 'V máx', 6.1, 20, valinit=Vmax_init, color='green')

# Slider para H₀ máximo
ax_Hmax = plt.axes([0.6, 0.02, 0.35, 0.02], facecolor='lightgray')
slider_Hmax = Slider(ax_Hmax, 'H₀ máx', 0.5, 15, valinit=H0max_init, color='purple')

# Título principal da figura
fig.suptitle('Convergência Numérica da Integral de Wz(V, H₀) — Soma de Riemann (esquerda)', 
             fontsize=14, fontweight='bold', y=0.985)


# ============================================================================== 
# SEÇÃO 8: FUNÇÃO DE ATUALIZAÇÃO (CALLBACK)
# ============================================================================== 

def update(val):
    """
    Função callback chamada quando qualquer slider é movido.
    """
    # Lê valores atuais dos sliders
    n = int(slider_n.val)
    Vmax = slider_Vmax.val
    Hmax = slider_Hmax.val
    
    # Recria grades com novos limites
    V_vals_new = np.linspace(6.1, Vmax, 50)
    H0_vals_new = np.linspace(0.5, Hmax, 50)
    V_mesh_new, H0_mesh_new = np.meshgrid(V_vals_new, H0_vals_new)
    
    # Recalcula superfícies
    W_num_new = compute_surface_numerica(V_vals_new, H0_vals_new, n)
    W_ana_new = compute_surface_analitica(V_vals_new, H0_vals_new)
    W_erro_new = np.abs(W_num_new - W_ana_new)
    
    # --- Atualiza subplot 1: Wz Numérica ---
    ax1.clear()
    ax1.plot_surface(V_mesh_new, H0_mesh_new, W_num_new, cmap='viridis', 
                     alpha=0.9, edgecolor='none', antialiased=True)
    ax1.set_title(f"Wz Numérica (n={n})", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("V", fontsize=11, labelpad=8)
    ax1.set_ylabel("H₀", fontsize=11, labelpad=8)
    ax1.set_zlabel("Wz", fontsize=11, labelpad=8)
    ax1.view_init(elev=25, azim=130)
    
    # --- Atualiza subplot 2: Wz Analítica ---
    ax2.clear()
    ax2.plot_surface(V_mesh_new, H0_mesh_new, W_ana_new, cmap='plasma', 
                     alpha=0.9, edgecolor='none', antialiased=True)
    ax2.set_title("Wz Analítica (Exata)", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("V", fontsize=11, labelpad=8)
    ax2.set_ylabel("H₀", fontsize=11, labelpad=8)
    ax2.set_zlabel("Wz", fontsize=11, labelpad=8)
    ax2.view_init(elev=25, azim=130)
    
    # --- Atualiza subplot 3: Curva de convergência ---
    # Note: erros_convergence foi calculado com a grade original n_convergence.
    # Mantemos esse comportamento para mostrar a curva de referência.
    erro_atual = np.interp(n, n_convergence, erros_convergence)
    point_current.set_data([n], [erro_atual])
    point_current.set_label(f'n = {n}')
    
    # Atualiza anotação
    annotation.xy = (n, erro_atual)
    annotation.set_text(f'Erro: {erro_atual:.2e}')
    
    ax3.legend(fontsize=11, loc='upper right', framealpha=0.9)
    
    # --- Atualiza subplot 4: Superfície de erro ---
    ax4.clear()
    ax4.plot_surface(V_mesh_new, H0_mesh_new, W_erro_new, cmap='hot_r', 
                     alpha=0.95, edgecolor='none', antialiased=True)
    
    # Recalcula estatísticas
    erro_max = np.max(W_erro_new)
    erro_medio = np.mean(W_erro_new)
    
    ax4.set_title(f"Erro: máx={erro_max:.2e}, médio={erro_medio:.2e}", 
                  fontsize=13, fontweight='bold', color='darkred', pad=12)
    ax4.set_xlabel("V", fontsize=11, labelpad=8)
    ax4.set_ylabel("H₀", fontsize=11, labelpad=8)
    ax4.set_zlabel("Erro", fontsize=11, labelpad=8)
    ax4.view_init(elev=25, azim=130)
    
    # Log no console
    print(f"n={n:3d} → Erro em (V=10, H0=2): {erro_atual:.2e} | "
          f"Erro máx superfície: {erro_max:.2e} | Erro médio: {erro_medio:.2e}")
    
    # Redesenha a figura
    fig.canvas.draw_idle()


# ============================================================================== 
# SEÇÃO 9: CONECTA SLIDERS E INICIA VISUALIZAÇÃO
# ============================================================================== 

slider_n.on_changed(update)
slider_Vmax.on_changed(update)
slider_Hmax.on_changed(update)

print("✅ Visualização carregada!")
print("💡 Ajuste o slider 'n' para ver a convergência em tempo real.\n")

# Exibe a figura
plt.show()
