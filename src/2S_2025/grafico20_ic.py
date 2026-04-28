# Compactei os codigos grafico17_ic.py e grafico18_ic.py e grafico19_ic.py em um unico arquivo
# O grafico20_ic.py é uma versão minimalista, sem plotly, apenas com matplotlib e sliders
# Mantive os dois códigos para referência futura, mas o grafico20_ic.py é o mais completo e funcional

# OBS: OS GRÁFICOS ESTÃO ERRADOS.

"""
grafico20_ic.py - Validação Numérica de Derivadas (Versão Minimalista)
======================================================================
Compara soluções analíticas vs numéricas para:
- Coeficiente de Rigidez (K)
- Coeficiente de Amortecimento (C)

Método: Diferenças finitas centrais
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider

# =============================================================================
# CÁLCULOS - RIGIDEZ (K)
# =============================================================================

def Wz_estatico(V, H0):
    """Wz = (6 - V) * [ln((H0+1)/H0) - 2/(2*H0+1)]"""
    H0_array = np.asanyarray(H0)
    with np.errstate(divide='ignore', invalid='ignore'):
        termo_log = np.log((H0_array + 1) / H0_array)
        termo_fracao = 2 / (2 * H0_array + 1)
        resultado = (6 - V) * (termo_log - termo_fracao)
    return np.where(H0_array <= 1e-9, 0.0, resultado)

def K_analitico(H0):
    """K = 24/(1+2*H0)^2 - 6/(H0*(H0+1))"""
    H0_array = np.asanyarray(H0)
    with np.errstate(divide='ignore', invalid='ignore'):
        termo1 = 24 / ((1 + 2 * H0_array)**2)
        termo2 = 6 / (H0_array * (H0_array + 1))
        resultado = termo1 - termo2
    return np.where(H0_array <= 1e-9, 0.0, resultado)

def K_numerico(V, H0_array, h=1e-8):
    """Derivada numérica por diferenças finitas centrais"""
    K_num = np.zeros_like(H0_array)
    for i, H0 in enumerate(H0_array):
        K_num[i] = (Wz_estatico(V, H0 + h) - Wz_estatico(V, H0 - h)) / (2 * h)
    return K_num

# =============================================================================
# CÁLCULOS - AMORTECIMENTO (C)
# =============================================================================

def C_analitico(H0):
    """C = ln((H0+1)/H0) - 2/(2*H0+1)"""
    H0_array = np.asanyarray(H0)
    with np.errstate(divide='ignore', invalid='ignore'):
        termo_log = np.log((H0_array + 1) / H0_array)
        termo_fracao = 2 / (2 * H0_array + 1)
        resultado = termo_log - termo_fracao
    return np.where(H0_array <= 1e-9, np.inf, resultado)

def C_numerico(H0_array, delta_V=1e-8):
    """Derivada numérica: C ≈ -[Wz(+dV) - Wz(-dV)] / (2*dV)"""
    Wz_mais = Wz_estatico(delta_V, H0_array)
    Wz_menos = Wz_estatico(-delta_V, H0_array)
    return -((Wz_mais - Wz_menos) / (2 * delta_V))

# =============================================================================
# INTERFACE INTERATIVA
# =============================================================================

def criar_interface():
    """Interface minimalista com sliders."""
    
    # Configuração inicial
    H0_max_init = 3.0
    n_points_init = 200
    V_init = 3.0
    
    # Criar figura
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#f8f9fa')
    
    # Layout: 2 linhas para gráficos - COM ESPAÇO À DIREITA PARA LEGENDAS
    # top=0.90 deixa MAIS ESPAÇO entre título principal e gráfico superior
    gs = GridSpec(2, 1, figure=fig, hspace=0.35, top=0.90, bottom=0.14, 
                  left=0.08, right=0.82)
    
    # Título principal BEM ACIMA
    fig.suptitle('Validação Numérica de Derivadas Analíticas - Mancal Hidrodinâmico',
                 fontsize=16, fontweight='bold', y=0.985)
    
    # =============================================================================
    # GRÁFICO K
    # =============================================================================
    ax_K = fig.add_subplot(gs[0])
    ax_K.set_facecolor('#ffffff')
    
    # Dados iniciais
    H0_values = np.linspace(0.1, H0_max_init, n_points_init)
    K_a = K_analitico(H0_values)
    K_n = K_numerico(V_init, H0_values)
    
    line_K_a, = ax_K.plot(H0_values, K_a, lw=3.5, color='#1e3a8a', 
                         label='Analítico', zorder=3)
    line_K_n, = ax_K.plot(H0_values, K_n, 'o', color='#ef4444', markersize=5, 
                         markeredgewidth=1, markeredgecolor='white',
                         label='Numérico', zorder=2, alpha=0.8)
    
    ax_K.axhline(0, color='gray', linestyle='--', lw=0.8, alpha=0.4)
    ax_K.grid(True, linestyle=':', alpha=0.3)
    
    # Erro relativo
    mask_K = np.abs(K_a) > 1e-12
    erro_K = np.zeros_like(K_a)
    erro_K[mask_K] = np.abs((K_a[mask_K] - K_n[mask_K]) / K_a[mask_K])
    max_erro_K = np.max(erro_K) * 100
    
    titulo_K = ax_K.set_title(f'Rigidez: K = ∂Wz/∂H₀  (V={V_init:.1f})  |  Erro: {max_erro_K:.2e}%',
                              fontsize=13, fontweight='bold', pad=10)
    ax_K.set_ylabel('K', fontsize=12, fontweight='bold')
    
    # Legenda FORA do gráfico (à direita)
    ax_K.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                frameon=True, fontsize=11, framealpha=0.95, edgecolor='gray')
    
    # Limites
    K_a_finite = K_a[np.isfinite(K_a)]
    if len(K_a_finite) > 0:
        y_min_K = np.min(K_a_finite) - 2
        y_max_K = np.max(K_a_finite) + 2
        ax_K.set_ylim(y_min_K, y_max_K)
    
    # =============================================================================
    # GRÁFICO C
    # =============================================================================
    ax_C = fig.add_subplot(gs[1])
    ax_C.set_facecolor('#ffffff')
    
    C_a = C_analitico(H0_values)
    C_n = C_numerico(H0_values)
    
    # LINHA MAIS GROSSA PARA C ANALÍTICO
    line_C_a, = ax_C.plot(H0_values, C_a, lw=4, color='#15803d', 
                         label='Analítico', zorder=3, alpha=0.9)
    # PONTOS POR CIMA DA LINHA
    line_C_n, = ax_C.plot(H0_values, C_n, 's', color='#a855f7', markersize=6, 
                         markeredgewidth=1.5, markeredgecolor='white',
                         label='Numérico', zorder=4, alpha=0.85)
    
    ax_C.axhline(0, color='gray', linestyle='--', lw=0.8, alpha=0.4)
    ax_C.grid(True, linestyle=':', alpha=0.3)
    
    # Erro relativo
    mask_C = np.abs(C_a) > 1e-12
    erro_C = np.zeros_like(C_a)
    erro_C[mask_C] = np.abs((C_a[mask_C] - C_n[mask_C]) / C_a[mask_C])
    max_erro_C = np.max(erro_C) * 100
    
    titulo_C = ax_C.set_title(f'Amortecimento: C = -∂Wz/∂V  |  Erro: {max_erro_C:.2e}%',
                              fontsize=13, fontweight='bold', pad=10)
    ax_C.set_ylabel('C', fontsize=12, fontweight='bold')
    ax_C.set_xlabel('H₀', fontsize=12, fontweight='bold')
    
    # Legenda FORA do gráfico (à direita)
    ax_C.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                frameon=True, fontsize=11, framealpha=0.95, edgecolor='gray')
    
    # Limites
    C_a_finite = C_a[np.isfinite(C_a)]
    if len(C_a_finite) > 0:
        ax_C.set_ylim(-0.3, max(3.5, np.max(C_a_finite) * 1.1))
    
    # =============================================================================
    # SLIDERS
    # =============================================================================
    ax_slider_V = plt.axes([0.12, 0.08, 0.22, 0.02], facecolor='#e5e7eb')
    ax_slider_H0 = plt.axes([0.12, 0.04, 0.22, 0.02], facecolor='#e5e7eb')
    ax_slider_pts = plt.axes([0.50, 0.06, 0.30, 0.02], facecolor='#e5e7eb')
    
    slider_V = Slider(ax_slider_V, 'V', 0.5, 5.5, valinit=V_init, 
                      valstep=0.5, color='#f59e0b')
    slider_H0 = Slider(ax_slider_H0, 'H₀ máx', 1.0, 5.0, valinit=H0_max_init, 
                       valstep=0.5, color='#3b82f6')
    slider_pts = Slider(ax_slider_pts, 'Pontos', 50, 500, valinit=n_points_init, 
                        valstep=50, color='#10b981')
    
    # Elementos para atualização
    elementos = {
        'line_K_a': line_K_a, 'line_K_n': line_K_n,
        'line_C_a': line_C_a, 'line_C_n': line_C_n,
        'ax_K': ax_K, 'ax_C': ax_C,
        'titulo_K': titulo_K, 'titulo_C': titulo_C
    }
    
    # Função de atualização
    def atualizar(val):
        H0_max = slider_H0.val
        n_points = int(slider_pts.val)
        V_param = slider_V.val
        
        # Novos dados
        H0_values = np.linspace(0.1, H0_max, n_points)
        
        K_a = K_analitico(H0_values)
        K_n = K_numerico(V_param, H0_values)
        C_a = C_analitico(H0_values)
        C_n = C_numerico(H0_values)
        
        # Erros
        mask_K = np.abs(K_a) > 1e-12
        erro_K = np.zeros_like(K_a)
        erro_K[mask_K] = np.abs((K_a[mask_K] - K_n[mask_K]) / K_a[mask_K])
        max_erro_K = np.max(erro_K) * 100
        
        mask_C = np.abs(C_a) > 1e-12
        erro_C = np.zeros_like(C_a)
        erro_C[mask_C] = np.abs((C_a[mask_C] - C_n[mask_C]) / C_a[mask_C])
        max_erro_C = np.max(erro_C) * 100
        
        # Atualizar linhas
        elementos['line_K_a'].set_data(H0_values, K_a)
        elementos['line_K_n'].set_data(H0_values, K_n)
        elementos['line_C_a'].set_data(H0_values, C_a)
        elementos['line_C_n'].set_data(H0_values, C_n)
        
        # Atualizar limites X
        elementos['ax_K'].set_xlim(0.1, H0_max)
        elementos['ax_C'].set_xlim(0.1, H0_max)
        
        # Atualizar limites Y
        K_a_finite = K_a[np.isfinite(K_a)]
        if len(K_a_finite) > 0:
            y_min_K = np.min(K_a_finite) - 2
            y_max_K = np.max(K_a_finite) + 2
            elementos['ax_K'].set_ylim(y_min_K, y_max_K)
        
        C_a_finite = C_a[np.isfinite(C_a)]
        if len(C_a_finite) > 0:
            elementos['ax_C'].set_ylim(-0.3, max(3.5, np.max(C_a_finite) * 1.1))
        
        # Atualizar títulos
        elementos['titulo_K'].set_text(
            f'Rigidez: K = ∂Wz/∂H₀  (V={V_param:.1f})  |  Erro: {max_erro_K:.2e}%'
        )
        elementos['titulo_C'].set_text(
            f'Amortecimento: C = -∂Wz/∂V  |  Erro: {max_erro_C:.2e}%'
        )
        
        fig.canvas.draw_idle()
    
    slider_V.on_changed(atualizar)
    slider_H0.on_changed(atualizar)
    slider_pts.on_changed(atualizar)
    
    plt.show()

# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("VALIDAÇÃO NUMÉRICA - MANCAL HIDRODINÂMICO")
    print("="*70)
    print("\n✓ Interface interativa com 3 sliders:")
    print("  • V (viscosidade) - afeta apenas K")
    print("  • H₀ máximo - faixa de análise")
    print("  • Pontos - resolução do gráfico")
    print("\n✓ Método: Diferenças finitas centrais (h = 10⁻⁸)")
    print("="*70 + "\n")
    
    try:
        criar_interface()
    except KeyboardInterrupt:
        print("\n✗ Execução cancelada.\n")
    except Exception as e:
        print(f"\n✗ Erro: {e}\n")