# grafico21.py
# Calculei numericamente e analiticamente Wz(V,H0) e comparei
# De modo que Wz = ∫[H0 to H0+1] P(V,H,H0) dH
# Onde P(V,H,H0) = (V-6) * [1/(2H0+1) - 1/H + H0(H0+1)/(H²(2H0+1))]
# Fiz o calculo numérico via regra do trapézio, pois a soma de Riemann é imprecisa
# A regra do trapezio consiste em calcular a integral como:
# ∫[a to b] f(x) dx ≈ (b-a) * [f(a) + f(b)]/2 


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec

# --- Função P(V,H,H0) ---
def P(V, H, H0):
    """
    Função P(V,H,H0) = (V-6) * [1/(2H0+1) - 1/H + H0(H0+1)/(H²(2H0+1))]
    """
    term1 = 1 / (2*H0 + 1)
    term2 = -1 / H
    term3 = H0 * (H0 + 1) / (H**2 * (2*H0 + 1))
    return (V - 6) * (term1 + term2 + term3)

# --- Integral numérica via regra do trapézio ---
def Wz_numerica(V, H0, n):
    """
    Calcula Wz(V,H0) = ∫[H0 to H0+1] P(V,H,H0) dH
    CORREÇÃO APLICADA: SEM sinal negativo extra!
    """
    H = np.linspace(H0, H0 + 1, n + 1)
    P_vals = P(V, H, H0)
    # MUDANÇA CRÍTICA: Removido o "-" que estava invertendo o sinal
    integral = np.trapz(P_vals, H)
    return integral  # Retorna ∫P dH diretamente

# --- Integral analítica ---
def Wz_analitica(V, H0):
    """
    Solução analítica: Wz(V,H0) = (6-V) * [ln((H0+1)/H0) - 2/(2H0+1)]
    """
    termo_log = np.log((H0 + 1) / H0)
    termo_frac = 2 / (2*H0 + 1)
    return (6 - V) * (termo_log - termo_frac)

# --- Função para gerar superfície numérica ---
def compute_surface_numerica(V_vals, H0_vals, n):
    W = np.zeros((len(H0_vals), len(V_vals)))
    for i, H0 in enumerate(H0_vals):
        for j, V in enumerate(V_vals):
            W[i, j] = Wz_numerica(V, H0, n)
    return W

# --- Função para gerar superfície analítica ---
def compute_surface_analitica(V_vals, H0_vals):
    W = np.zeros((len(H0_vals), len(V_vals)))
    for i, H0 in enumerate(H0_vals):
        for j, V in enumerate(V_vals):
            W[i, j] = Wz_analitica(V, H0)
    return W

# --- Diagnóstico detalhado ---
print("\n" + "="*80)
print("DIAGNÓSTICO: VERIFICAÇÃO APÓS CORREÇÃO")
print("="*80)

V_test, H0_test = 10.0, 2.0
print(f"\nPonto de teste: V={V_test}, H0={H0_test}")

# Cálculo manual da analítica
termo_log = np.log((H0_test + 1) / H0_test)
termo_frac = 2 / (2*H0_test + 1)
wz_ana_manual = (6 - V_test) * (termo_log - termo_frac)

print(f"\nCálculo analítico:")
print(f"  (6-V) × [ln((H0+1)/H0) - 2/(2H0+1)]")
print(f"  = {6-V_test:.1f} × [{termo_log:.6f} - {termo_frac:.6f}]")
print(f"  = {6-V_test:.1f} × {termo_log - termo_frac:.6f}")
print(f"  = {wz_ana_manual:.6f}")

# Analisa P(V,H,H0)
H_sample = np.linspace(H0_test, H0_test + 1, 1000)
P_sample = P(V_test, H_sample, H0_test)
integral_P = np.trapz(P_sample, H_sample)

print(f"\nCálculo numérico:")
print(f"  ∫P dH = {integral_P:.6f}")

wz_ana_func = Wz_analitica(V_test, H0_test)
wz_num_func = Wz_numerica(V_test, H0_test, 1000)

print(f"\nComparação:")
print(f"  Wz_analitica  = {wz_ana_func:.6f}")
print(f"  Wz_numerica   = {wz_num_func:.6f}")
print(f"  |Diferença|   = {abs(wz_num_func - wz_ana_func):.2e}")

if abs(wz_num_func - wz_ana_func) < 1e-5:
    print(f"\n✅ SUCESSO! Numérico e analítico concordam perfeitamente!")
    print(f"   As superfícies agora têm a MESMA forma.")
else:
    print(f"\n⚠️  Ainda há diferença...")
    print(f"   Sinais: numérico={'+' if wz_num_func > 0 else '-'}, analítico={'+' if wz_ana_func > 0 else '-'}")

# --- Parâmetros iniciais ---
Vmax_init = 12
H0max_init = 5
n_init = 100

V_vals = np.linspace(6.1, Vmax_init, 50)
H0_vals = np.linspace(0.5, H0max_init, 50)

# --- Superfícies iniciais ---
W_num = compute_surface_numerica(V_vals, H0_vals, n_init)
W_ana = compute_surface_analitica(V_vals, H0_vals)
V_mesh, H0_mesh = np.meshgrid(V_vals, H0_vals)

# --- Criação da figura ---
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.8], hspace=0.35, wspace=0.35)

# Gráfico 1: Wz numérica
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
surf_num = ax1.plot_surface(V_mesh, H0_mesh, W_num, cmap='viridis', alpha=0.85, 
                             edgecolor='none', antialiased=True, vmin=-1, vmax=1)
ax1.set_title(f"Wz Numérica (n={n_init})", fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel("V", fontsize=10, labelpad=8)
ax1.set_ylabel("H₀", fontsize=10, labelpad=8)
ax1.set_zlabel("Wz", fontsize=10, labelpad=8)
ax1.view_init(elev=20, azim=135)
cbar1 = fig.colorbar(surf_num, ax=ax1, shrink=0.6, aspect=8, pad=0.1)
cbar1.set_label('Wz', fontsize=9)

# Gráfico 2: Wz analítica
ax2 = fig.add_subplot(gs[0, 1], projection='3d')
surf_ana = ax2.plot_surface(V_mesh, H0_mesh, W_ana, cmap='plasma', alpha=0.85,
                             edgecolor='none', antialiased=True, vmin=-1, vmax=1)
ax2.set_title("Wz Analítica (Exata)", fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel("V", fontsize=10, labelpad=8)
ax2.set_ylabel("H₀", fontsize=10, labelpad=8)
ax2.set_zlabel("Wz", fontsize=10, labelpad=8)
ax2.view_init(elev=20, azim=135)
cbar2 = fig.colorbar(surf_ana, ax=ax2, shrink=0.6, aspect=8, pad=0.1)
cbar2.set_label('Wz', fontsize=9)

# Gráfico 3: Erro absoluto
ax3 = fig.add_subplot(gs[1, 0], projection='3d')
W_erro = np.abs(W_num - W_ana)
surf_erro = ax3.plot_surface(V_mesh, H0_mesh, W_erro, cmap='Reds', alpha=0.9,
                              edgecolor='none', antialiased=True)
erro_max_init = np.max(W_erro)
ax3.set_title(f"Erro Absoluto (máx: {erro_max_init:.2e})", 
              fontsize=12, fontweight='bold', color='darkred', pad=10)
ax3.set_xlabel("V", fontsize=10, labelpad=8)
ax3.set_ylabel("H₀", fontsize=10, labelpad=8)
ax3.set_zlabel("Erro", fontsize=10, labelpad=8)
ax3.view_init(elev=20, azim=135)
cbar3 = fig.colorbar(surf_erro, ax=ax3, shrink=0.6, aspect=8, pad=0.1)
cbar3.set_label('|Num - Ana|', fontsize=9)

# Gráfico 4: P(V,H,H0)
ax4 = fig.add_subplot(gs[1, 1])
V_diag, H0_diag = 10.0, 2.0
H_diag = np.linspace(H0_diag, H0_diag + 1, 300)
P_diag = P(V_diag, H_diag, H0_diag)
integral_P_diag = np.trapz(P_diag, H_diag)

ax4.plot(H_diag, P_diag, 'b-', linewidth=2.5, label=f'P(V={V_diag}, H, H₀={H0_diag})')
ax4.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax4.fill_between(H_diag, 0, P_diag, alpha=0.25, color='blue')
ax4.set_xlabel('H', fontsize=11, fontweight='bold')
ax4.set_ylabel('P(V,H,H₀)', fontsize=11, fontweight='bold')
ax4.set_title(f'Integrando: ∫P dH = {integral_P_diag:.6f} = Wz numérico', 
              fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
ax4.legend(fontsize=10, loc='best')

# Informações textuais
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('off')
info_text = f"""
✅ CORREÇÃO APLICADA COM SUCESSO!
• Wz numérico = ∫P dH (sem multiplicar por -1)
• Wz analítico = (6-V) × [ln((H0+1)/H0) - 2/(2H0+1)]
• Ambas as superfícies agora convergem para a mesma forma!

Para V > 6: Wz < 0 (negativo) ✓
O erro diminui quando n aumenta, demonstrando convergência perfeita.
"""
ax5.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=11, 
         family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))

plt.subplots_adjust(bottom=0.12, top=0.96)

# --- Sliders ---
ax_n = plt.axes([0.25, 0.055, 0.5, 0.025], facecolor='lightgoldenrodyellow')
slider_n = Slider(ax_n, 'n', 10, 500, valinit=n_init, valstep=5)

ax_Vmax = plt.axes([0.15, 0.02, 0.3, 0.025], facecolor='lightgoldenrodyellow')
slider_Vmax = Slider(ax_Vmax, 'V máx', 6.1, 20, valinit=Vmax_init)

ax_Hmax = plt.axes([0.6, 0.02, 0.3, 0.025], facecolor='lightgoldenrodyellow')
slider_Hmax = Slider(ax_Hmax, 'H₀ máx', 0.5, 15, valinit=H0max_init)

# --- Função de atualização ---
def update(val):
    n = int(slider_n.val)
    Vmax = slider_Vmax.val
    Hmax = slider_Hmax.val
    
    V_vals_new = np.linspace(6.1, Vmax, 50)
    H0_vals_new = np.linspace(0.5, Hmax, 50)
    V_mesh_new, H0_mesh_new = np.meshgrid(V_vals_new, H0_vals_new)
    
    # Recalcula superfícies
    W_num_new = compute_surface_numerica(V_vals_new, H0_vals_new, n)
    W_ana_new = compute_surface_analitica(V_vals_new, H0_vals_new)
    
    # Atualiza numérica
    ax1.clear()
    surf_num = ax1.plot_surface(V_mesh_new, H0_mesh_new, W_num_new, cmap='viridis', 
                                 alpha=0.85, edgecolor='none', antialiased=True, vmin=-1, vmax=1)
    ax1.set_title(f"Wz Numérica (n={n})", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("V", fontsize=10, labelpad=8)
    ax1.set_ylabel("H₀", fontsize=10, labelpad=8)
    ax1.set_zlabel("Wz", fontsize=10, labelpad=8)
    ax1.view_init(elev=20, azim=135)
    
    # Atualiza analítica
    ax2.clear()
    surf_ana = ax2.plot_surface(V_mesh_new, H0_mesh_new, W_ana_new, cmap='plasma', 
                                 alpha=0.85, edgecolor='none', antialiased=True, vmin=-1, vmax=1)
    ax2.set_title("Wz Analítica (Exata)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("V", fontsize=10, labelpad=8)
    ax2.set_ylabel("H₀", fontsize=10, labelpad=8)
    ax2.set_zlabel("Wz", fontsize=10, labelpad=8)
    ax2.view_init(elev=20, azim=135)
    
    # Atualiza erro
    W_erro_new = np.abs(W_num_new - W_ana_new)
    ax3.clear()
    surf_erro = ax3.plot_surface(V_mesh_new, H0_mesh_new, W_erro_new, cmap='Reds', 
                                  alpha=0.9, edgecolor='none', antialiased=True)
    erro_max = np.max(W_erro_new)
    erro_medio = np.mean(W_erro_new)
    ax3.set_title(f"Erro Absoluto (máx: {erro_max:.2e}, médio: {erro_medio:.2e})", 
                  fontsize=12, fontweight='bold', color='darkred', pad=10)
    ax3.set_xlabel("V", fontsize=10, labelpad=8)
    ax3.set_ylabel("H₀", fontsize=10, labelpad=8)
    ax3.set_zlabel("Erro", fontsize=10, labelpad=8)
    ax3.view_init(elev=20, azim=135)
    
    # Atualiza P(V,H,H0)
    H_diag = np.linspace(H0_diag, H0_diag + 1, 300)
    P_diag = P(V_diag, H_diag, H0_diag)
    integral_P = np.trapz(P_diag, H_diag)
    
    ax4.clear()
    ax4.plot(H_diag, P_diag, 'b-', linewidth=2.5, label=f'P(V={V_diag}, H, H₀={H0_diag})')
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax4.fill_between(H_diag, 0, P_diag, alpha=0.25, color='blue')
    ax4.set_xlabel('H', fontsize=11, fontweight='bold')
    ax4.set_ylabel('P(V,H,H₀)', fontsize=11, fontweight='bold')
    ax4.set_title(f'Integrando: ∫P dH = {integral_P:.6f} = Wz numérico', 
                  fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax4.legend(fontsize=10, loc='best')
    
    print(f"n={n:3d} | Erro máx: {erro_max:.4e} | Erro médio: {erro_medio:.4e}")
    
    fig.canvas.draw_idle()

slider_n.on_changed(update)
slider_Vmax.on_changed(update)
slider_Hmax.on_changed(update)

print("="*80)
print("✅ Código corrigido executando! As superfícies devem ser idênticas.")
print("="*80 + "\n")

plt.show()
