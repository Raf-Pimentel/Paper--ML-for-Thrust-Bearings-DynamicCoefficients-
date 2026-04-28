# Nesse código iremos comparar o resultado analitico com o numérico
# Iremos utilizar o metodo de diferenças finitas para calcular os coeficientes C e K
# Assim, podemos validar os nossos calculos analiticos com os resultados numéricos
# Ainda não consegui plotar o codigo do K corretamente, mas o C está ok
# O problema é que o Knumerico se mantem em y=0, o que não faz sentido
# Tentando consertar o gráfico do K no grafico18_ic.py

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# MÓDULO DE CÁLCULO
# =============================================================================

# --- Funções Analíticas (Nosso "Gabarito") ---

def calcular_K_analitico(Ho):
    """Calcula o Coeficiente de Rigidez Adimensional (K_analitico)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        termo1 = 6 * ((1 / (Ho + 1)) - (1 / Ho))
        termo2 = 24 / ((1 + 2 * Ho)**2)
        resultado = termo1 + termo2
    return np.where(np.asanyarray(Ho) <= 1e-9, -np.inf, resultado)

def calcular_C_analitico(Ho):
    """Calcula o Coeficiente de Amortecimento (C_analitico) - Versão corrigida."""
    with np.errstate(divide='ignore', invalid='ignore'):
        termo_log = np.log((Ho + 1) / Ho)
        termo_fracao = 2 / (2 * Ho + 1)
        resultado = termo_log - termo_fracao
    return np.where(np.asanyarray(Ho) <= 1e-9, np.inf, resultado)

# --- Funções para o Método Numérico ---

def calcular_Wz_estatico(Ho):
    """Calcula a carga normal estática Wz(Ho), base da rigidez. Função vetorizada."""
    Ho_array = np.asanyarray(Ho)
    with np.errstate(divide='ignore', invalid='ignore'):
        K_geom = 1 / Ho_array
        log_K = np.log(1 + K_geom)
        frac_K = 2 * K_geom / (2 + K_geom)
        resultado = (6 / K_geom**2) * (log_K - frac_K)
    
    resultado_final = np.where(Ho_array <= 1e-9, 0.0, np.nan_to_num(resultado))
    return resultado_final.item() if np.isscalar(Ho) else resultado_final

def calcular_Wz_dinamico(Ho, V):
    """Calcula a carga normal dinâmica Wz(Ho, V), base do amortecimento."""
    Ho_array = np.asanyarray(Ho)
    termo_constante = np.log((Ho_array + 1) / Ho_array) - (2 / (2 * Ho_array + 1))
    return (6 - V) * termo_constante

# --- [FUNÇÃO CORRIGIDA COM A SUGESTÃO DO COPILOT] ---
def calcular_K_numerico(Ho_array):
    """Derivada numérica robusta: K = - dWz/dHo."""
    Ho_array = np.asanyarray(Ho_array, dtype=float)
    Wz_valores = np.asanyarray(calcular_Wz_estatico(Ho_array), dtype=float)

    # Garante shapes compatíveis (broadcast seguro)
    if Wz_valores.shape != Ho_array.shape:
        Wz_valores = np.broadcast_to(Wz_valores, Ho_array.shape)

    # derivada numérica com passo variável; negativo para obter K
    dWz_dHo = np.gradient(Wz_valores, Ho_array)
    return -dWz_dHo

def calcular_C_numerico(Ho_array, delta_V=1e-6):
    """Estima C_analitico usando diferenças finitas centradas."""
    Wz_mais = calcular_Wz_dinamico(Ho_array, delta_V)
    Wz_menos = calcular_Wz_dinamico(Ho_array, -delta_V)
    # Inverte o sinal para retornar o coeficiente de amortecimento positivo.
    return -((Wz_mais - Wz_menos) / (2 * delta_V))

# --- Configuração dos Dados para o Gráfico ---
Ho_coords = np.linspace(0.1, 3.0, 100)

K_analitico_valores = calcular_K_analitico(Ho_coords)
C_analitico_valores = calcular_C_analitico(Ho_coords)
K_numerico_valores = calcular_K_numerico(Ho_coords) # Chama a nova função
C_numerico_valores = calcular_C_numerico(Ho_coords)

# --- Criação do Gráfico de Comparação ---
fig, (ax_K, ax_C) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig.suptitle('Validação Numérica com Derivada Robusta (np.gradient)', fontsize=18)

# --- PLOT SUPERIOR: COMPARAÇÃO DA RIGIDEZ (K) ---
ax_K.plot(Ho_coords, K_analitico_valores, lw=4, color='darkred', label='Solução Analítica', zorder=2)
ax_K.plot(Ho_coords, K_numerico_valores, 'o', color='cyan', markersize=6, label='Estimativa Numérica (np.gradient)', zorder=3)
ax_K.set_title('Coeficiente de Rigidez ($K_{analitico}$)', fontsize=14)
ax_K.set_ylabel(r'Rigidez Adimensional, $K_{analitico}$', fontsize=12)
ax_K.axhline(0, color='black', linestyle='--', lw=1)
ax_K.set_ylim(-40, 5); ax_K.grid(True, linestyle=':'); ax_K.legend()

# --- PLOT INFERIOR: COMPARAÇÃO DO AMORTECIMENTO (C) ---
ax_C.plot(Ho_coords, C_analitico_valores, lw=4, color='darkblue', label='Solução Analítica (Corrigida)', zorder=2)
ax_C.plot(Ho_coords, C_numerico_valores, 'o', color='magenta', markersize=6, label='Estimativa Numérica (Dif. Finitas)', zorder=3)
ax_C.set_title('Coeficiente de Amortecimento ($C_{analitico}$)', fontsize=14)
ax_C.set_ylabel(r'Amortecimento Adimensional, $C_{analitico}$', fontsize=12)
ax_C.set_xlabel(r'Razão de Espessura do Filme, $H_0 = h_0/s_h$', fontsize=12)
ax_C.set_ylim(-0.5, 3); ax_C.grid(True, linestyle=':'); ax_C.legend()

# --- Finalização ---
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()