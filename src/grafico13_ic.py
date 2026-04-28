# Gráfico do centro de pressão em função de Ho
# Gráfico baseado na Equação 8.40 do livro

import numpy as np
import matplotlib.pyplot as plt

# --- Funções de Cálculo Necessárias ---

def calcular_carga_normal_adimensional(Ho):
    """
    Calcula a Carga Normal Adimensional (Wz) usando a fórmula padrão consistente.
    """
    # Usamos K = 1/Ho para estabilidade numérica e simplicidade
    K = 1 / Ho
    log_K = np.log(1 + K)
    frac_K = 2 * K / (2 + K)
    Wz = (6 / K**2) * (log_K - frac_K)
    return Wz

def calcular_centro_pressao_adimensional(Ho):
    """
    Calcula o Centro de Pressão Adimensional (Xcp).
    Baseado na Equação 8.40 do livro.
    """
    # Calcula Wz primeiro, pois é necessário na fórmula de Xcp
    Wz = calcular_carga_normal_adimensional(Ho)
    
    # Termo do logaritmo na Eq. 8.40. Note que ln(Ho / (Ho + 1)) é negativo.
    termo_log = np.log(Ho / (Ho + 1))
    
    # Montando a fórmula de Xcp
    fator_principal = -6 / (Wz * (1 + 2 * Ho))
    termo_colchetes = (Ho + 1) * (3 * Ho + 1) * termo_log + 3 * Ho + 2.5
    
    Xcp = fator_principal * termo_colchetes
    return Xcp

# --- Configuração dos Dados para o Gráfico ---

# Criar um array de pontos para o eixo X (H₀).
# Começamos de um valor pequeno para evitar erros e vamos até um valor maior para ver a tendência.
Ho_coords = np.linspace(0.01, 5.0, 500)

# Calcular os valores de Y (Xcp) correspondentes
Xcp_valores = calcular_centro_pressao_adimensional(Ho_coords)

# --- Criação do Gráfico ---

fig, ax = plt.subplots(figsize=(10, 7))

# Plotar a curva de Xcp
ax.plot(Ho_coords, Xcp_valores, lw=2.5, color='black')

# --- Customização e Legendas ---
ax.set_xlabel(r'Razão de Espessura do Filme, $H_0 = h_0/s_h$', fontsize=12)
ax.set_ylabel(r'Centro de Pressão Adimensional, $X_{cp} = x_{cp}/\ell$', fontsize=12)
ax.set_title('Efeito da Razão de Espessura no Centro de Pressão (Fig. 8.13)', fontsize=14)

# Adicionar uma linha de referência no centro geométrico (0.5)
ax.axhline(0.5, color='red', linestyle='--', linewidth=1.0, label='Centro Geométrico (X=0.5)')

# Configurar os limites dos eixos para uma boa visualização
ax.set_xlim(0, 5.0)
ax.set_ylim(0.5, 1.0)

ax.legend()
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Mostrar o gráfico final
plt.show()