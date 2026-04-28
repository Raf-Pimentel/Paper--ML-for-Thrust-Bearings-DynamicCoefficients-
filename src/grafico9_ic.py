import numpy as np
import matplotlib.pyplot as plt

# --- Funções de Cálculo (Baseadas na Literatura Padrão de Tribologia) ---

def calcular_desempenho_correto(Ho):
    """
    Calcula os parâmetros de desempenho usando as fórmulas padrão e consistentes
    para um mancal de deslizamento de inclinação fixa.
    """
    # É mais fácil calcular usando K, onde K é o inverso de Ho.
    K = 1 / Ho
    
    # Termos comuns para evitar repetição
    log_K = np.log(1 + K)
    frac_K = 2 * K / (2 + K)
    
    # Carga Normal Adimensional (Wz) - Fórmula Padrão
    Wz = (6 / K**2) * (log_K - frac_K)
    
    # Força de Atrito na Superfície Móvel (Fa) - Fórmula Padrão
    Fa = (1 / K) * (4 * log_K - 3 * frac_K)
    
    # Parâmetro do Coeficiente de Atrito (μl/sₕ) = Fa / Wz
    parametro_atrito = np.divide(Fa, Wz, out=np.full_like(Fa, np.inf), where=Wz!=0)
    
    return parametro_atrito

# --- Configuração dos Dados para o Gráfico ---

# Criar um array de pontos para o eixo X (H₀).
Ho_coords = np.linspace(0.1, 2.0, 500)

# Calcular os valores de Y com a função final e correta
parametro_atrito_valores = calcular_desempenho_correto(Ho_coords)

# --- Criação do Gráfico (Fiel à Figura 8.10) ---

fig, ax = plt.subplots(figsize=(10, 7))

# Plotar a curva
ax.plot(Ho_coords, parametro_atrito_valores, lw=2.5, color='black')

# --- Customização e Legendas ---

ax.set_xlabel(r'Razão de Espessura do Filme, $H_0 = h_0/s_h$', fontsize=12)
ax.set_ylabel(r'Parâmetro do Coeficiente de Atrito, $\mu\ell/s_h$', fontsize=12)
ax.set_title('Figura 8.10: Efeito da Razão de Espessura no Coeficiente de Atrito (Versão Final Corrigida)', fontsize=14)

# Configurar os limites dos eixos para ficarem idênticos ao original
ax.set_xlim(0, 2.0)
ax.set_ylim(0, 20)

ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()