# OBS: OS GRÁFICOS ESTÃO ERRADOS.

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import IFrame, display
import matplotlib.pyplot as plt

# Configura o renderer padrão do Plotly para notebook
pio.renderers.default = "notebook"

# Função original Wz(V, H0)
def Wz(V, H0):
    return (6 - V) * (np.log((H0 + 1) / H0) - 2 / (2 * H0 + 1))

# Derivada analítica
def K_analitico(V, H0):
    return (24 / (1 + 2 * H0) ** 2) - (6 / (H0 * (H0 + 1)))

# Derivada numérica (diferenças finitas centrais)
def K_numerico(V, H0_array, h=1e-8):
    return np.array([(Wz(V, H0 + h) - Wz(V, H0 - h)) / (2 * h) for H0 in H0_array])

# Parâmetros
V = 3.0
H0_min, H0_max = 0.1, 2.0
num_points = 200
H0_values = np.linspace(H0_min, H0_max, num_points)

K_a = K_analitico(V, H0_values)
K_n = K_numerico(V, H0_values)

# Erro relativo máximo
mask = np.abs(K_a) > 1e-12
errors = np.zeros_like(K_a)
errors[mask] = np.abs((K_a[mask] - K_n[mask]) / K_a[mask])
max_error = np.max(errors)
print(f"Erro relativo máximo = {max_error*100:.2e}%")

# Gráfico interativo com Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=H0_values, y=K_a, mode="lines",
                         name="Analítico", line=dict(color="blue", width=3)))
fig.add_trace(go.Scatter(x=H0_values, y=K_n, mode="lines",
                         name="Numérico (Dif. Finitas)", line=dict(color="red", width=2, dash="dash")))

fig.update_layout(
    title=f"Validação de Derivada Analítica - Erro Relativo Máximo = {max_error*100:.2e}%",
    xaxis_title="H₀",
    yaxis_title="K = ∂Wz/∂H₀",
    template="plotly_white",
    width=900, height=500
)

try:
    fig.show()
except Exception as e:
    print("Exibição inline falhou:", e)
    fname = "grafico_plotly.html"
    fig.write_html(fname)
    display(IFrame(fname, width=900, height=500))

# Gráfico estático com Matplotlib (opcional)
plt.figure(figsize=(9,5))
plt.plot(H0_values, K_a, label="Analítico", linewidth=2)
plt.plot(H0_values, K_n, label="Numérico (Dif. Finitas)", linestyle="--", linewidth=1.5)
plt.xlabel("H₀")
plt.ylabel("K = ∂Wz/∂H₀")
plt.title(f"Validação - Erro Rel. Máximo = {max_error*100:.2e}%")
plt.grid(True)
plt.legend()
plt.show()
