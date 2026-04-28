import numpy as np
import plotly.graph_objects as go

# Função original Wz(V, H0)
def Wz(V, H0):
    return (6 - V) * (np.log((H0 + 1) / H0) - 2 / (2 * H0 + 1))

# Derivada analítica
def K_analitico(V, H0):
    return (24 / (1 + 2 * H0) ** 2) - (6 / (H0 * (H0 + 1)))

# Derivada numérica por diferenças finitas centrais
def K_numerico(V, H0, h=1e-8):
    return (Wz(V, H0 + h) - Wz(V, H0 - h)) / (2 * h)

# Parâmetros
V = 3.0
H0_min, H0_max = 0.1, 2.0
num_points = 200
H0_values = np.linspace(H0_min, H0_max, num_points)

# Calcular valores
K_a = [K_analitico(V, H0) for H0 in H0_values]
K_n = [K_numerico(V, H0) for H0 in H0_values]

# Calcular erro relativo máximo
errors = [
    abs((ka - kn) / ka) if abs(ka) > 1e-10 else 0
    for ka, kn in zip(K_a, K_n)
]
max_error = max(errors)

# Plot interativo com Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=H0_values, y=K_a,
    mode="lines",
    name="Analítico",
    line=dict(color="blue", width=3)
))

fig.add_trace(go.Scatter(
    x=H0_values, y=K_n,
    mode="lines",
    name="Numérico (Dif. Finitas)",
    line=dict(color="red", width=2, dash="dash")
))

fig.update_layout(
    title=f"Validação de Derivada Analítica - Erro Relativo Máximo = {max_error*100:.2e}%",
    xaxis_title="H₀",
    yaxis_title="K = ∂Wz/∂H₀",
    template="plotly_white"
)

fig.show()
