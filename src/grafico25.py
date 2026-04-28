# Esse código gera um gráfico 3D interativo da função W(V, H0) e seu campo de gradiente usando Plotly.
# Um campo gradiente é utilizado para visualizar a direção e magnitude das mudanças na função W em relação a V e H0.
# Ele é representado por cores que indicam a direção do maior aumento de W.

import numpy as np
import plotly.graph_objects as go

# === Definições matemáticas ===
def W(V, H0):
    return (6 - V) * (np.log((H0 + 1) / H0) - 2 / (2 * H0 + 1))

def dW_dV(V, H0):
    return - (np.log((H0 + 1) / H0) - 2 / (2 * H0 + 1))

def dW_dH0(V, H0):
    return (6 - V) * (4 / (2 * H0 + 1)**2 - 1 / (H0 * (H0 + 1)))

# === Geração da malha ===
V_vals = np.linspace(0, 6, 30)
H0_vals = np.linspace(0.5, 3, 30)
V, H0 = np.meshgrid(V_vals, H0_vals)
W_vals = W(V, H0)

# === Campo de gradiente ===
dV = dW_dV(V, H0)
dH0 = dW_dH0(V, H0)
magnitude = np.sqrt(dV**2 + dH0**2)

# Subamostragem para visualização mais limpa dos vetores
step = 3
Vq = V[::step, ::step]
Hq = H0[::step, ::step]
Wq = W(Vq, Hq)
dVq = dV[::step, ::step]
dH0q = dH0[::step, ::step]

# Normaliza vetores
scale = 0.3
dVq_n = scale * dVq / np.max(np.abs(dVq))
dH0q_n = scale * dH0q / np.max(np.abs(dH0q))

# === Superfície ===
surface = go.Surface(
    x=V, y=H0, z=W_vals,
    colorscale='Viridis', opacity=0.9, name='Superfície W(V,H₀)'
)

# === Vetores do gradiente ===
cones = go.Cone(
    x=Vq.flatten(),
    y=Hq.flatten(),
    z=Wq.flatten(),
    u=dVq_n.flatten(),
    v=dH0q_n.flatten(),
    w=np.zeros_like(dVq_n.flatten()),  # gradiente tangente à superfície
    colorscale='inferno',
    sizemode='absolute',
    sizeref=0.2,
    showscale=False,
    name='Gradiente ∇W'
)

# === Layout ===
fig = go.Figure(data=[surface, cones])
fig.update_layout(
    scene=dict(
        xaxis_title='V',
        yaxis_title='H₀',
        zaxis_title='W(V, H₀)',
        camera=dict(eye=dict(x=1.6, y=1.6, z=1)),
    ),
    title='Superfície e Campo de Gradiente de W(V, H₀)',
    template='plotly_dark',
)
fig.show()
