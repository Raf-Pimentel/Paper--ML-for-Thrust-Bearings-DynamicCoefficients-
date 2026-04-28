# OBS: OS GRÁFICOS ESTÃO ERRADOS.

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# Função original
def Wz(V, H0):
    return (6 - V) * (np.log((H0 + 1) / H0) - 2 / (2 * H0 + 1))

# Derivada analítica
def K_analitico(V, H0):
    return (24 / (1 + 2 * H0) ** 2) - (6 / (H0 * (H0 + 1)))

# Derivada numérica
def K_numerico(V, H0_array, h=1e-8):
    return np.array([(Wz(V, H0 + h) - Wz(V, H0 - h)) / (2 * h) for H0 in H0_array])

# Função de plotagem
def plot_derivatives(V, H0_max):
    H0_values = np.linspace(0.1, H0_max, 200)
    
    K_a = K_analitico(V, H0_values)
    K_n = K_numerico(V, H0_values)
    
    # Erro relativo máximo
    mask = np.abs(K_a) > 1e-12
    errors = np.zeros_like(K_a)
    errors[mask] = np.abs((K_a[mask] - K_n[mask]) / K_a[mask])
    max_error = np.max(errors)
    
    # Plot
    plt.figure(figsize=(9,6))
    plt.plot(H0_values, K_a, label="Analítico", color="blue", linewidth=2)
    plt.plot(H0_values, K_n, label="Numérico (Dif. Finitas)", color="red", linestyle="--", linewidth=2)
    plt.xlabel("H₀")
    plt.ylabel("K = ∂Wz/∂H₀")
    plt.title(f"Validação de Derivada Analítica\nErro Relativo Máximo = {max_error*100:.2e}%")
    plt.legend()
    plt.grid(True)
    plt.show()

# Criar sliders
V_slider = widgets.FloatSlider(value=3.0, min=0.5, max=5.5, step=0.1, description="V")
H0_slider = widgets.FloatSlider(value=2.0, min=1.0, max=5.0, step=0.5, description="H₀ máximo")

ui = widgets.VBox([V_slider, H0_slider])
out = widgets.interactive_output(plot_derivatives, {"V": V_slider, "H0_max": H0_slider})

display(ui, out)
