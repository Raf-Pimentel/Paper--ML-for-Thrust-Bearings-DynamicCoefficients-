import numpy as np

# --- Função original ---
def W(V, H0):
    return (6 - V) * (np.log((H0 + 1) / H0) - 2 / (2 * H0 + 1))

# --- Derivada numérica (diferença centrada) ---
def dW_dV_num(V, H0, h=1e-5):
    return (W(V + h, H0) - W(V - h, H0)) / (2 * h)

# --- Derivada analítica ---
def dW_dV_analitico(V, H0):
    return - (np.log((H0 + 1) / H0) - 2 / (2 * H0 + 1))

# --- Teste simples de valores ---
V = 2.0
H0 = 1.5
h = 1e-7

num = dW_dV_num(V, H0, h)
ana = dW_dV_analitico(V, H0)
erro_absoluto = abs(num - ana)
erro_relativo = abs(erro_absoluto / ana)

print(f"Derivada numérica:   {num:.8f}")
print(f"Derivada analítica:  {ana:.8f}")
print(f"Erro absoluto:       {erro_absoluto:.2e}")
print(f"Erro relativo:       {erro_relativo:.2e}")
