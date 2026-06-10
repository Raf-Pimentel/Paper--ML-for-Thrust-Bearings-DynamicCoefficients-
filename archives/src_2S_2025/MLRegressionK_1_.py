import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- 1. Definição das Funções Analíticas ---
# Baseado na Equação (5.2) do seu relatório para a Rigidez (K) [cite: 206]
def coeficiente_rigidez_K(H0):
    return (24 / (1 + 2*H0)**2) - (6 / (H0 * (H0 + 1)))

# Baseado na Equação (5.4) do seu relatório para o Amortecimento (C) [cite: 217]
def coeficiente_amortecimento_C(H0):
    return (2 / (1 + 2*H0)) - np.log((H0 + 1) / H0)

# --- 2. Geração do Dataset Sintético ---
# Cria 10.000 pontos para H0 no intervalo relevante (onde a não-linearidade é forte)
N_PONTOS = 10000
H0_min = 0.01  # Evita H0=0 (assintota)
H0_max = 3.0
H0_data = np.linspace(H0_min, H0_max, N_PONTOS)

# Calcula os valores de K e C para cada H0
K_data = coeficiente_rigidez_K(H0_data)
C_data = coeficiente_amortecimento_C(H0_data)

# Cria um DataFrame Pandas
data = pd.DataFrame({
    'H0': H0_data,          # Feature (Variável de Entrada)
    'K_analitico': K_data,  # Target 1 (Saída a prever)
    'C_analitico': C_data   # Target 2 (Saída a prever)
})

print("✅ Dados brutos gerados com sucesso.")
print(f"Dimensão do Dataset: {data.shape}")
print("\nPrimeiras 10000 linhas do Dataset:")
print(data.head(10000))


# --- 3. Preparação para o ML e Divisão do Dataset ---

# 3.1. Definição de Features (X) e Targets (y)
# X deve ser uma matriz 2D (N, 1). .values.reshape(-1, 1) garante isso.
X = data['H0'].values.reshape(-1, 1)

# Usaremos o modelo para prever K e C separadamente.
# Para este exemplo, vamos focar na previsão da Rigidez (K)
y_K = data['K_analitico'].values

# 3.2. Divisão em Treinamento e Teste (80%/20%)
X_train, X_test, y_train_K, y_test_K = train_test_split(
    X, y_K,
    test_size=0.2,   # 20% para teste
    random_state=42, # Garante reprodutibilidade
    shuffle=True     # Embaralha antes de dividir
)

print("\n✅ Dataset dividido em Treinamento e Teste.")
print(f"Treinamento (X_train): {X_train.shape[0]} amostras")
print(f"Teste (X_test): {X_test.shape[0]} amostras")