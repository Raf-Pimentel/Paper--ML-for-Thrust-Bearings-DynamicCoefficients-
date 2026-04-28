import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from MLRegressionK_1_ import X_train, X_test, H0_max, data, pd, X, C_data, H0_min


# Definindo o Target para o Amortecimento (C)
y_C = data['C_analitico'].values

# Reutilizando as Features de Treinamento (X_train, X_test) do Passo 1
# Divisão para C:
X_train, X_test, y_train_C, y_test_C = train_test_split(
    X, y_C,
    test_size=0.2,   # 20% para teste
    random_state=42, # Garante reprodutibilidade
    shuffle=True
)

# --- 2. Instanciação e Treinamento do Modelo para C ---
# Utilizando o mesmo tipo de modelo (Random Forest)
rf_model_C = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

print("\nIniciando o treinamento do Random Forest Regressor para C...")
rf_model_C.fit(X_train, y_train_C)
print("✅ Treinamento concluído para C.")

# --- 3. Predição nos Conjuntos de Teste e Treinamento ---
y_pred_C_train = rf_model_C.predict(X_train)
y_pred_C_test = rf_model_C.predict(X_test)

# --- 4. Avaliação de Métricas ---

# Métricas para o conjunto de TESTE (Generalização)
mse_test_C = mean_squared_error(y_test_C, y_pred_C_test)
r2_test_C = r2_score(y_test_C, y_pred_C_test)

print("\n--- Resultados da Avaliação do Modelo C (Amortecimento) ---")
print("Métricas de TESTE:")
print(f"  Root Mean Squared Error (RMSE): {np.sqrt(mse_test_C):.6f}")
print(f"  R² Score (Generalização): {r2_test_C:.6f}")

# --- 5. Visualização para Validação ---

# Combina X_test, os valores analíticos (y_test_C) e as previsões (y_pred_C_test)
# e ordena pelo H0 para garantir que o plot de dispersão seja sequencial.
plot_data_C = pd.DataFrame({
    'H0': X_test.flatten(),
    'C_Analitico': y_test_C,
    'C_Predito': y_pred_C_test
}).sort_values('H0')

# Define a faixa de H0 para o zoom
H0_zoom_max = 0.1

plt.figure(figsize=(12, 6))

# Plot da Solução Analítica Exata (Linha Contínua Fina)
plt.plot(data['H0'], data['C_analitico'], label='Solução Analítica (Teórica)', 
         color='green', linestyle='-', linewidth=2, alpha=0.7)

# Plot das Previsões do ML no conjunto de Teste (Pontos de Dispersão)
plt.scatter(plot_data_C['H0'], plot_data_C['C_Predito'], 
            label='Previsão ML (Random Forest)', color='darkorange', s=15, alpha=0.8)

plt.title('Modelo ML vs. Solução Analítica para Coeficiente de Amortecimento (C)', fontsize=14)
plt.xlabel('Razão de Espessura do Filme (H₀)', fontsize=12)
plt.ylabel('Amortecimento Adimensional (C)', fontsize=12)
plt.xlim(0, H0_max)
# Define o limite Y para mostrar o crescimento assintótico
plt.ylim(0, max(data['C_analitico']) * 1.05) 
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper right', fontsize=11)
plt.show()

plt.figure(figsize=(12, 6))

# Plot da Solução Analítica Exata (Linha Contínua Fina)
plt.plot(data['H0'], data['C_analitico'], label='Solução Analítica (Teórica)', 
         color='green', linestyle='-', linewidth=2, alpha=0.7)

# Plot das Previsões do ML no conjunto de Teste (Pontos de Dispersão)
plt.scatter(plot_data_C['H0'], plot_data_C['C_Predito'], 
            label='Previsão ML (Random Forest)', color='darkorange', s=15, alpha=0.8)

plt.title(f'ZOOM: Região Crítica (H₀ < {H0_zoom_max})', fontsize=14)
plt.xlabel('Razão de Espessura do Filme (H₀)', fontsize=12)
plt.ylabel('Amortecimento Adimensional (C)', fontsize=12)
# Foca o eixo X na região crítica
plt.xlim(H0_min, H0_zoom_max) 
# Foca o eixo Y na variação crítica
plt.ylim(min(data.loc[data['H0'] <= H0_zoom_max, 'C_analitico']) * 0.9, 
         max(data.loc[data['H0'] <= H0_zoom_max, 'C_analitico']) * 1.05) 

plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper right', fontsize=11)
plt.show()