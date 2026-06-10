from MLRegressionK_1_ import X_train, X_test, y_train_K, y_test_K, K_data, H0_max, data, pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Instanciação e Treinamento do Modelo ---
# n_estimators: número de árvores na floresta (mais árvores = maior precisão, mas mais lento)
# random_state: para reprodutibilidade
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

print("Iniciando o treinamento do Random Forest Regressor para K...")
rf_model.fit(X_train, y_train_K)
print("✅ Treinamento concluído.")


# --- 2. Predição nos Conjuntos de Teste e Treinamento ---
y_pred_K_train = rf_model.predict(X_train)
y_pred_K_test = rf_model.predict(X_test)


# --- 3. Avaliação de Métricas ---
# Métricas para o conjunto de TREINAMENTO
mse_train = mean_squared_error(y_train_K, y_pred_K_train)
r2_train = r2_score(y_train_K, y_pred_K_train)

# Métricas para o conjunto de TESTE (Representa a capacidade de generalização)
mse_test = mean_squared_error(y_test_K, y_pred_K_test)
r2_test = r2_score(y_test_K, y_pred_K_test)

print("\n--- Resultados da Avaliação do Modelo K ---")
print("Métricas de TREINAMENTO:")
print(f"  Root Mean Squared Error (RMSE): {np.sqrt(mse_train):.6f}")
print(f"  R² Score (Desempenho no treino): {r2_train:.6f}")
print("-" * 35)
print("Métricas de TESTE:")
print(f"  Root Mean Squared Error (RMSE): {np.sqrt(mse_test):.6f}")
print(f"  R² Score (Generalização): {r2_test:.6f}")
# Interpretação R²: Um valor próximo de 1.0 (ou 100%) indica que o modelo explica quase toda a variância dos dados.


# --- 4. Visualização para Validação ---
# Combina X_test e y_test_K com as previsões (Ordena para o plot ficar limpo)
plot_data = pd.DataFrame({
    'H0': X_test.flatten(),
    'K_Analitico': y_test_K,
    'K_Predito': y_pred_K_test
}).sort_values('H0')

plt.figure(figsize=(10, 6))

# Plot da Solução Analítica Exata (Linha)
plt.plot(data['H0'], data['K_analitico'], label='Solução Analítica (Dados Brutos)', color='red', linestyle='--')

# Plot das Previsões do ML no conjunto de Teste (Pontos)
plt.scatter(plot_data['H0'], plot_data['K_Predito'], label='Previsão do Random Forest', color='blue', s=10)

plt.title('Validação do Modelo ML: Coeficiente de Rigidez (K) vs H₀')
plt.xlabel('Razão de Espessura do Filme (H₀)')
plt.ylabel('Rigidez Adimensional (K)')
plt.xlim(0, H0_max)
plt.ylim(min(K_data)*1.1, 20) # Limita Y para focar na região de interesse
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

# Adicionando um plot de zoom para a região crítica (H0 < 0.5)
plt.figure(figsize=(10, 6))
plt.plot(data['H0'], data['K_analitico'], label='Solução Analítica', color='red', linestyle='--')
plt.scatter(plot_data['H0'], plot_data['K_Predito'], label='Previsão do Random Forest', color='blue', s=10)
plt.title('ZOOM: Região Crítica (H₀ < 0.5)')
plt.xlabel('Razão de Espessura do Filme (H₀)')
plt.ylabel('Rigidez Adimensional (K)')
plt.xlim(0.01, 0.5)
plt.ylim(min(K_data)*1.1, 5) 
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()