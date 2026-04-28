"""
a) Iremos começar a tratar da equação de Reynolds para um mancal finito, ou seja, onde a 
   pressão varia nas duas direções: P = P(X, Y).

b) Para isso, vamos considerar o plano superior como uma superfície h = h(x,y).
   (Nota: Para haver geração de pressão hidrodinâmica, o filme não pode ter espessura constante h = cte).

c) A equação de Reynolds para um mancal finito, considerando um fluido incompressível, é:
   ∂/∂x (h^3 * ∂P/∂x) + ∂/∂y (h^3 * ∂P/∂y) = 6μ*u_b * ∂h/∂x + 12μ * ∂h/∂t
   Iremos considerar a variação temporal da espessura do filme (esmagamento) 12μ * ∂h/∂t = V.

d) Iremos adimensionalizar essa equação usando as seguintes variáveis adimensionais:
   X = x/l, Y = y/l, H = h/s_h, P = Padimensional = p_real/p0, 
   p0 = 6*μ*u_b*lx/s_h^2, Razão de aspecto = Λ = lx/ly, v0 = u_b*s_h/(2*lx)

e) A equação de Reynolds adimensionalizada para um mancal finito resulta em:
   ∂/∂X (H^3 * ∂P/∂X) + (Λ^2) * ∂/∂Y (H^3 * ∂P/∂Y) = ∂H/∂X + ∂H/∂t

f) Para resolver essa equação, precisamos de condições de contorno de Dirichlet:
   - Contorno 1: P(X=0, Y) = 0 (pressão ambiente na entrada do mancal)
   - Contorno 2: P(X=1, Y) = 0 (pressão ambiente na saída do mancal)
   - Contorno 3: P(X, Y=0) = 0 (pressão ambiente na borda inferior do mancal)
   - Contorno 4: P(X, Y=1) = 0 (pressão ambiente na borda superior do mancal)

g) Para resolver a EDP (Equação Diferencial Parcial), usamos o método de diferenças finitas centradas.
   Discretizamos o domínio em uma grade de pontos (X_i, Y_j) e expandimos as derivadas.

h) Ao expandir a equação e isolar o termo central P(X_i, Y_j) que chamaremos de Pp, temos:
   - Termos Leste (Pe), Oeste (Pw), Norte (Pn) e Sul (Ps).
   - O termo de geração de pressão é Bp = ∂H/∂X + ∂H/∂t.
   - Isolando o Pp, o denominador da equação discretizada é estritamente dependente da geometria
     e da malha: Denominador = 2 * Hp^3 * [ 1/(ΔX^2) + (Λ^2)/(ΔY^2) ].

i) Para resolver o sistema linear resultante, utilizamos o método iterativo de Gauss-Seidel, 
   atualizando a matriz de pressão até que o erro residual seja menor que uma tolerância (alpha).
"""

import numpy as np
import matplotlib.pyplot as plt

def calcular_pressao_adimensional(X, Y, H, Lambda, dHdt=0.0, alpha=1e-6):
    """
    Calcula a pressão adimensional P em um mancal finito usando o método de Gauss-Seidel.
    """
    nx, ny = len(X), len(Y)
    
    # Inicializar a matriz de pressão adimensional P com zeros. 
    P = np.zeros((nx, ny))
    
    # Passos da malha (ΔX e ΔY)
    dx = X[1] - X[0]
    dy = Y[1] - Y[0]
    print(f"Passo da malha: ΔX = {dx:.4f}, ΔY = {dy:.4f}")
    
    # Iterar até a convergência usando o critério de erro máximo absoluto
    for iteracao in range(2000):
        P_old = P.copy()
        
        # Percorrendo apenas os pontos internos
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                
                Hp = H[i, j]
                Pe = P[i+1, j]  # Leste
                Pw = P[i-1, j]  # Oeste
                Pn = P[i, j+1]  # Norte
                Ps = P[i, j-1]  # Sul
                
                # Calculando as derivadas espaciais em X e em Y
                dHdX = (H[i+1, j] - H[i-1, j]) / (2 * dx)
                dHdY = (H[i, j+1] - H[i, j-1]) / (2 * dy)
                
                Bp = dHdX + dHdt
                
                # Agrupando os termos vizinhos decorrentes da expansão das diferenças finitas
                termo_X_dif = Hp**3 * (Pe + Pw) / (dx**2)
                
                # CORREÇÃO 2: Multiplicando pelo dHdX
                termo_X_conv = 3 * Hp**2 * dHdX * (Pe - Pw) / (2 * dx) 
                
                termo_Y_dif = Lambda**2 * Hp**3 * (Pn + Ps) / (dy**2)
                
                # CORREÇÃO 3: Multiplicando pelo dHdY
                termo_Y_conv = 3 * Lambda**2 * Hp**2 * dHdY * (Pn - Ps) / (2 * dy) 
                
                soma_vizinhos = termo_X_dif + termo_X_conv + termo_Y_dif + termo_Y_conv
                
                # Denominador correto
                denominador = 2 * Hp**3 * (1/(dx**2) + Lambda**2/(dy**2))
                
                # Atualizando o valor de P
                P[i, j] = (soma_vizinhos - Bp) / denominador
        
        # Verificar a convergência
        if np.max(np.abs(P - P_old)) < alpha:
            print(f"Convergência alcançada na iteração {iteracao}.")
            break
            
    return P

'''
j) O próximo passo será integrar a pressão adimensional P(X, Y) para obter a carga W(X,Y) suportada pelo mancal.
   Usando a relação W = ∫∫ P(X,Y) dA, onde dA é a área diferencial adimensional (dA = dX * dY), temos:
   W = ∫∫ P(X,Y) dA = ∫∫ P(X,Y) dx * dy ≈ ΣΣ P(X_i, Y_j) * ΔX * ΔY
   O resultado será um numero W (adimensional?) que representa a carga suportada pelo mancal finito 
'''

def calcular_carga_suportada(P, X, Y):
    """
    Calcula a carga suportada W pelo mancal finito integrando a pressão adimensional P(X, Y).
    """
    dx = X[1] - X[0]
    dy = Y[1] - Y[0]
    
    # A carga W é a soma de P(X_i, Y_j) * ΔX * ΔY para todos os pontos da malha
    W = np.sum(P) * dx * dy
    return W


# ================================================================================================
# DEFINIÇÃO DOS PARÂMETROS E EXECUÇÃO
# ================================================================================================

# Parâmetros de discretização
beta = 50  # Número de pontos no eixo X
gama = 50  # Número de pontos no eixo Y
X = np.linspace(0, 1, beta)  # Coordenadas adimensionais no eixo X
Y = np.linspace(0, 1, gama)  # Coordenadas adimensionais no eixo Y

# Criando as malhas para manipulação matricial e plotagem (indexing='ij' preserva a ordem X, Y)
X_grid, Y_grid = np.meshgrid(X, Y, indexing='ij')

# Definindo o perfil da espessura do filme (Passo b)
H = 2.0 - 0.5*X_grid - 0.5*Y_grid

# Parâmetros operacionais
Lambda = 1.0  # Razão de aspecto do mancal (Lambda = lx/ly)

# Chamando a função para calcular a pressão adimensional (Temro Squeeze: dHdt = 0)
P = calcular_pressao_adimensional(X, Y, H, Lambda, dHdt=0.0, alpha=1e-6)

# Calculando a carga suportada pelo mancal
W = calcular_carga_suportada(P, X, Y)
print(f"Carga suportada pelo mancal: W = {W:.4f}")

# ================================================================================================
# PLOTAGEM DO GRÁFICO 3D
# ================================================================================================

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plotando a superfície. O colormap 'viridis' ajuda a visualizar o pico da pressão.
surf = ax.plot_surface(X_grid, Y_grid, P, cmap='viridis', edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Pressão P (adimensional)')

ax.set_xlabel('Comprimento X (adimensional)')
ax.set_ylabel('Largura Y (adimensional)')
ax.set_zlabel('Pressão P (adimensional)')
ax.set_title('Distribuição de Pressão Adimensional P(X, Y)\nMancal Finito com Perfil de Cunha')

# Ajustando o ângulo de visão para ver bem a "colina de pressão"
ax.view_init(elev=30, azim=-135)

plt.show()
