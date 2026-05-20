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
from math import sqrt, pi
import matplotlib.pyplot as plt

def calcular_pressao_adimensional(X, Y, H, Lambda, dHdt=0.0, alpha=1e-6):
    """
    Calcula a pressão adimensional P em um mancal finito usando o método de Gauss-Seidel.
    """
    nx, ny = len(X), len(Y)

    # Calculamos o fator de amortecimento do método SOR para agilizar o Gauss-Seidel
    omega = 2 - sqrt(2)*pi*sqrt(1/nx**2+1/ny**2)

    # Inicializar a matriz de pressão adimensional P com zeros. 
    P = np.zeros((nx, ny))
    
    # Passos da malha (ΔX e ΔY)
    dx = X[1] - X[0]
    dy = Y[1] - Y[0]
    
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
                
                # Denominador
                denominador = 2 * Hp**3 * (1/(dx**2) + Lambda**2/(dy**2))
                
                # Atualizando o valor de P
                pij = (soma_vizinhos - Bp) / denominador
                P[i, j] = P[i,j] + omega*(pij-P[i,j])
        
        # Verificar a convergência
        if np.max(np.abs(P - P_old)) < alpha:
            print(f"Convergência alcançada na iteração {iteracao}.")
            break
            
    return P, omega

'''
j) O próximo passo será integrar a pressão adimensional P(X, Y) para obter a carga W(X,Y) suportada pelo mancal.
   Usando a relação W = ∫∫ P(X,Y) dA, onde dA é a área diferencial adimensional (dA = dX * dY), temos:
   W = ∫∫ P(X,Y) dA = ∫∫ P(X,Y) dx * dy ≈ ΣΣ P(X_i, Y_j) * ΔX * ΔY
   O resultado será um numero W (adimensional) que representa a carga suportada pelo mancal finito 
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

"""
k) O próximo passo será derivar a carga em relação à H0, isto nos dará o coeficiente dinâmico K.
   Note que W = W(H0,V).
   De modo que (∂W/∂H0 = K). Veja que K = K(W,H0,V)
   Vamos derivar por diferenças centradas, de modo que K ≈ (W(H0 + epsilon, V) - W(H0 - epsilon, V))/2*epsilon
   Veja que teremos que recalcular H, P e W duas vezes cada um.
   Isso ocorre pois teremos um H+ e um H-, um P+ e um P- e um W+ e W-.
   Basta chamarmos a função que fizemos anteriormente novamente. Veja que esse processo se torna custoso computacionalmente.
"""

def calcular_coeficiente_K(X, Y, H0, construir_H, Lambda, dHdt, epsilon=1e-4, alpha=1e-6):
    """
    Calcula o coeficiente de mola K = ∂W/∂H0 por diferenças centradas.

    construir_H: callable que recebe H0 e retorna a matriz H correspondente.
    """
    H_plus = construir_H(H0 + epsilon)
    P_plus, _ = calcular_pressao_adimensional(X, Y, H_plus, Lambda, dHdt, alpha=alpha)
    W_plus = calcular_carga_suportada(P_plus, X, Y)

    H_minus = construir_H(H0 - epsilon)
    P_minus, _ = calcular_pressao_adimensional(X, Y, H_minus, Lambda, dHdt, alpha=alpha)
    W_minus = calcular_carga_suportada(P_minus, X, Y)

    K = (W_plus - W_minus) / (2 * epsilon)
    return K

"""
l) O próximo passo será derivar a carga em relação à dHdt = V, isto nos dará o coeficiente dinâmico C.
   Note que W = W(H0,V).
   De modo que (∂W/∂V = C). Veja que C = C(W,H0,V)
   Vamos derivar por diferenças centradas, de modo que C ≈ (W(H0, V + epsilon) - W(H0, V - epsilon))/2*epsilon
"""

def calcular_coeficiente_C(X, Y, H, Lambda, dHdt, epsilon=1e-4, alpha=1e-6):
    """
    Calcula o coeficiente de amortecimento C = ∂W/∂V por diferenças centradas.
    """
    P_plus, _ = calcular_pressao_adimensional(X, Y, H, Lambda, dHdt + epsilon, alpha=alpha)
    W_plus = calcular_carga_suportada(P_plus, X, Y)

    P_minus, _ = calcular_pressao_adimensional(X, Y, H, Lambda, dHdt - epsilon, alpha=alpha)
    W_minus = calcular_carga_suportada(P_minus, X, Y)

    C = (W_plus - W_minus) / (2 * epsilon)
    return C


# ================================================================================================
# DEFINIÇÃO DOS PARÂMETROS E EXECUÇÃO
# ================================================================================================

# Parâmetros de discretização
beta = 100  # Número de pontos no eixo X
gama = 100  # Número de pontos no eixo Y
X = np.linspace(0, 1, beta)  # Coordenadas adimensionais no eixo X
Y = np.linspace(0, 1, gama)  # Coordenadas adimensionais no eixo Y

# Total de pontos na malha
total_pontos = beta*gama

# Criando as malhas para manipulação matricial e plotagem (indexing='ij' preserva a ordem X, Y)
X_grid, Y_grid = np.meshgrid(X, Y, indexing='ij')

# Definindo o perfil da espessura do filme
H0 = 0.5
construir_H = lambda h0: h0 + 1 - X_grid
H = construir_H(H0)

# Para o caso estático: termo Squeeze dHdt = 0
# Para casos não estáticos dHdt > 0 (Afastamento) e dHdt < 0 (Aproximação)
dHdt = 0.5

# Parâmetros operacionais
Lambda = 1.0  # Razão de aspecto do mancal (Lambda = lx/ly)

# Chamando a função para calcular a pressão adimensional e o omega do método SOR
P, omega = calcular_pressao_adimensional(X, Y, H, Lambda, dHdt, alpha=1e-6)
print(f"Valor do omega {omega:.2f}")

# Printando os passos da malha (ΔX e ΔY) e total de pontos na malha
dx = X[1] - X[0]
dy = Y[1] - Y[0]
print(f"Passo da malha: ΔX = {dx:.2f}, ΔY = {dy:.2f}")
print(f"Total de pontos na malha: {total_pontos}")


# Calculando a carga suportada pelo mancal
W = calcular_carga_suportada(P, X, Y)
print(f"Carga suportada pelo mancal: W = {W:.2f}")

# Calculando coeficientes dinâmicos
K = calcular_coeficiente_K(X, Y, H0, construir_H, Lambda, dHdt)
C = calcular_coeficiente_C(X, Y, H, Lambda, dHdt)
print(f"Coeficiente de Mola K: {K:.2f}")
print(f"Coeficiente de Amortecimento C: {C:.2f}")



# ================================================================================================
# SOLUÇÃO ANALÍTICA 1D (MANCAL INFINITO — APROXIMAÇÃO NWA)
# ================================================================================================
"""
n) A solução analítica para o mancal infinito (sem variação em Y) resolve a EDO 1D:
   d/dX(H^3 * dP/dX) = dH/dX + dHdt  com  H = H0+1-X  e  P(0)=P(1)=0.

   Integrando duas vezes e aplicando as condições de contorno, obtém-se a solução fechada:
     P(X) = k/2 * (h1/H² - 2/H + 1/h1) + C1/2 * (1/H² - 1/h1²)
   onde:
     k  = dH/dX + dHdt = -1 + dHdt   (fonte efetiva para perfil linear)
     h1 = H(X=0) = H0+1,  h2 = H(X=1) = H0
     C1 = -k * h1 / (h1+h2)           (constante de integração)

   Esta solução representa o limite de mancal infinito (sem vazamento lateral em Y).
   A solução numérica 2D tem pressão MENOR no centro devido ao efeito de borda.
"""

def calcular_pressao_analitica_1D(X, H0, dHdt):
    """
    Solução analítica da equação de Reynolds 1D (aproximação de mancal infinito).
    """
    h1 = H0 + 1.0
    h2 = float(H0)
    H = h1 - X
    k = -1.0 + dHdt          # dH/dX = −1 para perfil linear H = H0+1−X
    C1 = -k * h1 / (h1 + h2)
    return k/2 * (h1/H**2 - 2/H + 1/h1) + C1/2 * (1/H**2 - 1/h1**2)


# ================================================================================================
# PLOTAGEM: LADO A LADO DOS GRÁFICOS — NUMÉRICO 3D | COMPARAÇÃO ANALÍTICO vs. NUMÉRICO
# ================================================================================================

# Solução analítica 1D (mancal infinito)
P_analitica = calcular_pressao_analitica_1D(X, H0, dHdt)

# Plano central Y = 0.5: ponto mais afastado das bordas (onde P=0), logo mais próximo
# da solução analítica 1D (mancal infinito, sem efeito de borda lateral em Y).
j_center = np.argmin(np.abs(Y - 0.5))

fig = plt.figure(figsize=(16, 6))

# --- Plot 1: Superfície 3D numérica ---
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(X_grid, Y_grid, P, cmap='viridis', edgecolor='none')
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='P (adimensional)')
ax1.set_xlabel('X (adimensional)')
ax1.set_ylabel('Y (adimensional)')
ax1.set_zlabel('P (adimensional)')
ax1.set_title('Solução Numérica 2D\n(Mancal Finito — Diferenças Finitas + SOR)')
ax1.view_init(elev=30, azim=-135)

# --- Plot 2: Comparação perfil central ---
ax2 = fig.add_subplot(122)
ax2.plot(X, P_analitica,    'r--', linewidth=2, label='Analítico 1D (mancal infinito)')
ax2.plot(X, P[:, j_center], 'b-',  linewidth=2, label='Numérico 2D  (Y = 0.5)')
ax2.set_xlabel('X (adimensional)')
ax2.set_ylabel('Pressão P (adimensional)')
ax2.set_title('Analítico 1D vs. Numérico 2D\n'
              r'Perfil central (Y = 0.5)  —  $\Lambda$ = ' + f'{Lambda:.1f}')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
