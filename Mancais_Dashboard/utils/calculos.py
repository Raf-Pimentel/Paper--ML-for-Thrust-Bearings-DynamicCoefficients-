"""
Módulo de cálculos para análise de mancais hidrodinâmicos.
Baseado nas equações do livro "Hydrodynamic Lubrication" de Hamrock.
"""

import numpy as np

# ============================================================================
# FUNÇÕES DE DISTRIBUIÇÃO DE PRESSÃO
# ============================================================================

def calcular_pressao_adimensional(X, Ho):
    """
    Calcula a pressão adimensional P (Eq. 8.24).
    
    Args:
        X (array): Coordenadas adimensionais (0 a 1)
        Ho (float): Razão de espessura do filme (h₀/sₕ)
    
    Returns:
        array: Valores de pressão adimensional P
    """
    numerador = 6 * X * (1 - X)
    denominador = ((Ho + 1 - X)**2) * (1 + 2 * Ho)
    
    if (1 + 2 * Ho) <= 1e-9:
        return np.zeros_like(X)
    
    return np.divide(numerador, denominador, out=np.zeros_like(numerador), where=denominador!=0)

def encontrar_pico_pressao(Ho):
    """
    Calcula a localização (Xm) e valor (Pm) da pressão máxima (Eq. 8.25).
    
    Args:
        Ho (float): Razão de espessura do filme
    
    Returns:
        tuple: (Xm, Pm) - posição e valor da pressão máxima
    """
    if (1 + 2 * Ho) <= 1e-9:
        return 1.0, 0.0
    
    Xm = (1 + Ho) / (1 + 2 * Ho)
    Pm = calcular_pressao_adimensional(Xm, Ho)
    return Xm, Pm

# ============================================================================
# FUNÇÕES DE CAPACIDADE DE CARGA
# ============================================================================

def calcular_carga_normal(Ho):
    """
    Calcula a Carga Normal Adimensional (Wz) - Eq. 8.30.
    
    Args:
        Ho (float or array): Razão de espessura do filme
    
    Returns:
        float or array: Carga normal adimensional Wz
    """
    termo_log = np.log((Ho + 1) / Ho)
    termo_fracao = 12 / (1 + 2 * Ho)
    return 6 * termo_log - termo_fracao

def encontrar_Ho_otimo():
    """
    Retorna o valor ótimo teórico de H₀ para máxima capacidade de carga.
    
    Returns:
        float: H₀ ótimo ≈ 0.707
    """
    return 1 / np.sqrt(2)

# ============================================================================
# FUNÇÕES DE FORÇAS DE ATRITO
# ============================================================================

def calcular_atrito_movel(Ho):
    """
    Calcula a Força de Atrito na superfície móvel (Fa) - Eq. 8.33.
    
    Args:
        Ho (float or array): Razão de espessura do filme
    
    Returns:
        float or array: Força de atrito adimensional Fa
    """
    termo_log = np.log((Ho + 1) / Ho)
    termo_fracao = 6 / (1 + 2 * Ho)
    return 2 * termo_log + termo_fracao

def calcular_atrito_fixo(Ho):
    """
    Calcula a Força de Atrito na superfície fixa (Fb) - Eq. 8.32.
    
    Args:
        Ho (float or array): Razão de espessura do filme
    
    Returns:
        float or array: Força de atrito adimensional Fb
    """
    termo_log = np.log((Ho + 1) / Ho)
    termo_fracao = 6 / (1 + 2 * Ho)
    return -(4 * termo_log + termo_fracao)

def calcular_coeficiente_atrito(Ho):
    """
    Calcula o parâmetro do coeficiente de atrito (μl/sₕ) = Fa/Wz.
    
    Args:
        Ho (float or array): Razão de espessura do filme
    
    Returns:
        float or array: Parâmetro do coeficiente de atrito
    """
    Fa = calcular_atrito_movel(Ho)
    Wz = calcular_carga_normal(Ho)
    return np.divide(Fa, Wz, out=np.full_like(Fa, np.inf), where=Wz!=0)

# ============================================================================
# FUNÇÕES DE VAZÃO E TEMPERATURA
# ============================================================================

def calcular_vazao_adimensional(Ho):
    """
    Calcula a Vazão Volumétrica Adimensional (Q) - Eq. 8.36.
    
    Args:
        Ho (float or array): Razão de espessura do filme
    
    Returns:
        float or array: Vazão volumétrica adimensional Q
    """
    numerador = 2 * Ho * (1 + Ho)
    denominador = 1 + 2 * Ho
    return numerador / denominador

def calcular_perda_potencia_adimensional(Ho):
    """
    Calcula a Perda de Potência Adimensional (Hp) - Eq. 8.37.
    
    Args:
        Ho (float or array): Razão de espessura do filme
    
    Returns:
        float or array: Perda de potência adimensional Hp
    """
    termo_log = np.log((Ho + 1) / Ho)
    termo_fracao = 6 / (1 + 2 * Ho)
    return 4 * termo_log + termo_fracao

def calcular_aumento_temperatura_adimensional(Ho):
    """
    Calcula o Aumento de Temperatura Adimensional (HT) - Eq. 8.38.
    
    Args:
        Ho (float or array): Razão de espessura do filme
    
    Returns:
        float or array: Aumento de temperatura adimensional HT
    """
    Hp = calcular_perda_potencia_adimensional(Ho)
    Q = calcular_vazao_adimensional(Ho)
    return np.divide(Hp, Q, out=np.full_like(Hp, np.inf), where=Q!=0)

# ============================================================================
# FUNÇÕES DE CENTRO DE PRESSÃO
# ============================================================================

def calcular_centro_pressao_adimensional(Ho):
    """
    Calcula o Centro de Pressão Adimensional (Xcp) - Eq. 8.40.
    
    Args:
        Ho (float or array): Razão de espessura do filme
    
    Returns:
        float or array: Centro de pressão adimensional Xcp
    """
    Wz = calcular_carga_normal(Ho)
    termo_log = np.log(Ho / (Ho + 1))
    fator_principal = -6 / (Wz * (1 + 2 * Ho))
    termo_colchetes = (Ho + 1) * (3 * Ho + 1) * termo_log + 3 * Ho + 2.5
    return fator_principal * termo_colchetes

# ============================================================================
# FUNÇÕES DE COEFICIENTES DINÂMICOS
# ============================================================================

def calcular_rigidez_analitica(Ho):
    """
    Calcula o Coeficiente de Rigidez Adimensional.
    ATENÇÃO: Esta função pode apresentar valores negativos para Ho muito pequeno.
    
    Args:
        Ho (float or array): Razão de espessura do filme
    
    Returns:
        float or array: Coeficiente de rigidez adimensional
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        termo1 = 6 * ((1 / (Ho + 1)) - (1 / Ho))
        termo2 = 24 / ((1 + 2 * Ho)**2)
        resultado = termo1 + termo2
    
    return np.where(Ho <= 1e-9, -np.inf, resultado)

def calcular_amortecimento_analitico(Ho):
    """
    Calcula o Coeficiente de Amortecimento Adimensional.
    Versão corrigida e fisicamente consistente.
    
    Args:
        Ho (float or array): Razão de espessura do filme
    
    Returns:
        float or array: Coeficiente de amortecimento adimensional
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        termo_log = np.log((Ho + 1) / Ho)
        termo_fracao = 2 / (2 * Ho + 1)
        resultado = termo_log - termo_fracao
    
    return np.where(Ho <= 1e-9, np.inf, resultado)

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def validar_parametros(l=None, sh=None, h0=None, Ho=None):
    """
    Valida os parâmetros de entrada e retorna mensagens de aviso se necessário.
    
    Args:
        l (float): Comprimento
        sh (float): Altura da cunha
        h0 (float): Folga de saída
        Ho (float): Razão calculada
    
    Returns:
        list: Lista de mensagens de validação
    """
    mensagens = []
    
    if Ho is not None:
        Ho_otimo = encontrar_Ho_otimo()
        if Ho < 0.1:
            mensagens.append("⚠️ H₀ muito baixo pode causar instabilidades")
        elif Ho > 3.0:
            mensagens.append("⚠️ H₀ muito alto reduz significativamente a capacidade de carga")
        elif abs(Ho - Ho_otimo) < 0.05:
            mensagens.append("✅ Configuração próxima ao ótimo teórico")
    
    if l is not None and sh is not None:
        razao_l_sh = l / sh
        if razao_l_sh < 5:
            mensagens.append("⚠️ Razão l/sₕ baixa pode afetar a aproximação de lubrificação")
        elif razao_l_sh > 50:
            mensagens.append("ℹ️ Razão l/sₕ alta - mancal muito alongado")
    
    return mensagens

def calcular_metricas_resumo(Ho):
    """
    Calcula todas as métricas principais para um dado H₀.
    
    Args:
        Ho (float): Razão de espessura do filme
    
    Returns:
        dict: Dicionário com todas as métricas calculadas
    """
    Xm, Pm = encontrar_pico_pressao(Ho)
    
    metricas = {
        'Ho': Ho,
        'pressao_maxima': Pm,
        'posicao_pico': Xm,
        'carga_normal': calcular_carga_normal(Ho),
        'atrito_movel': calcular_atrito_movel(Ho),
        'atrito_fixo': calcular_atrito_fixo(Ho),
        'coef_atrito': calcular_coeficiente_atrito(Ho),
        'vazao': calcular_vazao_adimensional(Ho),
        'centro_pressao': calcular_centro_pressao_adimensional(Ho),
        'rigidez': calcular_rigidez_analitica(Ho),
        'amortecimento': calcular_amortecimento_analitico(Ho)
    }
    
    return metricas