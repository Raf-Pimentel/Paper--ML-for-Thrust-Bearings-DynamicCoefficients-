"""
Módulo de visualizações para o dashboard de mancais hidrodinâmicos.
Contém funções especializadas para criar gráficos usando Plotly.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from .calculos import *

# ============================================================================
# CORES E TEMAS
# ============================================================================

CORES = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17a2b8',
    'pressao': '#1f77b4',
    'carga': '#2ca02c',
    'atrito_movel': '#ff7f0e',
    'atrito_fixo': '#9467bd',
    'vazao': '#17becf',
    'temperatura': '#d62728'
}

LAYOUT_DEFAULT = dict(
    font=dict(size=12),
    hovermode='x unified',
    margin=dict(l=50, r=50, t=50, b=50)
)

# ============================================================================
# GRÁFICOS DE GEOMETRIA
# ============================================================================

def criar_grafico_geometria(l, sh, h0, delta_H=0, mostrar_anotacoes=True):
    """
    Cria visualização da geometria do mancal com opções de customização.
    
    Args:
        l (float): Comprimento
        sh (float): Altura da cunha
        h0 (float): Folga de saída
        delta_H (float): Perturbação vertical
        mostrar_anotacoes (bool): Se deve mostrar dimensões
    
    Returns:
        go.Figure: Figura do Plotly
    """
    # Calcular vértices da sapata
    altura_bloco = (sh + h0) * 0.4
    vertices_x = [0, l, l, 0, 0]
    vertices_y = [h0 + sh + delta_H, h0 + delta_H, h0 + sh + altura_bloco + delta_H, 
                  h0 + sh + altura_bloco + delta_H, h0 + sh + delta_H]
    
    fig = go.Figure()
    
    # Sapata superior
    fig.add_trace(go.Scatter(
        x=vertices_x, y=vertices_y,
        fill='toself',
        fillcolor='rgba(169, 169, 169, 0.5)',
        line=dict(color='black', width=2),
        name='Sapata Superior',
        hoverinfo='skip'
    ))
    
    # Pista inferior
    fig.add_trace(go.Scatter(
        x=[-l*0.1, l*1.1], y=[0, 0],
        mode='lines',
        line=dict(color='black', width=3),
        name='Pista Inferior',
        hoverinfo='skip'
    ))
    
    # Linha do filme de óleo (visualização da cunha)
    filme_x = [0, l]
    filme_y = [h0 + sh + delta_H, h0 + delta_H]
    fig.add_trace(go.Scatter(
        x=filme_x, y=filme_y,
        mode='lines',
        line=dict(color=CORES['info'], width=3, dash='dash'),
        name='Interface do Filme',
        hoverinfo='skip'
    ))
    
    if mostrar_anotacoes:
        Ho = h0 / sh
        angulo_graus = np.degrees(np.arctan(sh / l))
        
        # Anotações de dimensões
        anotacoes = [
            dict(x=l/2, y=-l*0.08, text=f'l = {l:.1f}', showarrow=False),
            dict(x=l*1.05, y=(h0 + delta_H)/2, text=f'h₀ = {h0:.1f}', showarrow=False),
            dict(x=-l*0.05, y=h0 + sh/2 + delta_H, text=f'sₕ = {sh:.1f}', showarrow=False),
        ]
        
        for anotacao in anotacoes:
            fig.add_annotation(**anotacao, font=dict(size=12, color='black'))
    
    # Layout
    Ho = h0 / sh
    angulo_graus = np.degrees(np.arctan(sh / l))
    
    fig.update_layout(
        title=f'Geometria do Mancal - H₀ = {Ho:.3f}, θ = {angulo_graus:.2f}°',
        xaxis_title='Comprimento',
        yaxis_title='Altura',
        height=400,
        font=dict(size=12),
        hovermode='x unified',
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, scaleanchor="x", scaleratio=1, zeroline=False)
    )
    
    return fig

# ============================================================================
# GRÁFICOS DE PRESSÃO
# ============================================================================

def criar_grafico_pressao(Ho_valores, mostrar_picos=True, titulo_customizado=None):
    """
    Cria gráfico de distribuição de pressão para um ou múltiplos valores de H₀.
    
    Args:
        Ho_valores (float or list): Valor(es) de H₀
        mostrar_picos (bool): Se deve marcar os picos de pressão
        titulo_customizado (str): Título personalizado
    
    Returns:
        go.Figure: Figura do Plotly
    """
    if not isinstance(Ho_valores, (list, tuple)):
        Ho_valores = [Ho_valores]
    
    X_coords = np.linspace(0, 1, 500)
    fig = go.Figure()
    
    cores_ciclo = [CORES['pressao'], CORES['danger'], CORES['success'], CORES['warning'], CORES['info']]
    
    for i, Ho in enumerate(Ho_valores):
        cor = cores_ciclo[i % len(cores_ciclo)]
        
        # Distribuição de pressão
        P_valores = calcular_pressao_adimensional(X_coords, Ho)
        
        label = f'H₀ = {Ho:.2f}' if len(Ho_valores) > 1 else 'Distribuição de Pressão'
        
        fig.add_trace(go.Scatter(
            x=X_coords, y=P_valores,
            mode='lines',
            name=label,
            line=dict(color=cor, width=3),
            hovertemplate='X: %{x:.3f}<br>P: %{y:.3f}<extra></extra>'
        ))
        
        if mostrar_picos:
            Xm, Pm = encontrar_pico_pressao(Ho)
            label_pico = f'Pico (H₀={Ho:.2f})' if len(Ho_valores) > 1 else f'P_max = {Pm:.3f}'
            
            fig.add_trace(go.Scatter(
                x=[Xm], y=[Pm],
                mode='markers',
                name=label_pico,
                marker=dict(color=cor, size=10, symbol='circle'),
                hovertemplate=f'Pico<br>X: {Xm:.3f}<br>P: {Pm:.3f}<extra></extra>',
                showlegend=(len(Ho_valores) == 1)
            ))
    
    titulo = titulo_customizado or 'Distribuição de Pressão Adimensional'
    
    fig.update_layout(
        title=titulo,
        xaxis_title='Coordenada Adimensional X = x/l',
        yaxis_title='Pressão Adimensional P',
        font=dict(size=12),
        hovermode='x unified',
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

# ============================================================================
# GRÁFICOS DE DESEMPENHO
# ============================================================================

def criar_grafico_capacidade_carga(Ho_range=None, Ho_atual=None, mostrar_otimo=True):
    """
    Cria gráfico da capacidade de carga vs H₀.
    
    Args:
        Ho_range (array): Range de valores de H₀
        Ho_atual (float): Valor atual para destacar
        mostrar_otimo (bool): Se deve mostrar o ponto ótimo
    
    Returns:
        go.Figure: Figura do Plotly
    """
    if Ho_range is None:
        Ho_range = np.linspace(0.05, 3.0, 400)
    
    Wz_valores = calcular_carga_normal(Ho_range)
    
    fig = go.Figure()
    
    # Curva principal
    fig.add_trace(go.Scatter(
        x=Ho_range, y=Wz_valores,
        mode='lines',
        name='Capacidade de Carga',
        line=dict(color=CORES['carga'], width=3),
        hovertemplate='H₀: %{x:.3f}<br>Wz: %{y:.3f}<extra></extra>'
    ))
    
    if mostrar_otimo:
        Ho_otimo = encontrar_Ho_otimo()
        Wz_otimo = calcular_carga_normal(Ho_otimo)
        
        fig.add_trace(go.Scatter(
            x=[Ho_otimo], y=[Wz_otimo],
            mode='markers',
            name=f'Ótimo: H₀={Ho_otimo:.3f}',
            marker=dict(color='gold', size=12, symbol='star'),
            hovertemplate=f'Ótimo<br>H₀: {Ho_otimo:.3f}<br>Wz: {Wz_otimo:.3f}<extra></extra>'
        ))
        
        # Linha vertical no ótimo
        fig.add_vline(x=Ho_otimo, line_dash="dash", line_color="gold", 
                     annotation_text="H₀ Ótimo", annotation_position="top")
    
    if Ho_atual is not None:
        Wz_atual = calcular_carga_normal(Ho_atual)
        fig.add_trace(go.Scatter(
            x=[Ho_atual], y=[Wz_atual],
            mode='markers',
            name=f'Atual: Wz={Wz_atual:.3f}',
            marker=dict(color=CORES['danger'], size=10),
            hovertemplate=f'Atual<br>H₀: {Ho_atual:.3f}<br>Wz: {Wz_atual:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Capacidade de Carga vs Razão de Espessura',
        xaxis_title='H₀ = h₀/sₕ',
        yaxis_title='Carga Normal Adimensional Wz',
        font=dict(size=12),
        hovermode='x unified',
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def criar_grafico_forcas_atrito(Ho_range=None, Ho_atual=None):
    """
    Cria gráfico das forças de atrito vs H₀.
    
    Args:
        Ho_range (array): Range de valores de H₀
        Ho_atual (float): Valor atual para destacar
    
    Returns:
        go.Figure: Figura do Plotly
    """
    if Ho_range is None:
        Ho_range = np.linspace(0.05, 3.0, 400)
    
    Fa_valores = calcular_atrito_movel(Ho_range)
    Fb_valores = calcular_atrito_fixo(Ho_range)
    
    fig = go.Figure()
    
    # Força na superfície móvel
    fig.add_trace(go.Scatter(
        x=Ho_range, y=Fa_valores,
        mode='lines',
        name='Fa (Superfície Móvel)',
        line=dict(color=CORES['atrito_movel'], width=3),
        hovertemplate='H₀: %{x:.3f}<br>Fa: %{y:.3f}<extra></extra>'
    ))
    
    # Força na superfície fixa
    fig.add_trace(go.Scatter(
        x=Ho_range, y=Fb_valores,
        mode='lines',
        name='Fb (Superfície Fixa)',
        line=dict(color=CORES['atrito_fixo'], width=3),
        hovertemplate='H₀: %{x:.3f}<br>Fb: %{y:.3f}<extra></extra>'
    ))
    
    # Pontos atuais
    if Ho_atual is not None:
        Fa_atual = calcular_atrito_movel(Ho_atual)
        Fb_atual = calcular_atrito_fixo(Ho_atual)
        
        fig.add_trace(go.Scatter(
            x=[Ho_atual], y=[Fa_atual],
            mode='markers',
            name=f'Fa atual: {Fa_atual:.3f}',
            marker=dict(color=CORES['atrito_movel'], size=10),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=[Ho_atual], y=[Fb_atual],
            mode='markers',
            name=f'Fb atual: {Fb_atual:.3f}',
            marker=dict(color=CORES['atrito_fixo'], size=10),
            showlegend=False
        ))
    
    # Linha de referência zero
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    fig.update_layout(
        title='Forças de Atrito vs Razão de Espessura',
        xaxis_title='H₀ = h₀/sₕ',
        yaxis_title='Forças de Atrito Adimensionais',
        font=dict(size=12),
        hovermode='x unified',
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def criar_grafico_vazao(Ho_range=None, Ho_atual=None):
    """
    Cria gráfico da vazão volumétrica vs H₀.
    
    Args:
        Ho_range (array): Range de valores de H₀
        Ho_atual (float): Valor atual para destacar
    
    Returns:
        go.Figure: Figura do Plotly
    """
    if Ho_range is None:
        Ho_range = np.linspace(0.01, 3.0, 400)
    
    Q_valores = calcular_vazao_adimensional(Ho_range)
    
    fig = go.Figure()
    
    # Curva de vazão
    fig.add_trace(go.Scatter(
        x=Ho_range, y=Q_valores,
        mode='lines',
        name='Vazão Volumétrica',
        line=dict(color=CORES['vazao'], width=3),
        hovertemplate='H₀: %{x:.3f}<br>Q: %{y:.3f}<extra></extra>'
    ))
    
    if Ho_atual is not None:
        Q_atual = calcular_vazao_adimensional(Ho_atual)
        fig.add_trace(go.Scatter(
            x=[Ho_atual], y=[Q_atual],
            mode='markers',
            name=f'Q atual: {Q_atual:.3f}',
            marker=dict(color=CORES['danger'], size=10)
        ))
    
    fig.update_layout(
        title='Vazão Volumétrica vs Razão de Espessura',
        xaxis_title='H₀ = h₀/sₕ',
        yaxis_title='Vazão Volumétrica Adimensional Q',
        font=dict(size=12),
        hovermode='x unified',
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

# ============================================================================
# GRÁFICOS COMPOSTOS
# ============================================================================

def criar_dashboard_completo(l, sh, h0, delta_H=0):
    """
    Cria um dashboard completo com múltiplos subplots.
    
    Args:
        l (float): Comprimento
        sh (float): Altura da cunha
        h0 (float): Folga de saída
        delta_H (float): Perturbação
    
    Returns:
        go.Figure: Figura composta
    """
    Ho = h0 / sh
    
    # Criar subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Distribuição de Pressão', 'Capacidade de Carga', 
                       'Forças de Atrito', 'Vazão Volumétrica'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Dados para os gráficos
    X_coords = np.linspace(0, 1, 200)
    Ho_range = np.linspace(0.1, 3.0, 200)
    
    # Gráfico 1: Distribuição de Pressão
    P_valores = calcular_pressao_adimensional(X_coords, Ho)
    Xm, Pm = encontrar_pico_pressao(Ho)
    
    fig.add_trace(go.Scatter(
        x=X_coords, y=P_valores, 
        mode='lines', 
        name='Pressão',
        line=dict(color=CORES['pressao'], width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=[Xm], y=[Pm], 
        mode='markers', 
        name=f'Pmax={Pm:.3f}',
        marker=dict(color=CORES['danger'], size=6),
        showlegend=False
    ), row=1, col=1)
    
    # Gráfico 2: Capacidade de Carga
    Wz_range = calcular_carga_normal(Ho_range)
    Wz_atual = calcular_carga_normal(Ho)
    
    fig.add_trace(go.Scatter(
        x=Ho_range, y=Wz_range, 
        mode='lines', 
        name='Wz',
        line=dict(color=CORES['carga'], width=2)
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=[Ho], y=[Wz_atual], 
        mode='markers', 
        name=f'Atual={Wz_atual:.3f}',
        marker=dict(color=CORES['danger'], size=6),
        showlegend=False
    ), row=1, col=2)
    
    # Gráfico 3: Forças de Atrito
    Fa_range = calcular_atrito_movel(Ho_range)
    Fb_range = calcular_atrito_fixo(Ho_range)
    
    fig.add_trace(go.Scatter(
        x=Ho_range, y=Fa_range, 
        mode='lines', 
        name='Fa',
        line=dict(color=CORES['atrito_movel'], width=2)
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=Ho_range, y=Fb_range, 
        mode='lines', 
        name='Fb',
        line=dict(color=CORES['atrito_fixo'], width=2)
    ), row=2, col=1)
    
    # Gráfico 4: Vazão
    Q_range = calcular_vazao_adimensional(Ho_range)
    Q_atual = calcular_vazao_adimensional(Ho)
    
    fig.add_trace(go.Scatter(
        x=Ho_range, y=Q_range, 
        mode='lines', 
        name='Vazão Q',
        line=dict(color=CORES['vazao'], width=2)
    ), row=2, col=2)
    
    fig.add_trace(go.Scatter(
        x=[Ho], y=[Q_atual], 
        mode='markers', 
        name=f'Q atual={Q_atual:.3f}',
        marker=dict(color=CORES['danger'], size=6),
        showlegend=False
    ), row=2, col=2)
    
    # Configuração dos eixos
    fig.update_xaxes(title_text="X = x/l", row=1, col=1)
    fig.update_yaxes(title_text="Pressão P", row=1, col=1)
    fig.update_xaxes(title_text="H₀", row=1, col=2)
    fig.update_yaxes(title_text="Carga Wz", row=1, col=2)
    fig.update_xaxes(title_text="H₀", row=2, col=1)
    fig.update_yaxes(title_text="Forças F", row=2, col=1)
    fig.update_xaxes(title_text="H₀", row=2, col=2)
    fig.update_yaxes(title_text="Vazão Q", row=2, col=2)
    
    fig.update_layout(
        height=600, 
        showlegend=False,
        title_text=f"Dashboard Completo - H₀ = {Ho:.3f}"
    )
    
    return fig

def criar_grafico_coeficientes_dinamicos(Ho_range=None, Ho_atual=None):
    """
    Cria gráfico dos coeficientes dinâmicos (rigidez e amortecimento).
    
    Args:
        Ho_range (array): Range de valores de H₀
        Ho_atual (float): Valor atual para destacar
    
    Returns:
        go.Figure: Figura com subplots
    """
    if Ho_range is None:
        Ho_range = np.linspace(0.01, 3.0, 500)
    
    K_valores = calcular_rigidez_analitica(Ho_range)
    C_valores = calcular_amortecimento_analitico(Ho_range)
    
    # Criar subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Coeficiente de Rigidez', 'Coeficiente de Amortecimento'),
        shared_xaxes=True
    )
    
    # Gráfico de rigidez
    fig.add_trace(go.Scatter(
        x=Ho_range, y=K_valores,
        mode='lines',
        name='Rigidez K',
        line=dict(color=CORES['danger'], width=3)
    ), row=1, col=1)
    
    # Gráfico de amortecimento
    fig.add_trace(go.Scatter(
        x=Ho_range, y=C_valores,
        mode='lines',
        name='Amortecimento C',
        line=dict(color=CORES['primary'], width=3)
    ), row=2, col=1)
    
    # Pontos atuais
    if Ho_atual is not None:
        K_atual = calcular_rigidez_analitica(Ho_atual)
        C_atual = calcular_amortecimento_analitico(Ho_atual)
        
        fig.add_trace(go.Scatter(
            x=[Ho_atual], y=[K_atual],
            mode='markers',
            name=f'K atual: {K_atual:.2f}',
            marker=dict(color=CORES['danger'], size=10)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=[Ho_atual], y=[C_atual],
            mode='markers',
            name=f'C atual: {C_atual:.2f}',
            marker=dict(color=CORES['primary'], size=10)
        ), row=2, col=1)
    
    # Linhas de referência
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=1)
    
    fig.update_layout(
        height=600,
        title_text='Coeficientes Dinâmicos do Mancal',
        font=dict(size=12),
        hovermode='x unified',
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_xaxes(title_text="H₀ = h₀/sₕ", row=2, col=1)
    fig.update_yaxes(title_text="Rigidez K", row=1, col=1)
    fig.update_yaxes(title_text="Amortecimento C", row=2, col=1)
    
    return fig

# ============================================================================
# FUNÇÕES DE COMPARAÇÃO
# ============================================================================

def criar_grafico_comparacao_configuracoes(configuracoes):
    """
    Cria gráfico comparativo entre diferentes configurações.
    
    Args:
        configuracoes (list): Lista de dicts com parâmetros {nome, l, sh, h0}
    
    Returns:
        go.Figure: Figura comparativa
    """
    fig = go.Figure()
    
    cores_config = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, config in enumerate(configuracoes):
        Ho = config['h0'] / config['sh']
        X_coords = np.linspace(0, 1, 200)
        P_valores = calcular_pressao_adimensional(X_coords, Ho)
        
        cor = cores_config[i % len(cores_config)]
        nome = config.get('nome', f'Configuração {i+1}')
        
        fig.add_trace(go.Scatter(
            x=X_coords, y=P_valores,
            mode='lines',
            name=f'{nome} (H₀={Ho:.3f})',
            line=dict(color=cor, width=2)
        ))
    
    fig.update_layout(
        title='Comparação de Configurações - Distribuição de Pressão',
        xaxis_title='X = x/l',
        yaxis_title='Pressão Adimensional P',
        font=dict(size=12),
        hovermode='x unified',
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

# ============================================================================
# FUNÇÕES DE EXPORTAÇÃO
# ============================================================================

def exportar_relatorio_grafico(metricas, nome_arquivo='relatorio_mancal'):
    """
    Cria um relatório gráfico completo em formato HTML.
    
    Args:
        metricas (dict): Métricas calculadas
        nome_arquivo (str): Nome do arquivo de saída
    
    Returns:
        str: HTML do relatório
    """
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Relatório de Análise do Mancal</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ text-align: center; color: #1f77b4; }}
            .metric {{ margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px; }}
            .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
            .success {{ background: #d1edff; border-left: 4px solid #28a745; }}
        </style>
    </head>
    <body>
        <h1 class="header">Relatório de Análise do Mancal Hidrodinâmico</h1>
        
        <h2>Parâmetros de Entrada</h2>
        <div class="metric">
            <strong>H₀ (Razão de Espessura):</strong> {metricas['Ho']:.3f}
        </div>
        
        <h2>Resultados Principais</h2>
        <div class="metric">
            <strong>Pressão Máxima:</strong> {metricas['pressao_maxima']:.3f}<br>
            <strong>Posição do Pico:</strong> {metricas['posicao_pico']:.3f}
        </div>
        
        <div class="metric">
            <strong>Capacidade de Carga (Wz):</strong> {metricas['carga_normal']:.3f}
        </div>
        
        <div class="metric">
            <strong>Força de Atrito Móvel (Fa):</strong> {metricas['atrito_movel']:.3f}<br>
            <strong>Força de Atrito Fixa (Fb):</strong> {metricas['atrito_fixo']:.3f}
        </div>
        
        <div class="metric">
            <strong>Vazão Volumétrica (Q):</strong> {metricas['vazao']:.3f}
        </div>
        
        <h2>Análise</h2>
        <div class="metric {'success' if abs(metricas['Ho'] - 0.707) < 0.05 else 'warning'}">
            Configuração {'próxima ao ótimo' if abs(metricas['Ho'] - 0.707) < 0.05 else 'pode ser otimizada'}
        </div>
    </body>
    </html>
    """
    
    return html_template