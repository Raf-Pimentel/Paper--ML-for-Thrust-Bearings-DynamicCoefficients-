import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configuração da página
st.set_page_config(
    page_title="Análise de Mancais Hidrodinâmicos",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para melhorar a aparência
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .section-header {
        color: #2e86de;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
        margin: 1rem 0;
        color: #333333;
    }
    
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Funções de cálculo (migradas dos seus códigos)
def calcular_pressao_adimensional(X, Ho):
    """Calcula a pressão adimensional P (Eq. 8.24)."""
    numerador = 6 * X * (1 - X)
    denominador = ((Ho + 1 - X)**2) * (1 + 2 * Ho)
    if (1 + 2 * Ho) <= 1e-9:
        return np.zeros_like(X)
    return np.divide(numerador, denominador, out=np.zeros_like(numerador), where=denominador!=0)

def encontrar_pico_pressao(Ho):
    """Calcula a localização (Xm) e o valor (Pm) da pressão máxima (Eq. 8.25)."""
    if (1 + 2 * Ho) <= 1e-9:
        return 1.0, 0.0
    Xm = (1 + Ho) / (1 + 2 * Ho)
    Pm = calcular_pressao_adimensional(Xm, Ho)
    return Xm, Pm

def calcular_carga_normal(Ho):
    """Calcula a Carga Normal Adimensional (Wz) (Eq. 8.30)."""
    return 6 * np.log((Ho + 1) / Ho) - (12 / (1 + 2 * Ho))

def calcular_atrito_movel(Ho):
    """Calcula a Força de Atrito Adimensional na superfície móvel (Fa) (Eq. 8.33)."""
    return 2 * np.log((Ho + 1) / Ho) + (6 / (1 + 2 * Ho))

def calcular_atrito_fixo(Ho):
    """Calcula a Força de Atrito Adimensional na superfície fixa (Fb) (Eq. 8.32)."""
    return -(4 * np.log((Ho + 1) / Ho) + (6 / (1 + 2 * Ho)))

# Função para criar gráfico de geometria (versão local para evitar problemas de importação)
def criar_grafico_geometria_local(l, sh, h0, delta_H=0):
    """Cria visualização da geometria do mancal - versão local."""
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
        showlegend=False
    ))
    
    # Pista inferior
    fig.add_trace(go.Scatter(
        x=[-l*0.1, l*1.1], y=[0, 0],
        mode='lines',
        line=dict(color='black', width=3),
        name='Pista Inferior',
        showlegend=False
    ))
    
    # Linha do filme de óleo
    filme_x = [0, l]
    filme_y = [h0 + sh + delta_H, h0 + delta_H]
    fig.add_trace(go.Scatter(
        x=filme_x, y=filme_y,
        mode='lines',
        line=dict(color='#17a2b8', width=3, dash='dash'),
        name='Interface do Filme',
        showlegend=False
    ))
    
    # Layout
    Ho = h0 / sh
    angulo_graus = np.degrees(np.arctan(sh / l))
    
    fig.update_layout(
        title=f'Geometria do Mancal - H₀ = {Ho:.3f}, θ = {angulo_graus:.2f}°',
        xaxis_title='Comprimento',
        yaxis_title='Altura',
        height=400,
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, scaleanchor="x", scaleratio=1, zeroline=False)
    )
    
    # Anotações
    fig.add_annotation(x=l/2, y=-l*0.05, text=f'l = {l:.1f}', showarrow=False, font=dict(size=12))
    fig.add_annotation(x=l*1.05, y=(h0 + delta_H)/2, text=f'h₀ = {h0:.1f}', showarrow=False, font=dict(size=12))
    fig.add_annotation(x=-l*0.05, y=h0 + sh/2 + delta_H, text=f'sₕ = {sh:.1f}', showarrow=False, font=dict(size=12))
    
    return fig

def calcular_vazao_adimensional(Ho):
    """Calcula a Vazão Volumétrica Adimensional (Q) (Eq. 8.36)."""
    numerador = 2 * Ho * (1 + Ho)
    denominador = 1 + 2 * Ho
    return numerador / denominador

# Função para criar gráfico de geometria
def criar_grafico_geometria(l, sh, h0, delta_H=0):
    """Cria visualização da geometria do mancal."""
    # Vértices da sapata
    altura_bloco = (sh + h0) * 0.4
    vertices_x = [0, l, l, 0, 0]
    vertices_y = [h0 + sh + delta_H, h0 + delta_H, h0 + sh + altura_bloco + delta_H, 
                  h0 + sh + altura_bloco + delta_H, h0 + sh + delta_H]
    
    fig = go.Figure()
    
    # Sapata superior
    fig.add_trace(go.Scatter(
        x=vertices_x, y=vertices_y,
        fill='toself',
        fillcolor='lightgray',
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
    
    # Anotações
    Ho = h0 / sh
    angulo_graus = np.degrees(np.arctan(sh / l))
    
    fig.add_annotation(
        x=l/2, y=-l*0.05,
        text=f'l = {l:.1f}',
        showarrow=False,
        font=dict(size=12)
    )
    
    fig.add_annotation(
        x=l*1.05, y=h0/2,
        text=f'h₀ = {h0:.1f}',
        showarrow=False,
        font=dict(size=12)
    )
    
    fig.add_annotation(
        x=-l*0.05, y=h0 + sh/2,
        text=f'sₕ = {sh:.1f}',
        showarrow=False,
        font=dict(size=12)
    )
    
    fig.update_layout(
        title=f'Geometria do Mancal - H₀ = {Ho:.3f}, θ = {angulo_graus:.2f}°',
        xaxis_title='Comprimento',
        yaxis_title='Altura',
        showlegend=False,
        height=400,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

# Interface principal
def main():
    st.markdown('<h1 class="main-header">⚙️ Análise de Mancais Hidrodinâmicos</h1>', unsafe_allow_html=True)
    
    # Sidebar para navegação
    st.sidebar.title("Navegação")
    page = st.sidebar.selectbox(
        "Escolha a página:",
        ["🏠 Dashboard Principal", "📐 Geometria Interativa", "📊 Distribuição de Pressão", 
         "⚖️ Capacidade de Carga", "🔧 Forças de Atrito", "📈 Análise Completa"]
    )
    
    if page == "🏠 Dashboard Principal":
        dashboard_principal()
    elif page == "📐 Geometria Interativa":
        geometria_interativa()
    elif page == "📊 Distribuição de Pressão":
        distribuicao_pressao()
    elif page == "⚖️ Capacidade de Carga":
        capacidade_carga()
    elif page == "🔧 Forças de Atrito":
        forcas_atrito()
    elif page == "📈 Análise Completa":
        analise_completa()

def dashboard_principal():
    """Página principal com visão geral."""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-header">Sobre o Projeto</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <p style="color: #333333; margin-bottom: 10px;">Este dashboard interativo foi desenvolvido para análise de <strong>Mancais de Escora de Inclinação Fixa</strong>, 
        baseado no trabalho de Iniciação Científica sobre lubrificação hidrodinâmica.</p>
        
        <p style="color: #333333; font-weight: bold; margin-bottom: 5px;">Funcionalidades:</p>
        <ul style="color: #333333; margin-left: 20px;">
        <li>Visualização interativa da geometria do mancal</li>
        <li>Análise da distribuição de pressão</li>
        <li>Cálculo de capacidade de carga</li>
        <li>Análise de forças de atrito</li>
        <li>Coeficientes dinâmicos</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-header">Configuração Rápida</div>', unsafe_allow_html=True)
        
        # Parâmetros básicos
        l = st.slider("Comprimento (l)", 50.0, 200.0, 100.0, 1.0)
        sh = st.slider("Altura da Cunha (sₕ)", 1.0, 50.0, 14.1, 0.1)
        h0 = st.slider("Folga de Saída (h₀)", 1.0, 50.0, 10.0, 0.1)
        
        Ho = h0 / sh
        
        # Métricas importantes
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Razão H₀", f"{Ho:.3f}")
            st.metric("Ângulo θ", f"{np.degrees(np.arctan(sh/l)):.2f}°")
        
        with col_m2:
            Wz = calcular_carga_normal(Ho)
            Xm, Pm = encontrar_pico_pressao(Ho)
            st.metric("Carga Wz", f"{Wz:.2f}")
            st.metric("Pressão Máx.", f"{Pm:.3f}")
    
    # Gráfico de geometria
    st.markdown('<div class="section-header">Visualização da Geometria</div>', unsafe_allow_html=True)
    fig_geom = criar_grafico_geometria_local(l, sh, h0)
    st.plotly_chart(fig_geom, use_container_width=True)
    
    # Gráficos de desempenho
    col3, col4 = st.columns(2)
    
    with col3:
        # Gráfico de pressão
        X_coords = np.linspace(0, 1, 500)
        P_valores = calcular_pressao_adimensional(X_coords, Ho)
        
        fig_pressao = go.Figure()
        fig_pressao.add_trace(go.Scatter(
            x=X_coords, y=P_valores,
            mode='lines',
            name='Distribuição de Pressão',
            line=dict(color='royalblue', width=3)
        ))
        
        # Marcar pico
        fig_pressao.add_trace(go.Scatter(
            x=[Xm], y=[Pm],
            mode='markers',
            name=f'Pico: P_max = {Pm:.3f}',
            marker=dict(color='red', size=10)
        ))
        
        fig_pressao.update_layout(
            title='Distribuição de Pressão',
            xaxis_title='X = x/l',
            yaxis_title='Pressão Adimensional P',
            height=350
        )
        st.plotly_chart(fig_pressao, use_container_width=True)
    
    with col4:
        # Gráfico de carga vs H0
        Ho_range = np.linspace(0.1, 3.0, 200)
        Wz_range = calcular_carga_normal(Ho_range)
        
        fig_carga = go.Figure()
        fig_carga.add_trace(go.Scatter(
            x=Ho_range, y=Wz_range,
            mode='lines',
            name='Capacidade de Carga',
            line=dict(color='green', width=3)
        ))
        
        # Marcar ponto atual
        fig_carga.add_trace(go.Scatter(
            x=[Ho], y=[Wz],
            mode='markers',
            name=f'Atual: Wz = {Wz:.2f}',
            marker=dict(color='red', size=10)
        ))
        
        fig_carga.update_layout(
            title='Capacidade de Carga vs H₀',
            xaxis_title='H₀ = h₀/sₕ',
            yaxis_title='Carga Adimensional Wz',
            height=350
        )
        st.plotly_chart(fig_carga, use_container_width=True)

def geometria_interativa():
    """Página de geometria interativa."""
    st.markdown('<div class="section-header">📐 Geometria Interativa do Mancal</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Parâmetros Geométricos")
        l = st.slider("Comprimento (l)", 50.0, 200.0, 100.0, 1.0, key="geom_l")
        sh = st.slider("Altura da Cunha (sₕ)", 1.0, 50.0, 14.1, 0.1, key="geom_sh")
        h0 = st.slider("Folga de Saída (h₀)", 1.0, 50.0, 10.0, 0.1, key="geom_h0")
        
        st.markdown("### Perturbações")
        delta_H = st.slider("Deslocamento ΔH", -10.0, 10.0, 0.0, 0.1)
        
        Ho = h0 / sh
        angulo = np.degrees(np.arctan(sh/l))
        
        st.markdown("### Parâmetros Calculados")
        st.info(f"""
        **H₀ = h₀/sₕ = {Ho:.3f}**
        
        **Ângulo da cunha: {angulo:.2f}°**
        
        **Razão l/sₕ: {l/sh:.2f}**
        """)
    
    with col2:
        fig_geom = criar_grafico_geometria(l, sh, h0, delta_H)
        st.plotly_chart(fig_geom, use_container_width=True)
        
        # Explicações
        st.markdown("""
        ### Interpretação da Geometria
        
        - **l**: Comprimento da sapata (direção do movimento)
        - **sₕ**: Altura da cunha (diferença de espessura do filme)
        - **h₀**: Folga de saída (espessura mínima do filme)
        - **H₀**: Razão adimensional que determina o desempenho
        - **ΔH**: Perturbação de deslocamento vertical
        
        O valor ótimo teórico para H₀ é aproximadamente **0.707** (1/√2).
        """)

def distribuicao_pressao():
    """Página de análise da distribuição de pressão."""
    st.markdown('<div class="section-header">📊 Distribuição de Pressão</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Parâmetros")
        Ho = st.slider("Razão H₀", 0.1, 3.0, 1.0, 0.01, key="press_Ho")
        
        # Mostrar diferentes curvas
        mostrar_multiplas = st.checkbox("Mostrar múltiplas curvas")
        
        if mostrar_multiplas:
            Ho_valores = [0.25, 0.5, 1.0, 2.0]
        else:
            Ho_valores = [Ho]
        
        # Cálculos
        Xm, Pm = encontrar_pico_pressao(Ho)
        
        st.markdown("### Resultados")
        st.metric("Posição do Pico (Xm)", f"{Xm:.3f}")
        st.metric("Pressão Máxima (Pm)", f"{Pm:.3f}")
        
        st.markdown("""
        ### Equação Base
        A distribuição de pressão é dada pela **Equação 8.24**:
        
        $$P = \\frac{6X(1-X)}{(H_0 + 1 - X)^2(1 + 2H_0)}$$
        
        O pico ocorre em:
        $$X_m = \\frac{1 + H_0}{1 + 2H_0}$$
        """)
    
    with col2:
        X_coords = np.linspace(0, 1, 500)
        
        fig = go.Figure()
        
        cores = ['royalblue', 'red', 'green', 'orange', 'purple']
        
        for i, Ho_val in enumerate(Ho_valores):
            P_valores = calcular_pressao_adimensional(X_coords, Ho_val)
            Xm_val, Pm_val = encontrar_pico_pressao(Ho_val)
            
            # Curva de pressão
            fig.add_trace(go.Scatter(
                x=X_coords, y=P_valores,
                mode='lines',
                name=f'H₀ = {Ho_val:.2f}' if mostrar_multiplas else 'Distribuição de Pressão',
                line=dict(color=cores[i], width=3)
            ))
            
            # Pico de pressão
            fig.add_trace(go.Scatter(
                x=[Xm_val], y=[Pm_val],
                mode='markers',
                name=f'Pico (H₀={Ho_val:.2f})' if mostrar_multiplas else f'Pico: {Pm_val:.3f}',
                marker=dict(color=cores[i], size=8, symbol='circle'),
                showlegend=not mostrar_multiplas or i == 0
            ))
        
        fig.update_layout(
            title='Distribuição de Pressão Adimensional',
            xaxis_title='Coordenada Adimensional X = x/l',
            yaxis_title='Pressão Adimensional P',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretação
        st.markdown("""
        ### Interpretação Física
        
        - A pressão é **máxima** próximo ao centro da sapata
        - Valores **maiores** de H₀ resultam em pressões **menores**
        - A posição do pico se desloca conforme H₀ varia
        - Pressão **zero** nas extremidades (condições de contorno)
        """)

def capacidade_carga():
    """Página de análise da capacidade de carga."""
    st.markdown('<div class="section-header">⚖️ Capacidade de Carga</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Configuração")
        Ho_atual = st.slider("H₀ Atual", 0.1, 3.0, 1.0, 0.01, key="carga_Ho")
        
        Wz_atual = calcular_carga_normal(Ho_atual)
        Ho_otimo = 1/np.sqrt(2)  # ~0.707
        Wz_otimo = calcular_carga_normal(Ho_otimo)
        
        st.markdown("### Resultados")
        st.metric("Carga Atual (Wz)", f"{Wz_atual:.3f}")
        st.metric("H₀ Ótimo", f"{Ho_otimo:.3f}")
        st.metric("Carga Ótima", f"{Wz_otimo:.3f}")
        
        eficiencia = (Wz_atual / Wz_otimo) * 100
        st.metric("Eficiência", f"{eficiencia:.1f}%")
        
        st.markdown("""
        ### Equação (8.30)
        $$W_z = 6\\ln\\left(\\frac{H_0+1}{H_0}\\right) - \\frac{12}{1+2H_0}$$
        
        ### Interpretação
        - Existe um valor **ótimo** de H₀ ≈ 0.707
        - Valores muito **baixos** ou **altos** reduzem a capacidade
        - A curva apresenta um **máximo bem definido**
        """)
    
    with col2:
        Ho_range = np.linspace(0.05, 3.0, 400)
        Wz_range = calcular_carga_normal(Ho_range)
        
        fig = go.Figure()
        
        # Curva principal
        fig.add_trace(go.Scatter(
            x=Ho_range, y=Wz_range,
            mode='lines',
            name='Capacidade de Carga',
            line=dict(color='green', width=3)
        ))
        
        # Ponto atual
        fig.add_trace(go.Scatter(
            x=[Ho_atual], y=[Wz_atual],
            mode='markers',
            name=f'Atual: H₀={Ho_atual:.3f}, Wz={Wz_atual:.3f}',
            marker=dict(color='red', size=12)
        ))
        
        # Ponto ótimo
        fig.add_trace(go.Scatter(
            x=[Ho_otimo], y=[Wz_otimo],
            mode='markers',
            name=f'Ótimo: H₀={Ho_otimo:.3f}, Wz={Wz_otimo:.3f}',
            marker=dict(color='gold', size=12, symbol='star')
        ))
        
        # Linha vertical no ótimo
        fig.add_vline(x=Ho_otimo, line_dash="dash", line_color="gold", 
                     annotation_text="H₀ Ótimo")
        
        fig.update_layout(
            title='Capacidade de Carga vs Razão de Espessura',
            xaxis_title='H₀ = h₀/sₕ',
            yaxis_title='Carga Normal Adimensional Wz',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

def forcas_atrito():
    """Página de análise das forças de atrito."""
    st.markdown('<div class="section-header">🔧 Forças de Atrito</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Configuração")
        Ho_atual = st.slider("H₀ Atual", 0.1, 3.0, 1.0, 0.01, key="atrito_Ho")
        
        Fa_atual = calcular_atrito_movel(Ho_atual)
        Fb_atual = calcular_atrito_fixo(Ho_atual)
        
        st.markdown("### Resultados")
        st.metric("Atrito Superfície Móvel (Fa)", f"{Fa_atual:.3f}")
        st.metric("Atrito Superfície Fixa (Fb)", f"{Fb_atual:.3f}")
        st.metric("Atrito Total", f"{abs(Fa_atual) + abs(Fb_atual):.3f}")
        
        # Coeficiente de atrito
        Wz_atual = calcular_carga_normal(Ho_atual)
        coef_atrito = Fa_atual / Wz_atual if Wz_atual != 0 else 0
        st.metric("μl/sₕ", f"{coef_atrito:.3f}")
        
        st.markdown("""
        ### Equações
        **Superfície Móvel (8.33):**
        $$F_a = 2\\ln\\left(\\frac{H_0+1}{H_0}\\right) + \\frac{6}{1+2H_0}$$
        
        **Superfície Fixa (8.32):**
        $$F_b = -\\left[4\\ln\\left(\\frac{H_0+1}{H_0}\\right) + \\frac{6}{1+2H_0}\\right]$$
        """)
    
    with col2:
        Ho_range = np.linspace(0.05, 3.0, 400)
        Fa_range = calcular_atrito_movel(Ho_range)
        Fb_range = calcular_atrito_fixo(Ho_range)
        
        fig = go.Figure()
        
        # Força na superfície móvel
        fig.add_trace(go.Scatter(
            x=Ho_range, y=Fa_range,
            mode='lines',
            name='Fa (Superfície Móvel)',
            line=dict(color='orange', width=3)
        ))
        
        # Força na superfície fixa
        fig.add_trace(go.Scatter(
            x=Ho_range, y=Fb_range,
            mode='lines',
            name='Fb (Superfície Fixa)',
            line=dict(color='purple', width=3)
        ))
        
        # Pontos atuais
        fig.add_trace(go.Scatter(
            x=[Ho_atual], y=[Fa_atual],
            mode='markers',
            name=f'Fa atual: {Fa_atual:.3f}',
            marker=dict(color='orange', size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=[Ho_atual], y=[Fb_atual],
            mode='markers',
            name=f'Fb atual: {Fb_atual:.3f}',
            marker=dict(color='purple', size=10)
        ))
        
        # Linha de referência zero
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig.update_layout(
            title='Forças de Atrito vs Razão de Espessura',
            xaxis_title='H₀ = h₀/sₕ',
            yaxis_title='Forças de Atrito Adimensionais',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Interpretação
        - **Fa** (laranja): Força na superfície que se move
        - **Fb** (roxo): Força na superfície fixa (sempre negativa)
        - O **sinal** indica a direção da força
        - Forças **diminuem** com o aumento de H₀
        """)

def analise_completa():
    """Página de análise completa integrada."""
    st.markdown('<div class="section-header">📈 Análise Completa do Mancal</div>', unsafe_allow_html=True)
    
    # Controles principais
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    
    with col_ctrl1:
        l = st.slider("Comprimento (l)", 50.0, 200.0, 100.0, 1.0, key="comp_l")
        sh = st.slider("Altura Cunha (sₕ)", 1.0, 50.0, 14.1, 0.1, key="comp_sh")
    
    with col_ctrl2:
        h0 = st.slider("Folga Saída (h₀)", 1.0, 50.0, 10.0, 0.1, key="comp_h0")
        delta_H = st.slider("Perturbação ΔH", -5.0, 5.0, 0.0, 0.1, key="comp_dH")
    
    with col_ctrl3:
        Ho = h0 / sh
        st.metric("H₀ = h₀/sₕ", f"{Ho:.3f}")
        st.metric("Ângulo θ", f"{np.degrees(np.arctan(sh/l)):.2f}°")
        Q = calcular_vazao_adimensional(Ho)
        st.metric("Vazão Q", f"{Q:.3f}")
    
    # Layout em grid 2x2
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Distribuição de Pressão', 'Capacidade de Carga', 
                       'Forças de Atrito', 'Comparação de Desempenho'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Dados para os gráficos
    X_coords = np.linspace(0, 1, 200)
    Ho_range = np.linspace(0.1, 3.0, 200)
    
    P_valores = calcular_pressao_adimensional(X_coords, Ho)
    Wz_range = calcular_carga_normal(Ho_range)
    Fa_range = calcular_atrito_movel(Ho_range)
    Fb_range = calcular_atrito_fixo(Ho_range)
    
    # Gráfico 1: Distribuição de Pressão
    Xm, Pm = encontrar_pico_pressao(Ho)
    fig.add_trace(go.Scatter(x=X_coords, y=P_valores, mode='lines', name='Pressão', 
                            line=dict(color='blue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[Xm], y=[Pm], mode='markers', name=f'Pmax={Pm:.3f}',
                            marker=dict(color='red', size=6)), row=1, col=1)
    
    # Gráfico 2: Capacidade de Carga
    Wz_atual = calcular_carga_normal(Ho)
    fig.add_trace(go.Scatter(x=Ho_range, y=Wz_range, mode='lines', name='Wz',
                            line=dict(color='green', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=[Ho], y=[Wz_atual], mode='markers', name=f'Atual={Wz_atual:.3f}',
                            marker=dict(color='red', size=6)), row=1, col=2)
    
    # Gráfico 3: Forças de Atrito
    fig.add_trace(go.Scatter(x=Ho_range, y=Fa_range, mode='lines', name='Fa',
                            line=dict(color='orange', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=Ho_range, y=Fb_range, mode='lines', name='Fb',
                            line=dict(color='purple', width=2)), row=2, col=1)
    
    # Gráfico 4: Vazão
    Q_range = calcular_vazao_adimensional(Ho_range)
    fig.add_trace(go.Scatter(x=Ho_range, y=Q_range, mode='lines', name='Vazão Q',
                            line=dict(color='teal', width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=[Ho], y=[Q], mode='markers', name=f'Q atual={Q:.3f}',
                            marker=dict(color='red', size=6)), row=2, col=2)
    
    # Configuração do layout
    fig.update_xaxes(title_text="X = x/l", row=1, col=1)
    fig.update_yaxes(title_text="Pressão P", row=1, col=1)
    fig.update_xaxes(title_text="H₀", row=1, col=2)
    fig.update_yaxes(title_text="Carga Wz", row=1, col=2)
    fig.update_xaxes(title_text="H₀", row=2, col=1)
    fig.update_yaxes(title_text="Forças F", row=2, col=1)
    fig.update_xaxes(title_text="H₀", row=2, col=2)
    fig.update_yaxes(title_text="Vazão Q", row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Resumo de resultados
    st.markdown("### Resumo dos Resultados")
    
    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
    
    with col_res1:
        st.markdown(f"""
        **Pressão:**
        - Máxima: {Pm:.3f}
        - Posição: {Xm:.3f}
        """)
    
    with col_res2:
        st.markdown(f"""
        **Carga:**
        - Wz: {Wz_atual:.3f}
        - Eficiência: {(Wz_atual/calcular_carga_normal(1/np.sqrt(2)))*100:.1f}%
        """)
    
    with col_res3:
        Fa_atual = calcular_atrito_movel(Ho)
        Fb_atual = calcular_atrito_fixo(Ho)
        st.markdown(f"""
        **Atrito:**
        - Fa: {Fa_atual:.3f}
        - Fb: {Fb_atual:.3f}
        """)
    
    with col_res4:
        st.markdown(f"""
        **Outros:**
        - Vazão Q: {Q:.3f}
        - μl/sₕ: {Fa_atual/Wz_atual if Wz_atual != 0 else 0:.3f}
        """)
    
    # Recomendações
    st.markdown("### Recomendações de Design")
    
    Ho_otimo = 1/np.sqrt(2)
    if abs(Ho - Ho_otimo) < 0.05:
        st.success(f"✅ Configuração próxima ao ótimo! H₀ = {Ho:.3f} (ótimo ≈ {Ho_otimo:.3f})")
    elif Ho < Ho_otimo - 0.1:
        st.warning(f"⚠️ H₀ muito baixo. Considere aumentar h₀ ou diminuir sₕ para melhorar a capacidade de carga.")
    elif Ho > Ho_otimo + 0.1:
        st.info(f"ℹ️ H₀ alto. Configuração estável, mas capacidade de carga pode ser melhorada diminuindo H₀.")
    
    if Pm > 2.0:
        st.warning("⚠️ Pressão máxima alta. Verifique se o material suporta esta carga.")
    
    if abs(Fa_atual) > 5.0:
        st.warning("⚠️ Força de atrito elevada. Considere otimizar a geometria para reduzir perdas.")

if __name__ == "__main__":
    main()