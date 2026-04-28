import streamlit as st
import numpy as np
# Importa os módulos da pasta utils
from utils import calculos, visualizacoes

st.set_page_config(layout="wide", page_title="Geometria do Mancal")

# Título da Página
st.markdown("## 📐 Geometria Interativa do Mancal")
st.markdown("---")
st.markdown("""
Nesta página, você pode manipular os parâmetros geométricos fundamentais de um mancal de escora de inclinação fixa e observar o impacto visual imediato. Use os sliders para explorar como `l`, `sₕ`, e `h₀` definem a forma da cunha de lubrificação.
""")

# Layout com duas colunas
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("#### Controles da Geometria")
    
    # Sliders para os parâmetros base
    l = st.slider("Comprimento da Sapata (l)", min_value=10.0, max_value=200.0, value=100.0, step=1.0)
    sh = st.slider("Altura da Cunha (sₕ)", min_value=0.1, max_value=50.0, value=14.1, step=0.1)
    h0 = st.slider("Folga de Saída (h₀)", min_value=0.1, max_value=50.0, value=10.0, step=0.1)
    
    # Cálculos derivados
    Ho = h0 / sh
    Ho_otimo = calculos.encontrar_Ho_otimo()
    
    st.markdown("---")
    st.markdown("#### Parâmetros Derivados")
    
    # Exibe os resultados
    st.metric("Razão de Espessura (H₀)", f"{Ho:.3f}")
    
    if abs(Ho - Ho_otimo) < 0.05:
        st.success(f"Configuração Próxima da Ótima! (H₀ ótimo ≈ {Ho_otimo:.3f})")
    
    st.metric("Ângulo da Cunha (θ)", f"{np.degrees(np.arctan(sh/l)):.2f}°")

with col2:
    # Chama a função de plotagem do nosso módulo de visualização
    fig_geometria = visualizacoes.criar_grafico_geometria(l, sh, h0)
    st.plotly_chart(fig_geometria, use_container_width=True)