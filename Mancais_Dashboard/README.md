# Dashboard de Análise de Mancais Hidrodinâmicos

## Descrição
Este dashboard interativo foi desenvolvido para análise de **Mancais de Escora de Inclinação Fixa**, baseado no trabalho de Iniciação Científica sobre lubrificação hidrodinâmica e nas equações do livro "Hydrodynamic Lubrication" de Hamrock.

## Estrutura do Projeto
```
mancais_dashboard/
├── main.py                    # Aplicação principal
├── requirements.txt           # Dependências
├── utils/
│   ├── __init__.py
│   ├── calculos.py           # Funções de cálculo
│   └── visualizacoes.py      # Funções de visualização
└── README.md
```

## Funcionalidades

### Páginas Principais
- **🏠 Dashboard Principal**: Visão geral com métricas principais
- **📐 Geometria Interativa**: Visualização 3D da geometria do mancal
- **📊 Distribuição de Pressão**: Análise da pressão ao longo da superfície
- **⚖️ Capacidade de Carga**: Otimização da capacidade de carga
- **🔧 Forças de Atrito**: Análise das forças resistivas
- **📈 Análise Completa**: Dashboard integrado com todos os parâmetros

### Parâmetros Analisados
- **Geometria**: Comprimento (l), Altura da cunha (sₕ), Folga de saída (h₀)
- **Adimensionais**: H₀ = h₀/sₕ, Pressão P, Carga Wz, Forças Fa e Fb
- **Desempenho**: Vazão Q, Coeficiente de atrito μl/sₕ
- **Dinâmicos**: Rigidez K, Amortecimento C

## Instalação e Execução

### Pré-requisitos
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Passo 1: Clone/Download do projeto
```bash
# Se usando Git
git clone <url-do-repositorio>
cd mancais_dashboard

# Ou baixe os arquivos e extraia em uma pasta
```

### Passo 2: Criar ambiente virtual (recomendado)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Passo 3: Instalar dependências
```bash
pip install -r requirements.txt
```

### Passo 4: Executar a aplicação
```bash
streamlit run main.py
```

A aplicação será aberta automaticamente no navegador em `http://localhost:8501`

## Uso do Dashboard

### Navegação
- Use a **barra lateral** para navegar entre as páginas
- Cada página tem **sliders interativos** para ajustar parâmetros
- Os gráficos são **atualizados em tempo real**

### Interpretação dos Resultados

#### Parâmetro H₀ (Razão de Espessura)
- **Valor ótimo**: ≈ 0.707 (1/√2) para máxima capacidade de carga
- **H₀ < 0.3**: Configuração instável, alta pressão
- **H₀ > 2.0**: Baixa capacidade de carga

#### Distribuição de Pressão
- **Pico de pressão** ocorre próximo ao centro da sapata
- **Posição do pico** varia com H₀
- **Pressão zero** nas extremidades (condições de contorno)

#### Capacidade de Carga (Wz)
- **Curva com máximo** bem definido em H₀ ≈ 0.707
- **Eficiência** = (Wz atual / Wz ótimo) × 100%

#### Forças de Atrito
- **Fa** (laranja): Força na superfície móvel
- **Fb** (roxo): Força na superfície fixa (sempre negativa)
- **Magnitude** diminui com aumento de H₀

### Dicas de Uso
1. **Comece** pelo Dashboard Principal para visão geral
2. **Ajuste** os parâmetros e observe as mudanças em tempo real
3. **Compare** diferentes configurações usando múltiplas curvas
4. **Use** as métricas de eficiência para otimização

## Equações Implementadas

### Distribuição de Pressão (Eq. 8.24)
```
P = 6X(1-X) / [(H₀ + 1 - X)²(1 + 2H₀)]
```

### Capacidade de Carga (Eq. 8.30)
```
Wz = 6ln((H₀+1)/H₀) - 12/(1+2H₀)
```

### Forças de Atrito (Eq. 8.32, 8.33)
```
Fa = 2ln((H₀+1)/H₀) + 6/(1+2H₀)
Fb = -[4ln((H₀+1)/H₀) + 6/(1+2H₀)]
```

### Vazão Volumétrica (Eq. 8.36)
```
Q = 2H₀(1+H₀)/(1+2H₀)
```

## Desenvolvimento e Extensões

### Estrutura Modular
- **calculos.py**: Todas as funções matemáticas
- **visualizacoes.py**: Funções especializadas de gráficos
- **main.py**: Interface e lógica de apresentação

### Possíveis Extensões
1. **Novos tipos de mancais** (pivô fixo, pivô móvel)
2. **Análise de estabilidade** dinâmica
3. **Otimização automática** de parâmetros
4. **Exportação de relatórios** em PDF
5. **Análise de sensibilidade** paramétrica
6. **Interface em outros idiomas**

### Contribuindo
Para adicionar novas funcionalidades:
1. **Equações**: Adicione em `calculos.py`
2. **Visualizações**: Implemente em `visualizacoes.py`
3. **Interface**: Crie nova página em `main.py`

## Deployment (Streamlit Cloud)

### Passo 1: Preparar repositório
```bash
git add .
git commit -m "Dashboard completo"
git push origin main
```

### Passo 2: Deploy no Streamlit Cloud
1. Acesse [share.streamlit.io](https://share.streamlit.io)
2. Conecte seu repositório GitHub
3. Configure: `main.py` como main file
4. Deploy automático

### Passo 3: Compartilhar
A aplicação ficará disponível em uma URL pública para compartilhamento.

## Solução de Problemas

### Erro de importação
```bash
# Reinstalar dependências
pip install --upgrade -r requirements.txt
```

### Porta em uso
```bash
# Especificar porta diferente
streamlit run main.py --server.port 8502
```

### Gráficos não carregam
- Verifique conexão com internet (Plotly CDN)
- Limpe cache do navegador

## Referências
- Hamrock, B. J. "Hydrodynamic Lubrication"
- Trabalho de Iniciação Científica sobre Mancais Hidrodinâmicos
- Documentação Streamlit: [docs.streamlit.io](https://docs.streamlit.io)
- Documentação Plotly: [plotly.com/python](https://plotly.com/python)

## Contato e Suporte
Para dúvidas sobre o uso ou desenvolvimento, consulte:
- Documentação técnica dos mancais
- Código-fonte comentado
- Issues no repositório GitHub

---

**Desenvolvido como ferramenta educacional para análise de mancais hidrodinâmicos**