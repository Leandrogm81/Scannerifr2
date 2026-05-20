# IFR2 Miner & Screener

## 📋 Visão Geral
Aplicação web para screening, backtesting, análise técnica e gerenciamento de risco de ações. Operando tanto na **B3 (Brasil)** quanto na **NYSE/NASDAQ (Estados Unidos)** de forma unificada e modular.

## 🚀 Instalação e Execução

### 1. Ambiente Virtual (Recomendado)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Execute a aplicação
```bash
streamlit run ifr2_app.py
```

## 📁 Estrutura de Pastas
```
IFR2/
├── ifr2_app.py              # Entry point (Streamlit UI - 100% Desacoplado)
├── requirements.txt         # Dependências do projeto
├── data/
│   └── universes/           # Snapshots em CSV (Top 100 B3, Top 500 US)
├── scripts/
│   └── refresh_universes.py # Coletor para atualização dos snapshots
├── src/                     # Módulos de lógica de negócio centralizados
│   ├── universe_registry.py # Gerenciador de Universos (B3/US)
│   ├── data_fetcher.py      # Busca paralela em lotes e cache (yfinance)
│   ├── indicators.py        # Cálculo de indicadores técnicos
│   ├── backtester.py        # Lógica de backtesting histórico
│   ├── screener.py          # Lógica de screening multi-moeda
│   ├── risk_manager.py      # Gerenciamento de risco
│   └── __init__.py          # Pacote Python
├── config/                  # Configurações e constantes
│   └── settings.py          # Parâmetros configuráveis
├── auditoria/               # Histórico e scripts do Cão de Guarda
└── tests/                   # Suíte de testes automatizados (pytest)
```

## 🔧 Funcionalidades Principais

### Módulo 1: Screener Multi-Mercado
- Varredura em tempo real para ativos na B3 e EUA.
- Filtros de liquidez com normalização cambial inteligente (BRL vs USD).
- Execução em lote (`batch_size`) e multi-threading para máximo desempenho.

### Módulo 2: O Garimpo (Backtester em Lote)
- Execução maciça de backtest em todos os ativos (até 600) simultaneamente.
- Métricas consolidadas: Win Rate, Profit Factor, Alpha (vs Buy&Hold).
- Plotagem de gráficos com evolução do patrimônio.

### Módulo 3: Gerenciamento de Universos (Registry)
- Abandonou as antigas listas 'hardcoded'.
- Universos B3 e EUA são mantidos de forma resiliente por snapshots versionados em disco, permitindo funcionamento offline parcial e reprodutibilidade.

## 📊 Roadmap de Desenvolvimento

| Fase | Épico | Status |
|-----------|-------|--------|
| Sprint 01 | Universe Registry | ✅ Concluído |
| Sprint 02 | Fetch Pipeline & UI | ✅ Concluído |
| Sprint 03 | Market Normalization| ✅ Concluído |
| Sprint 04 | Testes e Performance| ✅ Concluído |

## ⚙️ Tecnologias

- **Linguagem:** Python 3.12+
- **Bibliotecas Base:** pandas, numpy, yfinance, plotly
- **Interface:** Streamlit
- **Testes & Qualidade:** pytest, flake8, black

## 📝 Documentação Adicional
- [Documento de Requisitos (PRD Técnicos)](./prd/IFR2_PRD_Technical.md)
- [Relatórios de Auditoria e Cão de Guarda](./auditoria/README.md)

## 📧 Contato
Leandro Gobbo Menezes - leandrogmzl@gmail.com
