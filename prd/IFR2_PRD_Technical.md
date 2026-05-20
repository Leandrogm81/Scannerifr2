---
title: "Product Requirements Document - IFR2 Miner & Screener"
author: "Leandro Gobbo Menezes"
email: "leandrogmzl@gmail.com"
version: "2.0.0"
status: "em-desenvolvimento"
created_date: "2026-05-20"
last_updated: "2026-05-20"
description: "Documentação completa de requisitos do projeto IFR2 Miner & Screener"
---

# Product Requirements Document - IFR2 Miner & Screener

## 📋 Informações do Projeto

| Campo | Valor |
|-------|-------|
| **Projeto** | IFR2 Miner & Screener |
| **Versão** | 2.0.0 |
| **Autor** | Leandro Gobbo Menezes |
| **Email** | leandrogmzl@gmail.com |
| **Data** | 20 de Maio de 2026 |
| **Última Atualização** | 20 de Maio de 2026 |
| **Escopo** | Aplicações web para screening, backtesting e análise técnica de ações da B3 e mercado americano |
| **Status** | 🟢 Em Desenvolvimento |

## 🎯 Visão Geral

**IFR2 Miner & Screener** é uma aplicação web que ajuda investidores a identificar oportunidades no mercado financeiro através de análise técnica quantitativa. A ferramenta combina:

- **Screener em tempo real** - Varredura diária de ativos em condições de compra/venda
- **Backtesting automatizado** - Validação histórica de estratégias de investimento
- **Analytics avançado** - Métricas de desempenho e risco
- **Suporte multi-mercado** - B3 (Brasil) e NYSE/NASDAQ (EUA)

### ⚠️ Importante

Esta aplicação é uma **ferramenta de análise** apenas. Não executa ordens de compra/venda no mercado financeiro. Todas as decisões de investimento devem ser feitas pelo usuário final, preferencialmente com o auxílio de um profissional qualificado.

## 📈 Objetivos de Negócio

### Objetivos Primários (12-18 meses):

1. **✅ Aumentar a precisão das recomendações** - Reduzir sinais falsos através de filtros adicionais
2. **✅ Expandir para mercado americano** - Suporte completo a NYSE e NASDAQ
3. **⚠️ Melhorar experiência do usuário** - Interface mais intuitiva e responsiva
4. **⚠️ Adicionar gerenciamento de risco** - Stop loss, position sizing, diversificação
5. **🔧 Integração com APIs de notícias** - Análise de sentimento e eventos

### Metas SMART:

| Métrica | Meta Q4 2026 | Como Medir |
|---------|--------------|------------|
| **Usuários ativos mensais** | 500+ | Analytics (PostHog/Mixpanel) |
| **Taxa de retenção** | 40% | Usuários retornando após 30 dias |
| **Tempo médio sessão** | 8+ minutos | Google Analytics |
| **Conversão para premium** | 5% | Cadastro em newsletter/feature paga |

## 👥 Público-Alvo

### Personas:

#### Roberto - Investidor Autônomo (35 anos)
- **Profissão:** Analista de Sistemas
- **Experiência:** 5 anos investindo na B3
- **Objetivo:** Identificar oportunidades diárias com base em análise técnica
- **Dores:** Falta de tempo para análise manual, muitos sinais falsos
- **Necessidades:** Screening rápido, backtesting confiável, métricas claras

#### Clara - Assessora de Investimentos (42 anos)
- **Profissão:** Assessora de Investimentos (XP)
- **Experiência:** 12 anos no mercado financeiro
- **Objetivo:** Ferramenta quantitativa para complementar análise fundamentalista
- **Dores:** Viés emocional, necessidade de dados objetivos
- **Necessidades:** Métricas de risco, comparativo com índices, relatórios profissionais

## ✨ Funcionalidades Principais

### Módulo 1: Screener em Tempo Real
- **Varredura diária** - IFR2 < threshold + Preço > SMA200 + Volume adequado
- **Filtros customizados** - Por setor, liquidez, mercado (B3/EUA)
- **Resultados em tempo real** - Tabela interativa com destaque para sinais
- **Exportação** - CSV e Excel com formatação profissional

### Módulo 2: Backtester Avançado
- **Backtest histórico** - Até 5 anos de dados
- **Métricas completas** - Win rate, profit factor, Sharpe ratio, drawdown
- **Comparativo com buy & hold** - Alpha e beta calculation
- **Curvas de capital** - Visualização da evolução do patrimônio

### Módulo 3: Analytics & Insights
- **Ranking de ativos** - Ordenado por métricas (Alpha, Win Rate, etc.)
- **Heatmaps setoriais** - Performance por setor/indústria
- **Relatórios PDF** - Com análise quantitativa pronta para cliente
- **Exportação de gráficos** - PNG e SVG para apresentações

### Módulo 4: Gerenciamento de Risco
- **Position sizing** - Baseado em capital e stop loss
- **Stop loss automático** - Suporte para trailing stop
- **Limites de exposição** - Setorial e total
- **Monitoramento de drawdown** - Alertas quando exceder limites

## 📊 Roadmap de Desenvolvimento

| Trimestre | Épico | Features | Status |
|-----------|-------|----------|--------|
| **Q3 2026** | Refatoração | - Separação em módulos<br>- Testes unitários<br>- Logging estruturado | 🟢 Em Andamento |
| **Q4 2026** | Multi-mercado | - Suporte a NYSE/NASDAQ<br>- Calendários de feriados<br>- Conversão de moeda | ✅ Concluído |
| **Q1 2027** | Risco & Analytics | - Position sizing<br>- Stop loss<br>- Relatórios PDF | 🔴 Não Iniciado |
| **Q2 2027** | Integrações | - APIs de notícias<br>- Sentiment analysis<br>- Email/Telegram alerts | 🔴 Não Iniciado |

## ⚙️ Requisitos Técnicos

### Backend:

| Tecnologia | Versão | Propósito |
|------------|--------|-----------|
| **Python** | 3.11+ | Linguagem principal |
| **Streamlit** | 1.28+ | Framework web |
| **yfinance** | 0.2+ | Fonte de dados |
| **pandas** | 2.0+ | Manipulação de dados |
| **plotly** | 5.13+ | Visualização de gráficos |
| **celery + redis** | - | Processamento assíncrono (futuro) |

### Frontend:

| Tecnologia | Propósito |
|------------|-----------|
| **HTML/CSS/JavaScript** | Customizações |
| **Bootstrap 5** | Componentes responsivos |
| **Plotly.js** | Gráficos interativos |

### Infraestrutura:

| Componente | Propósito |
|------------|-----------|
| **Docker** | Containerização |
| **Docker Compose** | Orquestração local |
| **PostgreSQL + TimescaleDB** | Dados históricos |
| **Redis** | Cache e filas |
| **NGINX** | Reverse proxy |

### Deploy:

| Plataforma | Finalidade |
|------------|------------|
| **Vercel / Heroku** | Opções iniciais |
| **AWS EC2 / GCP** | Para produção |
| **Docker Hub** | Registro de imagens |

## 📈 Métricas de Sucesso

### Business Metrics:

| Métrica | Meta Q4 2026 | Como Medir |
|---------|--------------|------------|
| **Usuários ativos mensais** | 500+ | Analytics (PostHog/Mixpanel) |
| **Taxa de retenção** | 40% | Usuários retornando após 30 dias |
| **Tempo médio sessão** | 8+ minutos | Google Analytics |
| **Conversão para premium** | 5% | Cadastro em newsletter/feature paga |

### Technical Metrics:

| Métrica | Meta | Ferramenta |
|---------|------|------------|
| **Tempo de resposta (P95)** | < 3s | Prometheus + Grafana |
| **Uptime** | > 99.5% | UptimeRobot |
| **Erro rate** | < 0.1% | Sentry |
| **Backtest speed** | < 10s/100 ativos | Benchmark interno |

## ⚠️ Riscos e Mitigações

| Risco | Impacto | Probabilidade | Mitigação |
|-------|---------|---------------|-----------|
| **Alterações na API yfinance** | Alto | Média | - Fallback para Alpha Vantage/IEX Cloud<br>- Caching robusto de dados históricos<br>- Monitoramento de status da API |
| **Regulatório (CVM)** | Alto | Baixa | - Isenção clara de responsabilidade<br>- Não armazenar dados de clientes<br>- Consultoria jurídica especializada |
| **Concorrência (TradingView, etc.)** | Médio | Alta | - Foco em simplicidade e custo (gratuito)<br>- Comunidade open source<br>- Integração com workflow existente |
| **Performance com muitos usuários** | Médio | Média | - Arquitetura assíncrona desde o início<br>- Cache distribuído<br>- Auto-scaling na nuvem |
| **Dados incorretos/outdated** | Baixo | Alta | - Data de última atualização visível<br>- Cache com TTL de 1h<br>- Validação de dados (ex: volume > 0) |

## 🖥️ Wireframes e Mockups

### Layout Principal - Screener:
```
+-------------------------------------------------+
| IFR2 Miner & Screener v2.0                      |
| [Logo]                                          |
+-------------------------------------------------+
| Sidebar (Filtros)                               |
| - Mercado: [B3] [EUA] [Ambos]                   |
| - Setor: [Todos] [Financeiro] [Tecnologia]...   |
| - Liquidez: [> 1M] [> 5M] [> 10M]              |
| - IFR2: [5] [10] [15]                           |
| [Aplicar] [Resetar]                             |
+-------------------------------------------------+
| Tabela de Resultados                            |
| +----------------------------------------------+ |
| | Ticker | Preço | IFR2 | SMA200 | Volume | Sinal | |
| +----------------------------------------------+ |
| | PETR4.SA | R$ 18.45 | 8.2 | Sim | R$ 2.1B | COMPRA | |
| | VALE3.SA | R$ 68.30 | 12.5 | Sim | R$ 1.8B | COMPRA | |
| | ITUB4.SA | R$ 32.10 | 22.4 | Sim | R$ 1.2B | NEUTRO | |
| +----------------------------------------------+ |
+-------------------------------------------------+
```

### Backtester - Curva de Capital:
```
Curvas de Capital - Top 5 Ativos
(comp para 5 tickers diferentes mostrando evolução do capital)
```

### Analytics Dashboard:
```
Dashboard de Analytics
- Ranking de Ativos (tabela)
- Heatmap Setorial (gráfico de cores)
- Performance vs IBOV (gráfico de linhas)
- Distribuição de Retornos (histograma)
```

## 📚 Referências e Anexos

### Códigos Importantes:

| Código | Descrição |
|--------|-----------|
| **Ticker B3** | PETR4.SA, VALE3.SA, ITUB4.SA, BBDC4.SA |
| **Ticker EUA** | AAPL, TSLA, AMZN, GOOGL, MSFT |
| **Índices** | ^BVSP (IBOV), ^GSPC (S&P500) |
| **ETF** | SPY, QQQ, IWM |

### APIs Utilizadas:

| API | Finalidade |
|-----|------------|
| **yfinance** | Fonte de dados históricos |
| **Streamlit** | Framework web |
| **Alpha Vantage** | Fallback (opcional) |
| **IEX Cloud** | Fallback premium (futuro) |

### Glossário:

| Termo | Significado |
|-------|-------------|
| **IFR2** | Relative Strength Index de 2 períodos |
| **SMA200** | Média móvel simples de 200 períodos |
| **Alpha** | Excesso de retorno vs benchmark |
| **Sharpe Ratio** | Retorno ajustado por risco |
| **Drawdown** | Perda máxima desde o pico |

---

## 📋 Checklist de Implementação

- [x] Criar estrutura de pastas para PRD
- [x] Escrever versão HTML (formato leigo)
- [x] Escrever versão Markdown (formato técnico)
- [ ] Revisar com stakeholder
- [ ] Aprovar versão final
- [ ] Iniciar desenvolvimento

---

**Próximos Passos:**
1. Revisar este PRD com o time
2. Criar user stories e tarefas no Jira
3. Iniciar sprint de refatoração
4. Implementar suporte a mercado americano

---

*Product Requirements Document - IFR2 Miner & Screener v2.0.0*  
*Desenvolvido para: Leandro Gobbo Menezes | leandrogmzl@gmail.com*  
*Copyright © 2026 - Todos os direitos reservados*