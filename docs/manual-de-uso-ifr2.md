# Manual de Uso Completo — IFR2 Miner & Screener

Versão do manual: 1.0
Gerado em: 2026-05-20 00:07:16 -03
Projeto: /mnt/c/Dev/IFR2

## 1) O que é este app

O IFR2 Miner & Screener é uma aplicação Streamlit para análise quantitativa de ações.

Ele tem dois objetivos principais:
1. Encontrar ativos em possível ponto de entrada com base na estratégia IFR2.
2. Comparar historicamente o desempenho da estratégia em lote (backtest), contra buy & hold.

Mercados suportados:
- B3 (Brasil)
- NYSE/NASDAQ (EUA)
- Modo combinado (B3 + EUA)

## 2) Aviso importante

Este sistema é uma ferramenta de análise. Não executa ordens de compra/venda.
Retorno passado não garante retorno futuro.

## 3) Requisitos

- Python 3.12+
- Ambiente virtual recomendado (venv)
- Dependências do projeto instaladas
- Acesso à internet para baixar dados via yfinance

## 4) Instalação e execução

No diretório do projeto:

```bash
cd /mnt/c/Dev/IFR2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Rodar o app:

```bash
./venv/bin/streamlit run ifr2_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true --browser.gatherUsageStats false --server.fileWatcherType none
```

Acesso no navegador:
- http://localhost:8501

## 5) Estrutura funcional do sistema

Entrada principal:
- `ifr2_app.py`

Módulos de domínio:
- `src/universe_registry.py` — carrega/salva snapshots dos universos
- `src/data_fetcher.py` — download de dados em lote/paralelo
- `src/indicators.py` — cálculo de IFR2, SMA200, volume financeiro
- `src/screener.py` — lógica de varredura
- `src/backtester.py` — lógica de backtest da estratégia
- `src/risk_manager.py` — utilitários de risco (cálculo de posição, stop, alvo)

Configuração:
- `config/settings.py`

## 6) Conceitos de estratégia usados no app

### 6.1 Indicadores

O cálculo (`calc_indicators`) adiciona:
- `IFR2`: RSI de 2 períodos (Wilder)
- `SMA200`: média móvel simples de 200 períodos
- `Bullish`: `Close > SMA200`
- `Vol_Fin`: `Close * Volume`
- `Vol_Fin_Medio`: média de 21 períodos do volume financeiro
- `Max_Prev`: máxima anterior (`High.shift(1)`)

Observação: se o ativo tiver menos de 200 candles, os indicadores estratégicos não são processados.

### 6.2 Regra do Screener

Um ativo entra com sinal `COMPRA!` quando, no candle mais recente:
- `IFR2 < threshold escolhido`
- `Bullish == True` (preço acima da SMA200)
- `Vol_Fin_Medio >= limite de liquidez`

### 6.3 Regra do Backtest

Compra quando:
- `IFR2 < threshold`
- e `Bullish == True`

Sai da posição quando:
- `IFR2 > 70`
- ou `Close > Max_Prev`

Métricas calculadas:
- Win Rate
- Profit Factor
- Expectativa Matemática
- Número de Trades
- Retorno do sistema
- Retorno buy & hold
- Alpha (sistema - buy & hold)
- Curva de capital

## 7) Como usar a interface (passo a passo)

### 7.1 Barra lateral (Sidebar)

1. Escolha o mercado:
   - `B3`
   - `NYSE/NASDAQ`
   - `Ambos`

2. Escolha a cesta de ativos:
   - B3:
     - Top 100 B3 (Snapshot)
     - IBOVESPA (Legado)
     - SMLL (Legado)
     - Customizado
   - EUA:
     - Top 500 US (Snapshot)
     - Customizado
   - Ambos:
     - B3 Top 100 + US Top 500 (Snapshots)
     - Customizado

3. Refine no multiselect `Selecione/Filtre os Ativos para Execução`.

4. Defina os parâmetros:
   - `Threshold IFR2 de Compra` (slider: 1 a 30; padrão: 10)
   - Liquidez mínima (depende do mercado):
     - B3: `Volume Fin. Médio Mínimo (R$)`
     - EUA: `Volume Fin. Médio Mínimo (US$)`
     - Ambos: um limite para BRL e outro para USD
   - `Período Histórico (Dados)`: 1y, 2y, 3y, 5y

### 7.2 Aba 1 — Screener Diário

1. Clique em `Executar Varredura`.
2. O app baixa os dados com barra de progresso.
3. O app calcula indicadores e aplica filtros.
4. Resultado:
   - Tabela com sinais por ativo
   - Coluna `Sinal` pode mostrar `COMPRA!` ou `Neutro`
   - Mensagem de conclusão com tempo e quantidade de falhas de download (se houver)

Colunas principais da saída:
- Ticker
- Preço
- Moeda
- Mercado
- IFR2
- Acima SMA200
- Vol Fin Médio
- Sinal

### 7.3 Aba 2 — O Garimpo (Mining)

1. Clique em `Iniciar Mineração`.
2. O app baixa dados e roda backtest por ativo.
3. Só entram no ranking os ativos com:
   - liquidez mínima atendida
   - pelo menos 1 trade válido no backtest

Saída:
- Ranking ordenado por `Alpha %` (desc)
- Colunas de performance (Win Rate, Profit Factor, Retorno etc.)
- Gráfico com curvas de capital dos Top 5

## 8) Universos de ativos (snapshots)

Os universos não ficam hardcoded na UI principal. São carregados via snapshots CSV.

Arquivos canônicos:
- `data/universes/b3_top100.csv`
- `data/universes/us_top500.csv`

Histórico versionado:
- `data/universes/history/b3_top100/`
- `data/universes/history/us_top500/`

## 9) Atualizar universos

Script oficial:
- `scripts/refresh_universes.py`

Dry-run (valida sem gravar):

```bash
PYTHONPATH=. ./venv/bin/python scripts/refresh_universes.py --dry-run
```

Atualização real (grava canônico + versionado):

```bash
PYTHONPATH=. ./venv/bin/python scripts/refresh_universes.py
```

Parâmetros úteis:
- `--b3-source`
- `--us-source`
- `--us-fallback-source`
- `--project-root`
- `--dry-run`

Observação crítica:
- O pipeline B3 normaliza ticker para sufixo `.SA` para compatibilidade com Yahoo Finance.

## 10) Testes e qualidade

Executar testes:

```bash
PYTHONPATH=. ./venv/bin/pytest -q
```

Smoke test:

```bash
./venv/bin/python smoke_test.py
```

Integration test:

```bash
./venv/bin/python integration_test.py
```

Linter:

```bash
./venv/bin/python -m flake8 ifr2_app.py smoke_test.py integration_test.py src/ config/ scripts/ tests/
```

Formatador:

```bash
./venv/bin/python -m black ifr2_app.py smoke_test.py integration_test.py src/ config/ scripts/ tests/
```

Guardian (gate de entrega):

```bash
./venv/bin/python auditoria/guardian/guardian.py
```

## 11) Benchmark de escala

Script:

```bash
PYTHONPATH=. ./venv/bin/python benchmark_scale.py
```

Saída de benchmark salva em:
- `benchmark_report.md`

## 12) Solução de problemas (troubleshooting)

### Problema: app não abre em localhost:8501

Verificar porta:

```bash
ss -ltnp | grep :8501 || true
```

Se já tiver outro Streamlit na porta, finalize o processo e rode novamente.

### Problema: muitos ativos com falha de download

Causas comuns:
- instabilidade da fonte (Yahoo)
- ticker sem dados/disponibilidade temporária
- rate limit

Ações:
- reduza quantidade de ativos (multiselect)
- use período menor (1y)
- repita a execução
- valide snapshots atualizados

### Problema: ativo B3 não retorna dados

Confirme formato com `.SA` (ex.: `PETR4.SA`).

### Problema: resultado vazio no screener

Isso pode ser normal se:
- IFR2 não estiver abaixo do threshold
- ativo estiver abaixo da SMA200
- liquidez mínima estiver alta demais

## 13) Boas práticas operacionais

1. Atualizar universos antes de análises grandes.
2. Começar com subset de ativos para teste rápido.
3. Só depois rodar universo completo.
4. Guardar evidências de execução (benchmark e relatórios de auditoria).
5. Sempre rodar `pytest` antes de fechar sprint.

## 14) Comandos rápidos (cola)

Subir app:

```bash
cd /mnt/c/Dev/IFR2
source venv/bin/activate
./venv/bin/streamlit run ifr2_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true --browser.gatherUsageStats false --server.fileWatcherType none
```

Atualizar universos:

```bash
PYTHONPATH=. ./venv/bin/python scripts/refresh_universes.py
```

Rodar validação completa:

```bash
PYTHONPATH=. ./venv/bin/pytest -q
./venv/bin/python -m flake8 ifr2_app.py smoke_test.py integration_test.py src/ config/ scripts/ tests/
./venv/bin/python -m black --check ifr2_app.py smoke_test.py integration_test.py src/ config/ scripts/ tests/
./venv/bin/python auditoria/guardian/guardian.py
```

---

Se quiser, no próximo passo eu também posso gerar uma versão "manual rápido de 1 página" (somente operação diária) para deixar junto deste manual completo.