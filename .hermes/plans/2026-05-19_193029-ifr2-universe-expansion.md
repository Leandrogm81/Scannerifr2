# IFR2 Miner & Screener — Plano de Expansão do Universo

> Assunção atual: B3 = 100 ativos mais líquidos; EUA = 500 large caps usando S&P 500 como proxy pública e estável. Se você quiser outro critério, o plano deve ser ajustado antes da execução.

**Goal:** Expandir o IFR2 Miner & Screener de listas hardcoded e amostrais para um universo grande, atualizável e consistente, cobrindo 100 ativos da B3 e 500 ativos americanos, sem misturar moedas nem duplicar lógica na interface.

**Architecture:** Criar uma fonte única de universos em arquivos de dados + módulo Python de carregamento, refatorar a UI para consumir esse núcleo, separar regras de mercado/moeda do cálculo de indicadores e fechar o ciclo com testes e benchmark. O app continua em Streamlit, mas a lógica deixa de morar na tela.

**Tech Stack:** Python, Streamlit, pandas, yfinance, plotly, pytest. Dependências extras só se forem realmente necessárias para ler a fonte pública escolhida.

---

## Contexto verificado
- O entrypoint real é `ifr2_app.py`.
- `src/data_fetcher.py`, `src/indicators.py`, `src/backtester.py`, `src/screener.py` e `src/risk_manager.py` já existem.
- A UI ainda faz parte da lógica inline em vez de usar tudo o que já foi modularizado.
- `config/settings.py` tem `TOP100_B3 = DEFAULT_IBOV + DEFAULT_SMLL`, que hoje soma 30 tickers, não 100.
- Os universos americanos são listas de exemplo com 10 tickers.
- `Vol_Fin` e `Preço (R$)` hoje aparecem como BRL mesmo para tickers dos EUA, o que é enganoso.
- Não existe pasta `tests/` ainda; há só scripts soltos de smoke/integration.
- A busca de dados usa somente yfinance e não há cache funcional.
- Já existem fontes públicas viáveis para o universo: Dados de Mercado para B3 e DataHub para S&P 500.

## Objetivo do produto
O usuário quer:
1. pesquisar em um universo muito maior,
2. manter a experiência simples dentro do Streamlit,
3. ter B3 e EUA no mesmo app,
4. ver resultados confiáveis, com unidades corretas e sem surpresas na performance.

## Proposta de solução
1. Criar uma camada de universo:
   - carrega listas de tickers a partir de snapshots em `data/universes/`
   - registra mercado, origem, atualização e tamanho esperado
   - permite refresh controlado por script
2. Fazer a UI consumir essa camada:
   - nada de listas hardcoded na tela
   - o seletor mostra contagem real
   - B3, EUA e customizado continuam disponíveis
3. Tornar a coleta escalável:
   - batches menores
   - cache para dados e universos
   - erros parciais sem derrubar a tela toda
4. Ajustar a parte de mercado:
   - moeda e rótulos por mercado
   - possibilidade de modo combinado sem confundir BRL com USD
5. Fechar com CoVe:
   - validar contagem
   - validar labels
   - validar amostras
   - validar performance com a nova escala

## Modelo de dados proposto
### UniverseDefinition
```python
@dataclass(frozen=True)
class UniverseDefinition:
    code: str
    label: str
    market: str           # "B3" | "US"
    source_url: str
    snapshot_path: Path
    expected_size: int
    currency: str         # "BRL" | "USD"
    updated_at: str
```

### TickerMetadata
```python
@dataclass(frozen=True)
class TickerMetadata:
    ticker: str
    market: str
    currency: str
    source: str
    rank: int | None
    name: str | None = None
    sector: str | None = None
```

### ScreeningResult
- ticker
- market
- currency
- close
- ifr2
- volume_financeiro
- sinal
- liquidez
- observacao_erro

## Arquivos que provavelmente vão mudar
- `ifr2_app.py`
- `config/settings.py`
- `src/data_fetcher.py`
- `src/screener.py`
- `src/backtester.py`
- `src/indicators.py`
- `src/risk_manager.py` (se houver normalização/limites por mercado)
- `src/universe_registry.py` ou `src/universe.py` (novo)
- `src/market_normalizer.py` ou `src/asset_context.py` (novo)
- `data/universes/*.csv` ou `data/universes/*.json` (novo)
- `scripts/refresh_universes.py` (novo)
- `tests/` (novo)
- `README.md`
- `prd/IFR2_PRD_Technical.md`

## Testes / validação
- contagem dos universos:
  - B3 = 100
  - EUA = 500
- smoke test sem erro de import
- screener e backtester rodando com o novo loader
- labels de moeda corretos
- benchmark com universo grande sem travar a UI

Comandos esperados:
```bash
pytest -q
python smoke_test.py
streamlit run ifr2_app.py
```

## Riscos e tradeoffs
- Não existe uma fonte pública universalmente perfeita para "top 100 B3" e "top 500 EUA" sem algum compromisso.
- Se a fonte pública mudar de HTML/API, o refresh precisa de manutenção.
- yfinance pode ficar lento ou rate-limitado com 600 tickers.
- Misturar BRL e USD sem normalização correta vai gerar decisões erradas.
- Se o critério final virar market cap em vez de liquidez, o Sprint 01 precisa ser refeito.

## Open questions
- Você quer B3 por liquidez ou por market cap?
- Você quer EUA como S&P 500 ou realmente os 500 maiores por market cap fora de um índice?
- Você quer atualização automática diária, semanal ou manual?
- Você quer manter o modo "Customizado" como hoje? (recomendado: sim)

## Estimativa
- Sprint 01: 1-2 dias
- Sprint 02: 1-2 dias
- Sprint 03: 1 dia
- Sprint 04: 1-2 dias
- Total: 4-7 dias úteis, dependendo da fonte escolhida para os universos.

## Verificação Chain-of-Verification
1. Rascunho: definir fonte e estrutura.
2. Verificar: conferir que os números batem e que a origem é estável.
3. Revisar: checar labels, moeda, fallback e erros parciais.
4. Fechar: rodar testes, benchmark e revisão final do resultado.

## Próximo passo
Se você aprovar esse plano, eu começo pela Sprint 01.
