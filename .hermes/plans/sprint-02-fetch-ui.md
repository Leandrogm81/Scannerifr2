# Sprint 02 — Fetch pipeline e UI para universos grandes

**Projeto:** IFR2 Miner & Screener  
**Duração estimada:** 1-2 dias  
**Dependências:** Sprint 01  
**Objetivo:** fazer a interface e o pipeline consumirem a nova fonte sem duplicação.

---

## Tarefas

### 2.1 Refatorar `ifr2_app.py` para usar os módulos de domínio
- [ ] Remover listas hardcoded do app
- [ ] Usar `load_universe()` para popular os selects
- [ ] Mostrar tamanho real do universo no sidebar
- [ ] Parar de duplicar a regra de screening dentro da tela

**Validação:** a tela mostra 100/500 no seletor e usa a fonte única.

### 2.2 Tornar `fetch_all_data` robusto para 600 tickers
- [ ] Adicionar `batch_size`
- [ ] Agrupar tickers em lotes menores
- [ ] Registrar falhas por ticker sem abortar tudo
- [ ] Permitir ajuste de `max_workers`

**Validação:** corrida com universo grande conclui e mostra resultados parciais quando algum ticker falha.

### 2.3 Centralizar a lógica de screening e backtest
- [ ] Mover a regra inline do `ifr2_app.py` para `src/screener.py` e `src/backtester.py`
- [ ] Eliminar duplicação entre UI e módulos
- [ ] Manter a saída de resultado compatível com a tabela atual

**Validação:** resultados da UI batem com os módulos em dados sintéticos.

### 2.4 Cache e experiência de uso
- [ ] Usar `st.cache_data` ou cache equivalente para dados e snapshot
- [ ] Mostrar tempo de execução e contagem de erros
- [ ] Manter a barra de progresso responsiva

**Validação:** a segunda execução é visivelmente mais rápida e não refaz tudo do zero.

---

## Snippet base

```python
def fetch_all_data(tickers: list[str], period: str, batch_size: int = 25) -> dict[str, pd.DataFrame]:
    ...
```

---

## Comandos de validação

```bash
python -m pytest tests/test_fetcher.py -q
streamlit run ifr2_app.py
```

---

## Aceitação do Sprint
- [ ] Screener e backtester rodam com 100/500 tickers
- [ ] Sem lógica duplicada na UI
- [ ] Erros parciais não derrubam a corrida
- [ ] A barra de progresso chega ao fim

---

## Notas técnicas
- Evitar uma única chamada gigante do yfinance.
- Se a performance ainda ficar ruim, quebrar em lotes e devolver resultado incremental.
- Se algum ticker falhar, registrar no resultado final em vez de esconder o problema.
