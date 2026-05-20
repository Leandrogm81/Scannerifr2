# Sprint 04 — Testes, benchmark e documentação

**Projeto:** IFR2 Miner & Screener  
**Duração estimada:** 1-2 dias  
**Dependências:** Sprint 03  
**Objetivo:** fechar a expansão com segurança, medindo comportamento e documentando o novo fluxo.

---

## Tarefas

### 4.1 Criar suíte de testes real
- [ ] Criar `tests/`
- [ ] Criar `pytest.ini` com markers para testes lentos
- [ ] Cobrir loader, fetcher, screener e backtester com dados sintéticos
- [ ] Portar smoke/integration para pytest

**Validação:** `pytest -q` passa sem depender de yfinance real.

### 4.2 Corrigir o smoke test atual
- [ ] Importar `pandas` em `smoke_test.py` ou aposentar o script
- [ ] Garantir que nenhum teste solto quebre a execução
- [ ] Eliminar duplicação entre `smoke_test.py` e `integration_test.py`

**Validação:** não existe mais falha por `pd` indefinido.

### 4.3 Benchmark de escala
- [ ] Medir 100 B3 + 500 EUA
- [ ] Registrar tempo total e consumo aproximado
- [ ] Definir um limite de regressão aceitável
- [ ] Marcar os testes que tocam fonte pública como `slow`

**Validação:** sabemos se a UI ficou lenta e onde está o gargalo.

### 4.4 Documentação final
- [ ] Atualizar `README.md`
- [ ] Atualizar `prd/IFR2_PRD_Technical.md`
- [ ] Documentar as fontes de universo e os limites do app

**Validação:** a documentação descreve o que o app realmente faz.

---

## Snippet base

```python
def test_loads_expected_universes():
    assert len(load_universe("b3_top100").tickers) == 100
    assert len(load_universe("us_top500").tickers) == 500
```

---

## Comandos de validação

```bash
pytest -q
python smoke_test.py
streamlit run ifr2_app.py
```

---

## Aceitação do Sprint
- [ ] Testes automatizados verdes
- [ ] Benchmark registrado
- [ ] README e PRD alinhados com o código
- [ ] O usuário consegue repetir o fluxo sem adivinhar nada

---

## Notas técnicas
- Não usar dados reais em teste unitário.
- Manter testes de integração opcionais para fonte pública, marcados como slow.
- Se o smoke test continuar útil, ele deve depender dos mesmos helpers dos testes.
