# Sprint 01 — Universe registry and snapshots

**Projeto:** IFR2 Miner & Screener  
**Duração estimada:** 1-2 dias  
**Dependências:** nenhuma  
**Objetivo:** substituir listas hardcoded por fonte única de universos em arquivos de dados.

---

## Tarefas

### 1.1 Definir o formato do snapshot
- [ ] Criar `src/universe_registry.py`
- [ ] Definir `UniverseDefinition`, `TickerMetadata` e funções de carga
- [ ] Estabelecer colunas mínimas do snapshot: `ticker`, `market`, `currency`, `rank`, `source`, `updated_at`

**Validação:** `load_universe()` retorna um objeto com contagem correta e metadata visível.

### 1.2 Criar os arquivos de universo
- [ ] Criar `data/universes/b3_top100.csv`
- [ ] Criar `data/universes/us_top500.csv`
- [ ] Criar `data/universes/custom_watchlists/`

**Validação:** pandas abre os arquivos sem tratamento especial.

### 1.3 Criar o refresh controlado
- [ ] Criar `scripts/refresh_universes.py`
- [ ] Suportar B3 a partir da página pública `https://www.dadosdemercado.com.br/acoes` ou de um arquivo local gerado a partir dela
- [ ] Suportar EUA a partir de `https://datahub.io/core/s-and-p-500-companies-financials/_r/-/data/constituents.csv` ou arquivo local
- [ ] Salvar snapshot versionado com data da atualização

**Validação:** `python scripts/refresh_universes.py --dry-run` gera relatório sem mexer na UI.

### 1.4 Manter compatibilidade temporária
- [ ] Deixar `config/settings.py` expor aliases para as novas listas
- [ ] Não remover ainda os defaults antigos até a UI apontar para o novo loader
- [ ] Preservar o modo customizado

**Validação:** o app ainda abre antes da Sprint 02.

---

## Snippet base

```python
@dataclass(frozen=True)
class UniverseDefinition:
    code: str
    label: str
    market: str
    source_url: str
    snapshot_path: Path
    expected_size: int
```

---

## Comandos de validação

```bash
python -m pytest tests/test_universe_registry.py -q
python scripts/refresh_universes.py --dry-run
```

---

## Aceitação do Sprint
- [ ] Existe uma fonte única para B3 e EUA
- [ ] O projeto carrega 100 tickers da B3 e 500 dos EUA
- [ ] O modo customizado continua funcionando
- [ ] O refresh gera snapshots versionados

---

## Notas técnicas
- Preferir arquivos simples (CSV/JSON) antes de inventar banco ou API.
- Se a fonte pública exigir token, o script deve aceitar URL local/arquivo como fallback.
- O objetivo aqui é organizar a base; ainda não é otimizar performance.
