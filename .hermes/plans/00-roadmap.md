# 00 — Roadmap

Projeto: IFR2 Miner & Screener
Tema: expansão do universo de ativos e normalização multi-mercado

Assunção de trabalho:
- B3 = 100 ativos mais líquidos / mais negociados
- EUA = 500 large caps, usando S&P 500 como proxy pública e estável

## Índice de sprints

| Sprint | Arquivo | Duração | Depende de | Entrega |
| --- | --- | --- | --- | --- |
| 01 | [sprint-01-universe-registry.md](./sprint-01-universe-registry.md) | 1-2 dias | — | fonte única dos universos |
| 02 | [sprint-02-fetch-ui.md](./sprint-02-fetch-ui.md) | 1-2 dias | Sprint 01 | UI e pipeline usando 100+500 ativos |
| 03 | [sprint-03-market-normalization.md](./sprint-03-market-normalization.md) | 1 dia | Sprint 02 | BRL/USD corretos e filtros por mercado |
| 04 | [sprint-04-tests-performance-docs.md](./sprint-04-tests-performance-docs.md) | 1-2 dias | Sprint 03 | testes, benchmark e docs atualizados |

## Dependência visual

```text
Sprint 01
   ↓
Sprint 02
   ↓
Sprint 03
   ↓
Sprint 04
```

## Ordem recomendada
1. Sprint 01 — criar a fonte única dos universos.
2. Sprint 02 — ligar a UI e o back-end à nova fonte.
3. Sprint 03 — corrigir unidades, moedas e filtros.
4. Sprint 04 — fechar com testes, benchmark e documentação.

## Checkpoints
- Depois do Sprint 01: o projeto já sabe carregar 100 tickers da B3 e 500 dos EUA a partir de arquivos/snapshots.
- Depois do Sprint 02: o screener e o backtester já rodam nesses universos maiores sem lógica duplicada na interface.
- Depois do Sprint 03: os dados de B3 e EUA aparecem com moeda/labels corretos.
- Depois do Sprint 04: temos testes automatizados e documentação coerente com o que o app faz de verdade.

## Gate de entrega obrigatório
Ao terminar qualquer sprint/fase:
1. Rodar o Cão de Guarda manualmente (`auditoria/guardian/guardian.py`)
2. Rodar a Auditoria manualmente (`auditoria/PROMPT-AUDITOR.md`)
3. Corrigir qualquer CRÍTICO ou regressão nova antes de fechar a fase
4. Só considerar a entrega concluída quando o gate estiver verde ou com riscos explicitamente aceitos

## CoVe: como vamos verificar
Antes de considerar cada sprint concluído:
- conferir contagem de tickers
- conferir fonte usada
- conferir moeda/label exibido
- conferir comportamento com alguns tickers de amostra
- revisar com uma segunda leitura antes do merge
