# Sprint 03 — Normalização de mercado e filtros

**Projeto:** IFR2 Miner & Screener  
**Duração estimada:** 1 dia  
**Dependências:** Sprint 02  
**Objetivo:** corrigir moeda, labels e filtros para B3 e EUA.

---

## Tarefas

### 3.1 Adicionar contexto de mercado por ticker
- [ ] Incluir `market`, `currency` e `source` nos dados do universo
- [ ] Propagar metadata até o resultado do screener
- [ ] Deixar explícito o mercado de cada linha na tabela

**Validação:** cada linha da tabela sabe se é BRL ou USD.

### 3.2 Corrigir a exibição de preço e volume
- [ ] Trocar `"Preço (R$)"` por rótulo dinâmico
- [ ] Trocar `"Vol Fin Médio (R$)"` por moeda correta
- [ ] Impedir mistura de BRL/USD em tabelas sem aviso

**Validação:** tickers americanos deixam de ser mostrados como BRL.

### 3.3 Ajustar critérios de liquidez
- [ ] Deixar claro se o filtro é por volume financeiro, número de negócios ou ambos
- [ ] Usar thresholds diferentes por mercado se necessário
- [ ] Documentar a regra na sidebar

**Validação:** a regra exibida na UI corresponde ao cálculo real.

### 3.4 Habilitar modo combinado com segurança
- [ ] Adicionar visão "Ambos" se a normalização estiver pronta
- [ ] Separar ordenação e agregação por mercado quando necessário

**Validação:** modo combinado não confunde moeda nem liquidez.

---

## Snippet base

```python
display_currency = "R$" if meta.market == "B3" else "US$"
st.write(f"Preço ({display_currency})")
```

---

## Comandos de validação

```bash
python -m pytest tests/test_market_normalization.py -q
streamlit run ifr2_app.py
```

---

## Aceitação do Sprint
- [ ] BRL e USD aparecem corretamente
- [ ] O filtro de liquidez é coerente com o mercado
- [ ] O modo combinado não gera números enganosos

---

## Notas técnicas
- Se não houver FX, não tentar somar liquidez de BRL com USD como se fosse a mesma coisa.
- Se o usuário quiser comparabilidade real, então precisa de conversão cambial explícita.
- Não assumir que um valor “alto” em USD significa a mesma coisa que um valor “alto” em BRL.
