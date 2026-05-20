# Auditoria e Cão de Guarda — IFR2 Miner & Screener

Este diretório guarda os dois agentes de qualidade do IFR2.

Objetivo:
- rodar ao fim de cada fase/sprint de entrega
- impedir que uma fase seja marcada como concluída com problemas críticos
- registrar a saúde do projeto ao longo do tempo

Fluxo por fase:
1. terminar a implementação da fase
2. rodar o Cão de Guarda manualmente (`python3 auditoria/guardian/guardian.py --phase "<fase>"`)
3. rodar o Auditor manualmente
4. corrigir críticos e regressões
5. só então fechar a fase

Sem cron job:
- a execução é manual
- o projeto não depende de agendamento automático
- o script pode ser chamado quando a fase termina

Saídas esperadas:
- `auditoria/relatorio-DATA.html`
- `auditoria/relatorio-DATA.md`
- `auditoria/guardian/status-latest.html`
- `auditoria/guardian/status-YYYYMMDD-HHMMSS-PHASE.html`
- `auditoria/guardian/log.txt`
- `auditoria/guardian/last-state.json`

Notas:
- IFR2 é um app de análise/consulta, não executa ordens de mercado
- a auditoria deve olhar segurança, qualidade, estrutura, performance, dados e testes
- o guardião deve comparar o estado atual com o anterior e avisar se houver regressão
