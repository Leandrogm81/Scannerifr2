"""Script para medição de performance da busca de dados (benchmark)."""

import time
import pandas as pd
from src.data_fetcher import fetch_all_data
from src.universe_registry import load_universe_tickers

def main():
    print("Iniciando Benchmark de Coleta de Dados...")
    
    b3_tickers = load_universe_tickers("b3_top100")
    us_tickers = load_universe_tickers("us_top500")
    
    combined = b3_tickers + us_tickers
    print(f"Total de ativos para coletar: {len(combined)} (100 B3 + 500 US)")
    print("Coletando 1 ano de histórico. Pode demorar alguns minutos...\n")
    
    start_time = time.time()
    
    # max_workers=20 is an assumption for max parallel yfinance downloads
    data = fetch_all_data(combined, period="1y", batch_size=50, max_workers=20)
    
    elapsed = time.time() - start_time
    
    success = sum(1 for df in data.values() if not df.empty)
    failed = len(combined) - success
    
    print(f"\n--- Resultados do Benchmark ---")
    print(f"Tempo total: {elapsed:.2f} segundos")
    print(f"Ativos baixados com sucesso: {success}")
    print(f"Ativos que falharam: {failed}")
    
    if elapsed > 0:
        speed = len(combined) / elapsed
        print(f"Velocidade média: {speed:.2f} ativos/segundo")
        
        with open("benchmark_report.md", "w") as f:
            f.write("# Benchmark de Escala IFR2\n\n")
            f.write(f"- **Total de Ativos:** {len(combined)}\n")
            f.write(f"- **Sucesso:** {success}\n")
            f.write(f"- **Falhas:** {failed} (Ativos possivelmente delistados ou sem dados)\n")
            f.write(f"- **Tempo Total:** {elapsed:.2f} segundos\n")
            f.write(f"- **Velocidade:** {speed:.2f} ativos/segundo\n")
            f.write("\n## Limites de Regressão\n")
            f.write("- **Threshold Aceitável:** O tempo total para 600 ativos não deve ultrapassar 120 segundos em uma rede estável.\n")
            f.write("- **Threshold Crítico:** Velocidade inferior a 2.0 ativos/segundo indica estrangulamento de rede ou problema na API externa.\n")

if __name__ == "__main__":
    main()
