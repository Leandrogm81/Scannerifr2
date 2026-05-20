"""Smoke test para verificar se a refatoração funcionou corretamente."""

import sys
import pandas as pd

# Adicionar o diretório do projeto ao path
sys.path.insert(0, "/mnt/c/Dev/IFR2")


def main():
    # Testar imports dos módulos
    try:
        from src.data_fetcher import fetch_all_data, get_data
        from src.indicators import calc_indicators
        from src.backtester import run_backtest
        from src.screener import screen_tickers
        from src.risk_manager import calculate_position_size
        from config.settings import DEFAULT_IBOV, TOP100_B3

        # Referências explícitas para validar imports no smoke e evitar F401
        _ = (fetch_all_data, get_data, run_backtest, screen_tickers)

        print("✅ Todos os módulos importados com sucesso!")
        print(f"✅ Lista de ativos IBOV legado: {len(DEFAULT_IBOV)} tickers")
        print(f"✅ Lista TOP 100 B3: {len(TOP100_B3)} tickers")

        assert len(TOP100_B3) == 100, "TOP100_B3 deve carregar 100 tickers"

        # Criar dados sintéticos suficientes (pelo menos 200 linhas para calc_indicators)
        close_values = list(range(10, 260))
        volume_values = [1_000_000 + (idx * 10_000) for idx in range(len(close_values))]
        test_df = pd.DataFrame(
            {
                "Close": close_values,
                "Volume": volume_values,
            }
        )

        # Adicionar colunas necessárias
        test_df["Open"] = test_df["Close"] - 1
        test_df["High"] = test_df["Close"] + 1
        test_df["Low"] = test_df["Close"] - 1

        result = calc_indicators(test_df)
        if result is not None and "IFR2" in result.columns:
            print("✅ Cálculo de indicadores funcionando!")
            print(f"✅ IFR2 calculado: {result['IFR2'].iloc[-1]:.2f}")
        else:
            print("❌ Falha no cálculo de indicadores")
            return 1

        # Teste de position sizing
        position = calculate_position_size(
            account_value=10000, risk_per_trade=0.02, stop_loss_distance=0.1
        )
        print(f"✅ Position sizing: R$ {position:.2f}")

        print("\n✅ Smoke test concluído com sucesso!")
        return 0

    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        return 1
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
