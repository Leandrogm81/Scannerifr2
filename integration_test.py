"""Teste de integração para verificar se a refatoração está coerente."""

import sys

sys.path.insert(0, "/mnt/c/Dev/IFR2")


def main():
    # Testar imports dos módulos
    try:
        import src.backtester as backtester
        import src.data_fetcher as data_fetcher
        import src.indicators as indicators
        import src.risk_manager as risk_manager
        import src.screener as screener

        from config.settings import (
            DEFAULT_IBOV,
            DEFAULT_RSI_THRESHOLD,
            TOP100_B3,
            TOP500_US,
        )

        print("✅ Imports realizados com sucesso!")
        print(
            "✅ Módulos carregados: data_fetcher, indicators, backtester, screener, risk_manager, settings"
        )

        # Verificar se as funções existem
        modules = {
            "data_fetcher": data_fetcher,
            "indicators": indicators,
            "backtester": backtester,
            "screener": screener,
            "risk_manager": risk_manager,
        }
        functions = [
            ("data_fetcher", ["get_data", "fetch_all_data"]),
            ("indicators", ["calc_indicators"]),
            ("backtester", ["run_backtest"]),
            ("screener", ["screen_tickers"]),
            ("risk_manager", ["calculate_position_size"]),
        ]

        for module, func_names in functions:
            for func in func_names:
                if func in dir(modules[module]):
                    print(f"  ✅ {module}.{func} existe")
                else:
                    print(f"  ❌ {module}.{func} NÃO encontrado")
                    return 1

        # Verificar configurações e registry
        print("\n✅ Configurações carregadas:")
        print(f"   - DEFAULT_RSI_THRESHOLD: {DEFAULT_RSI_THRESHOLD}")
        print(f"   - DEFAULT_IBOV tem {len(DEFAULT_IBOV)} ativos")
        print(f"   - TOP100_B3 tem {len(TOP100_B3)} ativos")
        print(f"   - TOP500_US tem {len(TOP500_US)} ativos")

        assert len(TOP100_B3) == 100, "TOP100_B3 deve carregar 100 tickers"
        assert len(TOP500_US) == 500, "TOP500_US deve carregar 500 tickers"

        print("\n✅ Todas as verificações passaram! A refatoração está coerente.")
        return 0

    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        print("   Verifique se as dependências estão instaladas.")
        return 1
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
