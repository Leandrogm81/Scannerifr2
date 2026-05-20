"""Módulo para busca e cache de dados financeiros."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import yfinance as yf

# Configuração de cache (será movida para settings.py futuramente)
CACHE_TTL = 3600  # 1 hora em segundos


def get_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Busca dados históricos de um ativo usando yfinance.

    Args:
        ticker: Ticker do ativo (ex: 'PETR4.SA', 'AAPL')
        period: Período de dados ('1y', '2y', '5y', etc.)

    Returns:
        DataFrame com dados históricos ou DataFrame vazio se falhar
    """
    try:
        data = yf.download(
            ticker, period=period, interval="1d", auto_adjust=True, progress=False
        )

        if data is None or data.empty:
            return pd.DataFrame()

        # Ajustar colunas MultiIndex se necessário
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in data.columns for col in required_cols):
            return pd.DataFrame()

        # Filtrar volume > 0
        data = data[data["Volume"] > 0].copy()
        return data

    except Exception:
        return pd.DataFrame()


def fetch_all_data(
    tickers: list[str],
    period: str,
    batch_size: int = 25,
    max_workers: int = 10,
    progress_callback=None,
) -> dict[str, pd.DataFrame]:
    """Busca dados para múltiplos tickers em paralelo e em lotes (batches) para evitar rate limits.

    Args:
        tickers: Lista de tickers para buscar
        period: Período de dados
        batch_size: Tamanho do lote para execução sequencial dos lotes
        max_workers: Número máximo de threads em paralelo
        progress_callback: Função opcional para atualizar progresso (recebe float de 0.0 a 1.0)

    Returns:
        Dicionário com ticker como chave e DataFrame como valor
    """
    results: dict[str, pd.DataFrame] = {}
    total = len(tickers)
    if total == 0:
        return results

    # Dividir em lotes (batches)
    batches = [tickers[i : i + batch_size] for i in range(0, total, batch_size)]
    processed_count = 0

    def fetch_single(ticker: str) -> tuple[str, pd.DataFrame]:
        df = get_data(ticker, period)
        return ticker, df

    for batch in batches:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = list(executor.map(fetch_single, batch))
            for ticker, df in futures:
                results[ticker] = df.copy() if df is not None else pd.DataFrame()
                processed_count += 1
                if progress_callback:
                    progress_callback(processed_count / total)

    return results
