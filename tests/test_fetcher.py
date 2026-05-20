from __future__ import annotations

import pandas as pd
from unittest.mock import patch
from src.data_fetcher import fetch_all_data, get_data


def test_fetch_all_data_empty_input() -> None:
    results = fetch_all_data([], period="1y")
    assert results == {}


@patch("src.data_fetcher.get_data")
def test_fetch_all_data_with_mocked_fetch(mock_get_data) -> None:
    # Configurar mock para devolver DataFrame fictício
    df_mock = pd.DataFrame({"Close": [10, 20], "Volume": [100, 200]})
    mock_get_data.return_value = df_mock

    tickers = ["AAPL", "MSFT", "GOOGL"]
    results = fetch_all_data(tickers, period="1y", batch_size=2, max_workers=2)

    assert len(results) == 3
    assert "AAPL" in results
    assert "MSFT" in results
    assert "GOOGL" in results
    assert len(results["AAPL"]) == 2
    assert mock_get_data.call_count == 3


def test_get_data_on_invalid_ticker() -> None:
    # Ticker inválido que gera erro ou DataFrame vazio
    df = get_data("INVALID_TICKER_NAME_XYZ", period="1d")
    assert isinstance(df, pd.DataFrame)
    assert df.empty
