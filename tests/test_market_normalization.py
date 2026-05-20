from __future__ import annotations

import pandas as pd
from src.screener import screen_tickers


def test_screen_tickers_single_threshold() -> None:
    df_b3 = pd.DataFrame(
        {
            "Close": [10.0, 11.0],
            "Vol_Fin_Medio": [1_500_000, 1_500_000],
            "IFR2": [5.0, 5.0],
            "Bullish": [True, True],
            "market": ["B3", "B3"],
            "currency": ["BRL", "BRL"],
        }
    )

    data_dict = {"PETR4.SA": df_b3}
    results = screen_tickers(data_dict, rsi_threshold=10, min_vol_fin=1_000_000)

    assert len(results) == 1
    assert results[0]["Ticker"] == "PETR4.SA"
    assert results[0]["Moeda"] == "BRL"
    assert results[0]["Mercado"] == "B3"
    assert results[0]["Vol Fin Médio"] == "R$ 1.5M"
    assert results[0]["Sinal"] == "COMPRA!"


def test_screen_tickers_dict_thresholds() -> None:
    df_b3 = pd.DataFrame(
        {
            "Close": [10.0, 11.0],
            "Vol_Fin_Medio": [1_500_000, 1_500_000],
            "IFR2": [5.0, 5.0],
            "Bullish": [True, True],
            "market": ["B3", "B3"],
            "currency": ["BRL", "BRL"],
        }
    )

    df_us = pd.DataFrame(
        {
            "Close": [100.0, 101.0],
            "Vol_Fin_Medio": [300_000, 300_000],
            "IFR2": [8.0, 8.0],
            "Bullish": [True, True],
            "market": ["NYSE/NASDAQ", "NYSE/NASDAQ"],
            "currency": ["USD", "USD"],
        }
    )

    data_dict = {"PETR4.SA": df_b3, "AAPL": df_us}
    thresholds = {"BRL": 2_000_000, "USD": 200_000}  # PETR4 should be filtered out!

    results = screen_tickers(data_dict, rsi_threshold=10, min_vol_fin=thresholds)

    assert len(results) == 1
    assert results[0]["Ticker"] == "AAPL"
    assert results[0]["Moeda"] == "USD"
    assert results[0]["Mercado"] == "NYSE/NASDAQ"
    assert results[0]["Vol Fin Médio"] == "US$ 0.3M"
    assert results[0]["Sinal"] == "COMPRA!"
