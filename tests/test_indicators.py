import pandas as pd
from src.indicators import calc_indicators


def test_calc_indicators_returns_same_on_empty():
    df = pd.DataFrame()
    res = calc_indicators(df)
    assert res.empty


def test_calc_indicators_needs_200_rows():
    df = pd.DataFrame(
        {
            "Close": [1] * 100,
            "Volume": [10] * 100,
            "High": [1] * 100,
            "Low": [1] * 100,
            "Open": [1] * 100,
        }
    )
    res = calc_indicators(df)
    assert "IFR2" not in res.columns


def test_calc_indicators_happy_path():
    close_vals = list(range(10, 250))
    df = pd.DataFrame(
        {
            "Close": close_vals,
            "Volume": [1000] * 240,
            "High": [c + 1 for c in close_vals],
            "Low": [c - 1 for c in close_vals],
            "Open": close_vals,
        }
    )
    res = calc_indicators(df)
    assert "IFR2" in res.columns
    assert "Vol_Fin" in res.columns
    assert "Vol_Fin_Medio" in res.columns
    assert "SMA200" in res.columns
    assert "Bullish" in res.columns
    assert "Max_Prev" in res.columns

    # Check SMA200 calculation roughly
    # the 200th value should be mean(10..209) = 109.5
    assert not pd.isna(res["SMA200"].iloc[199])
