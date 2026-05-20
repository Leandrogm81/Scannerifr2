import pandas as pd
from src.backtester import run_backtest


def test_run_backtest_empty_or_no_ifr2():
    res = run_backtest(pd.DataFrame())
    assert res is None

    df = pd.DataFrame({"Close": [10, 11]})
    res = run_backtest(df)
    assert res is None


def test_run_backtest_happy_path():
    df = pd.DataFrame(
        {
            "Close": [10, 9, 8, 12, 13, 14],
            "IFR2": [50, 5, 4, 80, 80, 50],
            "SMA200": [5, 5, 5, 5, 5, 5],
            "Max_Prev": [11, 10, 9, 8, 12, 13],
            "Bullish": [True, True, True, True, True, True],
        }
    )

    # row 0: IFR2=50 (no buy)
    # row 1: IFR2=5, Bullish=True -> BUY_SIGNAL. in_pos becomes True. entry=9.
    # row 2: in_pos=True. IFR2=4 (<70), Close=8 (not > Max_Prev 9). Keep pos.
    # row 3: in_pos=True. IFR2=80 (>70) -> Sell! pnl = (12/9)-1 = 0.333. in_pos=False
    # row 4: in_pos=False. IFR2=80 (no buy)
    # row 5: in_pos=False. IFR2=50 (no buy)

    res = run_backtest(df, buy_threshold=10)
    assert res is not None
    assert res["Total_Trades"] == 1
    assert round(res["Exp_Math"], 2) == 0.33
    assert res["Win_Rate"] == 1.0
