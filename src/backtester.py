"""Módulo para lógica de backtesting."""

import pandas as pd


def run_backtest(df: pd.DataFrame, buy_threshold: int = 10) -> dict:
    """
    Executa backtest da estratégia IFR2.

    Args:
        df: DataFrame com dados históricos e indicadores calculados
        buy_threshold: Nível de IFR2 para sinal de compra

    Returns:
        Dicionário com métricas do backtest ou None se não houver trades
    """
    if df is None or "IFR2" not in df.columns:
        return None

    df = df.copy().dropna(subset=["IFR2", "SMA200", "Max_Prev"])

    if df.empty or len(df) < 2:
        return None

    # Sinal de compra: IFR2 < threshold E preço > SMA200 (tendência de alta)
    df["Buy_Signal"] = (df["IFR2"] < buy_threshold) & df["Bullish"]

    trades = []
    in_pos = False
    entry_price = 0

    # Simular trades
    for idx, row in df.iterrows():
        if not in_pos:
            if row["Buy_Signal"]:
                in_pos = True
                entry_price = row["Close"]
        else:
            # Saída: IFR2 > 70 (overbought) OU preço > máximo anterior
            if row["IFR2"] > 70 or row["Close"] > row["Max_Prev"]:
                in_pos = False
                pnl = (row["Close"] / entry_price) - 1
                trades.append({"Return": pnl})

    if not trades:
        return None

    # Calcular métricas
    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df["Return"] > 0]["Return"]
    losses = trades_df[trades_df["Return"] <= 0]["Return"]

    win_rate = len(wins) / len(trades_df)
    losses_sum = float(abs(losses.sum())) if len(losses) > 0 else 0
    profit_factor = 99.9 if losses_sum == 0 else float(wins.sum()) / losses_sum
    exp_math = trades_df["Return"].mean()
    cum_return = (trades_df["Return"] + 1).prod() - 1

    # Curva de capital
    equity = [1.0] + (trades_df["Return"] + 1).cumprod().tolist()

    # Retorno buy & hold
    buy_hold_return = (df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1

    return {
        "Win_Rate": win_rate,
        "Profit_Factor": profit_factor,
        "Total_Trades": len(trades_df),
        "Exp_Math": exp_math,
        "Cum_Return": cum_return,
        "Equity_Curve": equity,
        "Buy_Hold_Return": buy_hold_return,
    }
