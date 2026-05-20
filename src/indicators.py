"""Módulo para cálculo de indicadores técnicos."""

import pandas as pd


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula indicadores técnicos: IFR2, SMA200, Volume Financeiro.

    Args:
        df: DataFrame com colunas ['Open', 'High', 'Low', 'Close', 'Volume']

    Returns:
        DataFrame com indicadores adicionados
    """
    if df is None or df.empty or len(df) < 200:
        return df

    df = df.copy()

    # Volume Financeiro (Preço × Volume)
    df["Vol_Fin"] = df["Close"] * df["Volume"]
    df["Vol_Fin_Medio"] = df["Vol_Fin"].rolling(window=21).mean()

    # IFR2 (RSI de 2 períodos - Fórmula de Wilder)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(com=1, min_periods=2).mean()
    loss = (-delta.clip(upper=0)).ewm(com=1, min_periods=2).mean()
    rs = gain / loss
    df["IFR2"] = 100 - (100 / (1 + rs))

    # SMA200 (Média móvel simples de 200 períodos)
    df["SMA200"] = df["Close"].rolling(window=200).mean()
    df["Bullish"] = df["Close"] > df["SMA200"]

    # Máximo anterior (para saída)
    df["Max_Prev"] = df["High"].shift(1)
    df["Max_Prev"] = df["Max_Prev"].fillna(df["Close"])

    return df
