"""Módulo para lógica de screening (varredura de ativos)."""

from __future__ import annotations

import pandas as pd


def screen_tickers(
    data_dict: dict[str, pd.DataFrame],
    rsi_threshold: int,
    min_vol_fin: float | dict[str, float],
) -> list[dict[str, object]]:
    """Executa screening em múltiplos ativos, aplicando filtros de liquidez específicos de cada moeda.

    Args:
        data_dict: Dicionário de dados (ticker -> DataFrame)
        rsi_threshold: Limite de IFR2 para compra
        min_vol_fin: Volume financeiro médio mínimo (ou dict por moeda, ex: {'BRL': 1000000, 'USD': 200000})

    Returns:
        Lista de dicionários com resultados do screening
    """
    results: list[dict[str, object]] = []

    for ticker, df in data_dict.items():
        if df is not None and not df.empty:
            # Verificar se tem IFR2 calculado
            if "IFR2" in df.columns and not df["IFR2"].isna().iloc[-1]:
                last = df.iloc[-1]

                vol_medio = float(last.get("Vol_Fin_Medio", 0))
                market = str(df["market"].iloc[-1]) if "market" in df.columns else "B3"
                currency = (
                    str(df["currency"].iloc[-1]) if "currency" in df.columns else "BRL"
                )

                # Resolver o threshold de liquidez de acordo com a moeda
                if isinstance(min_vol_fin, dict):
                    threshold = min_vol_fin.get(currency, 0.0)
                else:
                    threshold = float(min_vol_fin)

                if vol_medio >= threshold:
                    close_val = float(last["Close"])
                    ifr2_val = float(last["IFR2"])
                    is_bullish = bool(last["Bullish"])

                    # Sinal: IFR2 < threshold E preço > SMA200 (tendência)
                    status = (
                        "COMPRA!"
                        if ifr2_val < rsi_threshold and is_bullish
                        else "Neutro"
                    )

                    symbol_currency = "R$" if currency == "BRL" else "US$"

                    results.append(
                        {
                            "Ticker": ticker,
                            "Preço": round(close_val, 2),
                            "Moeda": currency,
                            "Mercado": market,
                            "IFR2": round(ifr2_val, 2),
                            "Acima SMA200": "Sim" if is_bullish else "Não",
                            "Vol Fin Médio": f"{symbol_currency} {vol_medio/1e6:.1f}M",
                            "Sinal": status,
                        }
                    )

    return results
