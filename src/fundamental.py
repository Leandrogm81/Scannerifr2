"""Módulo para coleta e classificação de fundamentos de ativos."""

from __future__ import annotations

from dataclasses import dataclass

import yfinance as yf


@dataclass
class FundamentalData:
    """Dados fundamentalistas normalizados para uso no app."""

    ticker: str
    trailing_pe: float | None
    dividend_yield_pct: float | None
    net_income: float | None
    debt_to_equity: float | None
    roe_pct: float | None
    currency: str


@dataclass
class FundamentalSeal:
    """Classificação de saúde fundamentalista para leitura leiga."""

    label: str
    emoji: str
    score: int
    reasons: list[str]


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_fundamental_data(ticker: str) -> FundamentalData | None:
    """Busca fundamentos básicos de um ticker via Yahoo Finance.

    Returns None em caso de falha total de coleta.
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return None

    if not isinstance(info, dict) or len(info) == 0:
        return None

    trailing_pe = _safe_float(info.get("trailingPE"))

    # Em alguns papéis vem como fração (0.07 => 7%), em outros já como percentual.
    raw_dy = _safe_float(info.get("dividendYield"))
    if raw_dy is None:
        dividend_yield_pct = None
    else:
        dividend_yield_pct = raw_dy * 100 if raw_dy <= 1 else raw_dy

    net_income = _safe_float(info.get("netIncomeToCommon"))
    debt_to_equity = _safe_float(info.get("debtToEquity"))

    raw_roe = _safe_float(info.get("returnOnEquity"))
    if raw_roe is None:
        roe_pct = None
    else:
        roe_pct = raw_roe * 100 if raw_roe <= 1 else raw_roe

    currency = str(info.get("currency") or "N/A")

    return FundamentalData(
        ticker=ticker,
        trailing_pe=trailing_pe,
        dividend_yield_pct=dividend_yield_pct,
        net_income=net_income,
        debt_to_equity=debt_to_equity,
        roe_pct=roe_pct,
        currency=currency,
    )


def classify_fundamental_health(data: FundamentalData) -> FundamentalSeal:
    """Classifica saúde fundamentalista em score simples (0-100)."""

    score = 50
    reasons: list[str] = []

    # Lucro líquido
    if data.net_income is not None:
        if data.net_income > 0:
            score += 20
            reasons.append("Empresa lucrativa nos últimos 12 meses")
        else:
            score -= 25
            reasons.append("Empresa com prejuízo nos últimos 12 meses")
    else:
        reasons.append("Lucro indisponível no provedor")

    # Valuation simples pelo P/L
    if data.trailing_pe is not None:
        if 0 < data.trailing_pe <= 15:
            score += 12
            reasons.append("P/L em faixa saudável")
        elif data.trailing_pe > 40:
            score -= 10
            reasons.append("P/L elevado")
    else:
        reasons.append("P/L indisponível")

    # Dividendos
    if data.dividend_yield_pct is not None:
        if data.dividend_yield_pct >= 6:
            score += 10
            reasons.append("Dividend Yield atrativo")
        elif data.dividend_yield_pct < 1:
            score -= 4
            reasons.append("Dividend Yield baixo")

    # Endividamento
    if data.debt_to_equity is not None:
        if data.debt_to_equity <= 100:
            score += 8
            reasons.append("Endividamento sob controle")
        elif data.debt_to_equity > 300:
            score -= 12
            reasons.append("Endividamento elevado")

    # ROE
    if data.roe_pct is not None:
        if data.roe_pct >= 12:
            score += 8
            reasons.append("ROE consistente")
        elif data.roe_pct < 0:
            score -= 10
            reasons.append("ROE negativo")

    score = max(0, min(100, score))

    if score >= 70:
        return FundamentalSeal(
            label="Saudável", emoji="🟢", score=score, reasons=reasons
        )
    if score >= 45:
        return FundamentalSeal(label="Neutra", emoji="🟡", score=score, reasons=reasons)
    return FundamentalSeal(label="Atenção", emoji="🔴", score=score, reasons=reasons)
