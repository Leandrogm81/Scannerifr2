"""Lightweight bootstrap helpers for the Streamlit app.

This module intentionally avoids pandas/yfinance imports so the public Streamlit
page can render its initial UI before heavier market-data dependencies are
loaded. Heavy modules are imported lazily only after the user starts a scan.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_DIR = PROJECT_ROOT / "data" / "universes"

DEFAULT_IBOV = [
    "PETR4.SA",
    "VALE3.SA",
    "ITUB4.SA",
    "BBDC4.SA",
    "ABEV3.SA",
    "BBAS3.SA",
    "JBSS3.SA",
    "WEGE3.SA",
    "RENT3.SA",
    "LREN3.SA",
    "GGBR4.SA",
    "CMIG4.SA",
    "SUZB3.SA",
    "ELET3.SA",
    "PRIO3.SA",
    "RADL3.SA",
    "BRFS3.SA",
    "CSNA3.SA",
]

DEFAULT_SMLL = [
    "COGN3.SA",
    "SOMA3.SA",
    "RAIZ4.SA",
    "MRFG3.SA",
    "CVCB3.SA",
    "MOVI3.SA",
    "QUAL3.SA",
    "MYPK3.SA",
    "STBP3.SA",
    "LOGG3.SA",
    "WIZC3.SA",
    "ANIM3.SA",
]

LEGACY_TOP100_B3 = DEFAULT_IBOV + DEFAULT_SMLL
LEGACY_US_WATCHLIST = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "META",
    "NVDA",
    "BRK.B",
    "JPM",
    "V",
]

DEFAULT_RSI_THRESHOLD = 10


@dataclass(frozen=True)
class UniverseDefinition:
    code: str
    label: str
    market: str
    snapshot_path: Path
    expected_size: int
    currency: str


@dataclass(frozen=True)
class TickerMetadata:
    ticker: str
    name: str
    market: str
    currency: str
    rank: int
    source: str
    updated_at: str


@dataclass(frozen=True)
class UniverseSnapshot:
    definition: UniverseDefinition
    records: tuple[TickerMetadata, ...]

    @property
    def count(self) -> int:
        return len(self.records)

    @property
    def tickers(self) -> list[str]:
        return [record.ticker for record in self.records]


def _definition(code: str) -> UniverseDefinition:
    definitions = {
        "b3_top100": UniverseDefinition(
            code="b3_top100",
            label="Top 100 B3",
            market="B3",
            snapshot_path=UNIVERSE_DIR / "b3_top100.csv",
            expected_size=100,
            currency="BRL",
        ),
        "us_top500": UniverseDefinition(
            code="us_top500",
            label="Top 500 US",
            market="NYSE/NASDAQ",
            snapshot_path=UNIVERSE_DIR / "us_top500.csv",
            expected_size=500,
            currency="USD",
        ),
    }
    try:
        return definitions[code]
    except KeyError as exc:
        available = ", ".join(sorted(definitions))
        raise KeyError(f"Universo desconhecido: {code!r}. Disponíveis: {available}") from exc


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _safe_text(value: object, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


def _fallback_records(definition: UniverseDefinition, tickers: list[str]) -> tuple[TickerMetadata, ...]:
    return tuple(
        TickerMetadata(
            ticker=ticker,
            name="",
            market=definition.market,
            currency=definition.currency,
            rank=idx,
            source="fallback",
            updated_at="N/A",
        )
        for idx, ticker in enumerate(tickers, start=1)
    )


def load_universe(code: str) -> UniverseSnapshot:
    """Load ticker metadata with stdlib CSV only, avoiding heavy import cost."""

    definition = _definition(code)
    fallback = LEGACY_TOP100_B3 if code == "b3_top100" else LEGACY_US_WATCHLIST

    if not definition.snapshot_path.exists():
        return UniverseSnapshot(definition, _fallback_records(definition, fallback))

    records: list[TickerMetadata] = []
    with definition.snapshot_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            ticker = _safe_text(row.get("ticker"))
            if not ticker:
                continue
            records.append(
                TickerMetadata(
                    ticker=ticker,
                    name=_safe_text(row.get("name")),
                    market=_safe_text(row.get("market"), definition.market),
                    currency=_safe_text(row.get("currency"), definition.currency),
                    rank=_safe_int(row.get("rank"), idx),
                    source=_safe_text(row.get("source")),
                    updated_at=_safe_text(row.get("updated_at"), "N/A"),
                )
            )

    if not records:
        records = list(_fallback_records(definition, fallback))

    return UniverseSnapshot(definition, tuple(records))
