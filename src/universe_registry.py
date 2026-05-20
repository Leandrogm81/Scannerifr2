"""Registro de universos de ativos baseado em snapshots CSV.

Este módulo centraliza a leitura e a escrita dos universos de mercado usados
pelo IFR2. A ideia é manter os tickers em arquivos simples e versionados,
sem banco de dados e sem listas hardcoded espalhadas pelo app.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_DIR = PROJECT_ROOT / "data" / "universes"
CUSTOM_WATCHLIST_DIR = UNIVERSE_DIR / "custom_watchlists"
HISTORY_DIR = UNIVERSE_DIR / "history"

B3_SOURCE_URL = "https://www.dadosdemercado.com.br/acoes"
US_SOURCE_URL = "https://datahub.io/core/s-and-p-500-companies-financials/_r/-/data/constituents.csv"
US_SOURCE_FALLBACK_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"

CANONICAL_COLUMNS = ("ticker", "market", "currency", "rank", "source", "updated_at")


@dataclass(frozen=True)
class UniverseDefinition:
    """Describe onde encontrar e como interpretar um universo."""

    code: str
    label: str
    market: str
    source_url: str
    snapshot_path: Path
    expected_size: int
    currency: str


@dataclass(frozen=True)
class TickerMetadata:
    """Metadata normalizada por ticker dentro de um universo."""

    ticker: str
    name: str
    market: str
    currency: str
    rank: int
    source: str
    updated_at: str


@dataclass(frozen=True)
class UniverseSnapshot:
    """Snapshot carregado do disco, com frame e metadados por ticker."""

    definition: UniverseDefinition
    frame: pd.DataFrame
    records: tuple[TickerMetadata, ...]

    @property
    def count(self) -> int:
        return len(self.records)

    @property
    def tickers(self) -> list[str]:
        return [record.ticker for record in self.records]

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "code": self.definition.code,
            "label": self.definition.label,
            "market": self.definition.market,
            "currency": self.definition.currency,
            "expected_size": self.definition.expected_size,
            "count": self.count,
            "snapshot_path": str(self.definition.snapshot_path),
            "source_url": self.definition.source_url,
        }


def _resolve_root(project_root: str | Path | None = None) -> Path:
    if project_root is None:
        return PROJECT_ROOT
    return Path(project_root).resolve()


def _timestamp_label(moment: datetime | None = None) -> str:
    return (moment or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")


def _timestamp_slug(moment: datetime | None = None) -> str:
    return (moment or datetime.now()).strftime("%Y%m%d-%H%M%S")


@lru_cache(maxsize=None)
def default_universe_definitions(
    project_root: str | Path | None = None,
) -> dict[str, UniverseDefinition]:
    """Retorna o manifesto padrão de universos do projeto."""

    root = _resolve_root(project_root)
    universe_dir = root / "data" / "universes"
    return {
        "b3_top100": UniverseDefinition(
            code="b3_top100",
            label="Top 100 B3",
            market="B3",
            source_url=B3_SOURCE_URL,
            snapshot_path=universe_dir / "b3_top100.csv",
            expected_size=100,
            currency="BRL",
        ),
        "us_top500": UniverseDefinition(
            code="us_top500",
            label="Top 500 US",
            market="NYSE/NASDAQ",
            source_url=US_SOURCE_URL,
            snapshot_path=universe_dir / "us_top500.csv",
            expected_size=500,
            currency="USD",
        ),
    }


def get_universe_definition(
    code: str, project_root: str | Path | None = None
) -> UniverseDefinition:
    """Busca a definição de um universo pelo código."""

    definitions = default_universe_definitions(project_root)
    try:
        return definitions[code]
    except KeyError as exc:
        available = ", ".join(sorted(definitions))
        raise KeyError(
            f"Universo desconhecido: {code!r}. Disponíveis: {available}"
        ) from exc


def ensure_universe_directories(project_root: str | Path | None = None) -> None:
    """Garante que as pastas do registry existam."""

    root = _resolve_root(project_root)
    (root / "data" / "universes").mkdir(parents=True, exist_ok=True)
    (root / "data" / "universes" / "custom_watchlists").mkdir(
        parents=True, exist_ok=True
    )
    (root / "data" / "universes" / "history").mkdir(parents=True, exist_ok=True)


def list_custom_watchlists(project_root: str | Path | None = None) -> list[Path]:
    """Lista watchlists customizadas em CSV."""

    root = _resolve_root(project_root)
    watchlist_dir = root / "data" / "universes" / "custom_watchlists"
    if not watchlist_dir.exists():
        return []
    return sorted(path for path in watchlist_dir.glob("*.csv") if path.is_file())


def load_custom_watchlist(
    path: str | Path, project_root: str | Path | None = None
) -> pd.DataFrame:
    """Carrega uma watchlist customizada a partir de um CSV."""

    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (
            _resolve_root(project_root)
            / "data"
            / "universes"
            / "custom_watchlists"
            / resolved
        )
    return pd.read_csv(resolved)


def _fallback_frame(
    definition: UniverseDefinition, tickers: Sequence[str]
) -> pd.DataFrame:
    frame = pd.DataFrame({"ticker": list(tickers)})
    if frame.empty:
        return frame
    frame["name"] = ""
    frame["market"] = definition.market
    frame["currency"] = definition.currency
    frame["rank"] = range(1, len(frame) + 1)
    frame["source"] = f"fallback:{definition.source_url}"
    frame["updated_at"] = _timestamp_label()
    return frame


def _normalize_loaded_frame(
    frame: pd.DataFrame,
    definition: UniverseDefinition,
    *,
    source_label: str | None = None,
    updated_at: str | None = None,
) -> pd.DataFrame:
    df = frame.copy()
    rename_map = {
        "Symbol": "ticker",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
        "Headquarters Location": "headquarters_location",
        "Date added": "date_added",
        "CIK": "cik",
        "Founded": "founded",
    }
    df = df.rename(
        columns={
            old: new
            for old, new in rename_map.items()
            if old in df.columns and new not in df.columns
        }
    )

    if "ticker" not in df.columns:
        raise ValueError(
            f"Snapshot inválido para {definition.code}: coluna 'ticker' ausente"
        )

    if "name" not in df.columns:
        df["name"] = ""
    if "market" not in df.columns:
        df["market"] = definition.market
    if "currency" not in df.columns:
        df["currency"] = definition.currency
    if "rank" not in df.columns:
        df["rank"] = range(1, len(df) + 1)
    if "source" not in df.columns:
        df["source"] = source_label or definition.source_url
    if "updated_at" not in df.columns:
        df["updated_at"] = updated_at or _timestamp_label()

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["market"] = df["market"].astype(str).str.strip()
    df["currency"] = df["currency"].astype(str).str.strip()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").fillna(0).astype(int)
    df["source"] = df["source"].astype(str).str.strip()
    df["updated_at"] = df["updated_at"].astype(str).str.strip()

    ordered_columns = list(CANONICAL_COLUMNS) + [
        column for column in df.columns if column not in CANONICAL_COLUMNS
    ]
    return df.loc[:, ordered_columns].reset_index(drop=True)


def _safe_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_text(value: object, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    text = str(value).strip()
    return text if text else default


def _build_records(
    frame: pd.DataFrame, definition: UniverseDefinition
) -> tuple[TickerMetadata, ...]:
    records: list[TickerMetadata] = []
    for row in frame.itertuples(index=False):
        data = row._asdict()
        records.append(
            TickerMetadata(
                ticker=_safe_text(data.get("ticker")),
                name=_safe_text(data.get("name")),
                market=_safe_text(data.get("market"), definition.market),
                currency=_safe_text(data.get("currency"), definition.currency),
                rank=_safe_int(data.get("rank")),
                source=_safe_text(data.get("source"), definition.source_url),
                updated_at=_safe_text(data.get("updated_at"), _timestamp_label()),
            )
        )
    return tuple(records)


def load_universe(
    code: str,
    project_root: str | Path | None = None,
    *,
    fallback: Sequence[str] | None = None,
) -> UniverseSnapshot:
    """Carrega um universo do snapshot local ou de um fallback em memória."""

    definition = get_universe_definition(code, project_root)
    snapshot_path = definition.snapshot_path

    if snapshot_path.exists():
        frame = pd.read_csv(snapshot_path)
        normalized = _normalize_loaded_frame(frame, definition)
        return UniverseSnapshot(
            definition=definition,
            frame=normalized,
            records=_build_records(normalized, definition),
        )

    if fallback is None:
        raise FileNotFoundError(
            f"Snapshot ausente para {definition.code}: {snapshot_path}"
        )

    fallback_frame = _fallback_frame(definition, fallback)
    normalized = _normalize_loaded_frame(
        fallback_frame,
        definition,
        source_label=f"fallback:{definition.source_url}",
    )
    return UniverseSnapshot(
        definition=definition,
        frame=normalized,
        records=_build_records(normalized, definition),
    )


def load_universe_tickers(
    code: str,
    project_root: str | Path | None = None,
    *,
    fallback: Sequence[str] | None = None,
) -> list[str]:
    """Carrega apenas a lista de tickers de um universo."""

    return load_universe(code, project_root, fallback=fallback).tickers


def snapshot_history_path(
    definition: UniverseDefinition,
    project_root: str | Path | None = None,
    *,
    moment: datetime | None = None,
) -> Path:
    """Calcula o caminho versionado do snapshot."""

    root = _resolve_root(project_root)
    history_dir = root / "data" / "universes" / "history" / definition.code
    return history_dir / f"{definition.code}-{_timestamp_slug(moment)}.csv"


def save_snapshot(
    code: str,
    frame: pd.DataFrame,
    project_root: str | Path | None = None,
    *,
    source_label: str | None = None,
    updated_at: datetime | None = None,
    versioned: bool = True,
) -> tuple[Path, Path | None]:
    """Salva o snapshot canônico e, opcionalmente, uma cópia versionada."""

    definition = get_universe_definition(code, project_root)
    ensure_universe_directories(project_root)
    timestamp = updated_at or datetime.now()
    normalized = _normalize_loaded_frame(
        frame,
        definition,
        source_label=source_label,
        updated_at=_timestamp_label(timestamp),
    )

    canonical_path = definition.snapshot_path
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(canonical_path, index=False, encoding="utf-8")

    versioned_path: Path | None = None
    if versioned:
        versioned_path = snapshot_history_path(
            definition, project_root, moment=timestamp
        )
        versioned_path.parent.mkdir(parents=True, exist_ok=True)
        normalized.to_csv(versioned_path, index=False, encoding="utf-8")

    return canonical_path, versioned_path
