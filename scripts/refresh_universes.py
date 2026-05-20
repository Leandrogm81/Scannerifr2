#!/usr/bin/env python3
"""Atualiza os snapshots dos universos de ativos.

O script suporta fonte pública ou arquivo local como entrada e grava um
snapshot canônico junto com uma cópia versionada por data/hora.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from bs4 import BeautifulSoup

from src.universe_registry import (
    B3_SOURCE_URL,
    US_SOURCE_FALLBACK_URL,
    US_SOURCE_URL,
    get_universe_definition,
    save_snapshot,
)

B3_TARGET_ROWS = 100
US_TARGET_ROWS = 500


def _read_text_source(source: str) -> str:
    path = Path(source)
    if path.exists():
        return path.read_text(encoding="utf-8")

    with urlopen(source, timeout=60) as response:
        return response.read().decode("utf-8")


def _read_csv_source(source: str) -> pd.DataFrame:
    path = Path(source)
    if path.exists():
        return pd.read_csv(path)
    return pd.read_csv(source)


def _parse_br_int(value: object) -> int:
    text = str(value).strip()
    if not text:
        return 0
    return int(text.replace(".", ""))


def _parse_br_float(value: object) -> float:
    text = str(value).strip()
    if not text:
        return 0.0

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(".", "").replace(",", ".")
    else:
        text = text.replace(",", "")

    return float(text)


def _normalize_b3_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Ticker": "ticker",
        "Nome": "name",
        "Negócios": "trades",
        "Negocios": "trades",
        "Última (R$)": "last_price",
        "Ultima (R$)": "last_price",
        "Variação": "variation",
        "Variacao": "variation",
    }
    normalized = frame.rename(
        columns={old: new for old, new in rename_map.items() if old in frame.columns}
    )

    if "ticker" not in normalized.columns:
        raise ValueError("B3 source did not provide a ticker column")

    # Adicionar sufixo .SA para o Yahoo Finance
    normalized["ticker"] = (
        normalized["ticker"]
        .astype(str)
        .str.strip()
        .apply(lambda x: f"{x}.SA" if not x.endswith(".SA") else x)
    )

    if "name" not in normalized.columns:
        normalized["name"] = ""
    if "trades" in normalized.columns:
        normalized["trades"] = normalized["trades"].map(_parse_br_int)
    if "last_price" in normalized.columns:
        normalized["last_price"] = normalized["last_price"].map(_parse_br_float)
    if "variation" not in normalized.columns:
        normalized["variation"] = ""

    normalized = normalized.head(B3_TARGET_ROWS).copy()
    if len(normalized) < B3_TARGET_ROWS:
        raise ValueError(
            f"B3 source returned only {len(normalized)} rows; expected at least {B3_TARGET_ROWS}"
        )
    return normalized.reset_index(drop=True)


def _parse_b3_html(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError("B3 HTML source did not contain a table")

    rows: list[dict[str, object]] = []
    for tr in table.select("tbody tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.select("td")]
        if len(cells) < 5:
            continue
        rows.append(
            {
                "ticker": cells[0],
                "name": cells[1],
                "trades": _parse_br_int(cells[2]),
                "last_price": _parse_br_float(cells[3]),
                "variation": cells[4],
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("B3 HTML source did not yield any rows")
    return _normalize_b3_frame(frame)


def fetch_b3_snapshot(source: str) -> pd.DataFrame:
    path = Path(source)
    if path.exists() and path.suffix.lower() == ".csv":
        return _normalize_b3_frame(_read_csv_source(source))

    if path.exists():
        return _parse_b3_html(_read_text_source(source))

    if source.lower().endswith(".csv"):
        return _normalize_b3_frame(_read_csv_source(source))

    return _parse_b3_html(_read_text_source(source))


def _normalize_us_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Symbol": "ticker",
        "Name": "name",
        "Security": "name",
        "Sector": "sector",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
        "Headquarters Location": "headquarters_location",
        "Date added": "date_added",
        "CIK": "cik",
        "Founded": "founded",
    }
    normalized = frame.rename(
        columns={old: new for old, new in rename_map.items() if old in frame.columns}
    )

    if "ticker" not in normalized.columns:
        raise ValueError("US source did not provide a ticker column")
    if "name" not in normalized.columns:
        normalized["name"] = ""

    normalized = normalized.head(US_TARGET_ROWS).copy()
    if len(normalized) < US_TARGET_ROWS:
        raise ValueError(
            f"US source returned only {len(normalized)} rows; expected at least {US_TARGET_ROWS}"
        )
    return normalized.reset_index(drop=True)


def fetch_us_snapshot(source: str) -> pd.DataFrame:
    return _normalize_us_frame(_read_csv_source(source))


def _candidate_sources(primary: str, fallback: str | None = None) -> Iterable[str]:
    yield primary
    if fallback:
        yield fallback


def _refresh_one(
    code: str,
    source_candidates: Iterable[str],
    *,
    dry_run: bool,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    definition = get_universe_definition(code, project_root)
    last_error: Exception | None = None

    for source in source_candidates:
        try:
            if code == "b3_top100":
                frame = fetch_b3_snapshot(source)
            elif code == "us_top500":
                frame = fetch_us_snapshot(source)
            else:
                raise ValueError(f"Unsupported universe code: {code}")

            result = {
                "code": code,
                "source": source,
                "rows": len(frame),
                "expected": definition.expected_size,
                "canonical_path": str(definition.snapshot_path),
                "versioned_path": None,
                "dry_run": dry_run,
            }

            if not dry_run:
                canonical_path, versioned_path = save_snapshot(
                    code,
                    frame,
                    project_root,
                    source_label=source,
                    updated_at=datetime.now(),
                    versioned=True,
                )
                result["canonical_path"] = str(canonical_path)
                result["versioned_path"] = (
                    str(versioned_path) if versioned_path else None
                )

            return result
        except (
            FileNotFoundError,
            URLError,
            OSError,
            ValueError,
            pd.errors.ParserError,
        ) as exc:
            last_error = exc

    assert last_error is not None
    raise last_error


def _print_report(results: list[dict[str, object]], dry_run: bool) -> None:
    for item in results:
        state = "DRY-RUN" if dry_run else "OK"
        print(
            f"[{state}] {item['code']} -> {item['rows']} rows | "
            f"source={item['source']} | canonical={item['canonical_path']}"
        )
        if item.get("versioned_path"):
            print(f"          versioned={item['versioned_path']}")

    if dry_run:
        print("Dry-run concluído: nenhum arquivo foi alterado.")
    else:
        print("Refresh concluído: snapshots canônicos e versionados foram atualizados.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Atualiza os snapshots dos universos do IFR2."
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Raiz do projeto (padrão: detectada automaticamente)",
    )
    parser.add_argument(
        "--b3-source", default=B3_SOURCE_URL, help="URL ou arquivo local da lista B3"
    )
    parser.add_argument(
        "--us-source", default=US_SOURCE_URL, help="URL ou arquivo local da lista US"
    )
    parser.add_argument(
        "--us-fallback-source",
        default=US_SOURCE_FALLBACK_URL,
        help="Fallback adicional para o universo US",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Baixa e valida, mas não grava arquivos"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root

    results = [
        _refresh_one(
            "b3_top100",
            _candidate_sources(args.b3_source),
            dry_run=args.dry_run,
            project_root=project_root,
        ),
        _refresh_one(
            "us_top500",
            _candidate_sources(args.us_source, args.us_fallback_source),
            dry_run=args.dry_run,
            project_root=project_root,
        ),
    ]
    _print_report(results, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
