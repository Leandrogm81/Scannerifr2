from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.universe_registry import (
    CUSTOM_WATCHLIST_DIR,
    default_universe_definitions,
    list_custom_watchlists,
    load_universe,
    save_snapshot,
)


def test_default_definitions_expose_expected_sizes() -> None:
    definitions = default_universe_definitions()

    assert definitions["b3_top100"].expected_size == 100
    assert definitions["us_top500"].expected_size == 500
    assert definitions["b3_top100"].market == "B3"
    assert definitions["us_top500"].currency == "USD"


def test_b3_snapshot_loads_100_tickers() -> None:
    snapshot = load_universe("b3_top100")

    assert snapshot.count == 100
    assert snapshot.tickers[0] == "B3SA3.SA"
    assert {"ticker", "market", "currency", "rank", "source", "updated_at"}.issubset(
        snapshot.frame.columns
    )
    assert snapshot.metadata["expected_size"] == 100


def test_us_snapshot_loads_500_tickers() -> None:
    snapshot = load_universe("us_top500")

    assert snapshot.count == 500
    assert snapshot.frame["market"].nunique() == 1
    assert snapshot.frame["currency"].nunique() == 1
    assert snapshot.frame["currency"].iloc[0] == "USD"


def test_save_snapshot_writes_canonical_and_versioned_files(tmp_path) -> None:
    frame = pd.DataFrame({"ticker": ["AAA"], "name": ["Alpha"]})

    canonical, versioned = save_snapshot(
        "b3_top100",
        frame,
        project_root=tmp_path,
        source_label="unit-test",
        updated_at=datetime(2026, 5, 19, 12, 34, 56),
    )

    assert canonical.exists()
    assert versioned is not None and versioned.exists()

    saved = pd.read_csv(canonical)
    assert saved.loc[0, "source"] == "unit-test"
    assert saved.loc[0, "updated_at"] == "2026-05-19 12:34:56"
    assert saved.loc[0, "market"] == "B3"
    assert saved.loc[0, "rank"] == 1


def test_custom_watchlist_directory_is_available() -> None:
    assert CUSTOM_WATCHLIST_DIR.exists()
    assert list_custom_watchlists() == []
