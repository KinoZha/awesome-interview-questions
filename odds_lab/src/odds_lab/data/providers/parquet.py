"""ChainProvider backed by an already-ingested local `ChainStore`.

Lets the rest of the pipeline (engine/report code, or a second-pass re-ingest into
another store) treat "data already on disk" the same way it treats a live vendor,
via the same `ChainProvider` interface. Read-only: `chain_eod`/`underlying_eod`
here always come back schema-valid because `ChainStore` only ever persists
schema-valid frames.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from odds_lab.data.providers.base import ChainProvider
from odds_lab.data.store import ChainStore

__all__ = ["ParquetProvider"]


class ParquetProvider(ChainProvider):
    name = "parquet"

    def __init__(self, store: str | Path | ChainStore = "data"):
        self.store = store if isinstance(store, ChainStore) else ChainStore(store)

    def trading_dates(self, start: date, end: date) -> list[date]:
        roots = [r for r in self.store.coverage()["root"].tolist()]
        dates: set[date] = set()
        for r in roots:
            dates.update(self.store.trading_dates(r, start, end))
        return sorted(dates)

    def expirations(self, root: str, quote_date: date) -> list[date]:
        return self.store.expiries(root, quote_date, dte_min=0, dte_max=400)

    def chain_eod(self, root: str, quote_date: date, expiry: date | None = None) -> pd.DataFrame:
        return self.store.chain(root, quote_date, expiry=expiry)

    def underlying_eod(self, root: str, start: date, end: date) -> pd.DataFrame:
        return self.store.underlying(root, start=start, end=end)
