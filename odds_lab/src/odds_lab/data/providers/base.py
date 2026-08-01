"""ChainProvider ABC -- the single interface every data source implements.

Strategy/engine code never talks to a provider directly (CLAUDE.md rule 4); only
`data/ingest.py` does, writing normalized parquet that `data/store.py` then serves.
See ARCHITECTURE.md §0 "Provider-agnostic data layer".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

__all__ = ["ChainProvider"]


class ChainProvider(ABC):
    """Source of option chains + underlying EOD bars. One implementation per vendor."""

    name: str

    @abstractmethod
    def trading_dates(self, start: date, end: date) -> list[date]:
        """Trading dates in [start, end] for which this provider has data."""

    @abstractmethod
    def expirations(self, root: str, quote_date: date) -> list[date]:
        """Expirations listed as of `quote_date`, ascending, all > quote_date."""

    @abstractmethod
    def chain_eod(self, root: str, quote_date: date, expiry: date | None = None) -> pd.DataFrame:
        """EOD chain snapshot. One row per contract.

        Returns at least `schema.CHAIN_REQUIRED` columns (schema.validate_chain(strict=False)
        must pass). Greeks/iv are optional here -- `quant.bs.enrich_chain` fills them
        authoritatively downstream. If `expiry` is None, returns contracts across all
        expirations returned by `expirations(root, quote_date)`.
        """

    @abstractmethod
    def underlying_eod(self, root: str, start: date, end: date) -> pd.DataFrame:
        """Daily underlying bars in [start, end]. Conforms to schema.UNDERLYING_DTYPES."""
