"""CsvExportProvider -- reads a DIRECTORY of real ThetaData CSV bulk exports.

ARCHITECTURE.md "Real ThetaData CSV export ground truth" documents this in full;
this docstring is the code-adjacent summary. Ground truth below was confirmed by
directly parsing the two fixtures committed at:
  data/samples/thetadata_spy_eod_20250819.csv.gz
  data/samples/thetadata_spy_ohlc_1m_20250819_exp20251219.csv.gz
This is the path that matters most operationally: the user's real chain history is
hundreds of MB to GB of files shaped exactly like these, on their own machine, and
will never pass through this sandbox -- so this module, not thetadata.py's REST
client, is the adapter to trust for bulk CSV-export ingestion.

Confirmed EOD chain export header (exact column order, `EOD_HEADER` below):
  symbol,expiration,strike,right,created,last_trade,open,high,low,close,volume,count,
  bid_size,bid_exchange,bid,bid_condition,ask_size,ask_exchange,ask,ask_condition

Confirmed 1-minute OHLC export header (exact column order, `OHLC_1M_HEADER` below):
  symbol,expiration,strike,right,timestamp,open,high,low,close,volume,count,vwap

Traps, confirmed by measurement on the fixtures -- do not "fix" these later:
  - `strike` is DOLLARS in this export mode (690.000), not tenths-of-a-cent. Unlike
    thetadata.py's REST client, there is no ambiguity to auto-detect here; a strike
    grid that doesn't look like dollars relative to the joined underlying price is
    treated as a vendor format change and raises loudly (`_sanity_check_strike_scale`)
    rather than being silently rescaled.
  - `right` is the string "CALL"/"PUT", mapped explicitly to 'C'/'P' (`_map_right`).
  - `expiration` is an ISO date string; `created`/`last_trade` are ISO timestamps.
    There is no ms_of_day and no YYYYMMDD int in this export mode -- unlike
    thetadata.py's REST client this is a whole-day-at-once export, not an intraday
    snapshot, so there is no meaningful sub-day timestamp to report. `ms_of_day` is
    set to 0 for every row (matching thetadata.py's own "absent -> 0" default, and
    required by `schema.validate_chain(strict=True)`, which `ChainStore.write_chain`
    enforces) -- it is NOT a real intraday timestamp here and must not be read as one.
  - `close`/`last` is NOT a mark. 4107 of 8768 EOD fixture rows have close == 0 while
    carrying a live bid/ask (no trade that day). Every mid/mark computed from this
    provider's output MUST come from bid/ask (quant.bs.enrich_chain already does
    this), never from `last`. `close` is mapped to `last` purely for reporting/
    data-quality comparison, exactly as thetadata.py does.
  - `vwap` in the 1-minute export is not a trustworthy per-bar price either: 97.2% of
    bars in the fixture are all-zero (no trade that minute), and 43645 of those
    zero-close bars still carry vwap > 0 (a stale/carried value, not a real trade
    price). `read_ohlc_1m_csv` below refuses to emit a `price` for an all-zero bar
    even when vwap is nonzero -- see its docstring.
  - There is NO underlying_price, NO open_interest, NO iv, and NO greeks column in
    this export. `underlying_price` MUST be resolved from a separate underlying EOD
    source (see `underlying_path`/`underlying_glob`) -- this provider never fills
    NaN and never silently substitutes 0. If it cannot resolve one, it raises
    `CsvExportError` naming exactly which file(s)/pattern it looked for. A put-call
    parity fallback exists (`infer_underlying_from_parity=True`) but is OFF by
    default and, when used, is stamped into `source` ("csv_export+parity") so it is
    never confused with a real quote.
  - `open_interest` is absent from this export entirely -> every row gets the
    `OPEN_INTEREST_UNKNOWN` sentinel, never 0. See that constant's docstring for the
    policy this exists to protect (CostModel.min_open_interest silently rejecting
    every row if unknown OI came through as 0).

UNVERIFIED (no real fixture covers this; flagged rather than guessed silently):
  - The exact filename/column convention for a standalone underlying-stock EOD
    export. `underlying_path`/`underlying_glob` are therefore explicit knobs, not an
    assumed single convention -- see `_resolve_underlying_path`.
  - Whether a real multi-day export directory names files with the 8-digit date
    pattern used by the fixtures (`..._20250819...`). `_index_eod_files` assumes it
    and raises a clear, itemized error if no file matches for a requested date.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from odds_lab import schema
from odds_lab.data.providers.base import ChainProvider

__all__ = [
    "CsvExportProvider",
    "CsvExportError",
    "OPEN_INTEREST_UNKNOWN",
    "EOD_HEADER",
    "OHLC_1M_HEADER",
    "read_ohlc_1m_csv",
]


class CsvExportError(Exception):
    """Base for every CsvExportProvider error. Always names the file/pattern involved."""


OPEN_INTEREST_UNKNOWN = -1
"""Sentinel written to `open_interest` when a source does not report it at all
(the real ThetaData bulk EOD chain export never does).

Policy: unknown OI must never be represented as 0. `strategy/selector.py`'s
liquidity filter rejects the short leg whenever
`open_interest < costs.min_open_interest` (default 100) -- if a provider defaulted
missing OI to 0, every single row would fail that check and the backtest would
silently place zero trades with no error anywhere. -1 is unambiguous (real OI is
never negative) and sorts below every real threshand a caller can use, so
`open_interest < 0` is the correct guard for "unknown, don't use for liquidity
filtering" wherever a consumer reads this column. As of this change,
`strategy/selector.py` is out of scope for this module (owned by a concurrent
change) and still treats `open_interest < min_open_interest` as it always has;
this sentinel does not yet make that comparison "unknown-aware" on its own -- a
consumer must explicitly special-case `open_interest < 0`. That follow-up is
flagged, not hidden: see docs/ARCHITECTURE.md's data-source section.
"""

EOD_HEADER: list[str] = [
    "symbol", "expiration", "strike", "right", "created", "last_trade",
    "open", "high", "low", "close", "volume", "count",
    "bid_size", "bid_exchange", "bid", "bid_condition",
    "ask_size", "ask_exchange", "ask", "ask_condition",
]
"""Exact, in-order header of a real ThetaData EOD chain bulk export."""

OHLC_1M_HEADER: list[str] = [
    "symbol", "expiration", "strike", "right", "timestamp",
    "open", "high", "low", "close", "volume", "count", "vwap",
]
"""Exact, in-order header of a real ThetaData 1-minute OHLC bulk export."""

_RIGHT_MAP = {"CALL": "C", "PUT": "P", "C": "C", "P": "P"}
_DATE_RE = re.compile(r"(\d{8})")


def _map_right(series: pd.Series) -> pd.Series:
    upper = series.astype(str).str.upper().str.strip()
    mapped = upper.map(_RIGHT_MAP)
    bad = mapped.isna()
    if bad.any():
        bad_vals = sorted(upper[bad].unique())[:5]
        raise CsvExportError(
            f"unrecognized option `right` value(s): {bad_vals!r} "
            "(expected one of CALL/PUT/C/P)"
        )
    return mapped.astype("string")


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """`.csv` and `.csv.gz` both handled -- pandas infers compression from suffix."""
    return pd.read_csv(path, compression="infer", **kwargs)


def read_ohlc_1m_csv(path: str | Path, chunksize: int = 250_000) -> pd.DataFrame:
    """Load a ThetaData 1-minute OHLC bulk export, streamed in chunks.

    Adds two columns beyond the raw header: `all_zero_bar` (open==high==low==close==
    volume==0, i.e. no trade printed that minute) and `price`, which is `close` on a
    real bar and NaN on an all-zero bar -- `vwap` is explicitly NOT used as a
    fallback there (module docstring: 43645 all-zero-close bars in the fixture still
    carry vwap > 0, a stale/carried value, not a traded price). Anyone computing
    returns off this frame must use `price`, never `close` or `vwap` directly.
    """
    path = Path(path)
    chunks: list[pd.DataFrame] = []
    for chunk in _read_csv(path, chunksize=chunksize):
        missing = [c for c in OHLC_1M_HEADER if c not in chunk.columns]
        if missing:
            raise CsvExportError(
                f"{path}: 1-minute OHLC export missing expected column(s) {missing}; "
                f"got {list(chunk.columns)}"
            )
        all_zero = (chunk[["open", "high", "low", "close", "volume"]] == 0).all(axis=1)
        price = chunk["close"].where(~all_zero, other=np.nan)
        chunks.append(chunk.assign(all_zero_bar=all_zero, price=price))
    if not chunks:
        return pd.DataFrame(columns=[*OHLC_1M_HEADER, "all_zero_bar", "price"])
    return pd.concat(chunks, ignore_index=True)


@dataclass
class CsvExportProvider(ChainProvider):
    """`ChainProvider` over a directory of real ThetaData CSV bulk exports.

    One EOD chain file per (root, quote_date) is assumed (matches the fixture
    naming: `..._eod_YYYYMMDD.csv[.gz]`), discovered via `eod_glob` + an 8-digit
    date embedded in the filename. All expirations for that day live in the one
    file (unlike thetadata.py's REST client, which fetches per-expiry).

    `underlying_price` is resolved from a separate source -- pass `underlying_path`
    (a single consolidated stock-EOD CSV/CSV.GZ) explicitly, or let
    `underlying_glob` discover one in `directory`. Neither is assumed silently: if
    neither resolves, `chain_eod`/`underlying_eod` raise `CsvExportError` naming
    exactly what was tried. Set `infer_underlying_from_parity=True` to opt into an
    approximate put-call-parity fallback instead (off by default, flagged in
    `source` when used).
    """

    directory: str | Path
    name: str = field(default="csv_export", init=False)
    eod_glob: str = "*eod*.csv*"
    underlying_path: str | Path | None = None
    underlying_glob: str = "*underlying*.csv*"
    chunksize: int = 250_000
    infer_underlying_from_parity: bool = False
    risk_free_rate: float = 0.02
    progress: bool = True

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        if not self.directory.is_dir():
            raise CsvExportError(f"CsvExportProvider directory not found: {self.directory}")
        self._eod_index: dict[date, Path] | None = None
        self._underlying_cache: dict[str, pd.DataFrame] = {}

    # -- file discovery ------------------------------------------------------------

    def _index_eod_files(self) -> dict[date, Path]:
        if self._eod_index is not None:
            return self._eod_index
        idx: dict[date, Path] = {}
        for p in sorted(self.directory.glob(self.eod_glob)):
            m = _DATE_RE.search(p.name)
            if not m:
                continue
            try:
                d = datetime.strptime(m.group(1), "%Y%m%d").date()
            except ValueError:
                continue
            idx[d] = p
        self._eod_index = idx
        return idx

    def _eod_file(self, quote_date: date) -> Path:
        idx = self._index_eod_files()
        path = idx.get(quote_date)
        if path is None:
            raise CsvExportError(
                f"no EOD export found for {quote_date.isoformat()} in {self.directory} "
                f"matching glob {self.eod_glob!r} with an embedded YYYYMMDD date "
                f"(found dates: {sorted(d.isoformat() for d in idx)[:10]}...)"
                if idx
                else f"no EOD export files at all matched {self.eod_glob!r} in {self.directory}"
            )
        return path

    def _resolve_underlying_path(self) -> Path:
        if self.underlying_path is not None:
            p = Path(self.underlying_path)
            if not p.exists():
                raise CsvExportError(
                    f"CsvExportProvider(underlying_path={p}) does not exist"
                )
            return p
        candidates = sorted(self.directory.glob(self.underlying_glob))
        if not candidates:
            raise CsvExportError(
                "could not resolve an underlying EOD source: no `underlying_path` was "
                f"given, and nothing in {self.directory} matched underlying_glob="
                f"{self.underlying_glob!r}. Pass an explicit "
                "CsvExportProvider(underlying_path=...) pointing at a stock EOD CSV "
                "export, or set infer_underlying_from_parity=True to opt into the "
                "approximate put-call-parity fallback instead."
            )
        return candidates[0]

    # -- underlying ------------------------------------------------------------------

    def _load_underlying(self, root: str) -> pd.DataFrame:
        if root in self._underlying_cache:
            return self._underlying_cache[root]
        path = self._resolve_underlying_path()
        raw = _read_csv(path)
        cols = {c.lower(): c for c in raw.columns}

        def _col(name: str, required: bool = True) -> str | None:
            if name in cols:
                return cols[name]
            if required:
                raise CsvExportError(
                    f"underlying source {path} is missing required column {name!r}; "
                    f"got columns {list(raw.columns)}"
                )
            return None

        out = pd.DataFrame(
            {
                "root": root,
                "date": pd.to_datetime(raw[_col("date")]),
                "open": raw[_col("open")].astype(float),
                "high": raw[_col("high")].astype(float),
                "low": raw[_col("low")].astype(float),
                "close": raw[_col("close")].astype(float),
                "volume": raw[_col("volume")].astype("int64"),
            }
        )
        div_col = _col("dividend", required=False)
        out["dividend"] = raw[div_col].astype(float) if div_col else 0.0
        out = schema.validate_frame(out, schema.UNDERLYING_DTYPES, "underlying")
        self._underlying_cache[root] = out
        return out

    def underlying_eod(self, root: str, start: date, end: date) -> pd.DataFrame:
        df = self._load_underlying(root)
        mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        return df.loc[mask].reset_index(drop=True)

    def _underlying_price(self, root: str, quote_date: date) -> float:
        df = self._load_underlying(root)
        row = df.loc[df["date"] == pd.Timestamp(quote_date)]
        if row.empty:
            raise CsvExportError(
                f"underlying source has no row for {root} on {quote_date.isoformat()} "
                f"(dates available: {df['date'].min()} .. {df['date'].max()})"
            )
        return float(row["close"].iloc[0])

    # -- ChainProvider -----------------------------------------------------------

    def trading_dates(self, start: date, end: date) -> list[date]:
        return sorted(d for d in self._index_eod_files() if start <= d <= end)

    def expirations(self, root: str, quote_date: date) -> list[date]:
        path = self._eod_file(quote_date)
        raw = _read_csv(path, usecols=["expiration"])
        exps = pd.to_datetime(raw["expiration"]).dt.date
        return sorted(d for d in set(exps) if d > quote_date)

    def chain_eod(self, root: str, quote_date: date, expiry: date | None = None) -> pd.DataFrame:
        path = self._eod_file(quote_date)
        missing = [c for c in EOD_HEADER if c not in _read_csv(path, nrows=0).columns]
        if missing:
            raise CsvExportError(f"{path}: EOD export missing expected column(s) {missing}")

        expiry_str = expiry.isoformat() if expiry is not None else None
        t0 = time.perf_counter()
        n_read = 0
        chunks: list[pd.DataFrame] = []
        for chunk in _read_csv(path, chunksize=self.chunksize):
            n_read += len(chunk)
            if expiry_str is not None:
                chunk = chunk[chunk["expiration"] == expiry_str]
            if not chunk.empty:
                chunks.append(chunk)
        elapsed = time.perf_counter() - t0
        if self.progress:
            rate = n_read / elapsed if elapsed > 0 else float("inf")
            print(f"[csv_export] {path.name}: streamed {n_read} rows in {elapsed:.2f}s ({rate:,.0f} rows/s)")

        raw = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=EOD_HEADER)
        if raw.empty:
            return schema.empty_frame(schema.CHAIN_DTYPES)

        bad_symbol = set(raw["symbol"].astype(str).str.upper().unique()) - {root.upper()}
        if bad_symbol:
            raise CsvExportError(
                f"{path}: expected symbol {root!r} but file also contains {sorted(bad_symbol)} "
                "-- point CsvExportProvider.eod_glob at a directory holding one root per file, "
                "or pass the matching `root`."
            )

        right = _map_right(raw["right"])

        source = "csv_export"
        try:
            underlying_price = self._underlying_price(root, quote_date)
        except CsvExportError:
            if not self.infer_underlying_from_parity:
                raise
            underlying_price = self._infer_underlying_via_parity(raw, right, quote_date)
            source = "csv_export+parity"

        strike = raw["strike"].astype(float)
        self._sanity_check_strike_scale(strike, underlying_price, path)

        out = pd.DataFrame(
            {
                "root": root,
                "quote_date": pd.Timestamp(quote_date),
                "ms_of_day": 0,  # whole-day export, no intraday timestamp -- see module docstring
                "expiry": pd.to_datetime(raw["expiration"]),
                "strike": strike,
                "right": right,
                "bid": raw["bid"].astype(float),
                "ask": raw["ask"].astype(float),
                "bid_size": raw["bid_size"].astype("int32"),
                "ask_size": raw["ask_size"].astype("int32"),
                # `last`/`close` is NOT a mark -- module docstring. Kept only for
                # data-quality comparisons, exactly as thetadata.py does.
                "last": raw["close"].astype(float),
                "volume": raw["volume"].astype("int64"),
                "open_interest": OPEN_INTEREST_UNKNOWN,
                "underlying_price": float(underlying_price),
                "source": source,
                "is_synthetic": False,
            }
        )
        return schema.validate_chain(out, strict=False)

    # -- helpers ---------------------------------------------------------------------

    @staticmethod
    def _sanity_check_strike_scale(strike: pd.Series, underlying_price: float, path: Path) -> None:
        """Ground truth: strike is DOLLARS in this export mode (docstring). Fail loud
        instead of silently mis-scaling if a future export doesn't look like that."""
        med = float(strike.median())
        if med > 20 * max(underlying_price, 1e-6):
            raise CsvExportError(
                f"{path}: median strike {med:.1f} is >20x the joined underlying price "
                f"{underlying_price:.2f} -- this export does not look like the confirmed "
                "dollars-strike CSV format (ARCHITECTURE.md data-source section). Refusing "
                "to silently rescale; verify the export and, if the vendor format really "
                "changed, update CsvExportProvider accordingly."
            )

    def _infer_underlying_via_parity(
        self, raw: pd.DataFrame, right: pd.Series, quote_date: date
    ) -> float:
        """Approximate, OFF-by-default fallback: S ~= C_mid - P_mid + K*disc(r, T) at
        the nearest-expiry, most-ATM strike. Ignores dividends -- a real approximation,
        not a substitute for a real underlying feed. Never used unless
        `infer_underlying_from_parity=True`, and always stamped into `source`."""
        mid = (raw["bid"].astype(float) + raw["ask"].astype(float)) / 2.0
        df = pd.DataFrame(
            {
                "expiry": pd.to_datetime(raw["expiration"]),
                "strike": raw["strike"].astype(float),
                "right": right,
                "mid": mid,
            }
        )
        calls = df.loc[df["right"] == "C", ["expiry", "strike", "mid"]].rename(columns={"mid": "call_mid"})
        puts = df.loc[df["right"] == "P", ["expiry", "strike", "mid"]].rename(columns={"mid": "put_mid"})
        pairs = calls.merge(puts, on=["expiry", "strike"])
        if pairs.empty:
            raise CsvExportError(
                "cannot infer underlying via put-call parity: no matching call/put "
                "strikes found in this file"
            )
        nearest = pairs["expiry"].min()
        pairs = pairs[pairs["expiry"] == nearest]
        T = max((nearest.date() - quote_date).days, 0) / 365.0
        disc = math.exp(-self.risk_free_rate * T)
        implied_s = pairs["call_mid"] - pairs["put_mid"] + pairs["strike"] * disc
        # most-ATM pair (smallest |call_mid - put_mid|) is the least sensitive to the
        # discounting/dividend approximation above.
        atm_idx = (pairs["call_mid"] - pairs["put_mid"]).abs().idxmin()
        return float(implied_s.loc[atm_idx]) if len(implied_s) == 1 else float(implied_s.median())
