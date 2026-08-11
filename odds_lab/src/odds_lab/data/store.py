"""DuckDB query layer over hive-partitioned parquet. ARCHITECTURE.md §2.

Layout on disk (relative to `ChainStore.root`, the store *directory* -- not to be
confused with the ticker `root` argument every method takes):

    chains/underlying=<ROOT>/year=<Y>/month=<M>/part.parquet
    underlying/<ROOT>.parquet

Partition pruning: every read that knows its date range resolves the exact set of
(year, month) partition *files* on the Python side first and hands DuckDB that
explicit file list (plus column projection and a WHERE clause for the exact day/
expiry/etc). We never glob a whole root's directory to answer a single-day query --
see `_partition_paths_for_range`.

`closes()` boundary (documented precisely because callers use it for no-lookahead
lookback windows, CLAUDE.md rule 1): it returns every close with
`date <= end` -- **inclusive** of `end`. If the caller wants a lookback window that
must not see `end`'s own bar, they must pass `end - 1 trading day` themselves;
`closes()` will never return a date > `end`, but it WILL include `end` if the
store has a bar for it. This is intentional -- most callers use `closes(root, asof)`
where `asof` is itself the last permitted date.
"""

from __future__ import annotations

from datetime import date
from collections.abc import Sequence
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from odds_lab import schema

__all__ = ["ChainStore", "INTRADAY_DTYPES"]

_CHAIN_KEY = ["root", "quote_date", "expiry", "strike", "right"]
_UNDERLYING_KEY = ["root", "date"]

INTRADAY_DTYPES: dict[str, str] = {
    "root": "string",
    "expiry": "datetime64[ns]",
    "strike": "float64",
    "right": "string",  # 'C' | 'P'
    "quote_date": "datetime64[ns]",
    "ts": "datetime64[ns]",  # full minute-resolution timestamp
    "bid": "float64",
    "ask": "float64",
    "bid_size": "int32",
    "ask_size": "int32",
    "source": "string",
    "is_synthetic": "bool",
}
"""1-minute option QUOTE contract for the intraday-exit feature (task brief). Lives
here, not in `schema.py`: this change's edit scope is `data/store.py` +
`engine/*`/`config.py`/`cli.py`/`strategy/exits.py` only, and `schema.py` is out of
scope (owned/concurrently edited elsewhere) -- exactly the precedent
`engine/loop.py::FUNNEL_DTYPES` already sets for a table that doesn't belong in the
cross-module schema contract. Keyed on (root, expiry, strike, right, quote_date, ts).

The real ThetaData 1-minute QUOTE export schema is UNVERIFIED (ARCHITECTURE.md §0.5
only confirms `option_eod` and `option_ohlc_1m`/1-minute TRADE bars) -- this table's
shape is odds_lab's own normalized/canonical intraday-quote representation that
`data/providers/csv_export.py::read_quote_1m_csv` and
`data/providers/synthetic.py::SyntheticProvider.intraday_quotes` both produce rows
for; it is not itself a claim about the vendor's on-disk column names."""

_INTRADAY_KEY = ["root", "expiry", "strike", "right", "quote_date", "ts"]


def _validate_intraday(df: pd.DataFrame) -> pd.DataFrame:
    """Local, lightweight analog of `schema.validate_chain` for `INTRADAY_DTYPES`
    (kept here rather than in `schema.py` -- see `INTRADAY_DTYPES` docstring)."""
    missing = [c for c in INTRADAY_DTYPES if c not in df.columns]
    if missing:
        raise schema.SchemaError(f"intraday quotes missing columns: {missing}")
    extra = [c for c in df.columns if c not in INTRADAY_DTYPES]
    if extra:
        raise schema.SchemaError(f"intraday quotes has unknown columns: {extra}")
    out = df.copy()
    for col, dt in INTRADAY_DTYPES.items():
        out[col] = out[col].astype(dt)
    bad_right = ~out["right"].isin(["C", "P"])
    if bad_right.any():
        raise schema.SchemaError(f"{int(bad_right.sum())} intraday rows with right not in C/P")
    if (out["bid"] < 0).any() or (out["ask"] < 0).any():
        raise schema.SchemaError("negative intraday bid/ask")
    crossed = out["bid"] > out["ask"]
    if crossed.any():
        raise schema.SchemaError(f"{int(crossed.sum())} crossed intraday quotes (bid > ask)")
    return out[list(INTRADAY_DTYPES)]


def _empty_intraday() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in INTRADAY_DTYPES.items()})


def _ym(d: date) -> tuple[int, int]:
    return d.year, d.month


def _month_range(start: date, end: date) -> list[tuple[int, int]]:
    """Inclusive list of (year, month) tuples spanning [start, end]."""
    y, m = start.year, start.month
    out = []
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        m += 1
        if m == 13:
            m = 1
            y += 1
    return out


class ChainStore:
    """Read/write handle onto the parquet store. Cheap to construct repeatedly:
    the DuckDB connection is an in-memory, dependency-free handle opened lazily
    and cached on the instance."""

    def __init__(self, root: str | Path = "data"):
        self.root = Path(root)
        self._con: duckdb.DuckDBPyConnection | None = None
        self._intraday_avail_cache: dict[tuple[str, date], set[tuple]] = {}

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            self._con = duckdb.connect(database=":memory:")
        return self._con

    # -- paths -----------------------------------------------------------------

    def _chain_partition_path(self, root: str, year: int, month: int) -> Path:
        return self.root / "chains" / f"underlying={root}" / f"year={year}" / f"month={month:02d}" / "part.parquet"

    def _chain_root_dir(self, root: str) -> Path:
        return self.root / "chains" / f"underlying={root}"

    def _underlying_path(self, root: str) -> Path:
        return self.root / "underlying" / f"{root}.parquet"

    def _intraday_partition_path(self, root: str, quote_date: date) -> Path:
        """One file per (root, calendar day) -- deliberately NOT per (root, month)
        like the EOD chain partitions: pass 2 of the hybrid backtest only ever
        needs one contract-day at a time (task brief -- "a single contract-day is
        cheap to read"), and per-day files keep that read to exactly one small
        file instead of a whole month's worth of every-contract minute bars."""
        return self.root / "intraday" / f"underlying={root}" / f"date={quote_date.isoformat()}" / "part.parquet"

    def _partition_paths_for_range(self, root: str, start: date, end: date) -> list[Path]:
        paths = []
        for y, m in _month_range(start, end):
            p = self._chain_partition_path(root, y, m)
            if p.exists():
                paths.append(p)
        return paths

    # -- reads -------------------------------------------------------------------

    def chain(self, root: str, quote_date: date, expiry: date | None = None) -> pd.DataFrame:
        """One day's chain (optionally one expiry). Reads only that day's partition file."""
        paths = self._partition_paths_for_range(root, quote_date, quote_date)
        if not paths:
            return schema.empty_frame(schema.CHAIN_DTYPES)
        where = ['"root" = ?', '"quote_date" = ?']
        params: list = [root, pd.Timestamp(quote_date)]
        if expiry is not None:
            where.append('"expiry" = ?')
            params.append(pd.Timestamp(expiry))
        cols = ", ".join(f'"{c}"' for c in schema.CHAIN_COLUMNS)
        sql = (
            f"SELECT {cols} FROM read_parquet(?) WHERE " + " AND ".join(where)
        )
        df = self.con.execute(sql, [[str(p) for p in paths], *params]).fetchdf()
        if df.empty:
            return schema.empty_frame(schema.CHAIN_DTYPES)
        return schema.validate_chain(df, strict=True)

    def expiries(self, root: str, quote_date: date, dte_min: int = 0, dte_max: int = 400) -> list[date]:
        """Distinct expirations visible on `quote_date` with dte in [dte_min, dte_max]."""
        range_end = quote_date + pd.Timedelta(days=dte_max)
        paths = self._partition_paths_for_range(root, quote_date, range_end.date() if hasattr(range_end, "date") else range_end)
        if not paths:
            return []
        sql = (
            'SELECT DISTINCT "expiry" FROM read_parquet(?) '
            'WHERE "root" = ? AND "quote_date" = ? '
            'AND date_diff(\'day\', "quote_date", "expiry") BETWEEN ? AND ? '
            'ORDER BY "expiry"'
        )
        df = self.con.execute(sql, [[str(p) for p in paths], root, pd.Timestamp(quote_date), dte_min, dte_max]).fetchdf()
        if df.empty:
            return []
        return [d.date() if hasattr(d, "date") else d for d in pd.to_datetime(df["expiry"]).tolist()]

    def underlying(self, root: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        path = self._underlying_path(root)
        if not path.exists():
            return schema.empty_frame(schema.UNDERLYING_DTYPES)
        where = ['"root" = ?']
        params: list = [root]
        if start is not None:
            where.append('"date" >= ?')
            params.append(pd.Timestamp(start))
        if end is not None:
            where.append('"date" <= ?')
            params.append(pd.Timestamp(end))
        cols = ", ".join(f'"{c}"' for c in schema.UNDERLYING_DTYPES)
        sql = f"SELECT {cols} FROM read_parquet(?) WHERE " + " AND ".join(where) + ' ORDER BY "date"'
        df = self.con.execute(sql, [str(path), *params]).fetchdf()
        if df.empty:
            return schema.empty_frame(schema.UNDERLYING_DTYPES)
        return schema.validate_frame(df, schema.UNDERLYING_DTYPES, "underlying")

    def closes(self, root: str, end: date) -> pd.Series:
        """Date-indexed close series, `date <= end` inclusive. See module docstring
        for the exact boundary contract."""
        df = self.underlying(root, start=None, end=end)
        if df.empty:
            return pd.Series(dtype="float64", name="close")
        s = pd.Series(df["close"].to_numpy(), index=pd.DatetimeIndex(df["date"]).date, name="close")
        assert (pd.DatetimeIndex(df["date"]).date <= end).all(), "closes() returned a date > end"
        return s

    def trading_dates(self, root: str, start: date, end: date) -> list[date]:
        df = self.underlying(root, start=start, end=end)
        if df.empty:
            return []
        return [d.date() for d in pd.to_datetime(df["date"]).tolist()]

    def contains_synthetic(self, roots: Sequence[str] | None = None) -> bool:
        """True if any stored chain row is flagged synthetic.

        The run manifest must reflect the data actually consumed, not what the config
        asked for: a backtest pointed at a synthetic store via --store with the parquet
        provider would otherwise report real data and render no warning banner.
        """
        chains_dir = self.root / "chains"
        if not chains_dir.exists():
            return False
        wanted = set(roots) if roots else None
        for d in sorted(chains_dir.iterdir()):
            if not d.is_dir() or not d.name.startswith("underlying="):
                continue
            r = d.name.split("=", 1)[1]
            if wanted is not None and r not in wanted:
                continue
            pattern = str(d / "year=*" / "month=*" / "part.parquet")
            try:
                got = self._con.execute(
                    f"SELECT bool_or(is_synthetic) FROM read_parquet('{pattern}')"
                ).fetchone()
            except Exception:
                continue
            if got and got[0]:
                return True
        return False

    def coverage(self) -> pd.DataFrame:
        """One row per root with parquet chain data: min/max quote_date, row count,
        distinct expiry count. Full scan -- meant for occasional reporting, not the hot path."""
        chains_dir = self.root / "chains"
        rows = []
        if chains_dir.exists():
            for d in sorted(chains_dir.iterdir()):
                if not d.is_dir() or not d.name.startswith("underlying="):
                    continue
                r = d.name.split("=", 1)[1]
                pattern = str(d / "year=*" / "month=*" / "part.parquet")
                sql = (
                    "SELECT min(quote_date) AS min_date, max(quote_date) AS max_date, "
                    "count(*) AS n_rows, count(DISTINCT expiry) AS n_expiries "
                    "FROM read_parquet(?)"
                )
                res = self.con.execute(sql, [pattern]).fetchdf()
                if res.empty or pd.isna(res.loc[0, "min_date"]):
                    continue
                rows.append(
                    {
                        "root": r,
                        "min_date": res.loc[0, "min_date"],
                        "max_date": res.loc[0, "max_date"],
                        "n_rows": int(res.loc[0, "n_rows"]),
                        "n_expiries": int(res.loc[0, "n_expiries"]),
                    }
                )
        if not rows:
            return pd.DataFrame(columns=["root", "min_date", "max_date", "n_rows", "n_expiries"])
        return pd.DataFrame(rows)

    # -- intraday (1-minute quotes) -----------------------------------------------

    def intraday(self, root: str, expiry: date, strike: float, right: str, quote_date: date) -> pd.DataFrame:
        """One contract-day of 1-minute quotes, sorted by `ts`. Empty frame (never
        an error) if this contract-day was never ingested -- callers (engine/loop.py,
        engine/intraday.py) treat "no rows" as "fall back to EOD for this position on
        this day" and record that fallback explicitly (task requirement)."""
        path = self._intraday_partition_path(root, quote_date)
        if not path.exists():
            return _empty_intraday()
        df = pd.read_parquet(path)
        mask = (
            np.isclose(df["strike"].astype(float), float(strike))
            & (df["right"] == right)
            & (pd.to_datetime(df["expiry"]) == pd.Timestamp(expiry))
        )
        out = df.loc[mask].sort_values("ts").reset_index(drop=True)
        if out.empty:
            return _empty_intraday()
        return out[list(INTRADAY_DTYPES)]

    def intraday_available_keys(self, root: str, quote_date: date) -> set[tuple[date, float, str]]:
        """Distinct (expiry, strike, right) covered by this (root, quote_date)'s
        intraday partition -- cheap existence index, cached per instance (perf: the
        two-pass hybrid backtest calls this once per open position per day)."""
        cache_key = (root, quote_date)
        if cache_key in self._intraday_avail_cache:
            return self._intraday_avail_cache[cache_key]
        path = self._intraday_partition_path(root, quote_date)
        if not path.exists():
            keys: set[tuple[date, float, str]] = set()
        else:
            df = pd.read_parquet(path, columns=["expiry", "strike", "right"])
            keys = {
                ((e.date() if hasattr(e, "date") else e), float(s), str(r))
                for e, s, r in zip(pd.to_datetime(df["expiry"]), df["strike"], df["right"])
            }
        self._intraday_avail_cache[cache_key] = keys
        return keys

    def has_intraday(self, root: str, expiry: date, strike: float, right: str, quote_date: date) -> bool:
        return (expiry, float(strike), right) in self.intraday_available_keys(root, quote_date)

    def write_intraday(self, df: pd.DataFrame) -> None:
        """Merge-write 1-minute quotes, partitioned by (root, quote_date). Idempotent
        like `write_chain`: dedupes on `_INTRADAY_KEY`, new rows win."""
        if df.empty:
            return
        valid = _validate_intraday(df)
        qd = pd.to_datetime(valid["quote_date"]).dt.date
        for (root, d), part in valid.groupby([valid["root"], qd]):
            path = self._intraday_partition_path(str(root), d)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                existing = pd.read_parquet(path)
                merged = pd.concat([existing, part], ignore_index=True)
            else:
                merged = part
            merged = merged.drop_duplicates(subset=_INTRADAY_KEY, keep="last")
            merged = _validate_intraday(merged)
            merged = merged.sort_values(["expiry", "strike", "right", "ts"]).reset_index(drop=True)
            merged.to_parquet(path, index=False)
            self._intraday_avail_cache.pop((str(root), d), None)

    def has_data(self, root: str) -> bool:
        if self._underlying_path(root).exists():
            return True
        d = self._chain_root_dir(root)
        return d.exists() and any(d.rglob("part.parquet"))

    # -- writes ------------------------------------------------------------------

    def write_chain(self, df: pd.DataFrame) -> None:
        """Merge-write, partitioned by (root, year(quote_date), month(quote_date)).

        Idempotent: rows are deduped on `_CHAIN_KEY`, new rows win over any existing
        rows with the same key, so re-ingesting a day/expiry never duplicates rows.
        """
        if df.empty:
            return
        valid = schema.validate_chain(df, strict=True)
        qd = pd.to_datetime(valid["quote_date"])
        for (y, m), part in valid.groupby([qd.dt.year, qd.dt.month]):
            path = self._chain_partition_path(str(part["root"].iloc[0]), int(y), int(m))
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                existing = pd.read_parquet(path)
                merged = pd.concat([existing, part], ignore_index=True)
            else:
                merged = part
            merged = merged.drop_duplicates(subset=_CHAIN_KEY, keep="last")
            merged = schema.validate_chain(merged, strict=True)
            merged = merged.sort_values(["quote_date", "expiry", "right", "strike"]).reset_index(drop=True)
            merged.to_parquet(path, index=False)

    def write_underlying(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        valid = schema.validate_frame(df, schema.UNDERLYING_DTYPES, "underlying")
        for r, part in valid.groupby("root"):
            path = self._underlying_path(str(r))
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                existing = pd.read_parquet(path)
                merged = pd.concat([existing, part], ignore_index=True)
            else:
                merged = part
            merged = merged.drop_duplicates(subset=_UNDERLYING_KEY, keep="last")
            merged = schema.validate_frame(merged, schema.UNDERLYING_DTYPES, "underlying")
            merged = merged.sort_values("date").reset_index(drop=True)
            merged.to_parquet(path, index=False)
