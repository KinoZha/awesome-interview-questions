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
from pathlib import Path

import duckdb
import pandas as pd

from odds_lab import schema

__all__ = ["ChainStore"]

_CHAIN_KEY = ["root", "quote_date", "expiry", "strike", "right"]
_UNDERLYING_KEY = ["root", "date"]


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
