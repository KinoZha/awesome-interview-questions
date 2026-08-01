"""provider -> normalized -> partitioned parquet. ARCHITECTURE.md §1.

Resumable + idempotent:
  - Idempotent: `ChainStore.write_chain`/`write_underlying` dedupe on their natural
    key (root, quote_date, expiry, strike, right) / (root, date) -- re-running with
    the same inputs never duplicates rows, regardless of `force`.
  - Resumable: at the *start* of a call, the set of (root, year, month) chain
    partitions already present on disk is snapshotted once. Any date whose month
    falls in that set is skipped entirely (no provider calls, no writes) unless
    `force=True`. The snapshot is taken once per call so a run that itself creates
    a partition mid-way does not start skipping its own later dates in the same
    month.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd

from odds_lab import schema
from odds_lab.data.providers.base import ChainProvider
from odds_lab.data.store import ChainStore

__all__ = ["ingest"]


def _trailing_div_yield(underlying: pd.DataFrame) -> float:
    """Rough annualized trailing dividend yield, seeded from observed ex-div payments.
    Used only to seed `quant.bs.enrich_chain`'s `q` -- not a schema field.

    STRATEGY.md §7.5: SPY/QQQ/IWM all pay quarterly. Rather than dividing the total
    dividends seen by the *window* length (biased whenever the window isn't close to
    a whole number of years -- e.g. a 2-month slice containing one $0.86 payment
    would extrapolate to a wildly overstated yield), we annualize from the average
    *per-payment* amount times an assumed 4 payments/year. Falls back to the
    window-length method if fewer than 2 payments were observed (not enough to
    trust "quarterly" -- e.g. a single-month ingest).
    """
    if underlying.empty or underlying["dividend"].sum() <= 0:
        return 0.0
    avg_price = float(underlying["close"].mean())
    if avg_price <= 0:
        return 0.0
    payments = underlying.loc[underlying["dividend"] > 0, "dividend"]
    if len(payments) >= 2:
        return max(0.0, (4.0 * float(payments.mean())) / avg_price)
    n_years = max((underlying["date"].max() - underlying["date"].min()).days / 365.0, 1e-6)
    total_div = float(underlying["dividend"].sum())
    return max(0.0, (total_div / n_years) / avg_price)


def _enrich(df: pd.DataFrame, r: float, q: float) -> pd.DataFrame:
    """Recompute iv/greeks via quant.bs.enrich_chain (CLAUDE.md rule 3: recompute,
    never trust the vendor blindly). Falls back to NaN greeks + non-strict validation
    if quant.bs isn't importable yet (lets this module's tests run standalone before
    that module lands)."""
    try:
        from odds_lab.quant.bs import enrich_chain
    except ImportError:
        out = df.copy()
        for col in schema.CHAIN_COLUMNS:
            if col not in out.columns:
                out[col] = pd.NA
        if "source" not in df.columns:
            out["source"] = "unknown"
        if "is_synthetic" not in df.columns:
            out["is_synthetic"] = False
        return schema.validate_chain(out, strict=False)
    return enrich_chain(df, r=r, q=q)


def ingest(
    provider: ChainProvider,
    store: ChainStore,
    roots: Sequence[str],
    start: date,
    end: date,
    *,
    dte_max: int = 400,
    progress: bool = True,
    force: bool = False,
    risk_free_rate: float = 0.02,
) -> dict:
    """Pull `provider` data for `roots` over `[start, end]`, normalize, enrich, and
    write it into `store`. Returns the manifest dict also written to
    `<store.root>/manifest.json`."""
    manifest: dict = {"roots": {}}

    for root in roots:
        already_complete: set[tuple[int, int]] = set()
        if not force:
            existing = store._partition_paths_for_range(root, start, end)
            for p in existing:
                # .../year=YYYY/month=MM/part.parquet
                y = int(p.parent.parent.name.split("=", 1)[1])
                m = int(p.parent.name.split("=", 1)[1])
                already_complete.add((y, m))

        underlying = provider.underlying_eod(root, start, end)
        if not underlying.empty:
            store.write_underlying(underlying)
        q_root = _trailing_div_yield(underlying)

        trading_dates = [d for d in provider.trading_dates(start, end) if start <= d <= end]
        rows_written = 0
        dates_written = 0

        # Batch per (year, month) -- matches the partition granularity, and means
        # `enrich_chain`/`write_chain` (each with real per-call fixed cost) run once
        # per partition instead of once per trading day.
        month_buffer: dict[tuple[int, int], list[pd.DataFrame]] = {}

        def _flush(key: tuple[int, int]) -> None:
            nonlocal rows_written
            chunks = month_buffer.pop(key, None)
            if not chunks:
                return
            month_chain = pd.concat(chunks, ignore_index=True)
            month_chain = _enrich(month_chain, r=risk_free_rate, q=q_root)
            store.write_chain(month_chain)
            rows_written += len(month_chain)
            if progress:
                y, m = key
                print(f"[ingest] {root} {y}-{m:02d}: +{len(month_chain)} rows")

        current_key: tuple[int, int] | None = None
        for d in trading_dates:
            if not force and (d.year, d.month) in already_complete:
                continue
            key = (d.year, d.month)
            if current_key is not None and key != current_key:
                _flush(current_key)
            current_key = key
            expirations = [e for e in provider.expirations(root, d) if (e - d).days <= dte_max]
            frames = []
            for exp in expirations:
                chunk = provider.chain_eod(root, d, exp)
                if chunk is not None and not chunk.empty:
                    frames.append(chunk)
            if not frames:
                continue
            month_buffer.setdefault(key, []).append(pd.concat(frames, ignore_index=True))
            dates_written += 1
        if current_key is not None:
            _flush(current_key)

        manifest["roots"][root] = {
            "dates_written_this_run": dates_written,
            "rows_written_this_run": rows_written,
            "skipped_partitions_this_run": sorted(f"{y}-{m:02d}" for y, m in already_complete),
        }

    coverage = store.coverage()
    manifest["coverage"] = json.loads(coverage.to_json(orient="records", date_format="iso"))
    manifest["start"] = start.isoformat()
    manifest["end"] = end.isoformat()
    manifest["dte_max"] = dte_max

    manifest_path = store.root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest
