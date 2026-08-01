"""`odds-lab probe` -- confirm or correct provider UNVERIFIED assumptions without
guessing. ARCHITECTURE.md §0 "Schema is probed, not assumed." Three independent
capabilities live here:

1. `probe()` -- REST/JSON Terminal schema probe (original capability). Hits every
   endpoint `ThetaDataProvider` uses for one (root, quote_date), dumps the raw
   `header.format` array from each, diffs it against the field names `thetadata.py`
   looks for, and writes `runs/probe-<date>.md`. Degrades gracefully if the
   terminal isn't running.

2. `probe_csv()` -- profiles a DIRECTORY of ThetaData CSV bulk exports on the
   user's own machine (never uploaded here) without reading any file whole. The
   two committed fixtures (`data/samples/thetadata_spy_eod_20250819.csv.gz`,
   `data/samples/thetadata_spy_ohlc_1m_20250819_exp20251219.csv.gz`) are already
   ground truth for `option_eod`/`option_ohlc_1m` -- everything else in a real
   export directory (quote_1m, trade, tick, a standalone underlying/stock EOD
   file) is UNVERIFIED and classified by header-driven heuristic only, never by
   filename. Writes `runs/probe-csv-<timestamp>.md` plus a compact stdout summary.

3. `probe_coverage_rest()` / `probe_coverage_csv()` -- answers the single most
   consequential unresolved question (ARCHITECTURE.md §0 "provider-agnostic data
   layer": ThetaData's docs disagree on when SPY/IWM CTA-tape coverage starts,
   2017 vs 2020, while QQQ UTP-tape is documented from 2012-06). The REST variant
   binary-searches the earliest usable date per (root, data kind) against a live
   terminal -- O(log days) requests, not a linear day-by-day scan -- and reports a
   plain-language verdict. The CSV variant answers the same question for free by
   reading the local export directory's own date index. Writes
   `runs/coverage-<timestamp>.md`.

   Binary search assumes coverage-start is monotonic in calendar date, which is
   true for "does the vendor have this root/kind at all" but not perfectly true at
   day granularity across weekends/holidays (a closed-market day always reports
   "no data" regardless of vendor coverage). `_nudge_to_weekday` reduces -- it does
   not eliminate -- the chance of landing exactly on such a day mid-search; treat
   the reported floor as accurate to within a few calendar days, not to the day,
   and always sanity-check it against the sample-density table in the report.
"""

from __future__ import annotations

import csv as _csv
import gzip
import io
import math
import re
import struct
import tarfile
import time as _time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from odds_lab import schema
from odds_lab.data.providers import csv_export as csvx
from odds_lab.data.providers import thetadata as td

__all__ = ["probe", "probe_csv", "probe_coverage_rest", "probe_coverage_csv"]


def _endpoint_check(provider: td.ThetaDataProvider, path: str, params: dict) -> dict:
    """Call one endpoint and summarize what came back, never raising."""
    result: dict = {"path": path, "params": params}
    try:
        payload = provider._get(path, params)
    except td.ThetaNotRunning as exc:
        result["error"] = f"ThetaNotRunning: {exc}"
        return result
    except td.ThetaError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    except Exception as exc:  # noqa: BLE001 -- probe must never crash on a bad payload
        result["error"] = f"unexpected {type(exc).__name__}: {exc}"
        return result

    try:
        df, fmt = td._rows_to_frame(payload)
    except td.ThetaError as exc:
        result["error"] = f"could not parse rows: {exc}"
        result["raw_keys"] = list(payload) if isinstance(payload, dict) else None
        return result

    result["format"] = fmt
    result["n_rows"] = len(df)
    result["first_rows"] = df.head(3).to_dict(orient="records")
    return result


def _diff(expected_canonical: list[str], fmt: list[str] | None) -> tuple[list[str], list[str]]:
    """(fields we need and didn't find, fields present that we ignore)."""
    if fmt is None:
        return list(expected_canonical), []
    found_aliases = set()
    missing = []
    for canon in expected_canonical:
        alias = td._find_alias(fmt, canon)
        if alias is None:
            missing.append(canon)
        else:
            found_aliases.add(alias)
    ignored = [f for f in fmt if f not in found_aliases]
    return missing, ignored


_EOD_FIELDS = ["root", "expiration", "strike", "right", "date", "ms_of_day", "bid", "ask",
               "bid_size", "ask_size", "last", "volume", "open_interest"]
_GREEKS_FIELDS = ["root", "expiration", "strike", "right", "delta", "gamma", "theta", "vega", "rho", "iv"]
_STOCK_FIELDS = ["date", "open", "high", "low", "close", "volume"]


def probe(base_url: str | None, version: str, root: str, quote_date: date) -> Path:
    """Run the live schema probe and write `runs/probe-<date>.md`. Returns the
    report path (always -- even a "terminal not running" report is written)."""
    provider = td.ThetaDataProvider(base_url=base_url, version=version)
    paths = provider._paths

    # pick a plausible expiry to probe the option endpoints with: try to list
    # real expirations first, fall back to a synthetic guess so option_eod/greeks
    # still get *some* response (and a useful error) even if list/expirations itself
    # is broken.
    exp_check = _endpoint_check(provider, paths["expirations"], {"root": root})
    expiry = None
    if "format" in exp_check and exp_check["n_rows"] > 0:
        try:
            expiry = provider.expirations(root, quote_date)[0]
        except Exception:
            expiry = None
    if expiry is None:
        expiry = date(quote_date.year, quote_date.month, 28)

    option_params = {
        "root": root,
        "exp": expiry.strftime("%Y%m%d"),
        "start_date": quote_date.strftime("%Y%m%d"),
        "end_date": quote_date.strftime("%Y%m%d"),
        "strike": "*",
        "right": "*",
    }
    stock_params = {
        "root": root,
        "start_date": quote_date.strftime("%Y%m%d"),
        "end_date": quote_date.strftime("%Y%m%d"),
    }

    checks = {
        "expirations": exp_check,
        "strikes": _endpoint_check(provider, paths["strikes"], {"root": root, "exp": expiry.strftime("%Y%m%d")}),
        "option_eod": _endpoint_check(provider, paths["option_eod"], option_params),
        "option_greeks": _endpoint_check(provider, paths["option_greeks"], option_params),
        "stock_eod": _endpoint_check(provider, paths["stock_eod"], stock_params),
    }

    terminal_unreachable = all("ThetaNotRunning" in c.get("error", "") for c in checks.values())

    lines: list[str] = []
    lines.append(f"# ThetaData live probe -- {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"- base_url: `{provider.base_url}`")
    lines.append(f"- version: `{version}`")
    lines.append(f"- root: `{root}`  quote_date: `{quote_date.isoformat()}`  probed expiry: `{expiry.isoformat()}`")
    lines.append("")

    if terminal_unreachable:
        lines.append("## Terminal unreachable")
        lines.append("")
        lines.append(
            "Every endpoint returned a connection error. Launch ThetaTerminal.jar "
            "and confirm it is listening on the base_url/version above, then re-run "
            "`odds-lab probe`."
        )
        lines.append("")
        lines.append("```")
        for name, c in checks.items():
            lines.append(f"{name}: {c.get('error')}")
        lines.append("```")
        report = "\n".join(lines) + "\n"
        return _write(report, quote_date)

    expected = {
        "expirations": [],
        "strikes": [],
        "option_eod": _EOD_FIELDS,
        "option_greeks": _GREEKS_FIELDS,
        "stock_eod": _STOCK_FIELDS,
    }

    for name, check in checks.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- path: `{check['path']}`")
        lines.append(f"- params: `{check['params']}`")
        if "error" in check:
            lines.append(f"- **error**: {check['error']}")
            lines.append("")
            continue
        fmt = check.get("format")
        lines.append(f"- header.format: `{fmt}`")
        lines.append(f"- n_rows: {check.get('n_rows')}")

        missing, ignored = _diff(expected.get(name, []), fmt)
        if missing:
            lines.append(f"- **fields we need and did NOT find**: {missing}")
        else:
            lines.append("- fields we need and did not find: none")
        lines.append(f"- fields present we ignore: {ignored}")

        if name == "option_eod" and fmt and check.get("first_rows"):
            strikes = [r.get(td._find_alias(fmt, "strike")) for r in check["first_rows"]]
            strikes = [s for s in strikes if s is not None]
            if strikes:
                med = sorted(strikes)[len(strikes) // 2]
                lines.append(f"- sample raw strikes: {strikes} (median-ish {med})")
                lines.append(
                    "- detected strike scaling: "
                    + ("TENTHS-OF-CENT (>20x typical spot, will /1000)" if med > 2000 else "DOLLARS (no scaling)")
                )
            date_col = td._find_alias(fmt, "date")
            if date_col and check["first_rows"]:
                raw_date = check["first_rows"][0].get(date_col)
                kind = "YYYY-MM-DD string" if isinstance(raw_date, str) and "-" in raw_date else "YYYYMMDD int"
                lines.append(f"- detected date encoding: {kind} (raw value: {raw_date!r})")

        if check.get("first_rows"):
            lines.append("- first 3 raw rows:")
            lines.append("")
            lines.append("```")
            for row in check["first_rows"]:
                lines.append(str(row))
            lines.append("```")
        lines.append("")

    report = "\n".join(lines) + "\n"
    return _write(report, quote_date)


def _write(report: str, quote_date: date) -> Path:
    out_dir = Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"probe-{quote_date.isoformat()}.md"
    path.write_text(report)
    return path


# ========================================================================================
# probe_csv -- profile a directory of ThetaData CSV bulk exports without reading any
# file whole. See module docstring point 2.
# ========================================================================================

_DATE_TOKEN_RE = re.compile(r"(\d{8})")
_GZ_TAIL_DECOMPRESS_LIMIT = 20 * 1024 * 1024  # compressed bytes -- see _read_tail_gz

_NEEDED_ALIASES: dict[str, set[str]] = {
    "root": {"root", "symbol"},
    "quote_date": {"quote_date", "date", "created"},
    "expiry": {"expiry", "expiration"},
    "strike": {"strike"},
    "right": {"right"},
    "bid": {"bid"},
    "ask": {"ask"},
    "underlying_price": {"underlying_price"},  # 'close' counts only on a stock_eod file
    "open_interest": {"open_interest", "oi"},
}
_GREEK_COLS = {"delta", "gamma", "theta", "vega", "rho", "iv", "implied_vol"}


class _CountingReader:
    """Wraps a binary file object; tracks bytes actually pulled through it, so a
    profiling pass can prove (not just claim) it stayed bounded on a huge file."""

    __slots__ = ("_f", "bytes_read")

    def __init__(self, f):
        self._f = f
        self.bytes_read = 0

    def readline(self, *a, **kw) -> bytes:
        b = self._f.readline(*a, **kw)
        self.bytes_read += len(b)
        return b

    def read(self, *a, **kw) -> bytes:
        b = self._f.read(*a, **kw)
        self.bytes_read += len(b)
        return b

    def close(self) -> None:
        self._f.close()


def _is_gz(path: Path) -> bool:
    return path.name.endswith(".gz")


def _gzip_uncompressed_size(path: Path) -> int | None:
    """The last 4 bytes of a gzip stream are ISIZE: the original size mod 2**32.
    Exact for any file under 4GB, and reading it costs one seek + 4 bytes -- O(1)
    regardless of file size, unlike decompressing the whole thing to find out."""
    try:
        with open(path, "rb") as f:
            f.seek(-4, 2)
            (size,) = struct.unpack("<I", f.read(4))
        return size
    except OSError:
        return None


def _read_head(path: Path, is_gz: bool, n_rows: int) -> tuple[bytes, list[bytes], int]:
    """Header line + up to n_rows data lines, and nothing else. Bounded regardless
    of file size: `readline()` on a gzip stream only decompresses as far as it must
    to satisfy each call, so this never touches bytes past what we ask for."""
    raw = gzip.open(path, "rb") if is_gz else open(path, "rb")
    reader = _CountingReader(raw)
    try:
        header = reader.readline()
        lines: list[bytes] = []
        for _ in range(n_rows):
            line = reader.readline()
            if not line:
                break
            lines.append(line)
    finally:
        reader.close()
    return header, lines, reader.bytes_read


def _read_tail_plain(path: Path, n_rows: int, size_on_disk: int, max_chunk: int = 8 * 1024 * 1024) -> tuple[list[bytes], int]:
    """Last n_rows lines of an uncompressed file, found by seeking backward from
    EOF in growing chunks -- bounded by `max_chunk`, never by file size."""
    if size_on_disk == 0:
        return [], 0
    chunk = 65536
    with open(path, "rb") as f:
        while True:
            start = max(0, size_on_disk - chunk)
            f.seek(start)
            data = f.read(size_on_disk - start)
            lines = data.split(b"\n")
            if start > 0:
                lines = lines[1:]  # first line may be a partial line -- drop it
            if lines and lines[-1] == b"":
                lines = lines[:-1]  # trailing newline
            if len(lines) > n_rows or start == 0 or chunk >= max_chunk:
                return lines[-n_rows:], len(data)
            chunk *= 4


def _read_tail_gz(path: Path, n_rows: int, size_on_disk: int) -> tuple[list[bytes], int, str | None]:
    """A gzip stream cannot be seeked into, so the only way to get its true tail is
    to decompress the whole thing -- fine for the small fixtures in this repo, not
    fine for a GB-scale file. Files over `_GZ_TAIL_DECOMPRESS_LIMIT` (compressed)
    skip tail sampling and say why, rather than silently going slow. The real
    GB-scale exports this tool targets are plain .csv, not .csv.gz (module
    docstring), so this limit should not bite in practice."""
    if size_on_disk > _GZ_TAIL_DECOMPRESS_LIMIT:
        return (
            [],
            0,
            f"tail not sampled: compressed size {size_on_disk:,} bytes exceeds the "
            f"{_GZ_TAIL_DECOMPRESS_LIMIT:,}-byte full-decompression limit for .gz tail reads "
            "(a gzip stream cannot be seeked into). Real GB-scale ThetaData exports are "
            "plain .csv, not .csv.gz, so this should not come up for them.",
        )
    with gzip.open(path, "rb") as f:
        data = f.read()
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines = lines[:-1]
    return lines[-n_rows:], len(data), None


def _decode_lines(lines: list[bytes]) -> list[str]:
    return [ln.decode("utf-8", errors="replace").rstrip("\r\n") for ln in lines]


def _split_csv_line(text: str) -> list[str]:
    return next(_csv.reader([text]))


def _lines_to_df(header_text: str, rows_text: list[str]) -> pd.DataFrame:
    if not rows_text:
        return pd.DataFrame(columns=_split_csv_line(header_text))
    buf = io.StringIO("\n".join([header_text, *rows_text]))
    return pd.read_csv(buf)


def _infer_dtype(series: pd.Series) -> str:
    if series.empty:
        return "unknown (no sampled rows)"
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    non_null = series.dropna().astype(str)
    if not non_null.empty:
        # Mixed/unknown date layouts are exactly what we are here to discover, so the
        # per-element dateutil fallback is intended -- but pandas warns about it on
        # stdout, and this runs in the command the user is told to paste back to us.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(non_null, errors="coerce")
        if parsed.notna().mean() > 0.9:
            return "date/datetime (string)"
    return "string"


def _classify(header: list[str]) -> tuple[str, str]:
    """Classify by the header COLUMN SET only -- never by filename (that is a
    separate, explicitly-flagged check in `_infer_filename_semantics`). The first
    two branches are confirmed ground truth (ARCHITECTURE.md §0.5, fixtures
    committed); everything past that is an unverified heuristic guess, labelled as
    such, because no real fixture covers quote_1m/trade/tick/stock_eod shapes."""
    cols = {c.strip().lower() for c in header}
    if cols == {c.lower() for c in csvx.EOD_HEADER}:
        return "option_eod", "exact match to the confirmed EOD_HEADER (ARCHITECTURE.md §0.5)"
    if cols == {c.lower() for c in csvx.OHLC_1M_HEADER}:
        return "option_ohlc_1m", "exact match to the confirmed OHLC_1M_HEADER (ARCHITECTURE.md §0.5)"

    has_option_id = {"strike", "right"} <= cols and ("expiration" in cols or "expiry" in cols)
    has_ohlc_bars = {"open", "high", "low", "close"} <= cols
    has_quote = {"bid", "ask"} <= cols
    has_trade_fields = ({"price", "size"} <= cols) and not has_quote
    has_tick_marker = ("ms_of_day" in cols) or ("sequence" in cols)
    has_minute_ts = "timestamp" in cols

    if has_option_id and has_quote and not has_ohlc_bars:
        if has_tick_marker and not has_minute_ts:
            return (
                "option_quote_tick",
                "heuristic, UNVERIFIED (no committed fixture for this shape): option-identifying "
                "columns + bid/ask + a sub-day marker (ms_of_day/sequence), no OHLC bars, no bare "
                "`timestamp` column",
            )
        return (
            "option_quote_1m",
            "heuristic, UNVERIFIED (no committed fixture for this shape): option-identifying "
            "columns + bid/ask, no OHLC bars",
        )
    if has_option_id and has_trade_fields:
        return (
            "option_trade",
            "heuristic, UNVERIFIED (no committed fixture for this shape): option-identifying "
            "columns + price/size, no bid/ask",
        )
    if not has_option_id and has_ohlc_bars and "volume" in cols:
        return (
            "stock_eod",
            "heuristic, UNVERIFIED (no committed fixture for this shape): OHLC + volume, no "
            "option-identifying columns (strike/right/expiration)",
        )
    return "unknown", "header matches neither confirmed shape nor any heuristic ThetaData export shape"


def _infer_filename_semantics(path: Path, header: list[str], sample_df: pd.DataFrame) -> list[str]:
    """What the filename's embedded YYYYMMDD token(s) appear to encode, checked
    against the sampled columns -- and explicitly flagged, never silently resolved,
    when a filename carries more than one token that map to different roles (a
    per-expiration file whose name also happens to embed the trade date is exactly
    the kind of thing that silently corrupts an ingest if assumed away)."""
    tokens = _DATE_TOKEN_RE.findall(path.name)
    if not tokens:
        return ["filename has no embedded 8-digit YYYYMMDD token"]

    cols_lower = {c.lower(): c for c in header}
    exp_dates: set = set()
    if "expiration" in cols_lower and not sample_df.empty:
        exp_dates = set(pd.to_datetime(sample_df[cols_lower["expiration"]], errors="coerce").dt.date.dropna())
    trade_dates: set = set()
    for cand in ("date", "timestamp", "created", "quote_date"):
        if cand in cols_lower and not sample_df.empty:
            trade_dates |= set(pd.to_datetime(sample_df[cols_lower[cand]], errors="coerce").dt.date.dropna())

    notes: list[str] = []
    roles_found: list[tuple[str, tuple[str, ...]]] = []
    for tok in tokens:
        try:
            d = datetime.strptime(tok, "%Y%m%d").date()
        except ValueError:
            notes.append(f"filename token `{tok}` is not a valid YYYYMMDD date -- ignored")
            continue
        role: list[str] = []
        if d in exp_dates:
            role.append("matches the sampled `expiration` value(s) -> looks like a PER-EXPIRATION file")
        if d in trade_dates:
            role.append("matches a sampled date/timestamp/created value -> looks like a PER-TRADE-DATE file")
        if role:
            notes.append(f"filename token `{tok}` (parses as {d.isoformat()}): " + "; ".join(role))
            roles_found.append((tok, tuple(role)))
        else:
            notes.append(
                f"filename token `{tok}` (parses as {d.isoformat()}) matches NEITHER the sampled "
                "expiration nor date/timestamp columns"
            )

    if len(tokens) > 1:
        distinct_roles = {r for _, r in roles_found}
        if len(roles_found) >= 2 and len(distinct_roles) > 1:
            notes.append(
                f"AMBIGUITY: filename encodes multiple 8-digit dates ({tokens}) that map to DIFFERENT "
                "roles -- do not assume which one is the trade date and which is the expiration; the "
                "per-column matches above are the only reliable signal, not filename position."
            )
        else:
            notes.append(f"filename encodes multiple 8-digit dates ({tokens}); treat each independently.")
    return notes


def _missing_for_file(header: list[str], classification: str) -> tuple[list[str], list[str]]:
    cols = {c.lower() for c in header}
    missing_required = []
    for field in schema.CHAIN_REQUIRED:
        if field == "underlying_price" and classification == "stock_eod" and "close" in cols:
            continue
        aliases = _NEEDED_ALIASES.get(field, {field})
        if not (aliases & cols):
            missing_required.append(field)
    missing_extra = []
    if not (_NEEDED_ALIASES["open_interest"] & cols):
        missing_extra.append("open_interest")
    if not (_GREEK_COLS & cols):
        missing_extra.append("greeks/iv")
    return missing_required, missing_extra


def _file_supplies(header: list[str], classification: str) -> set[str]:
    cols = {c.lower() for c in header}
    supplies: set[str] = set()
    for field, aliases in _NEEDED_ALIASES.items():
        if field == "underlying_price":
            if classification == "stock_eod" and "close" in cols:
                supplies.add("underlying_price")
            continue
        if aliases & cols:
            supplies.add(field)
    if _GREEK_COLS & cols:
        supplies.add("greeks/iv")
    return supplies


def _profile_file(path: Path, head_rows: int, tail_rows: int) -> dict:
    is_gz = _is_gz(path)
    size_on_disk = path.stat().st_size

    header_bytes, head_lines_bytes, head_bytes_read = _read_head(path, is_gz, head_rows)
    if is_gz:
        tail_lines_bytes, tail_bytes_read, tail_note = _read_tail_gz(path, tail_rows, size_on_disk)
    else:
        tail_lines_bytes, tail_bytes_read = _read_tail_plain(path, tail_rows, size_on_disk)
        tail_note = None

    header_text = _decode_lines([header_bytes])[0]
    header = _split_csv_line(header_text)
    head_text = _decode_lines(head_lines_bytes)
    tail_text = _decode_lines(tail_lines_bytes)

    sample_df = _lines_to_df(header_text, head_text)
    full_sample_df = _lines_to_df(header_text, head_text + tail_text) if tail_text else sample_df

    dtypes = {
        col: _infer_dtype(sample_df[col]) for col in sample_df.columns
    } if not sample_df.empty else {c: "unknown (no sampled rows)" for c in header}
    first_3_rows = sample_df.head(3).to_dict(orient="records")

    date_columns_range: dict[str, tuple[str, str]] = {}
    for col in full_sample_df.columns:
        if not dtypes.get(col, "").startswith("date"):
            continue
        parsed = pd.to_datetime(full_sample_df[col].astype(str), errors="coerce").dropna()
        if not parsed.empty:
            date_columns_range[col] = (str(parsed.min()), str(parsed.max()))

    if is_gz:
        uncompressed = _gzip_uncompressed_size(path)
        data_size_bytes = uncompressed if uncompressed is not None else size_on_disk
        gz_size_note = (
            f"gzip ISIZE trailer reports {uncompressed:,} uncompressed bytes (used for the row estimate)"
            if uncompressed is not None
            else "could not read gzip ISIZE trailer; row estimate falls back to compressed size (unreliable)"
        )
    else:
        data_size_bytes = size_on_disk
        gz_size_note = None

    avg_row_bytes = None
    if head_text:
        avg_row_bytes = sum(len(b) for b in head_lines_bytes) / len(head_text)
    estimated_total_rows = None
    if avg_row_bytes:
        estimated_total_rows = max(int((data_size_bytes - len(header_bytes)) / avg_row_bytes), 0)

    classification, classification_why = _classify(header)
    filename_notes = _infer_filename_semantics(path, header, full_sample_df)

    return {
        "path": path,
        "size_on_disk_bytes": size_on_disk,
        "is_gz": is_gz,
        "gz_size_note": gz_size_note,
        "header": header,
        "dtypes": dtypes,
        "first_3_rows": first_3_rows,
        "estimated_total_rows": estimated_total_rows,
        "date_columns_range": date_columns_range,
        "classification": classification,
        "classification_why": classification_why,
        "filename_notes": filename_notes,
        "tail_note": tail_note,
        "bytes_read_for_profile": head_bytes_read + tail_bytes_read,
        "n_head_rows_sampled": len(head_text),
        "n_tail_rows_sampled": len(tail_text),
    }


def _fmt_row(row: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in row.items())


def _render_csv_report(directory: Path, profiles: list[dict], supplies_by_file: dict, any_underlying: bool) -> str:
    lines: list[str] = []
    lines.append(f"# CSV export probe -- {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"- directory: `{directory}`")
    lines.append(f"- files found: {len(profiles)}")
    lines.append("")

    if not any_underlying:
        lines.append("## No file in this directory supplies `underlying_price`")
        lines.append("")
        lines.append(
            "The option chain **cannot be ingested** without it -- `CsvExportProvider` will raise "
            "`CsvExportError` naming exactly this. Point `underlying_path` at a stock EOD export "
            "(date/open/high/low/close/volume or similar), or drop one into this directory matching "
            "`underlying_glob`."
        )
        lines.append("")

    for pf in profiles:
        path = pf["path"]
        rel = path.relative_to(directory)
        lines.append(f"## `{rel}`")
        lines.append("")
        if "error" in pf:
            lines.append(f"- **could not profile this file**: {pf['error']}")
            lines.append("")
            continue

        lines.append(f"- size on disk: {pf['size_on_disk_bytes']:,} bytes" + (" (gzip)" if pf["is_gz"] else ""))
        if pf.get("gz_size_note"):
            lines.append(f"- {pf['gz_size_note']}")
        lines.append(f"- classification: **{pf['classification']}** -- {pf['classification_why']}")
        lines.append(f"- header ({len(pf['header'])} columns, in order): `{pf['header']}`")
        lines.append("- inferred dtypes:")
        for c, dt in pf["dtypes"].items():
            lines.append(f"  - `{c}`: {dt}")
        if pf["date_columns_range"]:
            lines.append("- date-like column ranges (ESTIMATE, from sampled rows only):")
            for c, (lo, hi) in pf["date_columns_range"].items():
                lines.append(f"  - `{c}`: {lo} .. {hi}")
        if pf["estimated_total_rows"] is not None:
            lines.append(f"- estimated total row count (**ESTIMATE**, bytes/avg-sampled-row-length): {pf['estimated_total_rows']:,}")
        else:
            lines.append("- estimated total row count: n/a (no sampled data rows)")
        lines.append(
            f"- profiled by reading {pf['bytes_read_for_profile']:,} bytes "
            f"({pf['n_head_rows_sampled']} head rows + {pf['n_tail_rows_sampled']} tail rows) -- "
            "the whole file was never loaded"
        )
        if pf.get("tail_note"):
            lines.append(f"- {pf['tail_note']}")
        lines.append("- first 3 rows:")
        for row in pf["first_3_rows"]:
            lines.append(f"  - {_fmt_row(row)}")
        lines.append("- filename semantics:")
        for note in pf["filename_notes"]:
            lines.append(f"  - {note}")

        missing_required, missing_extra = _missing_for_file(pf["header"], pf["classification"])
        if missing_required or missing_extra:
            lines.append("- missing relative to `schema.CHAIN_REQUIRED` / engine needs:")
            for field in missing_required:
                key = field
                suppliers = [str(p2.relative_to(directory)) for p2, s in supplies_by_file.items() if key in s and p2 != path]
                if suppliers:
                    lines.append(f"  - `{field}` (CHAIN_REQUIRED) -- MISSING here; supplied by: {suppliers}")
                elif field == "underlying_price":
                    lines.append(f"  - `{field}` (CHAIN_REQUIRED) -- MISSING here, and **no file in this directory supplies it**")
                else:
                    lines.append(f"  - `{field}` (CHAIN_REQUIRED) -- MISSING here, and no other file in this directory supplies it either")
            for extra in missing_extra:
                suppliers = [str(p2.relative_to(directory)) for p2, s in supplies_by_file.items() if extra in s and p2 != path]
                if suppliers:
                    lines.append(f"  - `{extra}` (engine needs this) -- MISSING here; supplied by: {suppliers}")
                else:
                    lines.append(f"  - `{extra}` (engine needs this) -- MISSING here, and no file in this directory supplies it")
        else:
            lines.append("- nothing missing relative to CHAIN_REQUIRED/engine needs")
        lines.append("")
    return "\n".join(lines) + "\n"


def _write_probe_csv_report(text: str) -> Path:
    out_dir = Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"probe-csv-{ts}.md"
    path.write_text(text)
    return path


def _print_csv_summary(directory: Path, profiles: list[dict], any_underlying: bool, out_path: Path) -> None:
    print(f"[probe-csv] {directory}: profiled {len(profiles)} file(s)")
    for pf in profiles:
        if "error" in pf:
            print(f"  {pf['path'].name}: ERROR -- {pf['error']}")
            continue
        est = f"~{pf['estimated_total_rows']:,} rows est." if pf["estimated_total_rows"] is not None else "row estimate n/a"
        print(f"  {pf['path'].name}: {pf['classification']} ({pf['size_on_disk_bytes']:,} bytes, {est})")
    if not any_underlying:
        print("  ! no file supplies underlying_price -- the chain cannot be ingested without it")
    print(f"[probe-csv] full report: {out_path}")


def _write_sample_archive(profiles: list[dict], out_path: Path, n_rows: int = 50) -> Path:
    """One small .tar.gz containing header + first `n_rows` rows for each DISTINCT
    header shape found (deduped -- 33 identically-shaped ohlc/ files produce one
    sample, not 33). This is what to send back when a shape comes back `unknown`."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: dict[tuple, dict] = {}
    for pf in profiles:
        if "error" in pf:
            continue
        key = tuple(pf["header"])
        if key not in seen:
            seen[key] = pf

    with tarfile.open(out_path, "w:gz") as tar:
        for pf in seen.values():
            header_line, lines, _ = _read_head(pf["path"], pf["is_gz"], n_rows)
            content = header_line + b"".join(lines)
            stem = pf["path"].name
            if stem.endswith(".csv.gz"):
                stem = stem[: -len(".csv.gz")]
            elif stem.endswith(".csv"):
                stem = stem[: -len(".csv")]
            member_name = f"{pf['classification']}__{stem}.csv"
            info = tarfile.TarInfo(name=member_name)
            info.size = len(content)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(content))
    return out_path


def probe_csv(
    directory: str | Path,
    *,
    head_rows: int = 200,
    tail_rows: int = 5,
    sample_out: str | Path | None = None,
) -> Path:
    """Walk `directory` recursively (following `ohlc/`-style subdirs), profile
    every `.csv`/`.csv.gz` WITHOUT reading any file whole, classify it against
    known/heuristic ThetaData export shapes, and write `runs/probe-csv-<ts>.md`.
    Also prints a compact summary to stdout. If `sample_out` is given, additionally
    writes a small gzip schema-sample archive there (see `_write_sample_archive`).
    Returns the markdown report path.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"probe_csv: directory not found: {directory}")

    files = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and (p.name.endswith(".csv") or p.name.endswith(".csv.gz"))
    )
    if not files:
        raise FileNotFoundError(f"probe_csv: no .csv/.csv.gz files found under {directory}")

    profiles = []
    for p in files:
        try:
            profiles.append(_profile_file(p, head_rows, tail_rows))
        except Exception as exc:  # noqa: BLE001 -- one bad file must never kill the whole walk
            profiles.append({"path": p, "error": f"{type(exc).__name__}: {exc}"})

    supplies_by_file = {
        pf["path"]: _file_supplies(pf["header"], pf["classification"]) for pf in profiles if "error" not in pf
    }
    any_underlying = any("underlying_price" in s for s in supplies_by_file.values())

    report_text = _render_csv_report(directory, profiles, supplies_by_file, any_underlying)
    out_path = _write_probe_csv_report(report_text)

    if sample_out is not None:
        _write_sample_archive(profiles, Path(sample_out))

    _print_csv_summary(directory, profiles, any_underlying, out_path)
    return out_path


# ========================================================================================
# probe_coverage -- earliest usable date per (root, data kind). See module docstring
# point 3.
# ========================================================================================


class _RateLimiter:
    """Politely paces requests to at most `per_minute`, spread evenly rather than
    bursted. `per_minute<=0` disables pacing entirely (used by tests against a
    monkeypatched, no-network `_get`)."""

    def __init__(self, per_minute: int):
        self.min_interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self._last: float | None = None

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        now = _time.monotonic()
        if self._last is not None:
            elapsed = now - self._last
            if elapsed < self.min_interval:
                _time.sleep(self.min_interval - elapsed)
        self._last = _time.monotonic()


def _nudge_to_weekday(d: date) -> date:
    """Binary search assumes coverage-start is monotonic in calendar date; a
    closed-market weekend always reports 'no data' regardless of vendor coverage,
    which can bias a search midpoint that happens to land on one. Shifting to the
    following Monday reduces (does not eliminate -- holidays remain a gap) the
    chance of that. See module docstring's limitation note."""
    if d.weekday() == 5:  # Saturday
        return d + timedelta(days=2)
    if d.weekday() == 6:  # Sunday
        return d + timedelta(days=1)
    return d


def _find_floor_binary_search(
    check: Callable[[date], bool], floor: date, ceiling: date
) -> tuple[date | None, int]:
    """Earliest date in [floor, ceiling] for which `check(d)` is True, assuming
    monotonicity (once True, stays True through `ceiling`). O(log(ceiling-floor))
    calls to `check`, never a linear scan. Returns (None, n_requests) if `ceiling`
    itself has no data -- no floor exists in the searched range. `check` may raise;
    callers must not catch that as "no data" (that would silently under-report a
    real transport/auth failure as an early floor)."""
    n = 0

    def _check(d: date) -> bool:
        nonlocal n
        n += 1
        return check(d)

    if not _check(ceiling):
        return None, n
    if _check(floor):
        return floor, n

    lo, hi = floor, ceiling  # invariant: check(lo) is False, check(hi) is True
    while (hi - lo).days > 1:
        mid = lo + (hi - lo) // 2
        mid = _nudge_to_weekday(mid)
        if mid <= lo or mid >= hi:
            mid = lo + (hi - lo) // 2  # nudge pushed it out of range -- fall back
        if _check(mid):
            hi = mid
        else:
            lo = mid
    return hi, n


def _has_option_data(provider: td.ThetaDataProvider, root: str, d: date) -> bool:
    try:
        exps = provider.expirations(root, d)
    except td.ThetaNoData:
        return False
    return len(exps) > 0


def _has_greeks_data(provider: td.ThetaDataProvider, root: str, d: date) -> bool:
    try:
        exps = provider.expirations(root, d)
    except td.ThetaNoData:
        return False
    if not exps:
        return False
    params = {
        "root": root,
        "exp": exps[0].strftime("%Y%m%d"),
        "start_date": d.strftime("%Y%m%d"),
        "end_date": d.strftime("%Y%m%d"),
        "strike": "*",
        "right": "*",
    }
    try:
        payload = provider._get(provider._paths["option_greeks"], params)
    except td.ThetaNoData:
        return False
    df, _fmt = td._rows_to_frame(payload)
    return len(df) > 0


def _has_underlying_data(provider: td.ThetaDataProvider, root: str, d: date) -> bool:
    try:
        df = provider.underlying_eod(root, d, d)
    except td.ThetaNoData:
        return False
    return not df.empty


_REST_KINDS: list[tuple[str, Callable[[td.ThetaDataProvider, str, date], bool]]] = [
    ("option_eod", _has_option_data),
    ("option_greeks", _has_greeks_data),
    ("stock_eod", _has_underlying_data),
]


def _probe_root_kind(
    provider: td.ThetaDataProvider,
    root: str,
    kind: str,
    check_fn: Callable[[td.ThetaDataProvider, str, date], bool],
    floor: date,
    ceiling: date,
    limiter: _RateLimiter,
) -> dict:
    n_requests = 0

    def wrapped(d: date) -> bool:
        nonlocal n_requests
        limiter.wait()
        n_requests += 1
        return check_fn(provider, root, d)

    try:
        found, _n = _find_floor_binary_search(wrapped, floor, ceiling)
    except td.ThetaError as exc:
        return {"root": root, "kind": kind, "floor": None, "n_requests": n_requests, "error": f"{type(exc).__name__}: {exc}", "samples": []}
    return {"root": root, "kind": kind, "floor": found, "n_requests": n_requests, "error": None, "samples": []}


def _sample_option_density(
    provider: td.ThetaDataProvider, root: str, floor: date, ceiling: date, n_samples: int, limiter: _RateLimiter
) -> list[dict]:
    """A handful of dates spread across [floor, ceiling]: how many expirations and
    contracts actually came back. A floor where the chain is technically non-empty
    but has 3 strikes is not a usable floor -- this is what makes that visible."""
    if n_samples <= 0:
        return []
    span = (ceiling - floor).days
    if span <= 0:
        dates = [floor]
    else:
        dates = sorted({floor + timedelta(days=round(span * i / max(n_samples - 1, 1))) for i in range(n_samples)})

    out = []
    for d in dates:
        limiter.wait()
        try:
            exps = provider.expirations(root, d)
        except td.ThetaError as exc:
            out.append({"date": d, "n_expirations": None, "n_contracts": None, "error": f"{type(exc).__name__}: {exc}"})
            continue
        n_contracts = None
        if exps:
            limiter.wait()
            try:
                df = provider.chain_eod(root, d, exps[0])
                n_contracts = int(len(df))
            except td.ThetaError:
                n_contracts = None
        out.append({"date": d, "n_expirations": len(exps), "n_contracts": n_contracts})
    return out


def _render_coverage_report(
    base_url: str | None,
    version: str | None,
    roots: list[str],
    floor: date | None,
    ceiling: date | None,
    results: list[dict],
    *,
    mode: str,
    directory: Path | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"# ThetaData coverage probe -- {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    if mode == "rest":
        lines.append(f"- base_url: `{base_url}`  version: `{version}`")
        lines.append(f"- search range: `{floor.isoformat()}` .. `{ceiling.isoformat()}` (binary search)")
    else:
        lines.append(f"- csv directory: `{directory}` (local index scan, no requests)")
    lines.append(f"- roots: `{list(roots)}`")
    lines.append("")
    lines.append("## Summary table")
    lines.append("")
    lines.append("| root | kind | earliest usable date | requests | note/error |")
    lines.append("|---|---|---|---|---|")
    by_root: dict[str, list[dict]] = {}
    for r in results:
        by_root.setdefault(r["root"], []).append(r)
        floor_str = r["floor"].isoformat() if r["floor"] else "NOT FOUND"
        note = r.get("error") or r.get("note") or ""
        lines.append(f"| {r['root']} | {r['kind']} | {floor_str} | {r['n_requests']} | {note} |")
    lines.append("")

    lines.append("## Per-root verdict")
    lines.append("")
    for root, entries in by_root.items():
        eod = next((e for e in entries if e["kind"] == "option_eod"), None)
        if eod and eod["floor"]:
            lines.append(f"- **{root}**: option EOD usable from `{eod['floor'].isoformat()}`; "
                          f"a backtest starting before that date is not possible from this source.")
        elif eod and eod.get("error"):
            lines.append(f"- **{root}**: option EOD floor could not be determined -- {eod['error']}")
        else:
            lines.append(f"- **{root}**: no usable option EOD floor found in the searched range.")
        for e in entries:
            if e["kind"] == "option_eod":
                continue
            if e.get("error"):
                lines.append(f"  - {e['kind']}: error -- {e['error']}")
            elif e["floor"]:
                suffix = f" ({e['note']})" if e.get("note") else ""
                lines.append(f"  - {e['kind']}: usable from `{e['floor'].isoformat()}`{suffix}")
            else:
                suffix = f" -- {e['note']}" if e.get("note") else ""
                lines.append(f"  - {e['kind']}: not found{suffix}")
        lines.append("")

    density_rows = [r for r in results if r["kind"] == "option_eod" and r.get("samples")]
    if density_rows:
        lines.append("## Sample density (option_eod)")
        lines.append("")
        for r in density_rows:
            lines.append(f"### {r['root']}")
            lines.append("")
            lines.append("| date | expirations | contracts |")
            lines.append("|---|---|---|")
            for s in r["samples"]:
                exp_s = s.get("n_expirations", "?")
                con_s = s.get("n_contracts", "?")
                if s.get("error"):
                    lines.append(f"| {s['date'].isoformat()} | error | {s['error']} |")
                else:
                    lines.append(f"| {s['date'].isoformat()} | {exp_s} | {con_s} |")
            lines.append("")
    return "\n".join(lines) + "\n"


def _write_coverage_report(text: str) -> Path:
    out_dir = Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"coverage-{ts}.md"
    path.write_text(text)
    return path


def _print_coverage_summary(results: list[dict]) -> None:
    print(f"{'root':<6}{'kind':<16}{'earliest':<14}{'reqs':<6}note/error")
    for r in results:
        floor_str = r["floor"].isoformat() if r["floor"] else "NOT FOUND"
        note = r.get("error") or r.get("note") or ""
        print(f"{r['root']:<6}{r['kind']:<16}{floor_str:<14}{r['n_requests']:<6}{note}")


def probe_coverage_rest(
    base_url: str | None,
    version: str,
    roots: list[str],
    *,
    floor: date = date(2010, 1, 1),
    ceiling: date | None = None,
    sample_dates: int = 5,
    rate_limit_per_min: int = 20,
) -> Path:
    """Binary-search the earliest usable date per (root, data kind) against a live
    Theta Terminal. ~O(log(ceiling-floor)) requests per (root, kind), never a
    linear day-by-day scan -- the request budget is printed before any request is
    made. Kinds probed independently, because they do not share a floor: option
    EOD chain, option greeks, underlying/stock EOD. `open_interest` is not
    independently date-searched -- it travels inside the option_eod payload rather
    than behind its own endpoint, so it shares option_eod's floor by construction;
    the report says so rather than pretending to search it separately.
    Writes `runs/coverage-<timestamp>.md` and returns its path.
    """
    provider = td.ThetaDataProvider(base_url=base_url, version=version)
    ceiling = ceiling or (date.today() - timedelta(days=5))

    span_days = max((ceiling - floor).days, 1)
    per_kind_budget = int(math.ceil(math.log2(span_days))) + 2
    total_budget = len(roots) * len(_REST_KINDS) * per_kind_budget + len(roots) * sample_dates * 2
    print(
        f"[probe-coverage] up to ~{total_budget} requests across {len(roots)} root(s) x "
        f"{len(_REST_KINDS)} kind(s) (binary search over {floor.isoformat()}..{ceiling.isoformat()}), "
        f"rate-limited to {rate_limit_per_min}/min"
    )

    limiter = _RateLimiter(rate_limit_per_min)
    results: list[dict] = []
    for root in roots:
        for kind, check_fn in _REST_KINDS:
            res = _probe_root_kind(provider, root, kind, check_fn, floor, ceiling, limiter)
            if kind == "option_eod" and res["floor"] is not None:
                res["samples"] = _sample_option_density(provider, root, res["floor"], ceiling, sample_dates, limiter)
            results.append(res)
        results.append(
            {
                "root": root,
                "kind": "open_interest",
                "floor": next((r["floor"] for r in results if r["root"] == root and r["kind"] == "option_eod"), None),
                "n_requests": 0,
                "error": None,
                "samples": [],
                "note": "not independently date-searched -- travels inside the option_eod payload "
                        "(when present at all; some accounts/endpoints omit it entirely), so it "
                        "shares option_eod's floor by construction",
            }
        )

    report_text = _render_coverage_report(base_url, version, roots, floor, ceiling, results, mode="rest")
    out_path = _write_coverage_report(report_text)
    _print_coverage_summary(results)
    print(f"[probe-coverage] full report: {out_path}")
    return out_path


def probe_coverage_csv(directory: str | Path, roots: list[str]) -> Path:
    """Answer the same earliest-usable-date question from a local export directory
    -- free, no requests, just reading the directory's own date index."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"probe_coverage_csv: directory not found: {directory}")

    provider = csvx.CsvExportProvider(directory=directory, progress=False)
    wide = (date(1900, 1, 1), date(2100, 1, 1))
    eod_dates = provider.trading_dates(*wide)

    results: list[dict] = []
    for root in roots:
        if eod_dates:
            results.append(
                {
                    "root": root, "kind": "option_eod", "floor": eod_dates[0], "n_requests": 0, "error": None,
                    "samples": [], "note": f"local directory scan, {len(eod_dates)} EOD file(s) indexed by embedded YYYYMMDD",
                }
            )
        else:
            results.append(
                {
                    "root": root, "kind": "option_eod", "floor": None, "n_requests": 0,
                    "error": f"no EOD export files found in {directory}", "samples": [],
                }
            )
        results.append(
            {
                "root": root, "kind": "option_greeks", "floor": None, "n_requests": 0, "error": None, "samples": [],
                "note": "the ThetaData EOD CSV bulk export has no greeks/iv columns at all (ARCHITECTURE.md "
                        "§0.5) -- not answerable from this directory",
            }
        )
        results.append(
            {
                "root": root, "kind": "open_interest", "floor": None, "n_requests": 0, "error": None, "samples": [],
                "note": "the ThetaData EOD CSV bulk export has no open_interest column at all (ARCHITECTURE.md "
                        "§0.5) -- not answerable from this directory",
            }
        )
        try:
            u_df = provider.underlying_eod(root, *wide)
        except csvx.CsvExportError as exc:
            results.append({"root": root, "kind": "stock_eod", "floor": None, "n_requests": 0, "error": str(exc), "samples": []})
        else:
            u_floor = u_df["date"].min().date() if not u_df.empty else None
            results.append(
                {
                    "root": root, "kind": "stock_eod", "floor": u_floor, "n_requests": 0,
                    "error": None if u_floor else "underlying source resolved but has no rows", "samples": [],
                }
            )

    report_text = _render_coverage_report(None, None, roots, None, None, results, mode="csv", directory=directory)
    out_path = _write_coverage_report(report_text)
    _print_coverage_summary(results)
    print(f"[probe-coverage] full report: {out_path}")
    return out_path
