"""Theta Terminal local REST adapter. ARCHITECTURE.md §0 "Schema is probed, not assumed."

Everything below the endpoint URLs is coded against the *documented* ThetaData
wire format, which we could not verify against a live terminal (no outbound
network in this environment). Treat every constant in `_V3_PATHS`/`_V2_PATHS`,
every entry in `_FIELD_ALIASES`, and both scaling traps as **UNVERIFIED**
assumptions -- `odds_lab.data.probe.probe()` is the tool that confirms or
corrects them against a real subscription, and its report should be read
before trusting a real ingest run.

Parsing is driven entirely by the response's `header.format` array (the vendor's
own column-order manifest), never by hardcoded positions -- if a field we need
is missing from that array we raise `ThetaError` naming exactly what's missing
and the format array we actually got, rather than silently mis-column a row.

Two documented scaling traps, asserted rather than trusted:
  - `strike` may be in **tenths of a cent** (140000 == $140.00). Auto-detected by
    comparing the median strike to the day's underlying price (`strike_scale="auto"`,
    the default): if `median(strike) > 20 * underlying_price`, we divide by 1000.
    Overridable via `ThetaDataProvider(strike_scale=1000 | 1 | "auto")`.
  - `vega` and `rho`, when present in a greeks payload, are divided by 100 to match
    `quant.bs`'s "per 1 vol point" / "per 1% rate move" convention.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import requests

from odds_lab import schema
from odds_lab.data.providers.base import ChainProvider

__all__ = [
    "ThetaDataProvider",
    "ThetaError",
    "ThetaNotRunning",
    "ThetaNoData",
    "ThetaAuthError",
    "ThetaVersionGone",
]


# --------------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------------


class ThetaError(Exception):
    """Base for every ThetaData adapter error."""


class ThetaNotRunning(ThetaError):
    """Connection refused -- ThetaTerminal.jar is not running (or wrong port/version)."""

    def __init__(self, base_url: str, cause: Exception | None = None):
        super().__init__(
            f"Could not reach Theta Terminal at {base_url}. Launch ThetaTerminal.jar "
            "(https://http-docs.thetadata.us) and confirm it's listening on this "
            "base_url/version before retrying."
        )
        self.cause = cause


class ThetaNoData(ThetaError):
    """The terminal answered but has no data for this request (HTTP 472 or an
    explicit 'NO_DATA' payload)."""


class ThetaAuthError(ThetaError):
    """401/403 -- subscription tier does not cover this endpoint/root."""


class ThetaVersionGone(ThetaError):
    """HTTP 410 -- a v2 path was requested of a terminal that has moved to v3-only."""


# --------------------------------------------------------------------------------
# Endpoint map -- UNVERIFIED, see module docstring. probe.py checks these against a
# live terminal; correct them here once confirmed.
# --------------------------------------------------------------------------------

_V3_PATHS = {
    "option_eod": "/v3/option/history/eod",
    "option_greeks": "/v3/option/history/greeks",
    "expirations": "/v3/option/list/expirations",
    "strikes": "/v3/option/list/strikes",
    "stock_eod": "/v3/stock/history/eod",
    "stock_dates": "/v3/stock/list/dates",
}

_V2_PATHS = {
    "option_eod": "/v2/bulk_hist/option/eod",
    "option_greeks": "/v2/bulk_hist/option/greeks",
    "expirations": "/v2/list/expirations",
    "strikes": "/v2/list/strikes",
    "stock_eod": "/v2/hist/stock/eod",
    "stock_dates": "/v2/list/dates",
}

# canonical field -> acceptable vendor spellings (order = preference). UNVERIFIED.
_FIELD_ALIASES: dict[str, list[str]] = {
    "root": ["root", "symbol"],
    "expiration": ["expiration", "expiry", "exp"],
    "strike": ["strike"],
    "right": ["right", "option_right", "type"],
    "date": ["date"],
    "ms_of_day": ["ms_of_day", "ms_of_day2"],
    "bid": ["bid"],
    "bid_size": ["bid_size", "bid_condition_size"],
    "ask": ["ask"],
    "ask_size": ["ask_size", "ask_condition_size"],
    "last": ["last", "close"],
    "volume": ["volume"],
    "open_interest": ["open_interest", "oi"],
    "delta": ["delta"],
    "gamma": ["gamma"],
    "theta": ["theta"],
    "vega": ["vega"],
    "rho": ["rho"],
    "iv": ["implied_vol", "iv"],
    "open": ["open"],
    "high": ["high"],
    "low": ["low"],
    "close": ["close"],
}

_DEFAULT_BASE_URL = {"v3": "http://127.0.0.1:25503", "v2": "http://127.0.0.1:25510"}


def _find_alias(fmt: list[str], canonical: str) -> str | None:
    for alias in _FIELD_ALIASES.get(canonical, [canonical]):
        if alias in fmt:
            return alias
    return None


def _require(fmt: list[str], canonical: str) -> str:
    found = _find_alias(fmt, canonical)
    if found is None:
        raise ThetaError(
            f"ThetaData response is missing required field {canonical!r} "
            f"(looked for aliases {_FIELD_ALIASES.get(canonical, [canonical])!r}); "
            f"header.format was {fmt!r}. Run `odds-lab probe` and update "
            f"thetadata._FIELD_ALIASES / the endpoint path if the vendor renamed it."
        )
    return found


def _rows_to_frame(payload: dict) -> tuple[pd.DataFrame, list[str]]:
    """Turn a `{"header": {"format": [...]}, "response": [[...], ...]}` (or a v2
    shape with the rows directly under a different key) into a DataFrame whose
    columns are exactly `format`, in order -- never assume positions otherwise."""
    header = payload.get("header", {})
    fmt = header.get("format")
    if not fmt:
        raise ThetaError(f"ThetaData response has no header.format array: keys={list(payload)!r}")
    rows = payload.get("response")
    if rows is None:
        rows = payload.get("data")  # some v2 endpoints nest under 'data'
    if rows is None:
        raise ThetaError(f"ThetaData response has header.format but no row payload: keys={list(payload)!r}")
    df = pd.DataFrame(rows, columns=fmt)
    return df, fmt


def _parse_date_col(series: pd.Series) -> pd.Series:
    """`date` is YYYYMMDD int, or YYYY-MM-DD string if human_readable=True."""
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        return pd.to_datetime(series.astype("Int64").astype(str), format="%Y%m%d")
    return pd.to_datetime(series.astype(str))


@dataclass
class ThetaDataProvider(ChainProvider):
    """Theta Terminal local REST client. All network calls funnel through `_get`
    so tests can monkeypatch it with canned payloads (v2- and v3-shaped)."""

    name: str = field(default="thetadata", init=False)
    base_url: str | None = None
    version: Literal["v3", "v2"] = "v3"
    strike_scale: Literal["auto", 1, 1000] = "auto"
    timeout: float = 10.0
    max_retries: int = 5
    backoff_base: float = 0.5
    session: Any = None
    risk_free_rate: float = 0.02

    def __post_init__(self) -> None:
        if self.base_url is None:
            self.base_url = _DEFAULT_BASE_URL[self.version]
        self._paths = _V3_PATHS if self.version == "v3" else _V2_PATHS
        self._session = self.session or requests.Session()

    # -- transport ---------------------------------------------------------------

    def _get(self, path: str, params: dict | None = None) -> dict:
        """Single point of contact with the terminal. Retries 429/5xx with
        exponential backoff; raises the typed exceptions above on everything else."""
        url = f"{self.base_url}{path}"
        attempt = 0
        while True:
            try:
                resp = self._session.get(url, params=params or {}, timeout=self.timeout)
            except requests.exceptions.ConnectionError as exc:
                raise ThetaNotRunning(self.base_url, cause=exc) from exc
            except requests.exceptions.Timeout as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise ThetaError(f"Timed out calling {url} after {self.max_retries} retries") from exc
                time.sleep(self.backoff_base * (2**attempt))
                continue

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (401, 403):
                raise ThetaAuthError(
                    f"{resp.status_code} from {url} -- subscription does not cover this "
                    "endpoint/root."
                )
            if resp.status_code == 410:
                raise ThetaVersionGone(
                    f"410 Gone from {url}: a v2 path was called against a terminal that "
                    "has moved to v3-only. Set DataConfig.theta_api_version='v3'."
                )
            if resp.status_code == 472:
                raise ThetaNoData(f"No data for {url} params={params!r}")
            if resp.status_code == 429 or resp.status_code >= 500:
                attempt += 1
                if attempt > self.max_retries:
                    raise ThetaError(
                        f"{resp.status_code} from {url} after {self.max_retries} retries"
                    )
                time.sleep(self.backoff_base * (2**attempt))
                continue
            raise ThetaError(f"Unexpected {resp.status_code} from {url}: {resp.text[:500]!r}")

    # -- ChainProvider -------------------------------------------------------------

    def trading_dates(self, start: date, end: date) -> list[date]:
        payload = self._get(self._paths["stock_dates"], {"root": "SPY"})
        df, fmt = _rows_to_frame(payload)
        date_col = _require(fmt, "date")
        dates = _parse_date_col(df[date_col]).dt.date
        return sorted(d for d in dates if start <= d <= end)

    def expirations(self, root: str, quote_date: date) -> list[date]:
        payload = self._get(self._paths["expirations"], {"root": root})
        df, fmt = _rows_to_frame(payload)
        exp_col = _find_alias(fmt, "expiration")
        if exp_col is None:
            # some v2 shapes return a bare list of ints under one column
            if df.shape[1] == 1:
                exp_col = df.columns[0]
            else:
                raise ThetaError(f"could not find an expiration column in format {fmt!r}")
        exps = _parse_date_col(df[exp_col]).dt.date
        return sorted(d for d in exps if d > quote_date)

    def underlying_eod(self, root: str, start: date, end: date) -> pd.DataFrame:
        params = {
            "root": root,
            "start_date": start.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
        }
        payload = self._get(self._paths["stock_eod"], params)
        df, fmt = _rows_to_frame(payload)
        date_col = _require(fmt, "date")
        out = pd.DataFrame(
            {
                "root": root,
                "date": _parse_date_col(df[date_col]),
                "open": df[_require(fmt, "open")].astype(float),
                "high": df[_require(fmt, "high")].astype(float),
                "low": df[_require(fmt, "low")].astype(float),
                "close": df[_require(fmt, "close")].astype(float),
                "volume": df[_require(fmt, "volume")].astype("int64"),
                "dividend": 0.0,
            }
        )
        return schema.validate_frame(out, schema.UNDERLYING_DTYPES, "underlying")

    def _underlying_close(self, root: str, quote_date: date) -> float:
        """One day's underlying close, used only to seed the strike-scale auto-detect
        and as the chain's `underlying_price` column."""
        df = self.underlying_eod(root, quote_date, quote_date)
        if df.empty:
            raise ThetaNoData(f"no underlying EOD for {root} on {quote_date.isoformat()}")
        return float(df["close"].iloc[0])

    def chain_eod(self, root: str, quote_date: date, expiry: date | None = None) -> pd.DataFrame:
        if expiry is None:
            frames = [self.chain_eod(root, quote_date, e) for e in self.expirations(root, quote_date)]
            frames = [f for f in frames if not f.empty]
            if not frames:
                return schema.empty_frame(schema.CHAIN_DTYPES)
            return pd.concat(frames, ignore_index=True)

        params = {
            "root": root,
            "exp": expiry.strftime("%Y%m%d"),
            "start_date": quote_date.strftime("%Y%m%d"),
            "end_date": quote_date.strftime("%Y%m%d"),
            "strike": "*",
            "right": "*",
        }
        eod_payload = self._get(self._paths["option_eod"], params)
        eod_df, eod_fmt = _rows_to_frame(eod_payload)
        if eod_df.empty:
            return schema.empty_frame(schema.CHAIN_DTYPES)

        try:
            greeks_payload = self._get(self._paths["option_greeks"], params)
            greeks_df, greeks_fmt = _rows_to_frame(greeks_payload)
        except ThetaNoData:
            greeks_df, greeks_fmt = pd.DataFrame(), []

        underlying_price = self._underlying_close(root, quote_date)
        return self._parse_chain(root, quote_date, expiry, eod_df, eod_fmt, greeks_df, greeks_fmt, underlying_price)

    # -- parsing -------------------------------------------------------------------

    def _parse_chain(
        self,
        root: str,
        quote_date: date,
        expiry: date,
        eod_df: pd.DataFrame,
        eod_fmt: list[str],
        greeks_df: pd.DataFrame,
        greeks_fmt: list[str],
        underlying_price: float,
    ) -> pd.DataFrame:
        strike_col = _require(eod_fmt, "strike")
        right_col = _require(eod_fmt, "right")
        bid_col = _require(eod_fmt, "bid")
        ask_col = _require(eod_fmt, "ask")

        raw_strike = eod_df[strike_col].astype(float)

        scale = self.strike_scale
        if scale == "auto":
            scale = 1000 if raw_strike.median() > 20 * max(underlying_price, 1e-6) else 1
            print(
                f"[thetadata] strike scale auto-detected as {scale} "
                f"(median raw strike={raw_strike.median():.1f}, underlying={underlying_price:.2f})"
            )
        strike = raw_strike / scale if scale != 1 else raw_strike

        right = eod_df[right_col].astype(str).str.upper().str[0]

        ms_col = _find_alias(eod_fmt, "ms_of_day")
        ms_of_day = eod_df[ms_col].astype("int32") if ms_col else pd.Series(0, index=eod_df.index, dtype="int32")

        def _opt(col_name: str, fmt_list: list[str], src: pd.DataFrame, dtype=float, default=np.nan):
            col = _find_alias(fmt_list, col_name)
            if col is None:
                return pd.Series(default, index=eod_df.index, dtype=dtype if dtype != float else "float64")
            return src[col].astype(dtype)

        out = pd.DataFrame(
            {
                "root": root,
                "quote_date": pd.Timestamp(quote_date),
                "ms_of_day": ms_of_day,
                "expiry": pd.Timestamp(expiry),
                "strike": strike,
                "right": right,
                "bid": eod_df[bid_col].astype(float),
                "ask": eod_df[ask_col].astype(float),
                "bid_size": _opt("bid_size", eod_fmt, eod_df, "int32", 0),
                "ask_size": _opt("ask_size", eod_fmt, eod_df, "int32", 0),
                "last": _opt("last", eod_fmt, eod_df, float, np.nan),
                "volume": _opt("volume", eod_fmt, eod_df, "int64", 0),
                "open_interest": _opt("open_interest", eod_fmt, eod_df, "int64", 0),
                "underlying_price": float(underlying_price),
                "source": "thetadata",
                "is_synthetic": False,
            }
        )

        if not greeks_df.empty:
            g_strike_col = _require(greeks_fmt, "strike")
            g = pd.DataFrame(
                {
                    "strike": (
                        greeks_df[g_strike_col].astype(float) / scale
                        if scale != 1
                        else greeks_df[g_strike_col].astype(float)
                    ),
                    "right": greeks_df[_require(greeks_fmt, "right")].astype(str).str.upper().str[0],
                }
            )
            # schema.CHAIN_DTYPES only carries a vendor-comparison column for iv/delta
            # (ARCHITECTURE.md §2); gamma/theta/vega/rho have a single authoritative
            # slot that quant.bs.enrich_chain overwrites downstream -- so the raw
            # vendor greeks land straight in those columns here.
            for canon, col_out in [("delta", "delta_vendor"), ("iv", "iv_vendor"), ("gamma", "gamma"), ("theta", "theta")]:
                col = _find_alias(greeks_fmt, canon)
                g[col_out] = greeks_df[col].astype(float) if col else np.nan
            vega_col = _find_alias(greeks_fmt, "vega")
            rho_col = _find_alias(greeks_fmt, "rho")
            # vega/rho need dividing by 100 -- UNVERIFIED vendor scaling, see module docstring.
            g["vega"] = (greeks_df[vega_col].astype(float) / 100.0) if vega_col else np.nan
            g["rho"] = (greeks_df[rho_col].astype(float) / 100.0) if rho_col else np.nan
            out = out.merge(g, on=["strike", "right"], how="left")
        else:
            out["delta_vendor"] = np.nan
            out["iv_vendor"] = np.nan

        return schema.validate_chain(out, strict=False)
