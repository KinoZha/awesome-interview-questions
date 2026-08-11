"""Synthetic ChainProvider -- realistic dev/test data. ARCHITECTURE.md §0, §1.

Underlying: a regime-switching GBM with Poisson downward jumps (two states -- calm
and stressed -- with sticky Markov transitions), so realized returns are fat-tailed
and left-skewed the way STRATEGY.md §1/§2 requires the empirical distribution to be.

Chain: real weekly + monthly (3rd-Friday) expiration listing, a $1 strike grid
spanning roughly ±25% of spot (wider for far-dated expiries), priced by
Black-Scholes under a skewed smile (put skew + mild upward term structure, level
tracking the day's regime) with a realistic bid/ask spread and an OI/volume hump
around ATM and round strikes.

Every row is stamped `source='synthetic', is_synthetic=True` -- ARCHITECTURE.md
requires this never be silently mistaken for real data.

Simplification (documented, not hidden): the trading calendar is plain business
days (no US market holiday table). This only matters for exact date-count
comparisons against a real calendar, never for the shape of the data.
"""

from __future__ import annotations

import zlib
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from odds_lab import schema
from odds_lab.data.providers.base import ChainProvider

__all__ = ["SyntheticProvider", "make_sample_store"]

_CONTRACT_MULTIPLIER = schema.CONTRACT_MULTIPLIER

# Per-root baseline parameters, loosely matched to real SPY/QQQ/IWM circa 2012-2026.
_ROOT_PARAMS = {
    "SPY": {"s0": 200.0, "div_yield": 0.018, "avg_volume": 80_000_000},
    "QQQ": {"s0": 100.0, "div_yield": 0.006, "avg_volume": 40_000_000},
    "IWM": {"s0": 120.0, "div_yield": 0.014, "avg_volume": 30_000_000},
}
_DEFAULT_PARAMS = {"s0": 100.0, "div_yield": 0.012, "avg_volume": 20_000_000}

_MS_OF_DAY_EOD = 16 * 3600 * 1000  # 4:00pm ET close, in ms since midnight


def _root_seed(seed: int, root: str) -> int:
    """Deterministic per-root seed offset. Never uses python's randomized hash()."""
    return (seed * 1_000_003 + zlib.crc32(root.encode())) % (2**31 - 1)


def _day_seed(seed: int, root: str, d: date) -> int:
    return (_root_seed(seed, root) * 131 + d.toordinal()) % (2**31 - 1)


def _third_friday(year: int, month: int) -> date:
    """The real 3rd Friday of (year, month) -- standard monthly equity-option expiry."""
    d = date(year, month, 1)
    # weekday(): Monday=0 ... Friday=4
    first_friday = d + timedelta(days=(4 - d.weekday()) % 7)
    return first_friday + timedelta(days=14)


def _all_fridays(start: date, end: date) -> list[date]:
    if start > end:
        return []
    first = start + timedelta(days=(4 - start.weekday()) % 7)
    out = []
    d = first
    while d <= end:
        out.append(d)
        d += timedelta(days=7)
    return out


def _monthly_expiries(start: date, end: date) -> list[date]:
    out = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        tf = _third_friday(y, m)
        if start <= tf <= end:
            out.append(tf)
        m += 1
        if m == 13:
            m = 1
            y += 1
    return out


def _expirations_for(quote_date: date, near_horizon_days: int = 60, far_horizon_days: int = 400) -> list[date]:
    """Realistic listing: weekly (every Friday) density out to `near_horizon_days`,
    monthly (3rd Friday) density beyond that out to `far_horizon_days`."""
    near_end = quote_date + timedelta(days=near_horizon_days)
    far_end = quote_date + timedelta(days=far_horizon_days)
    weeklies = [f for f in _all_fridays(quote_date + timedelta(days=1), near_end) if f > quote_date]
    monthlies = [d for d in _monthly_expiries(near_end + timedelta(days=1), far_end)]
    out = sorted(set(weeklies) | set(monthlies))
    return [d for d in out if d > quote_date]


def _trading_dates(start: date, end: date) -> list[date]:
    return [d.date() for d in pd.bdate_range(start, end)]


def _simulate_underlying(root: str, start: date, end: date, seed: int) -> pd.DataFrame:
    """Regime-switching GBM + Poisson downward jumps. Returns UNDERLYING_DTYPES columns
    plus an internal `_regime` column (0=calm, 1=stressed), dropped before it leaves
    the provider's public surface."""
    params = _ROOT_PARAMS.get(root, _DEFAULT_PARAMS)
    dates = _trading_dates(start, end)
    n = len(dates)
    if n == 0:
        return pd.DataFrame(columns=[*schema.UNDERLYING_DTYPES, "_regime"])

    rng = np.random.default_rng(_root_seed(seed, root))

    # calm=0, stressed=1. Sticky Markov chain.
    p_calm_to_stress = 0.01
    p_stress_to_calm = 0.08
    regime = np.zeros(n, dtype=int)
    for i in range(1, n):
        u = rng.random()
        if regime[i - 1] == 0:
            regime[i] = 1 if u < p_calm_to_stress else 0
        else:
            regime[i] = 0 if u < p_stress_to_calm else 1

    vol = np.where(regime == 0, 0.12, 0.35)
    drift = np.where(regime == 0, 0.07, -0.20)  # annualized

    z = rng.standard_normal(n)
    daily_ret = (drift - 0.5 * vol**2) / 252.0 + vol / np.sqrt(252.0) * z

    jump_intensity = np.where(regime == 0, 0.002, 0.06)
    jump_flags = rng.random(n) < jump_intensity
    jump_size = -np.abs(rng.normal(0.035, 0.02, n))
    daily_ret = daily_ret + np.where(jump_flags, jump_size, 0.0)

    close = params["s0"] * np.exp(np.cumsum(daily_ret))

    intraday_noise = rng.normal(0.0, 0.003, n)
    open_ = np.empty(n)
    open_[0] = params["s0"]
    open_[1:] = close[:-1] * (1 + intraday_noise[1:])
    hi_bump = np.abs(rng.normal(0.0, 0.004, n))
    lo_bump = np.abs(rng.normal(0.0, 0.004, n))
    high = np.maximum(open_, close) * (1 + hi_bump)
    low = np.minimum(open_, close) * (1 - lo_bump)

    vol_noise = np.exp(rng.normal(0.0, 0.3, n))
    volume = (params["avg_volume"] * vol_noise * (1.0 + 0.8 * regime)).astype(np.int64)

    # quarterly-ish ex-dividend dates: every ~63 trading days, phase offset by root.
    phase = _root_seed(seed, root) % 63
    dividend = np.zeros(n)
    q_yield = params["div_yield"] / 4.0
    for i in range(n):
        if i >= phase and (i - phase) % 63 == 0:
            dividend[i] = close[i] * q_yield

    df = pd.DataFrame(
        {
            "root": root,
            "date": pd.to_datetime(dates),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "dividend": dividend,
            "_regime": regime,
        }
    )
    return df


def _smile_iv(K: np.ndarray, S: float, T: np.ndarray, regime: int) -> np.ndarray:
    """Skewed smile: put skew (iv up as strike falls) + mild upward term structure,
    level set by the day's regime."""
    base = 0.13 if regime == 0 else 0.32
    logm = np.log(np.maximum(K, 1e-6) / S)  # >0 above spot, <0 below spot
    term = base * (1.0 + 0.18 * np.sqrt(np.maximum(T, 0.0)))
    skew = -0.09 * logm  # K below S -> logm<0 -> positive add -> higher iv (put skew)
    curvature = 0.06 * logm**2
    iv = term + skew + curvature
    return np.clip(iv, 0.03, 2.5)


def _tick(price: np.ndarray) -> np.ndarray:
    return np.where(price < 3.0, 0.01, 0.05)


def _round_to_tick(price: np.ndarray, tick: np.ndarray, mode: str) -> np.ndarray:
    n = price / tick
    n = np.floor(n) if mode == "down" else np.ceil(n)
    return n * tick


def _minute_bridge(rng: np.random.Generator, s_open: float, s_close: float, n_bars: int, vol_scale: float = 0.15) -> np.ndarray:
    """Deterministic Brownian-bridge intraday path, anchored to hit `s_close`
    exactly at the last bar -- the day's open/close discipline must match the
    already-generated EOD chain for the same (root, quote_date) exactly, since
    both are consumed by the same backtest run. Intraday-exit tests that need an
    exact shape (e.g. a spike that reverts by the close) pass `s_path` explicitly
    to `SyntheticProvider.intraday_quotes` instead of using this default."""
    n_bars = max(int(n_bars), 1)
    log_open, log_close = np.log(max(s_open, 1e-6)), np.log(max(s_close, 1e-6))
    t = np.arange(1, n_bars + 1) / n_bars
    drift = log_open + t * (log_close - log_open)
    noise = rng.normal(0.0, vol_scale / np.sqrt(n_bars), n_bars)
    noise_cum = np.cumsum(noise)
    # subtract off the endpoint's cumulative noise, scaled by t, so the bridge lands
    # exactly on log_close at t=1 (standard Brownian-bridge construction).
    bridge_noise = noise_cum - t * noise_cum[-1]
    return np.exp(drift + bridge_noise)


def _empty_intraday_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["root", "expiry", "strike", "right", "quote_date", "ts", "bid", "ask", "bid_size", "ask_size", "source", "is_synthetic"]
    )


class SyntheticProvider(ChainProvider):
    """Generates a full arbitrage-consistent chain over [start, end] for `roots`,
    deterministic given `seed`. See module docstring."""

    name = "synthetic"

    def __init__(self, roots: Sequence[str], start: date, end: date, seed: int = 7):
        self.roots = list(roots)
        self.start = start
        self.end = end
        self.seed = seed
        self._underlying: dict[str, pd.DataFrame] = {
            r: _simulate_underlying(r, start, end, seed) for r in self.roots
        }

    # -- ChainProvider ---------------------------------------------------------

    def trading_dates(self, start: date, end: date) -> list[date]:
        return [d for d in _trading_dates(start, end) if self.start <= d <= self.end]

    def expirations(self, root: str, quote_date: date) -> list[date]:
        return _expirations_for(quote_date)

    def underlying_eod(self, root: str, start: date, end: date) -> pd.DataFrame:
        df = self._underlying.get(root)
        if df is None or df.empty:
            return schema.empty_frame(schema.UNDERLYING_DTYPES)
        mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        out = df.loc[mask, list(schema.UNDERLYING_DTYPES)].reset_index(drop=True)
        return schema.validate_frame(out, schema.UNDERLYING_DTYPES, "underlying")

    def chain_eod(self, root: str, quote_date: date, expiry: date | None = None) -> pd.DataFrame:
        under = self._underlying.get(root)
        if under is None:
            raise ValueError(f"synthetic provider was not built for root {root!r}")
        row = under.loc[under["date"] == pd.Timestamp(quote_date)]
        if row.empty:
            return schema.empty_frame(schema.CHAIN_DTYPES)
        S = float(row["close"].iloc[0])
        regime = int(row["_regime"].iloc[0])
        params = _ROOT_PARAMS.get(root, _DEFAULT_PARAMS)
        q = params["div_yield"]
        r = 0.02

        expiries = [expiry] if expiry is not None else self.expirations(root, quote_date)
        frames = []
        rng = np.random.default_rng(_day_seed(self.seed, root, quote_date))
        for exp in expiries:
            dte = (exp - quote_date).days
            if dte <= 0:
                continue
            T = dte / 365.0
            pct_span = 0.25 if dte <= 180 else 0.40
            lo = max(1, int(round(S * (1 - pct_span))))
            hi = int(round(S * (1 + pct_span)))
            strikes = np.arange(lo, hi + 1, 1.0)
            n_k = len(strikes)
            K = np.concatenate([strikes, strikes])
            right = np.array(["C"] * n_k + ["P"] * n_k)

            iv = _smile_iv(K, S, np.full(K.shape, T), regime)
            from odds_lab.quant.bs import bs_price  # lazy: keeps import order light

            theo = bs_price(S, K, T, r, q, iv, right)

            otm_amt = np.where(right == "C", np.maximum(K - S, 0.0), np.maximum(S - K, 0.0)) / S
            base_frac = 0.02
            spread_frac = (
                base_frac
                + 0.35 * otm_amt
                + 0.25 * (dte / 365.0)
                + 0.04 / np.maximum(theo, 0.05)
            )
            spread_frac = np.clip(spread_frac, 0.02, 0.9)
            half_spread = np.maximum(theo * spread_frac / 2.0, 0.005)
            tick = _tick(theo)
            # Deep-ITM contracts must trade at or above the BS no-arbitrage floor
            # (`quant.bs.implied_vol`'s exact discounted bound, S*disc_q - K*disc_r for
            # calls / K*disc_r - S*disc_q for puts, NOT plain undiscounted intrinsic --
            # for K < S with q < r that discounted bound can sit *above* intrinsic, and a
            # bid floored only at intrinsic then rounds a tick below it, silently NaN-ing
            # the IV inversion downstream). `ingest.py` recomputes greeks with its own
            # trailing-yield estimate of q (can differ from -- typically be lower than --
            # this generator's `params["div_yield"]`), so the floor here conservatively
            # uses q=0 for calls (call's arbitrage floor is *highest*, i.e. hardest to
            # clear, at q=0) so it stays valid for whatever q `enrich_chain` ends up using.
            disc_r = np.exp(-r * T)
            no_arb_floor = np.where(
                right == "C",
                np.maximum(S - K * disc_r, 0.0),
                np.maximum(K * disc_r - S * np.exp(-q * T), 0.0),
            )
            floor = no_arb_floor + tick
            bid = np.maximum(theo - half_spread, floor)
            ask = np.maximum(theo + half_spread, floor + tick)
            bid = _round_to_tick(bid, tick, "up")
            ask = _round_to_tick(ask, tick, "up")
            ask = np.maximum(ask, bid + tick)
            bid = np.maximum(bid, 0.0)
            mid = (bid + ask) / 2.0
            last = np.clip(mid + rng.normal(0.0, 1.0, K.shape) * tick, 0.0, None)

            atm_weight = np.exp(-0.5 * ((K - S) / (0.05 * S)) ** 2)
            round_bonus = 1.0 + 0.3 * (np.mod(K, 5) == 0) + 0.2 * (np.mod(K, 10) == 0)
            front_month_bonus = 1.0 / (1.0 + dte / 90.0)
            base_oi = 3000.0 * front_month_bonus
            oi_noise = np.exp(rng.normal(0.0, 0.5, K.shape))
            open_interest = np.maximum(
                0, (base_oi * atm_weight * round_bonus * oi_noise).astype(np.int64)
            )
            volume = np.maximum(
                0, (open_interest * rng.uniform(0.01, 0.25, K.shape)).astype(np.int64)
            )
            bid_size = np.clip((rng.integers(1, 50, K.shape) * (1 + atm_weight)).astype(np.int64), 1, None)
            ask_size = np.clip((rng.integers(1, 50, K.shape) * (1 + atm_weight)).astype(np.int64), 1, None)

            frames.append(
                pd.DataFrame(
                    {
                        "root": root,
                        "quote_date": pd.Timestamp(quote_date),
                        "ms_of_day": _MS_OF_DAY_EOD,
                        "expiry": pd.Timestamp(exp),
                        "strike": K.astype(float),
                        "right": right,
                        "bid": bid,
                        "ask": ask,
                        "bid_size": bid_size.astype(np.int32),
                        "ask_size": ask_size.astype(np.int32),
                        "last": last,
                        "volume": volume.astype(np.int64),
                        "open_interest": open_interest.astype(np.int64),
                        "underlying_price": S,
                        "source": "synthetic",
                        "is_synthetic": True,
                    }
                )
            )
        if not frames:
            return schema.empty_frame(schema.CHAIN_DTYPES)
        # Not calling schema.validate_chain here: it's a real cost at this per-(date,
        # expiry) call frequency and every column above is already built with the
        # right dtype, so it would pass strict=False unchanged. `ingest.py` validates
        # (via write_chain) once per batch instead of once per expiry.
        return pd.concat(frames, ignore_index=True)

    # -- intraday (1-minute quotes) ---------------------------------------------
    # Extends this generator for the intraday-exit feature only -- chain_eod/
    # underlying_eod above are untouched, per that change's scope.

    def intraday_quotes(
        self,
        root: str,
        quote_date: date,
        expiry: date,
        strike: float,
        right: str,
        n_bars: int = 390,
        s_path: "np.ndarray | Sequence[float] | None" = None,
    ) -> pd.DataFrame:
        """One synthetic contract-day of 1-minute quotes, BS-repriced bar-by-bar off
        an intraday underlying path, in `data/store.py::INTRADAY_DTYPES` shape.

        `s_path`: optional explicit length-`n_bars` array of underlying prices for
        the day -- lets a test construct an exact path (e.g. a spike that reverts by
        the close, the specific bias `docs/STRATEGY.md`/this feature exists to
        measure) instead of the default deterministic Brownian bridge between the
        day's already-generated EOD open and close.
        """
        under = self._underlying.get(root)
        if under is None:
            raise ValueError(f"synthetic provider was not built for root {root!r}")
        row = under.loc[under["date"] == pd.Timestamp(quote_date)]
        if row.empty:
            return _empty_intraday_frame()
        regime = int(row["_regime"].iloc[0])
        params = _ROOT_PARAMS.get(root, _DEFAULT_PARAMS)
        q = params["div_yield"]
        r = 0.02

        if s_path is None:
            rng = np.random.default_rng(_day_seed(self.seed, root, quote_date) ^ zlib.crc32(f"{strike}-{right}".encode()))
            s_open, s_close = float(row["open"].iloc[0]), float(row["close"].iloc[0])
            s_arr = _minute_bridge(rng, s_open, s_close, n_bars)
        else:
            s_arr = np.asarray(s_path, dtype=float)
            n_bars = len(s_arr)
        if n_bars <= 0:
            return _empty_intraday_frame()

        dte = (expiry - quote_date).days
        if dte <= 0:
            return _empty_intraday_frame()
        # time remaining ticks down across the day's bars too, not just day-to-day.
        T = np.maximum((dte - np.arange(n_bars) / n_bars) / 365.0, 1e-6)
        K = np.full(n_bars, float(strike))
        right_arr = np.full(n_bars, right, dtype=object)

        iv = _smile_iv(K, s_arr, T, regime)
        from odds_lab.quant.bs import bs_price

        theo = bs_price(s_arr, K, T, r, q, iv, right_arr)

        otm_amt = np.where(right == "C", np.maximum(K - s_arr, 0.0), np.maximum(s_arr - K, 0.0)) / np.maximum(s_arr, 1e-6)
        spread_frac = np.clip(0.02 + 0.35 * otm_amt + 0.04 / np.maximum(theo, 0.05), 0.02, 0.9)
        half_spread = np.maximum(theo * spread_frac / 2.0, 0.005)
        tick = _tick(theo)
        bid = np.maximum(_round_to_tick(np.maximum(theo - half_spread, 0.0), tick, "down"), 0.0)
        ask = np.maximum(_round_to_tick(theo + half_spread, tick, "up"), bid + tick)

        minutes = pd.date_range(
            pd.Timestamp(quote_date) + pd.Timedelta(hours=9, minutes=30), periods=n_bars, freq="1min"
        )
        return pd.DataFrame(
            {
                "root": root, "expiry": pd.Timestamp(expiry), "strike": float(strike), "right": right,
                "quote_date": pd.Timestamp(quote_date), "ts": minutes,
                "bid": bid, "ask": ask,
                "bid_size": np.full(n_bars, 10, dtype=np.int32), "ask_size": np.full(n_bars, 10, dtype=np.int32),
                "source": "synthetic", "is_synthetic": True,
            }
        )


def make_sample_store(
    path: str | Path,
    roots: Sequence[str] = ("SPY", "QQQ", "IWM"),
    start: date = date(2015, 1, 1),
    end: date = date(2016, 1, 1),
    seed: int = 7,
):
    """Generate and write a complete synthetic store, returned ready to query.

    Runs the full `data.ingest.ingest` pipeline (normalize -> enrich_chain ->
    validate -> partitioned parquet) so the fixture exercises the same code path
    real ingestion does. `dte_max=90` bounds the listed universe to what
    STRATEGY.md's entry/exit rules actually need (entry DTE <= 56, hold to
    expiry) -- a deliberate fixture-speed simplification, not a provider
    limitation; ThetaData ingestion is not bounded this way by default.
    """
    from odds_lab.data.ingest import ingest
    from odds_lab.data.store import ChainStore

    provider = SyntheticProvider(roots=roots, start=start, end=end, seed=seed)
    store = ChainStore(path)
    ingest(provider, store, roots, start, end, dte_max=90, progress=False)
    return store
