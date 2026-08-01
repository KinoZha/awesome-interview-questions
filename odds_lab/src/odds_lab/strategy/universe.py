"""Market-state classifier, IV rank, and realized vol -- STRATEGY.md §4.3.

Lookahead convention (documented once here, applies to every function below and is
exercised by tests/test_strategy.py):
  - `closes` series (underlying daily closes): only rows with index date STRICTLY BEFORE
    `asof` are usable. The rationale is that a daily-granularity backtest makes its entry
    decision using the EOD option-chain snapshot *at* asof, but the underlying's own close
    print for that same session is not a distinct, independently-observable data point at
    decision time -- it *is* the chain's `underlying_price` column, which is used directly
    where needed (e.g. `atm_iv`). Using `closes[idx < asof]` for SMA/momentum/realized-vol
    prevents smuggling today's close into a lookback statistic through the back door.
  - `atm_iv` is computed directly from a chain slice the caller already fetched at `asof`
    (not lookahead: it's the same snapshot driving the trade decision).
  - `iv_rank` treats its input series as a history of *already-computed* atm_iv values one
    per prior decision date, plus (optionally) today's; it uses `idx <= asof` because the
    caller is expected to have appended today's own atm_iv (computed above) before calling.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from odds_lab.schema import MarketState

__all__ = ["market_state", "iv_rank", "realized_vol", "atm_iv"]

_SMA_WINDOW = 200
_MOMENTUM_TRADING_DAYS = 126  # ~6 trading months


def market_state(closes: pd.Series, asof: date) -> MarketState:
    """SMA200 + 6-month (126 trading day) momentum classifier. STRATEGY.md §4.3.

    bullish: close > SMA200 and momentum > 0
    bearish: close < SMA200 and momentum < 0
    neutral: otherwise, and whenever there is insufficient history to judge (documented
    fallback -- a cold-start period never silently claims a directional edge).
    """
    idx = pd.DatetimeIndex(closes.index)
    usable = closes[idx < pd.Timestamp(asof)].sort_index()
    if len(usable) < _SMA_WINDOW:
        return MarketState.NEUTRAL

    sma200 = float(usable.iloc[-_SMA_WINDOW:].mean())
    last = float(usable.iloc[-1])

    if len(usable) > _MOMENTUM_TRADING_DAYS:
        momentum = last / float(usable.iloc[-_MOMENTUM_TRADING_DAYS - 1]) - 1.0
    else:
        momentum = 0.0

    if last > sma200 and momentum > 0:
        return MarketState.BULLISH
    if last < sma200 and momentum < 0:
        return MarketState.BEARISH
    return MarketState.NEUTRAL


def iv_rank(atm_iv: pd.Series, asof: date, window: int = 252) -> float:
    """Min-max IV rank of the latest value in `atm_iv` within the trailing `window`.

    `atm_iv` is a date-indexed series of previously-computed ATM IV values, one per prior
    decision date, including (by convention) today's own value at `asof` -- see module
    docstring. Rows with index date > asof are rejected as lookahead.
    """
    idx = pd.DatetimeIndex(atm_iv.index)
    if (idx > pd.Timestamp(asof)).any():
        raise ValueError("iv_rank: series contains dates after asof (lookahead)")
    usable = atm_iv[idx <= pd.Timestamp(asof)].dropna().sort_index()
    if usable.empty:
        raise ValueError(f"iv_rank: no usable IV history at or before asof={asof}")

    window_vals = usable.iloc[-window:]
    current = float(window_vals.iloc[-1])
    lo, hi = float(window_vals.min()), float(window_vals.max())
    if hi <= lo:
        return 0.5
    return float((current - lo) / (hi - lo))


def realized_vol(closes: pd.Series, asof: date, window: int = 21) -> float:
    """Annualized close-to-close realized vol over the trailing `window` trading days,
    using only closes strictly before `asof` (see module docstring)."""
    idx = pd.DatetimeIndex(closes.index)
    usable = closes[idx < pd.Timestamp(asof)].sort_index()
    if len(usable) < window + 1:
        raise ValueError(
            f"realized_vol: only {len(usable)} closes before asof={asof}, need >= {window + 1}"
        )
    vals = usable.iloc[-(window + 1):].to_numpy(dtype=float)
    log_rets = np.diff(np.log(vals))
    return float(np.std(log_rets, ddof=1) * np.sqrt(252.0))


def atm_iv(chain: pd.DataFrame, expiry: date) -> float:
    """Average IV of the call and put closest to the money for a given expiry, within a
    chain slice the caller has already restricted to a single (root, quote_date)."""
    sub = chain[chain["expiry"] == pd.Timestamp(expiry)]
    if sub.empty:
        raise ValueError(f"atm_iv: no rows for expiry={expiry}")
    S = float(sub["underlying_price"].iloc[0])
    dist = (sub["strike"].astype(float) - S).abs()
    sub = sub.assign(_dist=dist)
    best_idx = sub.groupby("right")["_dist"].idxmin()
    best = sub.loc[best_idx]
    ivs = best["iv"].dropna()
    if ivs.empty:
        raise ValueError(f"atm_iv: no IV data near the money for expiry={expiry}")
    return float(ivs.mean())
