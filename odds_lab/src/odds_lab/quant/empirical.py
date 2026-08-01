"""Empirical realized-return distribution -- the "Count" half of STRATEGY.md §2 step 2.

Builds the distribution of overlapping T-trading-day log returns from a trailing window
of underlying closes, optionally symmetrized, with a lazy stationary-bootstrap CI.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from odds_lab.config import EmpiricalConfig

__all__ = ["EmpiricalDist", "build_empirical"]


@dataclass(frozen=True)
class EmpiricalDist:
    """A realized-return distribution over `horizon` trading days, asof a decision date.

    `log_returns` is the (possibly symmetrized) sample used for prob_below/above/quantile.
    `daily_log_returns` retains the original one-day log returns (pre-overlap, pre-symmetrize)
    so ci_below can re-form horizon sums inside each stationary-bootstrap resample.
    """

    log_returns: np.ndarray
    horizon: int
    n_raw: int
    symmetrized: bool
    asof: date
    daily_log_returns: np.ndarray
    seed: int
    bootstrap_mean_block: int
    bootstrap_samples: int

    def prob_below(self, x: float) -> float:
        """P(log return <= x) under this distribution's empirical sample."""
        return float(np.mean(self.log_returns <= x))

    def prob_above(self, x: float) -> float:
        """P(log return >= x)."""
        return float(np.mean(self.log_returns >= x))

    def quantile(self, p: float) -> float:
        return float(np.quantile(self.log_returns, p))

    def ci_below(self, x: float, alpha: float = 0.05) -> tuple[float, float]:
        """Block-bootstrap CI for P(log return <= x), stationary bootstrap (Politis-Romano)
        over the ORIGINAL daily log returns, re-forming horizon sums inside each resample.

        This is deliberately NOT a naive binomial CI on the (autocorrelated, overlapping)
        horizon-window sample -- that CI is materially too narrow. See STRATEGY.md §2.
        """
        rng = np.random.default_rng(self.seed)
        daily = self.daily_log_returns
        n = len(daily)
        boot_probs = np.empty(self.bootstrap_samples)
        for b in range(self.bootstrap_samples):
            resampled = _stationary_bootstrap_resample(daily, n, self.bootstrap_mean_block, rng)
            windows = _overlapping_horizon_sums(resampled, self.horizon)
            if self.symmetrized:
                m = windows.mean()
                windows = np.concatenate([windows - m, -(windows - m)])
            boot_probs[b] = np.mean(windows <= x)
        lo = float(np.quantile(boot_probs, alpha / 2))
        hi = float(np.quantile(boot_probs, 1 - alpha / 2))
        return lo, hi


def _overlapping_horizon_sums(daily_log_returns: np.ndarray, horizon: int) -> np.ndarray:
    """r_i = sum of `horizon` consecutive daily log returns == ln(close[i+h]/close[i])."""
    n = len(daily_log_returns)
    if n < horizon:
        return np.empty(0)
    csum = np.concatenate([[0.0], np.cumsum(daily_log_returns)])
    return csum[horizon:] - csum[:-horizon]


def _stationary_bootstrap_resample(
    daily: np.ndarray, n_out: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap: geometric block lengths, wraps circularly."""
    n = len(daily)
    p = 1.0 / mean_block
    out = np.empty(n_out)
    filled = 0
    idx = int(rng.integers(0, n))
    while filled < n_out:
        block_len = rng.geometric(p)
        block_len = min(block_len, n_out - filled)
        for k in range(block_len):
            out[filled] = daily[(idx + k) % n]
            filled += 1
        idx = int(rng.integers(0, n))
    return out


_MEMO: dict[tuple, EmpiricalDist] = {}
_MEMO_MAX = 512
"""A 12-year x 3-root run builds ~10k distributions, each holding two float64 arrays of
a few thousand elements. Unbounded, that is hundreds of MB retained for nothing: the loop
only ever revisits the most recent few keys. Bound it and evict oldest-first."""


def build_empirical(closes: pd.Series, horizon: int, cfg: EmpiricalConfig, asof: date) -> EmpiricalDist:
    """Build the (optionally symmetrized) empirical horizon-return distribution.

    Uses only rows with index < asof, and only the trailing cfg.lookback_years. Raises
    ValueError if fewer than cfg.min_samples usable overlapping windows result.

    Symmetrization (STRATEGY.md §2, *Casino Secret* p.59): under no-arbitrage there is no
    persistent directional bias, so the reference distribution is the drift-centered sample
    unioned with its negation:
        m = mean(r)
        symmetrized_sample = concat(r - m, -(r - m))
    i.e. the symmetrized distribution is *centered at zero* (drift removed), not re-added.
    Both the raw and symmetrized log_returns are obtainable by calling this function with
    cfg.symmetrize True/False; `n_raw` always reports the pre-symmetrization window count.
    """
    asof_ts = pd.Timestamp(asof)
    idx = pd.DatetimeIndex(closes.index)

    usable = closes[idx < asof_ts]
    if usable.empty:
        raise ValueError(f"build_empirical: no data before asof={asof}")

    lookback_start = asof_ts - pd.Timedelta(days=int(cfg.lookback_years * 365.25))
    usable_idx = pd.DatetimeIndex(usable.index)
    usable = usable[usable_idx >= lookback_start]

    vals = usable.to_numpy(dtype=float)

    # Key on the *content* of the window, not id(closes). SPY/QQQ/IWM share trading
    # dates and lengths, so an identity-based key collides across roots the moment
    # CPython reuses an address -- silently returning one root's distribution for
    # another. Digesting ~2.5k float64 costs a few microseconds; that is the right
    # trade against a wrong-underlying bug that no test would catch.
    key = (
        hashlib.blake2b(vals.tobytes(), digest_size=16).digest(),
        horizon,
        cfg,
        asof_ts,
    )
    cached = _MEMO.get(key)
    if cached is not None:
        return cached
    if len(vals) < 2:
        raise ValueError(
            f"build_empirical: insufficient data in lookback window before asof={asof}"
        )
    daily_log_returns = np.diff(np.log(vals))

    windows = _overlapping_horizon_sums(daily_log_returns, horizon)
    n_raw = len(windows)
    if n_raw < cfg.min_samples:
        raise ValueError(
            f"build_empirical: only {n_raw} overlapping {horizon}-day windows, "
            f"need >= {cfg.min_samples} (asof={asof})"
        )

    if cfg.symmetrize:
        m = windows.mean()
        log_returns = np.concatenate([windows - m, -(windows - m)])
    else:
        log_returns = windows

    mean_block = cfg.bootstrap_mean_block if cfg.bootstrap_mean_block is not None else horizon

    result = EmpiricalDist(
        log_returns=log_returns,
        horizon=horizon,
        n_raw=n_raw,
        symmetrized=cfg.symmetrize,
        asof=asof,
        daily_log_returns=daily_log_returns,
        seed=cfg.seed,
        bootstrap_mean_block=mean_block,
        bootstrap_samples=cfg.bootstrap_samples,
    )
    if len(_MEMO) >= _MEMO_MAX:
        for stale in list(_MEMO)[: _MEMO_MAX // 4]:
            del _MEMO[stale]
    _MEMO[key] = result
    return result
