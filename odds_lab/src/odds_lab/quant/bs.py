"""Vectorized Black-Scholes-Merton pricing, greeks, and implied vol. numpy+scipy only.

`py_vollib_vectorized` is not usable here (pins Python<3.10) -- see ARCHITECTURE.md.

Conventions (all documented + tested in tests/test_quant.py):
  - T is in YEARS, calendar days / 365 (ACT/365).
  - delta: dPrice/dS, per $1 of underlying.
  - gamma: d2Price/dS2, per $1^2 of underlying.
  - vega:  dPrice/dsigma / 100, i.e. per 1 VOLATILITY POINT (sigma in 0..1 units, a "point"
    is 0.01 of sigma).
  - theta: dPrice/dT / 365, i.e. per CALENDAR DAY. Negative for a long option (time decay).
  - rho:   dPrice/dr / 100, i.e. per 1% (0.01) move in the risk-free rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

from odds_lab import schema

__all__ = ["bs_price", "bs_greeks", "implied_vol", "enrich_chain"]

_EPS = 1e-12
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _norm_cdf(x):
    # scipy.special.ndtr is a fast direct erf-based CDF; scipy.stats.norm.cdf carries
    # much heavier generic-distribution dispatch overhead and is materially slower at
    # the array sizes this module targets (enrich_chain: >=1e6 rows).
    return ndtr(x)


def _norm_pdf(x):
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def _bcast(*arrays):
    as_arrays = [np.asarray(a, dtype=float) for a in arrays]
    shapes = {a.shape for a in as_arrays}
    if len(shapes) == 1:
        # Fast path: everything is already the same shape (the common case for
        # enrich_chain, where every input is a length-n column) -- skip
        # np.broadcast_arrays' more general (and measurably slower) machinery.
        return as_arrays
    return np.broadcast_arrays(*as_arrays)


def _d1_d2(S, K, T, r, q, sigma):
    T_safe = np.maximum(T, _EPS)
    sigma_safe = np.maximum(sigma, _EPS)
    sqrtT = np.sqrt(T_safe)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * sqrtT)
    d2 = d1 - sigma_safe * sqrtT
    return d1, d2


def bs_price(S, K, T, r, q, sigma, right) -> np.ndarray:
    """Black-Scholes-Merton price. S,K,T,r,q,sigma broadcastable; `right` array-like of 'C'/'P'.

    T<=0 returns intrinsic value (settled/expired option).
    """
    S, K, T, r, q, sigma = _bcast(S, K, T, r, q, sigma)
    right_arr = np.broadcast_to(np.asarray(right, dtype=object), S.shape)
    is_call = right_arr == "C"

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_q = np.exp(-q * np.maximum(T, 0.0))
    disc_r = np.exp(-r * np.maximum(T, 0.0))

    call_px = S * disc_q * _norm_cdf(d1) - K * disc_r * _norm_cdf(d2)
    put_px = K * disc_r * _norm_cdf(-d2) - S * disc_q * _norm_cdf(-d1)

    price = np.where(is_call, call_px, put_px)

    intrinsic_call = np.maximum(S - K, 0.0)
    intrinsic_put = np.maximum(K - S, 0.0)
    intrinsic = np.where(is_call, intrinsic_call, intrinsic_put)
    price = np.where(T <= 0, intrinsic, price)
    return price


def bs_greeks(S, K, T, r, q, sigma, right) -> dict[str, np.ndarray]:
    """Analytic BSM greeks. See module docstring for units/conventions."""
    S, K, T, r, q, sigma = _bcast(S, K, T, r, q, sigma)
    right_arr = np.broadcast_to(np.asarray(right, dtype=object), S.shape)
    is_call = right_arr == "C"

    T_pos = T > 0
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    sqrtT = np.sqrt(np.maximum(T, _EPS))
    sigma_safe = np.maximum(sigma, _EPS)
    disc_q = np.exp(-q * np.maximum(T, 0.0))
    disc_r = np.exp(-r * np.maximum(T, 0.0))
    pdf_d1 = _norm_pdf(d1)

    delta_call = disc_q * _norm_cdf(d1)
    delta_put = -disc_q * _norm_cdf(-d1)
    delta = np.where(is_call, delta_call, delta_put)

    gamma = disc_q * pdf_d1 / (S * sigma_safe * sqrtT)

    vega_raw = S * disc_q * pdf_d1 * sqrtT  # dPrice/dsigma (sigma in decimal units)
    vega = vega_raw / 100.0  # per 1 vol POINT

    theta_call = (
        -S * disc_q * pdf_d1 * sigma_safe / (2 * sqrtT)
        - r * K * disc_r * _norm_cdf(d2)
        + q * S * disc_q * _norm_cdf(d1)
    )
    theta_put = (
        -S * disc_q * pdf_d1 * sigma_safe / (2 * sqrtT)
        + r * K * disc_r * _norm_cdf(-d2)
        - q * S * disc_q * _norm_cdf(-d1)
    )
    theta_annual = np.where(is_call, theta_call, theta_put)
    theta = theta_annual / 365.0  # per calendar day

    rho_call = K * T * disc_r * _norm_cdf(d2) / 100.0
    rho_put = -K * T * disc_r * _norm_cdf(-d2) / 100.0
    rho = np.where(is_call, rho_call, rho_put)

    # At/after expiry, greeks of the settled position are all zero (delta is technically
    # 0 or +-1 exactly at the strike boundary, but that's a measure-zero edge case).
    zeros = np.zeros_like(S)
    delta_exp_call = np.where(S > K, 1.0, 0.0)
    delta_exp_put = np.where(S < K, -1.0, 0.0)
    delta_exp = np.where(is_call, delta_exp_call, delta_exp_put)

    return {
        "delta": np.where(T_pos, delta, delta_exp),
        "gamma": np.where(T_pos, gamma, zeros),
        "vega": np.where(T_pos, vega, zeros),
        "theta": np.where(T_pos, theta, zeros),
        "rho": np.where(T_pos, rho, zeros),
    }


def _no_arbitrage_bounds(S, K, T, r, q, right):
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    is_call = right == "C"
    lower_call = np.maximum(S * disc_q - K * disc_r, 0.0)
    upper_call = S * disc_q
    lower_put = np.maximum(K * disc_r - S * disc_q, 0.0)
    upper_put = K * disc_r
    lower = np.where(is_call, lower_call, lower_put)
    upper = np.where(is_call, upper_call, upper_put)
    return lower, upper


def implied_vol(price, S, K, T, r, q, right) -> np.ndarray:
    """Invert BSM for sigma given a mid price. Vectorized Newton with bisection fallback.

    tol=1e-8 on price, max 100 iterations. Returns np.nan (never raises) where:
      - T <= 0
      - price violates no-arbitrage bounds (price <= lower bound i.e. <= discounted
        intrinsic, or price >= upper bound)
    """
    price, S, K, T, r, q = _bcast(price, S, K, T, r, q)
    right_arr = np.broadcast_to(np.asarray(right, dtype=object), price.shape)
    shape = price.shape

    valid = T > 0
    T_safe = np.where(valid, T, 1.0)
    lower, upper = _no_arbitrage_bounds(S, K, T_safe, r, q, right_arr)
    valid = valid & (price > lower + 1e-12) & (price < upper - 1e-12)

    result = np.full(price.size, np.nan)
    idx = np.flatnonzero(valid.ravel())
    if idx.size == 0:
        return result.reshape(shape)

    # Work on the ACTIVE SUBSET only, and shrink it further as elements converge -- for a
    # typical chain the bulk of rows converge in a handful of iterations, so recomputing
    # the full 1e6-length arrays on every one of the (rare) hard-to-converge stragglers'
    # iterations would waste >90% of the work. This loop is the hot path for enrich_chain.
    S_i = S.ravel()[idx]
    K_i = K.ravel()[idx]
    T_i = T_safe.ravel()[idx]
    r_i = r.ravel()[idx]
    q_i = q.ravel()[idx]
    price_i = price.ravel()[idx]
    is_call_i = (right_arr.ravel()[idx]) == "C"

    disc_q_i = np.exp(-q_i * T_i)
    disc_r_i = np.exp(-r_i * T_i)
    sqrtT_i = np.sqrt(T_i)
    log_moneyness_i = np.log(S_i / K_i)
    drift_i = (r_i - q_i) * T_i

    def _price_and_vega(sig, S_, K_, disc_q_, disc_r_, sqrtT_, log_m_, drift_, T_, is_call_):
        sigma_safe = np.maximum(sig, _EPS)
        d1 = (log_m_ + drift_ + 0.5 * sigma_safe**2 * T_) / (sigma_safe * sqrtT_)
        d2 = d1 - sigma_safe * sqrtT_
        call_px = S_ * disc_q_ * _norm_cdf(d1) - K_ * disc_r_ * _norm_cdf(d2)
        put_px = K_ * disc_r_ * _norm_cdf(-d2) - S_ * disc_q_ * _norm_cdf(-d1)
        px = np.where(is_call_, call_px, put_px)
        vega_raw = S_ * disc_q_ * _norm_pdf(d1) * sqrtT_
        return px, vega_raw

    lo = np.full(idx.size, 1e-6)
    hi = np.full(idx.size, 5.0)

    # ensure bracket actually brackets; expand hi if needed (deep ITM/high vol cases)
    f_hi, _ = _price_and_vega(hi, S_i, K_i, disc_q_i, disc_r_i, sqrtT_i, log_moneyness_i, drift_i, T_i, is_call_i)
    need_expand = (f_hi - price_i) < 0
    tries = 0
    while np.any(need_expand) and tries < 20:
        hi = np.where(need_expand, hi * 1.5, hi)
        f_hi, _ = _price_and_vega(
            hi, S_i, K_i, disc_q_i, disc_r_i, sqrtT_i, log_moneyness_i, drift_i, T_i, is_call_i
        )
        need_expand = (f_hi - price_i) < 0
        tries += 1

    sigma = np.sqrt(2 * np.pi / np.maximum(T_i, _EPS)) * price_i / np.maximum(S_i, _EPS)
    sigma = np.clip(sigma, 1e-4, 5.0)

    for _ in range(100):
        if sigma.size == 0:
            break
        px, vega_raw = _price_and_vega(
            sigma, S_i, K_i, disc_q_i, disc_r_i, sqrtT_i, log_moneyness_i, drift_i, T_i, is_call_i
        )
        diff = px - price_i
        newly_converged = np.abs(diff) < 1e-8
        if np.any(newly_converged):
            result[idx[newly_converged]] = sigma[newly_converged]

        keep = ~newly_converged
        if not np.any(keep):
            break

        # bisection bookkeeping: the invariant f(lo) < 0 < f(hi) holds throughout (lo/hi
        # were only ever replaced by a sigma whose residual had the matching sign), so
        # sign(f(lo)) is always negative -- no need to re-evaluate price at lo each
        # iteration, saving a redundant price/vega evaluation per iteration.
        go_hi = diff < 0  # f(sigma) has the same (negative) sign as f(lo) -> sigma becomes new lo
        lo = np.where(go_hi, sigma, lo)
        hi = np.where(~go_hi, sigma, hi)

        use_newton = vega_raw > 1e-8
        newton_step = np.where(use_newton, diff / np.where(use_newton, vega_raw, 1.0), 0.0)
        newton_sigma = sigma - newton_step
        # fall back to bisection if newton overshoots outside bracket or vega too small
        mid = 0.5 * (lo + hi)
        bad_newton = (newton_sigma <= lo) | (newton_sigma >= hi) | ~use_newton
        sigma = np.where(bad_newton, mid, newton_sigma)

        # compact: drop the converged elements from every working array
        idx = idx[keep]
        sigma = sigma[keep]
        lo = lo[keep]
        hi = hi[keep]
        S_i = S_i[keep]
        K_i = K_i[keep]
        T_i = T_i[keep]
        price_i = price_i[keep]
        is_call_i = is_call_i[keep]
        disc_q_i = disc_q_i[keep]
        disc_r_i = disc_r_i[keep]
        sqrtT_i = sqrtT_i[keep]
        log_moneyness_i = log_moneyness_i[keep]
        drift_i = drift_i[keep]

    return result.reshape(shape)


def enrich_chain(chain: pd.DataFrame, r: float | pd.Series = 0.02, q: float | pd.Series = 0.0) -> pd.DataFrame:
    """Compute T, mid, IV (inverted from mid), and greeks for a chain; validate against schema.

    Requires CHAIN_REQUIRED columns. Preserves iv_vendor/delta_vendor if present (else NaN),
    fills source/is_synthetic with defaults if absent. Fully vectorized (no python row loops).
    """
    out = chain.copy()

    missing = [c for c in schema.CHAIN_REQUIRED if c not in out.columns]
    if missing:
        raise schema.SchemaError(f"enrich_chain: missing required columns: {missing}")

    quote_date = pd.to_datetime(out["quote_date"])
    expiry = pd.to_datetime(out["expiry"])
    T = (expiry - quote_date).dt.days.astype(float) / 365.0

    S = out["underlying_price"].to_numpy(dtype=float)
    K = out["strike"].to_numpy(dtype=float)
    right = out["right"].to_numpy()
    mid = (out["bid"].to_numpy(dtype=float) + out["ask"].to_numpy(dtype=float)) / 2.0

    n = len(out)
    r_arr = np.full(n, r, dtype=float) if np.isscalar(r) else np.asarray(r, dtype=float)
    q_arr = np.full(n, q, dtype=float) if np.isscalar(q) else np.asarray(q, dtype=float)

    iv = implied_vol(mid, S, K, T.to_numpy(), r_arr, q_arr, right)
    greeks = bs_greeks(S, K, T.to_numpy(), r_arr, q_arr, np.nan_to_num(iv, nan=0.0), right)
    for name, val in greeks.items():
        val = np.where(np.isnan(iv), np.nan, val)
        out[name] = val
    out["iv"] = iv

    if "iv_vendor" not in out.columns:
        out["iv_vendor"] = np.nan
    if "delta_vendor" not in out.columns:
        out["delta_vendor"] = np.nan
    if "source" not in out.columns:
        out["source"] = "unknown"
    if "is_synthetic" not in out.columns:
        out["is_synthetic"] = False

    return schema.validate_chain(out, strict=True)
