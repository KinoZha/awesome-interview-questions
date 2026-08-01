"""Theoretical vs empirical edge -- STRATEGY.md §2 step 3, "Compare".

`edge_prob = p_theo_loss - p_actual_loss`: positive means the market's risk-neutral
probability of the short strike finishing ITM exceeds the empirically observed
frequency of that move, i.e. the option is overpriced relative to realized history
-> sell premium.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from odds_lab.quant.bs import _d1_d2, bs_price
from odds_lab.quant.empirical import EmpiricalDist

__all__ = ["EdgeResult", "theo_prob_itm", "spread_edge", "naked_edge"]


@dataclass(frozen=True)
class EdgeResult:
    p_theo_loss: float
    p_actual_loss: float
    edge_prob: float
    ev_theo: float
    edge_ev: float
    p_actual_ci: tuple[float, float] | None


def theo_prob_itm(S, K, T, r, q, sigma, right) -> float:
    """Risk-neutral P(finish ITM): N(d2) for a call, N(-d2) for a put."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    _, d2 = _d1_d2(S, K, T, r, q, sigma)
    if right == "C":
        p = norm.cdf(d2)
    else:
        p = norm.cdf(-d2)
    return float(p)


def _undiscounted_vanilla(
    S: float, K: float, T: float, r: float, q: float, sigma: float, right: str
) -> float:
    """E_Q[max(S_T - K, 0)] for a call / E_Q[max(K - S_T, 0)] for a put, under the
    risk-neutral lognormal terminal density.

    Exact, because the Black-Scholes price *is* that expectation discounted at r --
    so undiscounting it recovers the raw expectation with no quadrature error.
    """
    if T <= 0 or sigma <= 0:
        return float(max(S - K, 0.0) if right == "C" else max(K - S, 0.0))
    return float(np.exp(r * T) * np.asarray(bs_price(S, K, T, r, q, sigma, right)))


def _vertical_ev_theo(
    *,
    credit: float,
    S: float,
    short_strike: float,
    long_strike: float | None,
    right: str,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """Theoretical EV of a short vertical (or a bare short option when long_strike is
    None) under the risk-neutral lognormal.

    Every structure this system trades has a piecewise-linear payoff that decomposes
    into vanillas, so this is closed-form. It replaces an earlier Gauss-Hermite
    quadrature, which converges badly on kinked payoffs: at 96 nodes it mispriced a
    380/375 put spread by 0.018 on a 0.34 credit (5% of the premium), and numpy's
    hermgauss overflows to NaN past ~256 nodes, so the error could not be reduced.
    """
    owed = _undiscounted_vanilla(S, short_strike, T, r, q, sigma, right)
    if long_strike is not None and long_strike > 0:
        owed -= _undiscounted_vanilla(S, long_strike, T, r, q, sigma, right)
    return float(credit - owed)


def _payoff_leg(S_T: np.ndarray, strike: float, right: str) -> np.ndarray:
    if right == "C":
        return np.maximum(S_T - strike, 0.0)
    return np.maximum(strike - S_T, 0.0)


def _loss_threshold_log_return(S: float, strike: float) -> float:
    return float(np.log(strike / S))


def _empirical_prob_loss(dist: EmpiricalDist, S: float, short_strike: float, right: str) -> float:
    x = _loss_threshold_log_return(S, short_strike)
    if right == "P":
        return dist.prob_below(x)
    return dist.prob_above(x)


def _empirical_ci_loss(
    dist: EmpiricalDist, S: float, short_strike: float, right: str
) -> tuple[float, float]:
    x = _loss_threshold_log_return(S, short_strike)
    lo, hi = dist.ci_below(x)
    if right == "P":
        return lo, hi
    return 1.0 - hi, 1.0 - lo


def spread_edge(
    *,
    credit: float,
    width: float,
    short_strike: float,
    long_strike: float,
    right: str,
    S: float,
    T_years: float,
    iv: float,
    r: float,
    q: float,
    dist: EmpiricalDist,
    with_ci: bool = False,
) -> EdgeResult:
    """Edge computation for a defined-risk credit spread (S1/S2/S3 legs).

    edge_ev integrates the ACTUAL spread payoff over the empirical terminal-price
    distribution: S_T = S*exp(r_i) for each empirical log return r_i (no two-point
    win/lose approximation). ev_theo is the same payoff under the risk-neutral
    lognormal with the priced `iv`, in closed form (see _vertical_ev_theo).
    """
    p_theo_loss = theo_prob_itm(S, short_strike, T_years, r, q, iv, right)
    p_actual_loss = _empirical_prob_loss(dist, S, short_strike, right)

    ev_theo = _vertical_ev_theo(
        credit=credit,
        S=S,
        short_strike=short_strike,
        long_strike=long_strike,
        right=right,
        T=T_years,
        r=r,
        q=q,
        sigma=iv,
    )

    S_T_emp = S * np.exp(dist.log_returns)
    owed_emp = _payoff_leg(S_T_emp, short_strike, right) - _payoff_leg(S_T_emp, long_strike, right)
    edge_ev = float(np.mean(credit - owed_emp))

    p_actual_ci = _empirical_ci_loss(dist, S, short_strike, right) if with_ci else None

    return EdgeResult(
        p_theo_loss=p_theo_loss,
        p_actual_loss=p_actual_loss,
        edge_prob=p_theo_loss - p_actual_loss,
        ev_theo=ev_theo,
        edge_ev=edge_ev,
        p_actual_ci=p_actual_ci,
    )


def naked_edge(
    *,
    credit: float,
    short_strike: float,
    right: str,
    S: float,
    T_years: float,
    iv: float,
    r: float,
    q: float,
    dist: EmpiricalDist,
    loss_cap: float | None = None,
    with_ci: bool = False,
) -> EdgeResult:
    """Edge computation for an undefined-risk naked short option (S4/S5 legs).

    `loss_cap` optionally caps the per-share liability (e.g. for a stress-comparison
    report) -- both ev_theo and edge_ev respect it identically.
    """
    p_theo_loss = theo_prob_itm(S, short_strike, T_years, r, q, iv, right)
    p_actual_loss = _empirical_prob_loss(dist, S, short_strike, right)

    def _owed(S_T: np.ndarray) -> np.ndarray:
        owed = _payoff_leg(S_T, short_strike, right)
        if loss_cap is not None:
            owed = np.minimum(owed, loss_cap)
        return owed

    # A capped naked short is exactly a vertical: for a put, min(max(K-S,0), cap) equals
    # max(K-S,0) - max((K-cap)-S,0); mirrored for a call.
    cap_strike: float | None = None
    if loss_cap is not None:
        cap_strike = short_strike - loss_cap if right == "P" else short_strike + loss_cap
    ev_theo = _vertical_ev_theo(
        credit=credit,
        S=S,
        short_strike=short_strike,
        long_strike=cap_strike,
        right=right,
        T=T_years,
        r=r,
        q=q,
        sigma=iv,
    )

    S_T_emp = S * np.exp(dist.log_returns)
    edge_ev = float(np.mean(credit - _owed(S_T_emp)))

    p_actual_ci = _empirical_ci_loss(dist, S, short_strike, right) if with_ci else None

    return EdgeResult(
        p_theo_loss=p_theo_loss,
        p_actual_loss=p_actual_loss,
        edge_prob=p_theo_loss - p_actual_loss,
        ev_theo=ev_theo,
        edge_ev=edge_ev,
        p_actual_ci=p_actual_ci,
    )
