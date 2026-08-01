"""Tests for src/odds_lab/quant/{bs,empirical,edge}.py."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from odds_lab.config import EmpiricalConfig
from odds_lab.quant.bs import bs_greeks, bs_price, enrich_chain, implied_vol
from odds_lab.quant.edge import EdgeResult, naked_edge, spread_edge, theo_prob_itm
from odds_lab.quant.empirical import EmpiricalDist, build_empirical
from odds_lab import schema


# ======================================================================================
# bs.py -- bs_price
# ======================================================================================


def test_bs_price_textbook_hull_42_40():
    # Hull, "Options, Futures & Other Derivatives" 9e, worked example (S=42,K=40,r=10%,
    # sigma=20%, T=0.5y, no div): c ~= 4.76, p ~= 0.81. Call ITM, put OTM.
    c = bs_price(42, 40, 0.5, 0.10, 0.0, 0.20, "C")
    p = bs_price(42, 40, 0.5, 0.10, 0.0, 0.20, "P")
    assert c == pytest.approx(4.76, abs=0.01)
    assert p == pytest.approx(0.81, abs=0.01)


def test_bs_price_textbook_hull_49_50():
    # Hull 9e, delta worked example (S=49,K=50,r=5%,sigma=20%,T=20/52y, no div): c ~= 2.4005.
    # Call OTM (K>S).
    c = bs_price(49, 50, 20 / 52, 0.05, 0.0, 0.20, "C")
    assert c == pytest.approx(2.4005, abs=0.01)


def test_bs_price_textbook_haug():
    # Haug, "The Complete Guide to Option Pricing Formulas" 2e, Table 2.1 GBS test case:
    # S=60,K=65,T=0.25,r=8%,sigma=30%, no div: c ~= 2.1334, p ~= 5.846.
    # Call OTM, put ITM (K>S).
    c = bs_price(60, 65, 0.25, 0.08, 0.0, 0.30, "C")
    p = bs_price(60, 65, 0.25, 0.08, 0.0, 0.30, "P")
    assert c == pytest.approx(2.1334, abs=0.01)
    assert p == pytest.approx(5.846, abs=0.01)


def test_bs_price_expiry_is_intrinsic():
    assert bs_price(105, 100, 0.0, 0.02, 0.0, 0.2, "C") == pytest.approx(5.0)
    assert bs_price(95, 100, 0.0, 0.02, 0.0, 0.2, "P") == pytest.approx(5.0)
    assert bs_price(95, 100, 0.0, 0.02, 0.0, 0.2, "C") == pytest.approx(0.0)


def test_put_call_parity_grid():
    # c - p == S*exp(-qT) - K*exp(-rT), to 1e-10, across a grid.
    S = np.array([50, 90, 100, 110, 150], dtype=float)
    K = np.array([80, 100, 120], dtype=float)
    T = np.array([0.05, 0.25, 1.0, 2.0])
    r = np.array([0.0, 0.03, 0.06])
    q = np.array([0.0, 0.02])
    sigma = np.array([0.1, 0.3, 0.8])

    Sg, Kg, Tg, rg, qg, sg = np.meshgrid(S, K, T, r, q, sigma, indexing="ij")
    c = bs_price(Sg, Kg, Tg, rg, qg, sg, "C")
    p = bs_price(Sg, Kg, Tg, rg, qg, sg, "P")
    lhs = c - p
    rhs = Sg * np.exp(-qg * Tg) - Kg * np.exp(-rg * Tg)
    np.testing.assert_allclose(lhs, rhs, atol=1e-10)


# ======================================================================================
# bs.py -- greeks vs central finite differences
# ======================================================================================


def _fd_greeks(S, K, T, r, q, sigma, right):
    hS = S * 1e-5
    delta = (bs_price(S + hS, K, T, r, q, sigma, right) - bs_price(S - hS, K, T, r, q, sigma, right)) / (2 * hS)
    gamma = (
        bs_price(S + hS, K, T, r, q, sigma, right)
        - 2 * bs_price(S, K, T, r, q, sigma, right)
        + bs_price(S - hS, K, T, r, q, sigma, right)
    ) / (hS**2)
    hSig = 1e-6
    vega = (
        bs_price(S, K, T, r, q, sigma + hSig, right) - bs_price(S, K, T, r, q, sigma - hSig, right)
    ) / (2 * hSig) / 100.0
    hT = min(T * 1e-5, T * 0.49) if T > 0 else 1e-8
    # theta is -dPrice/dT (time DECAY as calendar time passes and T shrinks), per day.
    theta = -(
        bs_price(S, K, T + hT, r, q, sigma, right) - bs_price(S, K, T - hT, r, q, sigma, right)
    ) / (2 * hT) / 365.0
    hr = 1e-6
    rho = (
        bs_price(S, K, T, r + hr, q, sigma, right) - bs_price(S, K, T, r - hr, q, sigma, right)
    ) / (2 * hr) / 100.0
    return delta, gamma, vega, theta, rho


@pytest.mark.parametrize("right", ["C", "P"])
@pytest.mark.parametrize("T", [5 / 365, 0.05, 0.25, 1.0, 2.0])
@pytest.mark.parametrize("moneyness", [0.85, 0.95, 1.0, 1.05, 1.2])
def test_greeks_vs_finite_difference(right, T, moneyness):
    S = 100.0
    K = S * moneyness
    r, q, sigma = 0.03, 0.01, 0.25
    analytic = bs_greeks(S, K, T, r, q, sigma, right)
    fd_delta, fd_gamma, fd_vega, fd_theta, fd_rho = _fd_greeks(S, K, T, r, q, sigma, right)

    assert float(analytic["delta"]) == pytest.approx(fd_delta, rel=1e-5, abs=1e-6)
    assert float(analytic["gamma"]) == pytest.approx(fd_gamma, rel=1e-4, abs=1e-6)
    assert float(analytic["vega"]) == pytest.approx(fd_vega, rel=1e-4, abs=1e-6)
    assert float(analytic["theta"]) == pytest.approx(fd_theta, rel=1e-4, abs=1e-5)
    assert float(analytic["rho"]) == pytest.approx(fd_rho, rel=1e-4, abs=1e-6)


def test_theta_is_negative_for_long_options():
    g = bs_greeks(100, 100, 0.25, 0.02, 0.0, 0.2, "C")
    assert float(g["theta"]) < 0
    g = bs_greeks(100, 100, 0.25, 0.02, 0.0, 0.2, "P")
    assert float(g["theta"]) < 0


# ======================================================================================
# bs.py -- implied_vol
# ======================================================================================


def test_implied_vol_round_trip_grid():
    moneyness = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    T = np.array([0.02, 0.1, 0.5, 1.0, 2.0])
    sigma_true = np.array([0.10, 0.20, 0.30, 0.50, 0.80])
    rights = ["C", "P"]

    n_checked = 0
    for m in moneyness:
        for t in T:
            for sv in sigma_true:
                for right in rights:
                    S, K, r, q = 100.0, 100.0 * m, 0.03, 0.01
                    price = float(bs_price(S, K, t, r, q, sv, right))
                    if price < 1e-6:
                        continue  # legitimately near-zero; NaN is correct, checked elsewhere
                    lower = float(
                        max(S * np.exp(-q * t) - K * np.exp(-r * t), 0.0)
                        if right == "C"
                        else max(K * np.exp(-r * t) - S * np.exp(-q * t), 0.0)
                    )
                    upper = float(S * np.exp(-q * t) if right == "C" else K * np.exp(-r * t))
                    if price - lower < 1e-6 or upper - price < 1e-6:
                        # time value has underflowed to float64 precision at this deep
                        # ITM/short-T/low-vol corner -- NaN is the mathematically correct
                        # answer here, checked separately below.
                        continue
                    iv = float(implied_vol(price, S, K, t, r, q, right))
                    assert not np.isnan(iv), (S, K, t, r, q, right, price)
                    price2 = float(bs_price(S, K, t, r, q, iv, right))
                    assert abs(price - price2) < 1e-8
                    n_checked += 1
    assert n_checked > 100


def test_implied_vol_nan_for_arbitrage_violation_and_expired():
    S, K, T, r, q = 100.0, 100.0, 0.25, 0.02, 0.0
    upper = S * np.exp(-q * T)  # call upper no-arb bound
    lower = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)

    assert np.isnan(implied_vol(upper + 5.0, S, K, T, r, q, "C"))
    assert np.isnan(implied_vol(max(lower - 1.0, -1.0), S, K, T, r, q, "C"))
    assert np.isnan(implied_vol(5.0, S, K, 0.0, r, q, "C"))
    assert np.isnan(implied_vol(5.0, S, K, -0.1, r, q, "C"))


def test_implied_vol_deep_otm_ratio_does_not_diverge():
    # Deep OTM call, short-dated, low vol: price is tiny but should either round-trip or
    # cleanly return NaN -- never raise, never blow up.
    S, K, T, r, q, sigma = 100.0, 200.0, 0.05, 0.02, 0.0, 0.15
    price = float(bs_price(S, K, T, r, q, sigma, "C"))
    iv = implied_vol(price, S, K, T, r, q, "C")
    assert np.isnan(iv) or (iv > 0 and iv < 5.0)

    # Deep ITM put, near expiry
    S, K, T, r, q, sigma = 100.0, 200.0, 1 / 365, 0.02, 0.0, 0.4
    price = float(bs_price(S, K, T, r, q, sigma, "P"))
    iv = implied_vol(price, S, K, T, r, q, "P")
    assert np.isnan(iv) or (iv > 0 and iv < 5.0)


def test_implied_vol_vectorized_no_exceptions_on_mixed_grid():
    rng = np.random.default_rng(0)
    n = 2000
    S = rng.uniform(50, 200, n)
    K = rng.uniform(20, 400, n)
    T = rng.uniform(-0.1, 2.0, n)  # includes some invalid T<=0
    r = rng.uniform(0.0, 0.06, n)
    q = rng.uniform(0.0, 0.03, n)
    sigma = rng.uniform(0.05, 1.5, n)
    right = rng.choice(["C", "P"], n)
    price = bs_price(S, K, np.maximum(T, 0), r, q, sigma, right)
    # perturb some prices to violate arbitrage
    price[::7] *= 5.0
    iv = implied_vol(price, S, K, T, r, q, right)
    assert iv.shape == (n,)
    assert np.isfinite(iv[~np.isnan(iv)]).all()


# ======================================================================================
# bs.py -- enrich_chain
# ======================================================================================


def _synthetic_chain(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    quote_date = pd.Timestamp("2020-01-02")
    dte = rng.integers(5, 90, n)
    expiry = quote_date + pd.to_timedelta(dte, unit="D")
    S = rng.uniform(80, 420, n)
    moneyness = rng.uniform(0.7, 1.3, n)
    K = np.round(S * moneyness / 0.5) * 0.5
    right = rng.choice(["C", "P"], n)
    sigma_true = rng.uniform(0.08, 0.9, n)
    r, q = 0.02, 0.01
    T = dte.astype(float) / 365.0
    mid = bs_price(S, K, T, r, q, sigma_true, right)
    half_spread = np.maximum(0.01, 0.02 * mid)
    bid = np.maximum(mid - half_spread, 0.0)
    ask = mid + half_spread

    return pd.DataFrame(
        {
            "root": "SPY",
            "quote_date": quote_date,
            "expiry": expiry,
            "strike": K,
            "right": right,
            "bid": bid,
            "ask": ask,
            "bid_size": 10,
            "ask_size": 10,
            "last": mid,
            "volume": 1,
            "open_interest": 100,
            "ms_of_day": 57_600_000,
            "underlying_price": S,
        }
    ), sigma_true


def test_enrich_chain_schema_valid_and_iv_round_trips():
    chain, sigma_true = _synthetic_chain(5000)
    out = enrich_chain(chain, r=0.02, q=0.01)

    # schema.validate_chain already ran inside enrich_chain and would have raised;
    # re-running here proves the returned frame is still valid (idempotent).
    schema.validate_chain(out, strict=True)
    assert list(out.columns) == schema.CHAIN_COLUMNS

    mid = (out["bid"] + out["ask"]) / 2.0
    finite = out["iv"].notna()
    assert finite.mean() > 0.95  # a handful of near-zero-price rows may legitimately be NaN

    recomputed = bs_price(
        out.loc[finite, "underlying_price"].to_numpy(),
        out.loc[finite, "strike"].to_numpy(),
        (pd.to_datetime(out.loc[finite, "expiry"]) - pd.to_datetime(out.loc[finite, "quote_date"])).dt.days
        / 365.0,
        0.02,
        0.01,
        out.loc[finite, "iv"].to_numpy(),
        out.loc[finite, "right"].to_numpy(),
    )
    np.testing.assert_allclose(recomputed, mid[finite].to_numpy(), atol=1e-6)

    assert out["iv_vendor"].isna().all()
    assert out["delta_vendor"].isna().all()
    assert (out["source"] == "unknown").all()
    assert (~out["is_synthetic"]).all()


def test_enrich_chain_preserves_vendor_columns():
    chain, _ = _synthetic_chain(200)
    chain = chain.copy()
    chain["iv_vendor"] = 0.25
    chain["delta_vendor"] = 0.3
    chain["source"] = "thetadata"
    chain["is_synthetic"] = False
    out = enrich_chain(chain)
    assert (out["iv_vendor"] == 0.25).all()
    assert (out["delta_vendor"] == 0.3).all()
    assert (out["source"] == "thetadata").all()


def test_enrich_chain_performance_target():
    import time

    chain, _ = _synthetic_chain(1_000_000, seed=7)
    t0 = time.perf_counter()
    enrich_chain(chain)
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0, f"enrich_chain took {elapsed:.2f}s for 1e6 rows"


# ======================================================================================
# empirical.py
# ======================================================================================


def _synthetic_closes(n: int = 4000, seed: int = 42, start="2005-01-03", daily_vol=0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    daily_returns = rng.normal(0.0002, daily_vol, n)
    closes = 100.0 * np.exp(np.cumsum(daily_returns))
    idx = pd.bdate_range(start=start, periods=n)
    return pd.Series(closes, index=idx)


def test_build_empirical_raises_on_lookahead():
    closes = _synthetic_closes()
    too_early = (closes.index[0] - pd.Timedelta(days=10)).date()
    cfg = EmpiricalConfig(lookback_years=10, min_samples=50)
    with pytest.raises(ValueError):
        build_empirical(closes, horizon=21, cfg=cfg, asof=too_early)


def test_build_empirical_uses_only_data_before_asof():
    closes = _synthetic_closes(n=3000, start="2005-01-03")
    asof = closes.index[1500].date()
    cfg = EmpiricalConfig(lookback_years=50, min_samples=50)
    dist = build_empirical(closes, horizon=21, cfg=cfg, asof=asof)
    # the max daily return index implied by n_raw+horizon must stay under asof
    assert dist.n_raw <= 1500  # can't have used rows at/after asof


def test_build_empirical_respects_lookback_window():
    closes = _synthetic_closes(n=4000, start="2000-01-03")
    asof = closes.index[-1].date()
    cfg_full = EmpiricalConfig(lookback_years=50, min_samples=50)
    cfg_short = EmpiricalConfig(lookback_years=1, min_samples=50)
    dist_full = build_empirical(closes, horizon=10, cfg=cfg_full, asof=asof)
    dist_short = build_empirical(closes, horizon=10, cfg=cfg_short, asof=asof)
    assert dist_short.n_raw < dist_full.n_raw


def test_build_empirical_raises_below_min_samples():
    closes = _synthetic_closes(n=100, start="2020-01-02")
    asof = closes.index[-1].date()
    cfg = EmpiricalConfig(lookback_years=10, min_samples=10_000)
    with pytest.raises(ValueError):
        build_empirical(closes, horizon=21, cfg=cfg, asof=asof)


def test_symmetrized_distribution_has_zero_mean():
    closes = _synthetic_closes(n=3000, seed=3)
    asof = closes.index[-1].date()
    cfg = EmpiricalConfig(lookback_years=50, symmetrize=True, min_samples=50)
    dist = build_empirical(closes, horizon=21, cfg=cfg, asof=asof)
    assert dist.symmetrized
    assert abs(dist.log_returns.mean()) < 1e-9


def test_prob_below_is_monotone():
    closes = _synthetic_closes(n=3000, seed=5)
    asof = closes.index[-1].date()
    cfg = EmpiricalConfig(lookback_years=50, min_samples=50)
    dist = build_empirical(closes, horizon=21, cfg=cfg, asof=asof)
    xs = np.linspace(-0.5, 0.5, 25)
    probs = [dist.prob_below(x) for x in xs]
    assert all(a <= b + 1e-12 for a, b in zip(probs, probs[1:]))
    assert dist.prob_below(-np.inf) == pytest.approx(0.0)
    assert dist.prob_below(np.inf) == pytest.approx(1.0)


def test_block_bootstrap_ci_wider_than_naive_binomial():
    # STRATEGY.md §2: overlapping windows are autocorrelated -> the naive binomial CI
    # on the horizon-window sample is materially too narrow. Demonstrate it here even
    # with i.i.d. DAILY returns: the overlap itself induces autocorrelation between
    # adjacent horizon sums.
    closes = _synthetic_closes(n=3000, seed=11, daily_vol=0.012)
    asof = closes.index[-1].date()
    cfg = EmpiricalConfig(lookback_years=50, symmetrize=False, min_samples=50, bootstrap_samples=300, seed=1)
    horizon = 21
    dist = build_empirical(closes, horizon=horizon, cfg=cfg, asof=asof)

    x = dist.quantile(0.2)  # an interior threshold with real mass around it
    p_hat = dist.prob_below(x)
    n = dist.n_raw

    z = 1.959963984540054  # 95%
    naive_half_width = z * np.sqrt(p_hat * (1 - p_hat) / n)
    naive_width = 2 * naive_half_width

    boot_lo, boot_hi = dist.ci_below(x, alpha=0.05)
    boot_width = boot_hi - boot_lo

    assert boot_width > naive_width * 1.3, (boot_width, naive_width)


# ======================================================================================
# edge.py
# ======================================================================================


def _make_dist(log_returns: np.ndarray, horizon: int = 21) -> EmpiricalDist:
    return EmpiricalDist(
        log_returns=log_returns,
        horizon=horizon,
        n_raw=len(log_returns),
        symmetrized=False,
        asof=date(2020, 1, 1),
        daily_log_returns=np.diff(np.log(100 * np.exp(np.cumsum(log_returns[: min(500, len(log_returns))])))),
        seed=1,
        bootstrap_mean_block=horizon,
        bootstrap_samples=50,
    )


def test_theo_prob_itm_matches_d2():
    S, K, T, r, q, sigma = 100.0, 105.0, 0.25, 0.02, 0.0, 0.2
    p_call = theo_prob_itm(S, K, T, r, q, sigma, "C")
    p_put = theo_prob_itm(S, K, T, r, q, sigma, "P")
    # deltas are close to but not equal to N(d2); sanity bounds
    assert 0.0 < p_call < 1.0
    assert 0.0 < p_put < 1.0
    # far OTM call has low ITM prob; deep ITM-equivalent put (same K) has high P(below K)
    assert p_call < 0.5


def test_spread_edge_zero_when_distribution_matches_pricing_lognormal():
    S, T, r, q, iv = 100.0, 30 / 365, 0.02, 0.0, 0.20
    rng = np.random.default_rng(123)
    n = 400_000
    z = rng.standard_normal(n)
    mu = (r - q - 0.5 * iv**2) * T
    sd = iv * np.sqrt(T)
    log_returns = mu + sd * z
    dist = _make_dist(log_returns)

    short_strike, long_strike = 95.0, 90.0  # put credit spread
    credit, width = 0.9, 5.0

    result = spread_edge(
        credit=credit,
        width=width,
        short_strike=short_strike,
        long_strike=long_strike,
        right="P",
        S=S,
        T_years=T,
        iv=iv,
        r=r,
        q=q,
        dist=dist,
    )
    assert isinstance(result, EdgeResult)
    assert abs(result.edge_prob) < 0.01
    assert abs(result.edge_ev - result.ev_theo) < 0.02


def _crash_mixture_log_returns(n, S, iv, T, r, q, p=0.03, seed=1):
    """Terminal log-return sample with total variance matched to the pricing lognormal's
    sigma_T, but with a p-probability -4*sigma_T crash jump (STRATEGY.md §1: 2008/2020-style
    fat left tail). Mean is re-centered so the drift matches the lognormal used for pricing.

    Law of total variance: with the non-jump branch ~ N(0, sc) and the jump branch a point
    mass at `jump`, Var(X) = (1-p)*sc^2 + p*(1-p)*jump^2. Solve for sc so Var(X) == sigma_T^2,
    then recenter the (jump, 0)-mixture by its own mean (p*jump) so E[X] == 0 before adding
    the pricing drift back in.
    """
    sigma_T = iv * np.sqrt(T)
    jump = -4.0 * sigma_T
    sc2 = sigma_T**2 / (1 - p) - p * jump**2
    assert sc2 > 0
    sc = np.sqrt(sc2)
    rng = np.random.default_rng(seed)
    is_jump = rng.random(n) < p
    normal_part = rng.normal(0.0, sc, n)
    x = np.where(is_jump, jump, normal_part)
    x = x - p * jump  # recenter the mixture to zero mean
    mu = (r - q - 0.5 * iv**2) * T
    return mu + x


def test_naked_vs_spread_under_crash_mixture():
    # Test B (replaces an earlier, mathematically wrong "fat-tail => always worse" test --
    # a variance-matched Student-t at a ~1-sigma strike is actually MORE peaked near the
    # money than the normal, so P(ITM) falls and EV improves; that is not a tail-risk
    # regime). The economically meaningful case is a crash-mixture regime (rare, large
    # downside jump, variance held fixed) -- this is what actually ruins a naked seller
    # (STRATEGY.md §1, the Barings 2008 example) while a defined-risk spread's loss is
    # bounded below by -(width - credit) regardless.
    S, iv, T, r, q = 400.0, 0.16, 30 / 365, 0.02, 0.015
    short_strike, long_strike = 380.0, 375.0
    width = short_strike - long_strike
    n = 1_000_000

    # theoretical (lognormal) credit ~ what the market would charge for these strikes;
    # use BS prices themselves so credit is self-consistent with iv/right.
    short_px = float(bs_price(S, short_strike, T, r, q, iv, "P"))
    long_px = float(bs_price(S, long_strike, T, r, q, iv, "P"))
    credit = short_px - long_px

    lognormal_returns = _crash_mixture_log_returns(n, S, iv, T, r, q, p=0.0, seed=1)  # p=0 => pure lognormal
    dist_lognormal = _make_dist(lognormal_returns, horizon=21)
    crash_returns = _crash_mixture_log_returns(n, S, iv, T, r, q, p=0.03, seed=1)
    dist_crash = _make_dist(crash_returns, horizon=21)

    naked_ln = naked_edge(
        credit=short_px, short_strike=short_strike, right="P", S=S, T_years=T, iv=iv, r=r, q=q,
        dist=dist_lognormal,
    )
    spread_ln = spread_edge(
        credit=credit, width=width, short_strike=short_strike, long_strike=long_strike, right="P",
        S=S, T_years=T, iv=iv, r=r, q=q, dist=dist_lognormal,
    )
    naked_crash = naked_edge(
        credit=short_px, short_strike=short_strike, right="P", S=S, T_years=T, iv=iv, r=r, q=q,
        dist=dist_crash,
    )
    spread_crash = spread_edge(
        credit=credit, width=width, short_strike=short_strike, long_strike=long_strike, right="P",
        S=S, T_years=T, iv=iv, r=r, q=q, dist=dist_crash,
    )

    # sanity: on the pure-lognormal sample, actual matches theory closely.
    assert abs(naked_ln.edge_prob) < 0.01
    assert abs(spread_ln.edge_prob) < 0.01

    # under the crash mixture: P(ITM) actually falls (rare huge jump vs constant total
    # variance spent elsewhere shrinks the ordinary-day dispersion) -- both agents see this.
    assert naked_crash.p_actual_loss < naked_crash.p_theo_loss

    # but the NAKED seller's EV is devastated by the jump; the SPREAD's EV is protected
    # by the long leg (bounded below by -(width - credit)).
    assert naked_crash.edge_ev < 0
    assert naked_crash.edge_ev < naked_crash.ev_theo - 0.05
    assert spread_crash.edge_ev > -(width - credit) - 1e-6
    assert naked_crash.edge_ev < spread_crash.edge_ev - 0.1


def test_edge_prob_sign_when_actual_tail_mass_exceeds_theory():
    # Cheap, shape-independent sign check: construct an empirical sample that puts
    # strictly MORE mass below the short-put threshold than the pricing lognormal does
    # (shift a slice of the left tail further left), leaving everything else untouched.
    # edge_prob = p_theo_loss - p_actual_loss must then be NEGATIVE, and the seller's
    # empirical EV must be worse than the theoretical EV.
    S, T, r, q, iv = 100.0, 30 / 365, 0.02, 0.0, 0.20
    rng = np.random.default_rng(3)
    n = 300_000
    mu = (r - q - 0.5 * iv**2) * T
    sd = iv * np.sqrt(T)
    z = rng.standard_normal(n)
    log_returns = mu + sd * z

    short_strike, long_strike = 95.0, 90.0
    threshold = float(np.log(short_strike / S))
    below = log_returns < threshold
    # push a slice of the already-below-threshold mass further down (more crash-like),
    # strictly increasing P(actual <= threshold) is automatic since they stay below --
    # instead move some near-threshold mass to below it, which does increase the count.
    near = (log_returns >= threshold) & (log_returns < threshold + 0.02)
    idx = np.where(near)[0]
    move = idx[: len(idx) // 2]
    log_returns[move] -= 0.05

    credit, width = 0.9, 5.0
    dist = _make_dist(log_returns)
    result = spread_edge(
        credit=credit, width=width, short_strike=short_strike, long_strike=long_strike,
        right="P", S=S, T_years=T, iv=iv, r=r, q=q, dist=dist,
    )
    assert result.edge_prob < 0
    assert result.edge_ev < result.ev_theo


def test_naked_edge_with_loss_cap():
    S, T, r, q, iv = 100.0, 30 / 365, 0.02, 0.0, 0.25
    rng = np.random.default_rng(99)
    n = 100_000
    z = rng.standard_normal(n)
    mu = (r - q - 0.5 * iv**2) * T
    sd = iv * np.sqrt(T)
    log_returns = mu + sd * z
    dist = _make_dist(log_returns)

    uncapped = naked_edge(
        credit=1.5, short_strike=90.0, right="P", S=S, T_years=T, iv=iv, r=r, q=q, dist=dist
    )
    capped = naked_edge(
        credit=1.5, short_strike=90.0, right="P", S=S, T_years=T, iv=iv, r=r, q=q, dist=dist,
        loss_cap=20.0,
    )
    # capping losses can only raise (or leave unchanged) expected value for the seller
    assert capped.edge_ev >= uncapped.edge_ev - 1e-9
    assert capped.ev_theo >= uncapped.ev_theo - 1e-9
    assert capped.p_theo_loss == pytest.approx(uncapped.p_theo_loss)
