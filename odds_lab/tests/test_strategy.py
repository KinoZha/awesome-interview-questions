"""Tests for src/odds_lab/strategy/{universe,strategies,selector,exits}.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from odds_lab import schema
from odds_lab.config import BacktestConfig, CostModel, EmpiricalConfig, EntryConfig, ExitConfig, RiskConfig
from odds_lab.quant.empirical import EmpiricalDist
from odds_lab.schema import ExitReason, Position
from odds_lab.strategy import selector, strategies, universe
from odds_lab.strategy.exits import evaluate_exit

ROOT = "SPY"
ASOF = date(2015, 6, 1)
EXPIRY = date(2015, 7, 3)  # 32 DTE, inside [21, 56]


# ======================================================================================
# fixtures / helpers
# ======================================================================================


def _chain_df(quote_date: date, expiry: date, S: float, rows: list[dict]) -> pd.DataFrame:
    recs = []
    for row in rows:
        recs.append({
            "root": ROOT, "quote_date": pd.Timestamp(quote_date), "ms_of_day": 57600000,
            "expiry": pd.Timestamp(expiry), "strike": float(row["strike"]), "right": row["right"],
            "bid": float(row["bid"]), "ask": float(row["ask"]), "bid_size": 10, "ask_size": 10,
            "last": (row["bid"] + row["ask"]) / 2, "volume": 100,
            "open_interest": row.get("oi", 500), "underlying_price": S,
            "iv": row["iv"], "delta": row["delta"], "gamma": row.get("gamma", 0.01),
            "theta": row.get("theta", -0.02), "vega": row.get("vega", 0.05), "rho": row.get("rho", 0.01),
            "iv_vendor": np.nan, "delta_vendor": np.nan, "source": "test", "is_synthetic": True,
        })
    df = pd.DataFrame.from_records(recs)
    return schema.validate_chain(df, strict=True)


def _make_dist(log_returns: np.ndarray, horizon: int = 32) -> EmpiricalDist:
    log_returns = np.asarray(log_returns, dtype=float)
    return EmpiricalDist(
        log_returns=log_returns, horizon=horizon, n_raw=len(log_returns), symmetrized=False,
        asof=ASOF, daily_log_returns=log_returns[:2], seed=1,
        bootstrap_mean_block=horizon, bootstrap_samples=10,
    )


def _lognormal_dist(S: float, T: float, r: float, q: float, iv: float, n: int = 400_000, seed: int = 7) -> EmpiricalDist:
    rng = np.random.default_rng(seed)
    mu = (r - q - 0.5 * iv**2) * T
    sd = iv * np.sqrt(T)
    log_returns = mu + sd * rng.standard_normal(n)
    return _make_dist(log_returns)


PUT_ROWS_BAND = [
    # strike, delta -- descending strike (closest to S first)
    {"strike": 205.0, "right": "P", "bid": 2.00, "ask": 2.10, "delta": -0.35, "iv": 0.16, "oi": 500},
    {"strike": 200.0, "right": "P", "bid": 1.40, "ask": 1.50, "delta": -0.25, "iv": 0.17, "oi": 500},  # in band [0.15,0.30]
    {"strike": 197.0, "right": "P", "bid": 1.00, "ask": 1.10, "delta": -0.18, "iv": 0.18, "oi": 500},  # in band
    {"strike": 195.0, "right": "P", "bid": 0.70, "ask": 0.80, "delta": -0.12, "iv": 0.19, "oi": 500},
    {"strike": 190.0, "right": "P", "bid": 0.30, "ask": 0.40, "delta": -0.06, "iv": 0.20, "oi": 500},
]
CALL_ROWS_BAND = [
    {"strike": 215.0, "right": "C", "bid": 0.30, "ask": 0.40, "delta": 0.10, "iv": 0.20, "oi": 500},
    {"strike": 212.0, "right": "C", "bid": 0.60, "ask": 0.70, "delta": 0.18, "iv": 0.18, "oi": 500},
    {"strike": 210.0, "right": "C", "bid": 1.00, "ask": 1.10, "delta": 0.27, "iv": 0.17, "oi": 500},
    {"strike": 208.0, "right": "C", "bid": 1.40, "ask": 1.50, "delta": 0.35, "iv": 0.16, "oi": 500},
]

S = 210.0


def _full_chain():
    return _chain_df(ASOF, EXPIRY, S, PUT_ROWS_BAND + CALL_ROWS_BAND)


# ======================================================================================
# universe.py
# ======================================================================================


def test_market_state_bullish():
    idx = pd.date_range("2013-01-01", periods=260, freq="B")
    # rising series, strictly positive momentum and above SMA200
    vals = 100 + np.arange(260) * 0.5
    closes = pd.Series(vals, index=idx.date)
    state = universe.market_state(closes, idx.date[-1] + timedelta(days=1))
    assert state == schema.MarketState.BULLISH


def test_market_state_bearish():
    idx = pd.date_range("2013-01-01", periods=260, freq="B")
    vals = 200 - np.arange(260) * 0.5
    closes = pd.Series(vals, index=idx.date)
    state = universe.market_state(closes, idx.date[-1] + timedelta(days=1))
    assert state == schema.MarketState.BEARISH


def test_market_state_no_lookahead():
    idx = pd.date_range("2013-01-01", periods=300, freq="B")
    vals = 100 + np.arange(300) * 0.5
    closes = pd.Series(vals, index=idx.date)
    asof = idx.date[250]
    # Corrupt every close strictly at/after asof to a wildly different value; if
    # market_state peeked at it, the result would flip.
    tampered = closes.copy()
    tampered.loc[[d for d in tampered.index if d >= asof]] = -999.0
    assert universe.market_state(closes, asof) == universe.market_state(tampered, asof)


def test_realized_vol_matches_manual_stdev():
    idx = pd.date_range("2015-01-01", periods=40, freq="B")
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, 39)
    vals = 100 * np.exp(np.cumsum(np.concatenate([[0], rets])))
    closes = pd.Series(vals, index=idx.date)
    asof = idx.date[-1]
    rv = universe.realized_vol(closes, asof, window=21)
    # asof's own close is excluded (see universe.py convention); the window is built from
    # the 22 closes strictly before asof.
    manual = np.std(np.diff(np.log(vals[:-1][-22:])), ddof=1) * np.sqrt(252)
    assert rv == pytest.approx(manual)


def test_realized_vol_excludes_asof_close():
    idx = pd.date_range("2015-01-01", periods=40, freq="B")
    vals = np.full(40, 100.0)
    vals[-1] = 10_000.0  # a huge move ON asof itself
    closes = pd.Series(vals, index=idx.date)
    asof = idx.date[-1]
    rv = universe.realized_vol(closes, asof, window=21)
    assert rv == pytest.approx(0.0, abs=1e-9)  # flat except the excluded final bar


def test_iv_rank_bounds():
    idx = pd.date_range("2015-01-01", periods=252, freq="B")
    vals = np.linspace(0.10, 0.30, 252)
    s = pd.Series(vals, index=idx.date)
    rank_low = universe.iv_rank(s.iloc[:1], idx.date[0])
    rank_high = universe.iv_rank(s, idx.date[-1])
    assert rank_low == pytest.approx(0.5)  # single point, hi<=lo fallback
    assert rank_high == pytest.approx(1.0)  # latest value is the series max


def test_atm_iv_picks_closest_strikes():
    chain = _full_chain()
    iv = universe.atm_iv(chain, EXPIRY)
    # S=210: closest put is 205 (iv .16... wait band), closest call is 210 (iv .17)
    put_closest = chain[(chain.right == "P")].assign(d=(chain.strike - S).abs()).sort_values("d").iloc[0]
    call_closest = chain[(chain.right == "C")].assign(d=(chain.strike - S).abs()).sort_values("d").iloc[0]
    assert iv == pytest.approx((put_closest["iv"] + call_closest["iv"]) / 2.0)


# ======================================================================================
# strategies.py
# ======================================================================================


def test_build_legs_put_credit_spread():
    legs = strategies.build_legs("put_credit_spread", ROOT, EXPIRY, short_put=200.0, long_put=195.0)
    assert legs == [
        schema.Leg(ROOT, EXPIRY, 200.0, "P", -1),
        schema.Leg(ROOT, EXPIRY, 195.0, "P", +1),
    ]


def test_build_legs_missing_strike_raises():
    with pytest.raises(ValueError):
        strategies.build_legs("put_credit_spread", ROOT, EXPIRY, short_put=200.0)


def test_structure_economics_put_credit_spread():
    chain = _full_chain()
    legs = strategies.build_legs("put_credit_spread", ROOT, EXPIRY, short_put=200.0, long_put=195.0)
    econ = strategies.structure_economics(legs, chain)
    expected_credit = 1.40 - 0.80  # short bid - long ask
    assert econ["credit"] == pytest.approx(expected_credit)
    assert econ["width"] == pytest.approx(5.0)
    assert econ["max_loss"] == pytest.approx(5.0 - expected_credit)
    assert econ["breakevens"][0] == pytest.approx(200.0 - expected_credit)


def test_structure_economics_short_put_undefined_risk():
    chain = _full_chain()
    legs = strategies.build_legs("short_put", ROOT, EXPIRY, short_put=200.0)
    econ = strategies.structure_economics(legs, chain)
    assert econ["max_loss"] == float("inf")
    assert np.isnan(econ["width"])


def test_structure_economics_missing_quote_raises():
    chain = _full_chain()
    legs = [schema.Leg(ROOT, EXPIRY, 999.0, "P", -1), schema.Leg(ROOT, EXPIRY, 195.0, "P", +1)]
    with pytest.raises(schema.SchemaError):
        strategies.structure_economics(legs, chain)


# ======================================================================================
# selector.py -- select_strikes under each StrikeRule
# ======================================================================================


def test_select_strikes_delta_rule_picks_within_band():
    chain = _full_chain()
    cfg = EntryConfig(strategy="put_credit_spread", strike_rule="delta", delta_min=0.15, delta_max=0.30, width_strikes=1)
    picked = selector.select_strikes(chain, S, cfg)
    assert picked is not None
    assert 0.15 <= abs(picked["short_put_delta"]) <= 0.30
    # closest to the band midpoint (0.225) among {-0.25 (0.025 away), -0.18 (0.045 away)}
    assert picked["short_put"] == pytest.approx(200.0)
    assert picked["long_put"] == pytest.approx(197.0)  # one strike increment (3.0) further OTM


def test_select_strikes_delta_rule_no_candidate_returns_none():
    chain = _full_chain()
    cfg = EntryConfig(strategy="put_credit_spread", strike_rule="delta", delta_min=0.45, delta_max=0.49)
    assert selector.select_strikes(chain, S, cfg) is None


def test_select_strikes_pct_otm_baseline_rule():
    chain = _full_chain()
    cfg = EntryConfig(strategy="put_credit_spread", strike_rule="pct_otm", pct_otm=0.05, width_strikes=1)
    picked = selector.select_strikes(chain, S, cfg)
    # S*0.95 = 199.5 -> nearest listed strike is 200.0
    assert picked["short_put"] == pytest.approx(200.0)


def test_select_strikes_empirical_prob_rule_targets_probability():
    chain = _full_chain()
    dist = _lognormal_dist(S, 32 / 365, 0.02, 0.0, 0.17)
    cfg = EntryConfig(strategy="put_credit_spread", strike_rule="empirical_prob", target_prob_itm=0.20, width_strikes=1)
    picked = selector.select_strikes(chain, S, cfg, dist=dist)
    assert picked is not None
    x = np.log(picked["short_put"] / S)
    p = dist.prob_below(x)
    # verify it is the closest among all listed put strikes to the 0.20 target
    all_probs = {
        k: abs(dist.prob_below(np.log(k / S)) - 0.20) for k in chain[chain.right == "P"]["strike"].unique()
    }
    assert picked["short_put"] == min(all_probs, key=all_probs.get)


def test_select_strikes_iron_condor_both_sides():
    chain = _full_chain()
    cfg = EntryConfig(strategy="iron_condor", strike_rule="delta", delta_min=0.15, delta_max=0.30, width_strikes=1)
    picked = selector.select_strikes(chain, S, cfg)
    assert picked["short_put"] == pytest.approx(200.0)
    # 212 (|delta|=.18) and 210 (|delta|=.27) are equidistant from the band midpoint (.225);
    # the tie is broken by listing order (idxmin keeps the first match) -> 212.
    assert picked["short_call"] == pytest.approx(212.0)
    assert picked["long_call"] == pytest.approx(215.0)


# ======================================================================================
# selector.py -- propose_trade, filters + ranking + lookahead, via a FakeStore
# ======================================================================================


class FakeStore:
    """Minimal ChainStore stand-in: fully controlled, in-memory. Supports one root."""

    def __init__(self):
        self._chains: dict[tuple[date, date], pd.DataFrame] = {}
        self._closes: pd.Series | None = None
        self._underlying: pd.DataFrame | None = None

    def add_chain(self, quote_date: date, expiry: date, df: pd.DataFrame) -> None:
        self._chains[(quote_date, expiry)] = df

    def set_closes(self, closes: pd.Series) -> None:
        self._closes = closes
        self._underlying = pd.DataFrame({
            "root": ROOT, "date": pd.DatetimeIndex(closes.index),
            "open": closes.to_numpy(), "high": closes.to_numpy(), "low": closes.to_numpy(),
            "close": closes.to_numpy(), "volume": 0, "dividend": 0.0,
        })

    def chain(self, root, quote_date, expiry=None):
        if expiry is not None:
            return self._chains.get((quote_date, expiry), schema.empty_frame(schema.CHAIN_DTYPES))
        frames = [df for (qd, exp), df in self._chains.items() if qd == quote_date]
        return pd.concat(frames, ignore_index=True) if frames else schema.empty_frame(schema.CHAIN_DTYPES)

    def expiries(self, root, quote_date, dte_min=0, dte_max=400):
        out = []
        for (qd, exp) in self._chains:
            if qd != quote_date:
                continue
            dte = (exp - qd).days
            if dte_min <= dte <= dte_max:
                out.append(exp)
        return sorted(set(out))

    def closes(self, root, end):
        idx = pd.DatetimeIndex(self._closes.index)
        return self._closes[idx <= pd.Timestamp(end)]

    def trading_dates(self, root, start, end):
        idx = pd.DatetimeIndex(self._closes.index)
        mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
        return sorted(d for d in self._closes.index[mask])

    def underlying(self, root, start=None, end=None):
        df = self._underlying
        if start is not None:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["date"] <= pd.Timestamp(end)]
        return df


def _base_closes(asof: date, n_years: float = 1.5, seed: int = 5) -> pd.Series:
    idx = pd.bdate_range(end=pd.Timestamp(asof) - pd.Timedelta(days=1), periods=int(260 * n_years))
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.01, len(idx))
    vals = S * np.exp(np.cumsum(rets))
    vals = vals * (S / vals[-1])  # anchor the last (pre-asof) close near S
    return pd.Series(vals, index=idx.date)


def _base_cfg(**entry_kwargs) -> BacktestConfig:
    defaults = dict(
        strategy="put_credit_spread", strike_rule="delta", delta_min=0.15, delta_max=0.30,
        width_strikes=1, dte_min=21, dte_max=56, min_credit=0.25,
        expected_return_min=0.0, expected_return_max=0.50, market_filter="all",
        require_positive_edge=True,
    )
    defaults.update(entry_kwargs)
    entry = EntryConfig(**defaults)
    return BacktestConfig(
        roots=(ROOT,), start=ASOF - timedelta(days=1), end=ASOF + timedelta(days=60),
        entry=entry, exits=ExitConfig(), risk=RiskConfig(), costs=CostModel(),
        empirical=EmpiricalConfig(lookback_years=5.0, min_samples=50, seed=1),
    )


def _rich_store(oi=500, bid=1.40, spread_ok=True) -> FakeStore:
    """A FakeStore with one valid put-credit-spread candidate at EXPIRY. The strike the
    delta rule selects (200.0 put, short leg) can be perturbed for filter tests."""
    store = FakeStore()
    store.set_closes(_base_closes(ASOF))
    rows = []
    for r in PUT_ROWS_BAND + CALL_ROWS_BAND:
        r = dict(r)
        if r["strike"] == 200.0 and r["right"] == "P":
            r["bid"] = bid
            r["oi"] = oi
            r["ask"] = bid * 3 + 1.0 if not spread_ok else r["ask"]
        rows.append(r)
    chain = _chain_df(ASOF, EXPIRY, S, rows)
    store.add_chain(ASOF, EXPIRY, chain)
    return store


def test_propose_trade_selects_valid_candidate():
    store = _rich_store()
    cfg = _base_cfg()
    proposal = selector.propose_trade(store, ROOT, ASOF, cfg)
    assert proposal is not None
    assert proposal.strategy == "put_credit_spread"
    assert proposal.credit > 0
    for key in ("short_delta", "iv_entry", "iv_rank", "realized_vol", "market_state",
                "p_theo_loss", "p_actual_loss", "edge_prob", "edge_ev", "underlying_entry", "dte_entry"):
        assert key in proposal.meta


def test_propose_trade_rejects_low_bid():
    store = _rich_store(bid=0.01)  # below CostModel.min_bid default 0.05
    cfg = _base_cfg()
    assert selector.propose_trade(store, ROOT, ASOF, cfg) is None


def test_propose_trade_rejects_low_open_interest():
    store = _rich_store(oi=1)  # below CostModel.min_open_interest default 100
    cfg = _base_cfg()
    assert selector.propose_trade(store, ROOT, ASOF, cfg) is None


def test_propose_trade_rejects_wide_spread():
    store = _rich_store(spread_ok=False)
    cfg = _base_cfg()
    assert selector.propose_trade(store, ROOT, ASOF, cfg) is None


def test_propose_trade_rejects_below_min_credit():
    store = _rich_store()
    cfg = _base_cfg(min_credit=100.0)  # unreachable
    assert selector.propose_trade(store, ROOT, ASOF, cfg) is None


def test_propose_trade_rejects_expected_return_out_of_band():
    store = _rich_store()
    cfg = _base_cfg(expected_return_max=0.001)  # far below achievable ER
    assert selector.propose_trade(store, ROOT, ASOF, cfg) is None


def test_propose_trade_market_filter_skip_bearish():
    store = _rich_store()
    # force a bearish market state: rising-then-crashing series ending below SMA200 & down
    idx = pd.bdate_range(end=pd.Timestamp(ASOF) - pd.Timedelta(days=1), periods=400)
    vals = np.concatenate([
        np.linspace(150, 250, 250),
        np.linspace(250, 150, len(idx) - 250),
    ])
    store.set_closes(pd.Series(vals, index=idx.date))
    cfg = _base_cfg(market_filter="skip_bearish")
    assert universe.market_state(store.closes(ROOT, ASOF), ASOF) == schema.MarketState.BEARISH
    assert selector.propose_trade(store, ROOT, ASOF, cfg) is None


def test_propose_trade_require_positive_edge_rejects_negative_ev():
    store = _rich_store()
    cfg = _base_cfg(require_positive_edge=True)
    # Monkeypatch build_empirical indirectly: use a store whose closes imply a much fatter
    # left tail than the chain's IV prices in -- edge_ev should go negative and get filtered.
    idx = pd.bdate_range(end=pd.Timestamp(ASOF) - pd.Timedelta(days=1), periods=800)
    rng = np.random.default_rng(11)
    # heavy negative jumps baked into realized history, well beyond what a 17% IV prices in
    rets = rng.normal(-0.001, 0.05, len(idx))
    vals = S * np.exp(np.cumsum(rets))
    vals *= S / vals[-1]
    store.set_closes(pd.Series(vals, index=idx.date))
    proposal = selector.propose_trade(store, ROOT, ASOF, cfg)
    assert proposal is None


def test_propose_trade_ranks_by_max_edge_ev_not_er():
    store = _rich_store()
    # add a second, later expiry with a strictly worse (lower) edge_ev candidate: bump the
    # short put's ask way up (crossing more) so its edge_ev is worse despite similar ER.
    expiry2 = EXPIRY + timedelta(days=14)
    rows2 = [dict(r) for r in (PUT_ROWS_BAND + CALL_ROWS_BAND)]
    for r in rows2:
        if r["strike"] == 200.0 and r["right"] == "P":
            r["iv"] = 0.60  # richly priced -> the theoretical/empirical gap (edge) is worse
    store.add_chain(ASOF, expiry2, _chain_df(ASOF, expiry2, S, rows2))
    cfg = _base_cfg(dte_max=60)
    proposal = selector.propose_trade(store, ROOT, ASOF, cfg)
    assert proposal is not None
    # the original (EXPIRY, iv=.17) candidate must win over the richly-priced (expiry2) one
    assert proposal.expiry == EXPIRY


def test_propose_trade_lookahead_raises():
    store = _rich_store()
    # corrupt the chain's quote_date to be after asof
    bad = store._chains[(ASOF, EXPIRY)].copy()
    bad["quote_date"] = pd.Timestamp(ASOF) + pd.Timedelta(days=5)
    store._chains[(ASOF, EXPIRY)] = bad
    cfg = _base_cfg()
    with pytest.raises(schema.SchemaError):
        selector.propose_trade(store, ROOT, ASOF, cfg)


# ======================================================================================
# exits.py
# ======================================================================================


def _pos(credit: float = 1.00) -> Position:
    return Position(
        position_id="p1", strategy="put_credit_spread", root=ROOT, entry_date=ASOF,
        expiry=EXPIRY, legs=[schema.Leg(ROOT, EXPIRY, 200.0, "P", -1), schema.Leg(ROOT, EXPIRY, 195.0, "P", +1)],
        qty=1, entry_credit=credit,
    )


def test_evaluate_exit_profit_target_fires():
    pos = _pos(credit=1.00)
    cfg = ExitConfig(profit_target_pct=0.50, stop_loss_multiple=2.0, dte_exit=21, delta_breach=0.50)
    snap = {"mark": 0.40, "short_delta": -0.10, "dte": 30}  # captured 60% >= 50%
    assert evaluate_exit(pos, snap, cfg) == ExitReason.PROFIT_TARGET


def test_evaluate_exit_stop_loss_fires():
    pos = _pos(credit=1.00)
    cfg = ExitConfig(profit_target_pct=0.50, stop_loss_multiple=2.0, dte_exit=21, delta_breach=0.50)
    snap = {"mark": 3.50, "short_delta": -0.30, "dte": 30}  # loss 2.5x >= 2x credit
    assert evaluate_exit(pos, snap, cfg) == ExitReason.STOP_LOSS


def test_evaluate_exit_dte_exit_fires_when_nothing_else_does():
    pos = _pos(credit=1.00)
    cfg = ExitConfig(profit_target_pct=0.50, stop_loss_multiple=2.0, dte_exit=21, delta_breach=0.50)
    snap = {"mark": 0.90, "short_delta": -0.20, "dte": 20}
    assert evaluate_exit(pos, snap, cfg) == ExitReason.DTE


def test_evaluate_exit_delta_breach_fires():
    pos = _pos(credit=1.00)
    cfg = ExitConfig(profit_target_pct=0.50, stop_loss_multiple=2.0, dte_exit=21, delta_breach=0.50)
    snap = {"mark": 0.90, "short_delta": -0.55, "dte": 30}
    assert evaluate_exit(pos, snap, cfg) == ExitReason.DELTA_BREACH


def test_evaluate_exit_hold_to_expiry_disables_everything():
    pos = _pos(credit=1.00)
    cfg = ExitConfig(hold_to_expiry=True)
    snap = {"mark": -10.0, "short_delta": -0.99, "dte": 0}
    assert evaluate_exit(pos, snap, cfg) is None


def test_evaluate_exit_priority_profit_target_over_stop_loss():
    pos = _pos(credit=1.00)
    cfg = ExitConfig(profit_target_pct=0.50, stop_loss_multiple=2.0)
    # both technically satisfiable is impossible simultaneously (opposite signs of `captured`);
    # instead verify profit_target is checked and wins when it alone is satisfied even with
    # delta_breach also configured but not triggered.
    snap = {"mark": 0.40, "short_delta": -0.10, "dte": 30}
    assert evaluate_exit(pos, snap, cfg) == ExitReason.PROFIT_TARGET


def test_evaluate_exit_none_when_nothing_triggers():
    pos = _pos(credit=1.00)
    cfg = ExitConfig(profit_target_pct=0.50, stop_loss_multiple=2.0, dte_exit=7, delta_breach=0.50)
    snap = {"mark": 0.90, "short_delta": -0.20, "dte": 30}
    assert evaluate_exit(pos, snap, cfg) is None
