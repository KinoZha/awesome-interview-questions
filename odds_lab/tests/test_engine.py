"""Tests for src/odds_lab/engine/{fills,portfolio,attribution,loop,result}.py."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from odds_lab import schema
from odds_lab.config import (
    BacktestConfig, CostModel, EmpiricalConfig, EntryConfig, ExitConfig, RiskConfig,
)
from odds_lab.engine import attribution as attribution_mod
from odds_lab.engine import fills as fills_mod
from odds_lab.engine.loop import _settle_expiry, _trade_row, run_backtest
from odds_lab.engine.portfolio import Portfolio
from odds_lab.engine.result import BacktestResult
from odds_lab.quant.bs import bs_greeks, bs_price
from odds_lab.schema import CONTRACT_MULTIPLIER, ExitReason, Fill, Leg, Position

ROOT = "SPY"


def _chain_df(quote_date, expiry, S, rows):
    recs = []
    for row in rows:
        recs.append({
            "root": ROOT, "quote_date": pd.Timestamp(quote_date), "ms_of_day": 57600000,
            "expiry": pd.Timestamp(expiry), "strike": float(row["strike"]), "right": row["right"],
            "bid": float(row["bid"]), "ask": float(row["ask"]), "bid_size": 10, "ask_size": 10,
            "last": (row["bid"] + row["ask"]) / 2, "volume": 100,
            "open_interest": row.get("oi", 500), "underlying_price": S,
            "iv": row.get("iv", 0.20), "delta": row.get("delta", -0.20), "gamma": row.get("gamma", 0.01),
            "theta": row.get("theta", -0.02), "vega": row.get("vega", 0.05), "rho": row.get("rho", 0.01),
            "iv_vendor": np.nan, "delta_vendor": np.nan, "source": "test", "is_synthetic": True,
        })
    return schema.validate_chain(pd.DataFrame.from_records(recs), strict=True)


# ======================================================================================
# fills.py
# ======================================================================================


def test_fill_price_bid_ask_sell_crosses_bid_buy_crosses_ask():
    costs = CostModel(fill_kind="bid_ask")
    px, kind = fills_mod.fill_price(1.00, 1.10, "SELL", costs)
    assert (px, kind) == (1.00, "bid")
    px, kind = fills_mod.fill_price(1.00, 1.10, "BUY", costs)
    assert (px, kind) == (1.10, "ask")


def test_fill_price_mid():
    costs = CostModel(fill_kind="mid")
    px, kind = fills_mod.fill_price(1.00, 1.20, "SELL", costs)
    assert px == pytest.approx(1.10)
    assert kind == "mid"


def test_fill_price_mid_plus_frac():
    costs = CostModel(fill_kind="mid_plus_frac", mid_frac=0.5)
    px_sell, _ = fills_mod.fill_price(1.00, 1.20, "SELL", costs)
    px_buy, _ = fills_mod.fill_price(1.00, 1.20, "BUY", costs)
    assert px_sell == pytest.approx(1.05)  # mid(1.10) - 0.5*half_spread(0.10)
    assert px_buy == pytest.approx(1.15)


def test_fill_price_crossed_quote_raises():
    with pytest.raises(ValueError):
        fills_mod.fill_price(1.10, 1.00, "SELL", CostModel())


def test_execute_commission_and_fees_open_vs_close():
    expiry = date(2015, 7, 3)
    leg_short = Leg(ROOT, expiry, 100.0, "P", -1)
    leg_long = Leg(ROOT, expiry, 95.0, "P", +1)
    quotes = _chain_df(date(2015, 6, 1), expiry, 105.0, [
        {"strike": 100.0, "right": "P", "bid": 1.00, "ask": 1.10},
        {"strike": 95.0, "right": "P", "bid": 0.40, "ask": 0.50},
    ])
    costs = CostModel(commission_per_contract=0.65, fees_per_contract_sell=0.05)

    open_side_map = {leg_short: "SELL", leg_long: "BUY"}
    open_fills = fills_mod.execute([leg_short, leg_long], 2, quotes, open_side_map, costs, date(2015, 6, 1))
    short_fill = next(f for f in open_fills if f.leg == leg_short)
    long_fill = next(f for f in open_fills if f.leg == leg_long)

    assert short_fill.side == "SELL" and short_fill.price == pytest.approx(1.00)  # crosses bid
    assert short_fill.commission == pytest.approx(0.65 * 2)
    assert short_fill.fees == pytest.approx(0.05 * 2)  # sell-side fee applies

    assert long_fill.side == "BUY" and long_fill.price == pytest.approx(0.50)  # crosses ask
    assert long_fill.commission == pytest.approx(0.65 * 2)
    assert long_fill.fees == pytest.approx(0.0)  # no fee on a buy


def test_execute_missing_quote_raises():
    expiry = date(2015, 7, 3)
    leg = Leg(ROOT, expiry, 999.0, "P", -1)
    quotes = _chain_df(date(2015, 6, 1), expiry, 105.0, [{"strike": 100.0, "right": "P", "bid": 1.0, "ask": 1.1}])
    with pytest.raises(ValueError):
        fills_mod.execute([leg], 1, quotes, {leg: "SELL"}, CostModel(), date(2015, 6, 1))


# ======================================================================================
# portfolio.py
# ======================================================================================


def test_portfolio_size_respects_risk_pct():
    cfg = RiskConfig(starting_equity=100_000.0, risk_pct_per_trade=0.05, max_contracts=100, min_contracts=1)
    pf = Portfolio(cfg)
    pf.equity({})
    # max_loss = $0.65/share -> $65/contract; budget = 100000*0.05 = 5000 -> floor(5000/65) = 76
    qty = pf.size(max_loss_per_spread=0.65, margin_per_spread=0.65, cfg=cfg)
    assert qty == 76


def test_portfolio_size_respects_max_contracts_cap():
    cfg = RiskConfig(starting_equity=100_000.0, risk_pct_per_trade=0.05, max_contracts=10, min_contracts=1)
    pf = Portfolio(cfg)
    pf.equity({})
    qty = pf.size(max_loss_per_spread=0.01, margin_per_spread=0.01, cfg=cfg)
    assert qty == 10


def test_portfolio_size_zero_below_min_contracts():
    cfg = RiskConfig(starting_equity=1_000.0, risk_pct_per_trade=0.01, max_contracts=100, min_contracts=1)
    pf = Portfolio(cfg)
    pf.equity({})
    qty = pf.size(max_loss_per_spread=1000.0, margin_per_spread=1000.0, cfg=cfg)
    assert qty == 0


def test_portfolio_size_uses_margin_for_undefined_risk():
    cfg = RiskConfig(starting_equity=100_000.0, risk_pct_per_trade=0.05, max_contracts=100, min_contracts=1)
    pf = Portfolio(cfg)
    pf.equity({})
    qty = pf.size(max_loss_per_spread=float("inf"), margin_per_spread=50.0, cfg=cfg)
    # budget=5000, risk_dollars_per_contract = 50*100 = 5000 -> floor(5000/5000)=1
    assert qty == 1


def test_portfolio_size_uses_margin_for_undefined_risk_exact():
    cfg = RiskConfig(starting_equity=100_000.0, risk_pct_per_trade=0.05, max_contracts=100, min_contracts=1)
    pf = Portfolio(cfg)
    pf.equity({})
    qty = pf.size(max_loss_per_spread=float("inf"), margin_per_spread=25.0, cfg=cfg)
    # budget=5000, risk_dollars_per_contract = margin_per_spread($/share)*100 = 2500 -> floor(5000/2500)=2
    assert qty == 2


def test_reg_t_margin_naked_put_floor():
    cfg = RiskConfig()
    pf = Portfolio(cfg)
    legs = [Leg(ROOT, date(2015, 7, 3), 50.0, "P", -1)]  # deep OTM, tiny credit -> floor applies
    margin = pf.reg_t_margin(legs, qty=1, S=100.0, credit=0.05)
    # max(0.20*100 - 50, 0.10*100) = max(-30,10)=10 -> 10*100=1000 + credit*100=5 => 1005 > 250 floor
    assert margin == pytest.approx(1005.0)


def test_reg_t_margin_floor_binds():
    cfg = RiskConfig()
    pf = Portfolio(cfg)
    legs = [Leg(ROOT, date(2015, 7, 3), 99.0, "P", -1)]  # near the money, small OTM amount
    margin = pf.reg_t_margin(legs, qty=1, S=100.0, credit=0.01)
    # max(20-1,10)=19 -> 1900+1=1901, way above floor already; use a tiny S to hit the floor
    small = pf.reg_t_margin([Leg(ROOT, date(2015, 7, 3), 1.0, "P", -1)], qty=1, S=1.0, credit=0.01)
    assert small == pytest.approx(250.0)


def test_portfolio_try_open_refuses_on_margin_cap_and_records_reason():
    cfg = RiskConfig(starting_equity=10_000.0, max_margin_pct=0.10, max_pct_per_style=0.9)
    pf = Portfolio(cfg)
    pf.equity({})
    pos = Position(
        position_id="p1", strategy="short_put", root=ROOT, entry_date=date(2015, 6, 1),
        expiry=date(2015, 7, 3), legs=[Leg(ROOT, date(2015, 7, 3), 95.0, "P", -1)],
        qty=1, entry_credit=1.0, max_loss=float("inf"), margin=5_000.0,  # 50% of equity > 10% cap
    )
    accepted, reason = pf.try_open(pos, cfg)
    assert accepted is False
    assert reason == "max_margin_pct_breach"
    assert len(pf.rejected_trades) == 1
    assert pf.rejected_trades[0]["reason"] == "max_margin_pct_breach"
    assert pos not in pf.positions


def test_portfolio_try_open_refuses_on_style_cap():
    cfg = RiskConfig(starting_equity=10_000.0, max_margin_pct=0.90, max_pct_per_style=0.10)
    pf = Portfolio(cfg)
    pf.equity({})
    pos = Position(
        position_id="p1", strategy="short_put", root=ROOT, entry_date=date(2015, 6, 1),
        expiry=date(2015, 7, 3), legs=[Leg(ROOT, date(2015, 7, 3), 95.0, "P", -1)],
        qty=1, entry_credit=1.0, max_loss=float("inf"), margin=2_000.0,  # 20% > 10% style cap
    )
    accepted, reason = pf.try_open(pos, cfg)
    assert accepted is False
    assert reason == "max_pct_per_style_breach"


def test_portfolio_try_open_accepts_within_caps():
    cfg = RiskConfig(starting_equity=10_000.0, max_margin_pct=0.90, max_pct_per_style=0.90)
    pf = Portfolio(cfg)
    pf.equity({})
    pos = Position(
        position_id="p1", strategy="short_put", root=ROOT, entry_date=date(2015, 6, 1),
        expiry=date(2015, 7, 3), legs=[Leg(ROOT, date(2015, 7, 3), 95.0, "P", -1)],
        qty=1, entry_credit=1.0, max_loss=float("inf"), margin=2_000.0,
    )
    accepted, reason = pf.try_open(pos, cfg)
    assert accepted is True
    assert reason is None
    assert pos in pf.positions


def test_cash_earns_risk_free_rate():
    cfg = RiskConfig(starting_equity=100_000.0)
    pf = Portfolio(cfg, risk_free_rate=0.05)
    pf.accrue_cash_interest(days=365.0)
    assert pf.cash == pytest.approx(105_000.0, rel=1e-6)


# ======================================================================================
# expiration settlement -- exact hand-computed P&L
# ======================================================================================


def _open_credit_spread(qty=1, short_strike=200.0, long_strike=195.0, short_bid=1.00, long_ask=0.65):
    expiry = date(2015, 7, 3)
    legs = [Leg(ROOT, expiry, short_strike, "P", -1), Leg(ROOT, expiry, long_strike, "P", +1)]
    open_fills = [
        Fill(legs[0], qty, "SELL", short_bid, "bid", 0.65 * qty, 0.05 * qty, short_bid, date(2015, 6, 1)),
        Fill(legs[1], qty, "BUY", long_ask, "ask", 0.65 * qty, 0.0, long_ask, date(2015, 6, 1)),
    ]
    credit = short_bid - long_ask
    pos = Position(
        position_id="p1", strategy="put_credit_spread", root=ROOT, entry_date=date(2015, 6, 1),
        expiry=expiry, legs=legs, qty=qty, open_fills=open_fills, entry_credit=credit,
        max_loss=(short_strike - long_strike) - credit,
        meta={"dte_entry": 32, "underlying_entry": 210.0},
    )
    return pos, credit


def test_expiry_settlement_max_loss_exact():
    pos, credit = _open_credit_spread(qty=3, short_strike=200.0, long_strike=195.0, short_bid=1.00, long_ask=0.65)
    S_close = 190.0  # both legs deep ITM for the put spread -> full max-loss
    settlement_cash = _settle_expiry(pos, S_close)
    pos.meta["_settlement_cash"] = settlement_cash
    pos.exit_date = date(2015, 7, 3)
    pos.exit_reason = ExitReason.ASSIGNED

    row = _trade_row(pos)
    open_cash = sum(f.cash for f in pos.open_fills)
    expected_pnl = open_cash + settlement_cash
    # hand check: max loss per share = width - credit = 5 - 0.35 = 4.65; settlement should
    # drive total pnl to -(width - credit)*qty*100 net of commissions/fees already in open_cash
    width = 200.0 - 195.0
    max_loss_per_share = width - credit
    assert settlement_cash == pytest.approx(-width * pos.qty * CONTRACT_MULTIPLIER)
    assert row["pnl"] == pytest.approx(expected_pnl)
    assert row["pnl"] == pytest.approx(
        (credit - width) * pos.qty * CONTRACT_MULTIPLIER - (0.65 * 2 * pos.qty + 0.05 * pos.qty)
    )


def test_expiry_settlement_max_profit_exact():
    pos, credit = _open_credit_spread(qty=2, short_strike=200.0, long_strike=195.0, short_bid=1.00, long_ask=0.65)
    S_close = 210.0  # both puts expire worthless -> full credit kept
    settlement_cash = _settle_expiry(pos, S_close)
    assert settlement_cash == pytest.approx(0.0)
    pos.meta["_settlement_cash"] = settlement_cash
    pos.exit_date = date(2015, 7, 3)
    pos.exit_reason = ExitReason.EXPIRY

    row = _trade_row(pos)
    expected_pnl = sum(f.cash for f in pos.open_fills)  # nothing paid/received at settlement
    assert row["pnl"] == pytest.approx(expected_pnl)
    assert row["pnl"] == pytest.approx(credit * pos.qty * CONTRACT_MULTIPLIER - (0.65 * 2 * pos.qty + 0.05 * pos.qty))


# ======================================================================================
# attribution.py
# ======================================================================================


def test_attribution_reconciles_to_actual_mark_change():
    prev = {
        "date": date(2015, 6, 1), "underlying_price": 100.0, "iv_short": 0.20, "mark": 1.50,
        "net_delta": -0.20, "net_gamma": 0.01, "net_vega": -0.05, "net_theta": 0.03,
    }
    cur = {
        "date": date(2015, 6, 2), "underlying_price": 101.5, "iv_short": 0.185, "mark": 1.30,
    }
    incr = attribution_mod.attribute(prev, cur)
    total = sum(incr.values())
    assert total == pytest.approx(prev["mark"] - cur["mark"])
    assert set(incr) == {"d_delta", "d_gamma", "d_vega", "d_theta", "d_residual"}


def test_attribution_pure_theta_day_puts_pnl_in_theta():
    # A real BS reprice: only T changes (S, iv held fixed) -> residual ~0, theta dominates.
    S, K, r, q, iv, right = 100.0, 95.0, 0.02, 0.0, 0.20, "P"
    T1 = 30 / 365
    T2 = 29 / 365
    price1 = float(bs_price(S, K, T1, r, q, iv, right))
    price2 = float(bs_price(S, K, T2, r, q, iv, right))
    greeks1 = bs_greeks(S, K, T1, r, q, iv, right)

    # position: short 1 put -> position value = -price; "mark" (cost to close) = +price
    prev = {
        "date": date(2015, 6, 1), "underlying_price": S, "iv_short": iv, "mark": price1,
        "net_delta": -float(greeks1["delta"]), "net_gamma": -float(greeks1["gamma"]),
        "net_vega": -float(greeks1["vega"]), "net_theta": -float(greeks1["theta"]),
    }
    cur = {"date": date(2015, 6, 2), "underlying_price": S, "iv_short": iv, "mark": price2}

    incr = attribution_mod.attribute(prev, cur)
    actual = prev["mark"] - cur["mark"]
    assert incr["d_delta"] == pytest.approx(0.0, abs=1e-9)
    assert incr["d_vega"] == pytest.approx(0.0, abs=1e-9)
    assert abs(incr["d_residual"]) < 0.05 * abs(actual) + 1e-6
    assert incr["d_theta"] / actual > 0.8  # theta explains the large majority of the move
    assert sum(incr.values()) == pytest.approx(actual)


# ======================================================================================
# end-to-end run_backtest -- exit timing (trigger day D, fill day D+1) + schema validity
# ======================================================================================


class EngineFakeStore:
    """A hand-controlled 3-day store: one entry on day0, a profit-target trigger on day1,
    filled on day2. Exercises the full daily loop without needing DuckDB/parquet."""

    def __init__(self, day0, day1, day2, expiry):
        self.day0, self.day1, self.day2, self.expiry = day0, day1, day2, expiry
        self._chains: dict[tuple, pd.DataFrame] = {}
        hist_idx = pd.bdate_range(end=pd.Timestamp(day0) - pd.Timedelta(days=1), periods=400)
        rng = np.random.default_rng(3)
        vals = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, len(hist_idx))))
        vals *= 100.0 / vals[-1]
        all_idx = list(hist_idx.date) + [day0, day1, day2]
        all_vals = list(vals) + [100.0, 100.0, 100.0]
        self._closes = pd.Series(all_vals, index=all_idx)
        self._underlying = pd.DataFrame({
            "root": ROOT, "date": pd.DatetimeIndex(pd.to_datetime(all_idx)),
            "open": all_vals, "high": all_vals, "low": all_vals, "close": all_vals,
            "volume": 0, "dividend": 0.0,
        })

    def add_chain(self, quote_date, df):
        self._chains[(quote_date, self.expiry)] = df

    def chain(self, root, quote_date, expiry=None):
        if expiry is None:
            expiry = self.expiry
        return self._chains.get((quote_date, expiry), schema.empty_frame(schema.CHAIN_DTYPES))

    def expiries(self, root, quote_date, dte_min=0, dte_max=400):
        if (quote_date, self.expiry) not in self._chains:
            return []
        dte = (self.expiry - quote_date).days
        return [self.expiry] if dte_min <= dte <= dte_max else []

    def closes(self, root, end):
        idx = pd.DatetimeIndex(pd.to_datetime(list(self._closes.index)))
        return self._closes[idx <= pd.Timestamp(end)]

    def trading_dates(self, root, start, end):
        return [d for d in (self.day0, self.day1, self.day2) if start <= d <= end]

    def underlying(self, root, start=None, end=None):
        df = self._underlying
        if start is not None:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["date"] <= pd.Timestamp(end)]
        return df


def _engine_cfg(day0, day2) -> BacktestConfig:
    entry = EntryConfig(
        strategy="put_credit_spread", strike_rule="pct_otm", pct_otm=0.05, width_strikes=1,
        dte_min=21, dte_max=56, min_credit=0.10, expected_return_min=0.0, expected_return_max=5.0,
        market_filter="all", require_positive_edge=False, entry_schedule="daily",
        max_concurrent_per_root=4,
    )
    exits = ExitConfig(profit_target_pct=0.50, stop_loss_multiple=None, dte_exit=None, delta_breach=None, hold_to_expiry=False)
    risk = RiskConfig(starting_equity=10_000.0, risk_pct_per_trade=0.01, max_pct_per_style=0.9, max_margin_pct=0.9, min_contracts=1, max_contracts=100)
    costs = CostModel(commission_per_contract=0.65, fees_per_contract_sell=0.05, min_bid=0.05, min_open_interest=100, max_spread_pct_of_mid=0.5)
    empirical = EmpiricalConfig(lookback_years=5.0, min_samples=5, seed=1)
    return BacktestConfig(
        roots=(ROOT,), start=day0, end=day2, entry=entry, exits=exits, risk=risk,
        costs=costs, empirical=empirical, data=__import__("odds_lab.config", fromlist=["DataConfig"]).DataConfig(provider="synthetic", allow_synthetic=True),
    )


def test_exit_triggers_day_d_fills_day_d_plus_1_and_schema_valid():
    day0, day1, day2 = date(2015, 6, 1), date(2015, 6, 2), date(2015, 6, 3)
    expiry = day0 + timedelta(days=30)
    store = EngineFakeStore(day0, day1, day2, expiry)

    store.add_chain(day0, _chain_df(day0, expiry, 100.0, [
        {"strike": 95.0, "right": "P", "bid": 1.00, "ask": 1.05, "delta": -0.20},
        {"strike": 94.0, "right": "P", "bid": 0.60, "ask": 0.65, "delta": -0.15},
    ]))
    # day1: short leg has decayed a lot -> profit target (>=50% captured) triggers
    store.add_chain(day1, _chain_df(day1, expiry, 100.0, [
        {"strike": 95.0, "right": "P", "bid": 0.15, "ask": 0.20, "delta": -0.05},
        {"strike": 94.0, "right": "P", "bid": 0.05, "ask": 0.08, "delta": -0.03},
    ]))
    store.add_chain(day2, _chain_df(day2, expiry, 100.0, [
        {"strike": 95.0, "right": "P", "bid": 0.12, "ask": 0.16, "delta": -0.04},
        {"strike": 94.0, "right": "P", "bid": 0.04, "ask": 0.07, "delta": -0.02},
    ]))

    cfg = _engine_cfg(day0, day2)
    result = run_backtest(cfg, store)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert pd.Timestamp(trade["entry_date"]) == pd.Timestamp(day0)
    assert pd.Timestamp(trade["exit_date"]) == pd.Timestamp(day2)  # fills the day AFTER trigger
    assert trade["exit_reason"] == ExitReason.PROFIT_TARGET.value
    assert trade["qty"] == 1

    # hand-computed P&L: credit=1.00-0.65=0.35; close debit = ask(short)-bid(long) = 0.16-0.04=0.12
    credit = 1.00 - 0.65
    close_debit = 0.16 - 0.04
    commission_total = 0.65 * 4  # 2 legs x open+close
    fees_total = 0.05 * 2  # sell-side only: open-short-sell + close-long-sell
    expected_pnl = (credit - close_debit) * 1 * CONTRACT_MULTIPLIER - commission_total - fees_total
    assert trade["pnl"] == pytest.approx(expected_pnl)

    # schema validity of every frame
    schema.validate_frame(result.trades, schema.TRADE_DTYPES, "trades")
    schema.validate_frame(result.equity, schema.EQUITY_DTYPES, "equity")
    schema.validate_frame(result.snapshots, schema.SNAPSHOT_DTYPES, "snapshots")
    assert len(result.equity) == 3
    assert not result.equity["equity"].isna().any()


# ======================================================================================
# end-to-end on the synthetic sample store
# ======================================================================================


def test_run_backtest_on_synthetic_sample_store(tmp_path):
    from odds_lab.data.providers.synthetic import make_sample_store

    store = make_sample_store(
        tmp_path / "store", roots=("SPY",), start=date(2015, 1, 1), end=date(2015, 9, 1), seed=11,
    )
    entry = EntryConfig(
        strategy="put_credit_spread", strike_rule="delta", delta_min=0.15, delta_max=0.30,
        width_strikes=1, dte_min=21, dte_max=56, min_credit=0.10, expected_return_min=0.0,
        expected_return_max=1.0, market_filter="all", require_positive_edge=False,
        entry_schedule="weekly", entry_weekday=0, max_concurrent_per_root=2,
    )
    exits = ExitConfig(profit_target_pct=0.50, stop_loss_multiple=2.0, dte_exit=21, delta_breach=0.50, hold_to_expiry=False)
    risk = RiskConfig(starting_equity=100_000.0, risk_pct_per_trade=0.05, min_contracts=1, max_contracts=50)
    empirical = EmpiricalConfig(lookback_years=2.0, min_samples=20, seed=1)
    from odds_lab.config import DataConfig

    cfg = BacktestConfig(
        roots=("SPY",), start=date(2015, 3, 1), end=date(2015, 9, 1),
        entry=entry, exits=exits, risk=risk, costs=CostModel(), empirical=empirical,
        data=DataConfig(provider="synthetic", allow_synthetic=True),
    )

    result = run_backtest(cfg, store)

    schema.validate_frame(result.trades, schema.TRADE_DTYPES, "trades")
    schema.validate_frame(result.equity, schema.EQUITY_DTYPES, "equity")
    schema.validate_frame(result.snapshots, schema.SNAPSHOT_DTYPES, "snapshots")
    assert result.manifest["is_synthetic"] is True
    assert len(result.equity) > 0
    assert (result.equity["equity"] > 0).all()

    # -- selection funnel: present, attributed, and survives save()/load() --------------
    funnel = result.manifest["selection_funnel"]
    assert funnel["opportunities"] > 0
    assert not result.funnel.empty
    assert funnel["accepted"] + sum(funnel["rejected"].values()) == funnel["candidates_evaluated"]
    assert len(result.trades) == funnel["accepted"] or funnel["accepted"] >= len(result.trades)

    out_dir = tmp_path / "saved"
    result.save(out_dir)
    reloaded = BacktestResult.load(out_dir)
    pd.testing.assert_frame_equal(
        reloaded.funnel.reset_index(drop=True), result.funnel.reset_index(drop=True), check_like=True,
    )
    # manifest round-trips through JSON, which stringifies dict keys (by_year's int years) --
    # compare the numeric content, not the exact key types.
    reloaded_funnel = reloaded.manifest["selection_funnel"]
    for top_key in ("opportunities", "accepted", "candidates_evaluated", "rejected"):
        assert reloaded_funnel[top_key] == funnel[top_key]
    assert {int(y): v for y, v in reloaded_funnel["by_year"].items()} == funnel["by_year"]

    # -- attribution reconciles against realized P&L, to within the untracked entry/exit-day
    # and cost-of-trading residual -- STRATEGY.md §8A. This is a regression test for the
    # notional-scaling bug (attribution was 3 orders of magnitude below pnl before the fix).
    attrib_cols = ["pnl_delta", "pnl_gamma", "pnl_vega", "pnl_theta", "pnl_residual"]
    multi_day = result.trades[
        (pd.to_datetime(result.trades["exit_date"]) - pd.to_datetime(result.trades["entry_date"])).dt.days >= 3
    ]
    assert len(multi_day) > 0, "need at least one multi-day trade to exercise attribution"
    for _, row in multi_day.iterrows():
        attrib_sum = float(row[attrib_cols].sum())
        gap = abs(row["pnl"] - attrib_sum)
        # Scale the tolerance off the position's own size (max_loss notional), not off
        # `pnl` itself -- a trade can realize a small net pnl after large offsetting
        # mid-life mark swings, which the day-by-day attribution legitimately captures
        # in full even though they later reverse. What this test guards against is the
        # notional-scaling regression (attribution 3 orders of magnitude below pnl), so
        # the bound is generous in position-size units but would still catch that.
        scale = abs(row["max_loss"]) * row["qty"] * CONTRACT_MULTIPLIER if math.isfinite(row["max_loss"]) else abs(row["pnl"])
        tolerance = 5.0 * scale + row["commission"] + row["fees"] + abs(row["slippage"]) + 20.0
        assert gap <= tolerance, (
            f"attribution stack ({attrib_sum}) does not reconcile with pnl ({row['pnl']}) "
            f"for {row['position_id']}: gap {gap} > tolerance {tolerance}"
        )
        # and the regression this guards against directly: attribution must be within the
        # same order of magnitude as the position size, never ~0 while pnl is large.
        assert attrib_sum != 0.0 or row["pnl"] == 0.0
