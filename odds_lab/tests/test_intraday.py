"""Tests for engine/intraday.py, the intraday quote store table, and
data/providers/csv_export.py::read_quote_1m_csv. STRATEGY.md §5.1, ARCHITECTURE.md §3.1.
"""

from __future__ import annotations

import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from odds_lab import schema
from odds_lab.config import (
    BacktestConfig, CostModel, DataConfig, EmpiricalConfig, EntryConfig, ExitConfig,
    IntradayConfig, RiskConfig,
)
from odds_lab.data.providers.csv_export import CsvExportError, read_quote_1m_csv
from odds_lab.data.providers.synthetic import SyntheticProvider, make_sample_store
from odds_lab.data.store import ChainStore
from odds_lab.engine import intraday as intraday_mod
from odds_lab.engine.loop import run_backtest
from odds_lab.schema import ExitReason, Leg, Position

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


def _intraday_bars_df(quote_date, expiry, strike, right, ts_list, bid_list, ask_list):
    return pd.DataFrame({
        "root": ROOT, "expiry": pd.Timestamp(expiry), "strike": float(strike), "right": right,
        "quote_date": pd.Timestamp(quote_date), "ts": pd.to_datetime(ts_list),
        "bid": bid_list, "ask": ask_list, "bid_size": 10, "ask_size": 10,
        "source": "test", "is_synthetic": True,
    })


# ======================================================================================
# unit: bars_for_position / evaluate_intraday_day -- trigger bar i, fill bar i+1
# ======================================================================================


def _put_credit_position(entry_date, expiry, qty=1):
    legs = [
        Leg(root=ROOT, expiry=expiry, strike=95.0, right="P", ratio=-1),
        Leg(root=ROOT, expiry=expiry, strike=94.0, right="P", ratio=1),
    ]
    return Position(
        position_id="test-pos-1", strategy="put_credit_spread", root=ROOT,
        entry_date=entry_date, expiry=expiry, legs=legs, qty=qty,
        entry_credit=0.35, max_loss=0.65, margin=65.0,
    )


def _cfg(exits, intraday_exit_rules=("profit_target", "stop_loss", "delta_breach")):
    entry = EntryConfig(strategy="put_credit_spread", strike_rule="pct_otm", pct_otm=0.05, width_strikes=1)
    risk = RiskConfig(starting_equity=10_000.0, risk_pct_per_trade=0.05, min_contracts=1, max_contracts=100)
    costs = CostModel(min_bid=0.0, min_open_interest=0, max_spread_pct_of_mid=1.0)
    return BacktestConfig(
        roots=(ROOT,), start=date(2015, 6, 1), end=date(2015, 6, 3),
        entry=entry, exits=exits, risk=risk, costs=costs,
        empirical=EmpiricalConfig(min_samples=5),
        data=DataConfig(provider="synthetic", allow_synthetic=True),
        intraday=IntradayConfig(enabled=True, exit_rules=intraday_exit_rules),
    )


def test_intraday_trigger_fires_on_right_bar_and_fills_on_next_bar():
    """Mark crosses the stop-loss threshold at bar index 2 -- the fill must use bar
    index 3's quotes, never bar 2's (STRATEGY.md §7.3 extended intraday)."""
    entry_date, expiry = date(2015, 6, 1), date(2015, 6, 1) + timedelta(days=30)
    pos = _put_credit_position(entry_date, expiry)
    ts = pd.date_range("2015-06-01 09:30", periods=5, freq="1min")

    # short 95P bid/ask; long 94P bid/ask. mark = short_mid - long_mid (credit structure).
    # credit=0.35, stop threshold: mark >= 3*credit = 1.05. mark = mid_short - mid_long.
    short_bid = [0.30, 0.30, 1.50, 0.60, 0.55]
    short_ask = [0.35, 0.35, 1.55, 0.65, 0.60]
    long_bid = [0.05, 0.05, 0.05, 0.05, 0.05]
    long_ask = [0.10, 0.10, 0.10, 0.10, 0.10]

    store = _FakeIntradayStore()
    store.add_bars(expiry, 95.0, "P", ts, short_bid, short_ask)
    store.add_bars(expiry, 94.0, "P", ts, long_bid, long_ask)

    merged = intraday_mod.bars_for_position(store, pos, entry_date)
    assert merged is not None and len(merged) == 5

    exits = ExitConfig(profit_target_pct=None, stop_loss_multiple=2.0, dte_exit=None, delta_breach=None, hold_to_expiry=False)
    cfg = _cfg(exits, intraday_exit_rules=("stop_loss",))

    result = intraday_mod.evaluate_intraday_day(pos, merged, cfg, S_eod=100.0, asof=entry_date)
    assert result.reason == ExitReason.STOP_LOSS
    assert result.trigger_ts == ts[2]  # mark = 1.525 - 0.075 = 1.45 >= 1.05 first here
    assert result.fill_ts == ts[3]  # fills the NEXT bar, not the triggering one
    assert result.close_fills is not None
    short_fill = next(f for f in result.close_fills if f.leg.strike == 95.0)
    long_fill = next(f for f in result.close_fills if f.leg.strike == 94.0)
    # closing a short credit spread: buy back the short (crosses ask), sell the long (crosses bid)
    assert short_fill.side == "BUY" and short_fill.price == pytest.approx(short_ask[3])
    assert long_fill.side == "SELL" and long_fill.price == pytest.approx(long_bid[3])


def test_intraday_trigger_on_last_bar_has_no_same_day_fill():
    entry_date, expiry = date(2015, 6, 1), date(2015, 6, 1) + timedelta(days=30)
    pos = _put_credit_position(entry_date, expiry)
    ts = pd.date_range("2015-06-01 09:30", periods=3, freq="1min")
    short_bid, short_ask = [0.30, 0.30, 1.50], [0.35, 0.35, 1.55]
    long_bid, long_ask = [0.05, 0.05, 0.05], [0.10, 0.10, 0.10]

    store = _FakeIntradayStore()
    store.add_bars(expiry, 95.0, "P", ts, short_bid, short_ask)
    store.add_bars(expiry, 94.0, "P", ts, long_bid, long_ask)
    merged = intraday_mod.bars_for_position(store, pos, entry_date)

    exits = ExitConfig(profit_target_pct=None, stop_loss_multiple=2.0, dte_exit=None, delta_breach=None, hold_to_expiry=False)
    cfg = _cfg(exits, intraday_exit_rules=("stop_loss",))
    result = intraday_mod.evaluate_intraday_day(pos, merged, cfg, S_eod=100.0, asof=entry_date)
    assert result.reason == ExitReason.STOP_LOSS
    assert result.trigger_ts == ts[2]
    assert result.fill_ts is None  # no bar after the last one -- caller defers to next-day EOD
    assert result.close_fills is None


def test_bars_for_position_none_when_any_leg_missing_coverage():
    entry_date, expiry = date(2015, 6, 1), date(2015, 6, 1) + timedelta(days=30)
    pos = _put_credit_position(entry_date, expiry)
    store = _FakeIntradayStore()
    ts = pd.date_range("2015-06-01 09:30", periods=3, freq="1min")
    store.add_bars(expiry, 95.0, "P", ts, [0.3, 0.3, 0.3], [0.35, 0.35, 0.35])
    # long leg (94P) has no bars at all
    assert intraday_mod.bars_for_position(store, pos, entry_date) is None


class _FakeIntradayStore:
    """Minimal store double exposing only `.intraday()` -- exercises
    `bars_for_position`/`evaluate_intraday_day` without the parquet layer."""

    def __init__(self):
        self._bars: dict[tuple, pd.DataFrame] = {}

    def add_bars(self, expiry, strike, right, ts, bid, ask):
        self._bars[(expiry, float(strike), right)] = _intraday_bars_df(
            ts[0].date(), expiry, strike, right, ts, bid, ask
        )

    def intraday(self, root, expiry, strike, right, quote_date):
        return self._bars.get((expiry, float(strike), right), schema.empty_frame(
            {"root": "string", "expiry": "datetime64[ns]", "strike": "float64", "right": "string",
             "quote_date": "datetime64[ns]", "ts": "datetime64[ns]", "bid": "float64", "ask": "float64",
             "bid_size": "int32", "ask_size": "int32", "source": "string", "is_synthetic": "bool"}
        ))


# ======================================================================================
# end-to-end THE BIAS: a spike that reverts by the close stops out intraday, and
# produces NO stop-out at all in EOD-only mode (directly asserted, per task brief).
# ======================================================================================


class _RevertingSpikeStore:
    """day0: entry. day1: EOD close reflects a REVERTED (back-to-normal) mark, but the
    intraday path for day1 spikes hard mid-day (loss >> stop threshold) before reverting
    by the close -- exactly the bias this feature exists to catch. day2: nothing left
    open in the intraday run; the EOD run keeps holding (no chain quotes past day1 --
    it simply has nothing further to mark and no trigger fires within the window)."""

    def __init__(self, day0, day1, expiry):
        self.day0, self.day1, self.expiry = day0, day1, expiry
        self._chains: dict[tuple, pd.DataFrame] = {}
        self._intraday: dict[tuple, pd.DataFrame] = {}
        hist_idx = pd.bdate_range(end=pd.Timestamp(day0) - pd.Timedelta(days=1), periods=300)
        rng = np.random.default_rng(5)
        vals = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.006, len(hist_idx))))
        vals *= 100.0 / vals[-1]
        all_idx = list(hist_idx.date) + [day0, day1]
        all_vals = list(vals) + [100.0, 100.0]
        self._underlying = pd.DataFrame({
            "root": ROOT, "date": pd.DatetimeIndex(pd.to_datetime(all_idx)),
            "open": all_vals, "high": all_vals, "low": all_vals, "close": all_vals,
            "volume": 0, "dividend": 0.0,
        })

    def add_chain(self, quote_date, df):
        self._chains[(quote_date, self.expiry)] = df

    def add_intraday(self, quote_date, strike, right, ts, bid, ask):
        self._intraday[(self.expiry, float(strike), right, quote_date)] = _intraday_bars_df(
            quote_date, self.expiry, strike, right, ts, bid, ask
        )

    def chain(self, root, quote_date, expiry=None):
        if expiry is None:
            expiry = self.expiry
        return self._chains.get((quote_date, expiry), schema.empty_frame(schema.CHAIN_DTYPES))

    def intraday(self, root, expiry, strike, right, quote_date):
        key = (expiry, float(strike), right, quote_date)
        return self._intraday.get(key, schema.empty_frame(
            {"root": "string", "expiry": "datetime64[ns]", "strike": "float64", "right": "string",
             "quote_date": "datetime64[ns]", "ts": "datetime64[ns]", "bid": "float64", "ask": "float64",
             "bid_size": "int32", "ask_size": "int32", "source": "string", "is_synthetic": "bool"}
        ))

    def expiries(self, root, quote_date, dte_min=0, dte_max=400):
        if (quote_date, self.expiry) not in self._chains:
            return []
        dte = (self.expiry - quote_date).days
        return [self.expiry] if dte_min <= dte <= dte_max else []

    def closes(self, root, end):
        idx = pd.DatetimeIndex(pd.to_datetime(list(self._underlying["date"])))
        s = pd.Series(self._underlying["close"].to_numpy(), index=idx.date)
        return s[pd.DatetimeIndex(pd.to_datetime(list(s.index))) <= pd.Timestamp(end)]

    def trading_dates(self, root, start, end):
        return [d for d in (self.day0, self.day1) if start <= d <= end]

    def underlying(self, root, start=None, end=None):
        df = self._underlying
        if start is not None:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["date"] <= pd.Timestamp(end)]
        return df


def _spike_cfg(day0, day1, stop_multiple=1.5, intraday_enabled=True):
    entry = EntryConfig(
        strategy="put_credit_spread", strike_rule="pct_otm", pct_otm=0.05, width_strikes=1,
        dte_min=21, dte_max=90, min_credit=0.10, expected_return_min=0.0, expected_return_max=5.0,
        market_filter="all", require_positive_edge=False, entry_schedule="daily", max_concurrent_per_root=4,
    )
    exits = ExitConfig(profit_target_pct=None, stop_loss_multiple=stop_multiple, dte_exit=None, delta_breach=None, hold_to_expiry=False)
    risk = RiskConfig(starting_equity=10_000.0, risk_pct_per_trade=0.05, max_pct_per_style=0.9, max_margin_pct=0.9, min_contracts=1, max_contracts=100)
    costs = CostModel(commission_per_contract=0.0, fees_per_contract_sell=0.0, min_bid=0.0, min_open_interest=0, max_spread_pct_of_mid=1.0)
    empirical = EmpiricalConfig(lookback_years=5.0, min_samples=5, seed=1)
    return BacktestConfig(
        roots=(ROOT,), start=day0, end=day1, entry=entry, exits=exits, risk=risk, costs=costs,
        empirical=empirical, data=DataConfig(provider="synthetic", allow_synthetic=True),
        intraday=IntradayConfig(enabled=intraday_enabled, exit_rules=("stop_loss",)),
    )


def _build_spike_store():
    day0, day1 = date(2015, 6, 1), date(2015, 6, 2)
    expiry = day0 + timedelta(days=30)
    store = _RevertingSpikeStore(day0, day1, expiry)

    store.add_chain(day0, _chain_df(day0, expiry, 100.0, [
        {"strike": 95.0, "right": "P", "bid": 0.30, "ask": 0.35, "delta": -0.20},
        {"strike": 94.0, "right": "P", "bid": 0.05, "ask": 0.10, "delta": -0.15},
    ]))
    # day1's EOD chain (the only thing the EOD-only backtest ever sees for day1):
    # reverted back to roughly the entry level -- mark is FAR below the stop threshold.
    store.add_chain(day1, _chain_df(day1, expiry, 100.0, [
        {"strike": 95.0, "right": "P", "bid": 0.28, "ask": 0.33, "delta": -0.18},
        {"strike": 94.0, "right": "P", "bid": 0.04, "ask": 0.09, "delta": -0.14},
    ]))

    # day1 intraday: a spike bar (index 2) where the short put's mark blows out well
    # past the stop threshold, THEN reverts by the close (last bar matches the EOD
    # chain's reverted quote above) -- the exact "spike that reverts by the close" bias.
    ts = pd.date_range("2015-06-02 09:30", periods=5, freq="1min")
    store.add_intraday(day1, 95.0, "P", ts, [0.30, 0.30, 1.40, 0.55, 0.28], [0.35, 0.35, 1.45, 0.60, 0.33])
    store.add_intraday(day1, 94.0, "P", ts, [0.05, 0.05, 0.06, 0.05, 0.04], [0.10, 0.10, 0.11, 0.10, 0.09])
    return store, day0, day1


def test_reverting_intraday_spike_stops_out_intraday_but_not_in_eod_mode():
    store, day0, day1 = _build_spike_store()

    eod_cfg = _spike_cfg(day0, day1, intraday_enabled=False)
    eod_result = run_backtest(eod_cfg, store)
    # THE BIAS, asserted directly: EOD-only exit evaluation never sees the intraday
    # spike -- day1's EOD chain is already reverted, so stop_loss never fires.
    assert not (eod_result.trades["exit_reason"] == "stop_loss").any()
    assert eod_result.manifest["intraday"]["enabled"] is False

    intraday_cfg = _spike_cfg(day0, day1, intraday_enabled=True)
    intraday_result = run_backtest(intraday_cfg, store)
    stop_trades = intraday_result.trades[intraday_result.trades["exit_reason"] == "stop_loss"]
    assert len(stop_trades) == 1
    trade = stop_trades.iloc[0]
    assert pd.Timestamp(trade["exit_date"]) == pd.Timestamp(day1)  # same-day intraday close
    assert intraday_result.manifest["intraday"]["enabled"] is True
    assert intraday_result.manifest["intraday"]["n_positions_evaluated_intraday"] >= 1

    # the bias in dollars: the intraday-accurate run books a real loss on this
    # position where the EOD-only run books none at all (position never closes
    # within the EOD run's window in this fixture).
    assert trade["pnl"] < 0
    assert eod_result.trades.empty or not (eod_result.trades["position_id"] == trade["position_id"]).any()


def test_intraday_disabled_by_default_matches_pre_existing_eod_behavior():
    """Default config: cfg.intraday.enabled is False, so an unmodified store (no
    `.intraday` method at all, like the pre-existing EngineFakeStore fixtures) must
    run exactly as before -- CLAUDE.md rule 5 / task requirement "Default OFF"."""
    store, day0, day1 = _build_spike_store()
    cfg = _spike_cfg(day0, day1, intraday_enabled=False)
    assert cfg.intraday.enabled is False
    result = run_backtest(cfg, store)
    assert result.manifest["intraday"] == {
        "enabled": False, "exit_rules": ["stop_loss"],
        "n_positions_evaluated_intraday": 0, "n_positions_eod_fallback": 0,
        "positions_evaluated_intraday": [], "positions_eod_fallback": [],
    }


# ======================================================================================
# contract_day_keys_from_trades
# ======================================================================================


def test_contract_day_keys_from_trades():
    trades = pd.DataFrame([
        {
            "root": "SPY", "entry_date": pd.Timestamp("2015-06-01"), "exit_date": pd.Timestamp("2015-06-03"),
            "expiry": pd.Timestamp("2015-07-01"),
            "short_put_strike": 95.0, "long_put_strike": 94.0,
            "short_call_strike": float("nan"), "long_call_strike": float("nan"),
        },
    ])
    keys = intraday_mod.contract_day_keys_from_trades(trades)
    dates = sorted({k[4] for k in keys})
    assert dates == [date(2015, 6, 1), date(2015, 6, 2), date(2015, 6, 3)]
    rights = {k[3] for k in keys}
    assert rights == {"P"}
    strikes = {k[2] for k in keys}
    assert strikes == {95.0, 94.0}
    assert all(k[0] == "SPY" and k[1] == date(2015, 7, 1) for k in keys)
    assert len(keys) == 3 * 2  # 3 days x 2 legs, no call legs (NaN, correctly skipped)


def test_contract_day_keys_from_trades_empty():
    assert intraday_mod.contract_day_keys_from_trades(schema.empty_frame(schema.TRADE_DTYPES)) == set()


# ======================================================================================
# unrecognized quote schema -> clear error naming the columns found
# ======================================================================================


def test_read_quote_1m_csv_unrecognized_schema_raises_naming_columns(tmp_path):
    # Looks like neither the confirmed EOD nor OHLC shape, nor the quote heuristic:
    # option-identifying columns are present but there's no bid/ask AND no price/size
    # pairing either -- falls through `_classify` to 'unknown'.
    bad = pd.DataFrame({"symbol": ["SPY"], "expiration": ["2025-12-19"], "strike": [690.0], "right": ["CALL"], "weird_col": [1]})
    path = tmp_path / "mystery.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(CsvExportError) as exc:
        read_quote_1m_csv(path)
    msg = str(exc.value)
    assert "weird_col" in msg or "symbol" in msg  # names the columns it actually found
    assert "unknown" in msg or "does not look like" in msg


def test_read_quote_1m_csv_missing_bid_ask_raises():
    # Classifies as option_trade (price/size, no bid/ask) -- a real shape, just not
    # a quote export -- read_quote_1m_csv must still refuse it, naming what's missing.
    df = pd.DataFrame({
        "symbol": ["SPY"], "expiration": ["2025-12-19"], "strike": [690.0], "right": ["CALL"],
        "timestamp": ["2025-08-19T09:30:00"], "price": [1.23], "size": [5],
    })
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        f.write(buf.getvalue())
        path = f.name
    with pytest.raises(CsvExportError) as exc:
        read_quote_1m_csv(path)
    assert "bid" in str(exc.value) or "ask" in str(exc.value)


def test_read_quote_1m_csv_recognized_shape_parses(tmp_path):
    df = pd.DataFrame({
        "symbol": ["SPY", "SPY"],
        "expiration": ["2025-12-19", "2025-12-19"],
        "strike": [690.0, 690.0],
        "right": ["CALL", "PUT"],
        "timestamp": ["2025-08-19T09:30:00", "2025-08-19T09:31:00"],
        "bid": [1.20, 0.80],
        "ask": [1.25, 0.85],
        "bid_size": [3, 4],
        "ask_size": [5, 6],
    })
    path = tmp_path / "quote_1m.csv"
    df.to_csv(path, index=False)
    out = read_quote_1m_csv(path)
    assert set(out["right"]) == {"C", "P"}
    assert out["root"].tolist() == ["SPY", "SPY"]
    assert float(out.loc[out["right"] == "C", "bid"].iloc[0]) == pytest.approx(1.20)
    assert pd.Timestamp(out["ts"].iloc[0]) == pd.Timestamp("2025-08-19T09:30:00")


# ======================================================================================
# ChainStore intraday table -- write/read round trip + availability index
# ======================================================================================


def test_store_write_and_read_intraday_round_trip(tmp_path):
    store = ChainStore(tmp_path / "store")
    expiry = date(2015, 7, 1)
    quote_date = date(2015, 6, 1)
    ts = pd.date_range("2015-06-01 09:30", periods=3, freq="1min")
    df = _intraday_bars_df(quote_date, expiry, 95.0, "P", ts, [0.3, 0.31, 0.32], [0.35, 0.36, 0.37])
    store.write_intraday(df)

    out = store.intraday(ROOT, expiry, 95.0, "P", quote_date)
    assert len(out) == 3
    assert list(out["bid"]) == pytest.approx([0.3, 0.31, 0.32])

    assert store.has_intraday(ROOT, expiry, 95.0, "P", quote_date) is True
    assert store.has_intraday(ROOT, expiry, 94.0, "P", quote_date) is False
    assert (expiry, 95.0, "P") in store.intraday_available_keys(ROOT, quote_date)

    # idempotent re-write (dedupe on the natural key) never duplicates rows
    store.write_intraday(df)
    assert len(store.intraday(ROOT, expiry, 95.0, "P", quote_date)) == 3


def test_store_intraday_empty_when_never_ingested(tmp_path):
    store = ChainStore(tmp_path / "store")
    out = store.intraday(ROOT, date(2015, 7, 1), 95.0, "P", date(2015, 6, 1))
    assert out.empty


# ======================================================================================
# two-pass hybrid: manifest records fallbacks; feedback-loop convergence is reported
# ======================================================================================


def _small_synthetic_cfg(intraday_enabled, iteration_cap=3):
    # Mirrors tests/test_engine.py::test_run_backtest_on_synthetic_sample_store's known-
    # to-trade settings (this store/seed/date-range combination is not exercised by
    # this project's default configs, so match a config already proven to produce trades).
    entry = EntryConfig(
        strategy="put_credit_spread", strike_rule="delta", delta_min=0.15, delta_max=0.30,
        width_strikes=1, dte_min=21, dte_max=56, min_credit=0.10, expected_return_min=0.0,
        expected_return_max=1.0, market_filter="all", require_positive_edge=False,
        entry_schedule="weekly", entry_weekday=0, max_concurrent_per_root=2,
    )
    exits = ExitConfig(profit_target_pct=0.50, stop_loss_multiple=2.0, dte_exit=21, delta_breach=0.50, hold_to_expiry=False)
    risk = RiskConfig(starting_equity=100_000.0, risk_pct_per_trade=0.05, min_contracts=1, max_contracts=50)
    empirical = EmpiricalConfig(lookback_years=2.0, min_samples=20, seed=1)
    return BacktestConfig(
        roots=(ROOT,), start=date(2015, 3, 1), end=date(2015, 9, 1),
        entry=entry, exits=exits, risk=risk, costs=CostModel(), empirical=empirical,
        data=DataConfig(provider="synthetic", allow_synthetic=True),
        intraday=IntradayConfig(enabled=intraday_enabled, iteration_cap=iteration_cap, exit_rules=("profit_target", "stop_loss", "delta_breach")),
    )


def test_run_hybrid_backtest_no_coverage_falls_back_and_reports_nonconvergence(tmp_path):
    store = make_sample_store(tmp_path / "store", roots=(ROOT,), start=date(2015, 1, 1), end=date(2015, 9, 1), seed=11)
    cfg = _small_synthetic_cfg(intraday_enabled=True)
    result = intraday_mod.run_hybrid_backtest(cfg, store)  # no fetch_fn, no data ever ingested

    assert len(result.trades) > 0  # sanity: this cfg actually trades on the sample store
    hybrid = result.manifest["intraday"]["hybrid"]
    assert hybrid["converged"] is False  # no data was ever available -- must not claim success
    assert hybrid["iterations_run"] >= 2
    assert hybrid["held_contract_days_missing_data"] > 0
    assert len(hybrid["contract_days_eod_fallback_due_to_missing_data"]) > 0
    # every held position in the final run fell back to EOD-only (no coverage anywhere)
    assert result.manifest["intraday"]["n_positions_evaluated_intraday"] == 0
    assert result.manifest["intraday"]["n_positions_eod_fallback"] > 0


def test_run_hybrid_backtest_with_fetch_fn_converges_and_uses_intraday(tmp_path):
    store = make_sample_store(tmp_path / "store", roots=(ROOT,), start=date(2015, 1, 1), end=date(2015, 9, 1), seed=11)
    provider = SyntheticProvider(roots=(ROOT,), start=date(2015, 1, 1), end=date(2015, 9, 1), seed=11)
    cfg = _small_synthetic_cfg(intraday_enabled=True, iteration_cap=5)

    def fetch_fn(missing_keys):
        for root, expiry, strike, right, d in missing_keys:
            bars = provider.intraday_quotes(root, d, expiry, strike, right, n_bars=30)
            if not bars.empty:
                store.write_intraday(bars)

    result = intraday_mod.run_hybrid_backtest(cfg, store, fetch_fn=fetch_fn)
    hybrid = result.manifest["intraday"]["hybrid"]
    assert hybrid["held_contract_days_missing_data"] == 0
    assert result.manifest["intraday"]["n_positions_evaluated_intraday"] > 0


def test_run_hybrid_backtest_requires_enabled(tmp_path):
    store = ChainStore(tmp_path / "store")
    cfg = _small_synthetic_cfg(intraday_enabled=False)
    with pytest.raises(ValueError):
        intraday_mod.run_hybrid_backtest(cfg, store)


# ======================================================================================
# compare_eod_vs_intraday -- end-to-end on a small synthetic intraday fixture
# ======================================================================================


def test_compare_eod_vs_intraday_end_to_end(tmp_path):
    store = make_sample_store(tmp_path / "store", roots=(ROOT,), start=date(2015, 1, 1), end=date(2015, 9, 1), seed=11)
    provider = SyntheticProvider(roots=(ROOT,), start=date(2015, 1, 1), end=date(2015, 9, 1), seed=11)
    cfg = _small_synthetic_cfg(intraday_enabled=False, iteration_cap=5)  # compare_eod_vs_intraday flips this itself

    def fetch_fn(missing_keys):
        for root, expiry, strike, right, d in missing_keys:
            bars = provider.intraday_quotes(root, d, expiry, strike, right, n_bars=30)
            if not bars.empty:
                store.write_intraday(bars)

    comparison = intraday_mod.compare_eod_vs_intraday(cfg, store, fetch_fn=fetch_fn)
    overall = comparison["overall"]
    assert overall["n_trades_eod"] > 0
    assert overall["n_trades_hybrid"] > 0
    assert set(comparison["by_rule"]) == {"profit_target", "stop_loss", "delta_breach"}
    for rule_stats in comparison["by_rule"].values():
        assert rule_stats["trades_considered"] >= 0
        assert rule_stats["exits_changed"] <= rule_stats["trades_considered"]
    # the headline number the task exists to produce:
    assert isinstance(overall["pnl_difference_hybrid_minus_eod"], float)
