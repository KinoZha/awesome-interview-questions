"""Tests for engine/stress.py -- STRATEGY.md §10."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from odds_lab.config import BacktestConfig, EmpiricalConfig, EntryConfig, ExitConfig, RiskConfig, CostModel, DataConfig
from odds_lab.engine import stress
from odds_lab.quant.bs import bs_price
from odds_lab.schema import CONTRACT_MULTIPLIER


# ======================================================================================
# helpers
# ======================================================================================


def _write_history_csv(tmp_path, closes: pd.Series, name="history.csv"):
    path = tmp_path / name
    pd.DataFrame({"date": closes.index, "close": closes.to_numpy()}).to_csv(path, index=False)
    return path


def _synthetic_long_history(seed: int = 0) -> pd.Series:
    """Deterministic multi-decade close series covering every default scenario window
    (1987 -> 2021), so `build_scenarios` can successfully construct every scenario in
    `DEFAULT_SCENARIO_NAMES` from it. The exact numbers are not meant to be realistic --
    only to exercise the plumbing; scenario tests that need a controlled shape build
    their own short series instead."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("1986-01-02", "2021-01-04")
    log_rets = rng.normal(loc=0.0002, scale=0.011, size=len(dates) - 1)
    prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_rets)]))
    return pd.Series(prices, index=dates, name="close")


def _trade_row(
    *,
    position_id="P1",
    root="SPY",
    strategy="put_credit_spread",
    entry_date=date(2016, 3, 1),
    expiry=date(2016, 4, 15),
    dte_entry=45,
    qty=2,
    short_put_strike=190.0,
    long_put_strike=185.0,
    entry_credit=0.80,
    underlying_entry=200.0,
    iv_entry=0.18,
    max_loss=4.20,
) -> pd.Series:
    return pd.Series(
        {
            "position_id": position_id, "root": root, "strategy": strategy,
            "entry_date": pd.Timestamp(entry_date), "expiry": pd.Timestamp(expiry),
            "dte_entry": dte_entry, "qty": qty,
            "short_put_strike": short_put_strike, "long_put_strike": long_put_strike,
            "short_call_strike": np.nan, "long_call_strike": np.nan,
            "short_strike": short_put_strike, "long_strike": long_put_strike,
            "entry_credit": entry_credit, "underlying_entry": underlying_entry,
            "iv_entry": iv_entry, "max_loss": max_loss,
        }
    )


def _flat_exits() -> ExitConfig:
    """hold_to_expiry so replay_position tests aren't at the mercy of early-exit triggers
    unless a test wants them."""
    return ExitConfig(hold_to_expiry=True)


# ======================================================================================
# load_underlying_history
# ======================================================================================


def test_load_underlying_history_parses_date_close(tmp_path):
    closes = pd.Series([100.0, 101.0, 99.5], index=pd.bdate_range("2020-01-01", periods=3))
    path = _write_history_csv(tmp_path, closes)
    s = stress.load_underlying_history(path)
    assert len(s) == 3
    assert s.iloc[0] == pytest.approx(100.0)


def test_load_underlying_history_accepts_yahoo_style_columns(tmp_path):
    df = pd.DataFrame({"Date": pd.bdate_range("2020-01-01", periods=3), "Close": [1.0, 2.0, 3.0]})
    path = tmp_path / "yahoo.csv"
    df.to_csv(path, index=False)
    s = stress.load_underlying_history(path)
    assert len(s) == 3


def test_load_underlying_history_missing_columns_raises(tmp_path):
    df = pd.DataFrame({"foo": [1, 2, 3]})
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="need 'date' and 'close'"):
        stress.load_underlying_history(path)


# ======================================================================================
# build_scenarios: refuses to fabricate, correct construction
# ======================================================================================


def test_build_scenarios_skips_windows_the_series_does_not_cover():
    # Series only covers 2015-2020 -- every "historical" 20th-century/2008/2011 window
    # must be skipped, not fabricated.
    closes = pd.Series(
        100.0 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 1500))),
        index=pd.bdate_range("2015-01-01", periods=1500),
    )
    lib = stress.build_scenarios(closes, names=["1987-10", "2008-09", "2018-02"])
    assert "1987-10" in lib.skipped
    assert "2008-09" in lib.skipped
    assert "2018-02" in lib.scenarios
    for reason in lib.skipped.values():
        assert "does not cover" in reason


def test_build_scenarios_unknown_name_is_reported_not_silently_dropped():
    closes = pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2020-01-01", periods=3))
    lib = stress.build_scenarios(closes, names=["not-a-real-scenario"])
    assert lib.scenarios == {}
    assert "not-a-real-scenario" in lib.skipped


def test_build_scenarios_worst_2008_subwindow_is_actually_the_worst():
    # Small controlled series over the 2008-09 window with one obviously bad 21-day
    # stretch injected -- confirm the "worst" scenario picks THAT stretch, not just any.
    n = 130
    idx = pd.bdate_range("2008-09-01", periods=n)
    rets = np.full(n - 1, 0.001)
    crash_start = 40
    rets[crash_start:crash_start + 21] = -0.03  # a brutal 21-day stretch
    prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(rets)]))
    closes = pd.Series(prices, index=idx)

    lib = stress.build_scenarios(closes, names=["2008-09-worst-21d"])
    sc = lib.scenarios["2008-09-worst-21d"]
    assert sc.n_days == 21
    assert np.sum(sc.daily_log_returns) == pytest.approx(-0.03 * 21, abs=1e-6)
    # every OTHER possible 21-day window in the source series must have a cumulative
    # return >= the chosen one (it really is the worst).
    full_rets = np.diff(np.log(closes.to_numpy(dtype=float)))
    csum = np.concatenate([[0.0], np.cumsum(full_rets)])
    all_windows = csum[21:] - csum[:-21]
    assert np.sum(sc.daily_log_returns) == pytest.approx(float(all_windows.min()), abs=1e-9)


def test_build_scenarios_full_library_on_long_synthetic_history():
    closes = _synthetic_long_history()
    lib = stress.build_scenarios(closes)
    for name in stress.DEFAULT_SCENARIO_NAMES:
        assert name in lib.scenarios, f"{name} should have been built: skipped={lib.skipped}"
        sc = lib.scenarios[name]
        assert sc.n_days > 0
        assert np.isfinite(sc.realized_vol)


# ======================================================================================
# scenario_iv: the IV-mapping assumption + its sensitivity range
# ======================================================================================


def _make_scenario(vol_expansion: float, name="s") -> stress.Scenario:
    idx = pd.bdate_range("2020-01-01", periods=3)
    return stress.Scenario(
        name=name, category="historical", description="test",
        start=idx[0].date(), end=idx[-1].date(),
        daily_log_returns=np.array([-0.05, -0.03]), source_closes=pd.Series([100.0, 95.0, 92.0], index=idx),
        realized_vol=0.60, baseline_vol=0.60 / vol_expansion if vol_expansion else 1.0, vol_expansion=vol_expansion,
    )


def test_scenario_iv_scales_with_vol_expansion_and_beta():
    sc = _make_scenario(vol_expansion=2.0)
    iv_entry = 0.20
    assert stress.scenario_iv(iv_entry, sc, beta=1.0) == pytest.approx(0.40)
    assert stress.scenario_iv(iv_entry, sc, beta=1.3) == pytest.approx(0.52)


def test_scenario_iv_sensitivity_range_is_monotonic_and_documented():
    sc = _make_scenario(vol_expansion=1.5)
    ivs = [stress.scenario_iv(0.20, sc, beta=b) for b in stress.IV_BETA_SENSITIVITY_RANGE]
    assert ivs == sorted(ivs)  # monotonically increasing in beta
    assert stress.DEFAULT_IV_BETA in stress.IV_BETA_SENSITIVITY_RANGE


def test_scenario_iv_handles_degenerate_vol_expansion():
    sc = _make_scenario(vol_expansion=float("nan"))
    # falls back to vol_expansion=1.0 rather than propagating NaN into pricing
    assert np.isfinite(stress.scenario_iv(0.20, sc, beta=1.0))


# ======================================================================================
# replay_position: hand-computed P&L for one position on one scenario
# ======================================================================================


def test_replay_position_naked_matches_hand_computed_bs_pnl():
    # Single-day scenario, single short put, no early exit (hold_to_expiry) -- so the
    # ENTIRE path is just: mark to BS on day 1, hold to day 2 = expiry (dte_entry=2).
    row = _trade_row(
        short_put_strike=190.0, long_put_strike=np.nan, entry_credit=0.80,
        underlying_entry=200.0, iv_entry=0.18, dte_entry=2, qty=1,
    )
    row["long_put_strike"] = np.nan
    log_rets = np.array([-0.02, -0.01])
    sc = stress.Scenario(
        name="hand", category="historical", description="hand", start=date(2020, 1, 1), end=date(2020, 1, 3),
        daily_log_returns=log_rets, source_closes=pd.Series([200.0, 196.0, 194.0]),
        realized_vol=0.60, baseline_vol=0.30, vol_expansion=2.0,
    )
    beta = 1.0
    rep = stress.replay_position(row, sc, naked=True, exits_cfg=_flat_exits(), r=0.02, q=0.0, beta=beta)
    assert rep is not None

    # Hand-compute: iv_scenario = 0.18 * 2.0 * 1.0 = 0.36
    iv_scen = 0.18 * 2.0 * beta
    S0 = 200.0
    S1 = S0 * np.exp(log_rets[0])
    S2 = S1 * np.exp(log_rets[1])
    # day 1: dte_remaining = 1 -> T = 1/365 (not expiry yet, dte_remaining>0)
    mark1 = bs_price(S1, 190.0, 1 / 365.0, 0.02, 0.0, iv_scen, "P")
    # day 2: dte_remaining = 0 -> exits at expiry, final_mark uses T=0 -> intrinsic
    mark2 = bs_price(S2, 190.0, 0.0, 0.02, 0.0, iv_scen, "P")
    expected_pnl = (0.80 - float(mark2)) * 1 * CONTRACT_MULTIPLIER

    assert rep.pnl == pytest.approx(expected_pnl, abs=1e-6)
    assert rep.exit_day == 2
    assert rep.exit_reason == "expiry"
    assert rep.variant == "undefined"


def test_replay_position_returns_none_for_unpriceable_row():
    row = _trade_row()
    row["iv_entry"] = np.nan
    assert stress.replay_position(row, _make_scenario(1.5), naked=False, exits_cfg=_flat_exits()) is None

    row2 = _trade_row(short_put_strike=np.nan, long_put_strike=np.nan)
    row2["short_strike"] = np.nan
    row2["long_strike"] = np.nan
    assert stress.replay_position(row2, _make_scenario(1.5), naked=False, exits_cfg=_flat_exits()) is None


# ======================================================================================
# defined vs undefined divergence on a real crash path -- STRATEGY.md §2.1 / §10.3
# ======================================================================================


def test_defined_vs_undefined_diverges_violently_on_a_crash_scenario():
    # A brutal short put credit spread position, replayed through a scenario that blows
    # straight through both strikes. The defined-risk spread's loss is capped at
    # width - credit; the naked put's is not.
    row = _trade_row(
        short_put_strike=190.0, long_put_strike=185.0, entry_credit=0.80,
        underlying_entry=200.0, iv_entry=0.18, dte_entry=30, qty=5,
    )
    n = 25
    log_rets = np.full(n, np.log(0.85) / n)  # smooth ~15% cumulative crash over 25 days
    sc = stress.Scenario(
        name="crash", category="historical", description="crash", start=date(2008, 9, 1), end=date(2008, 10, 1),
        daily_log_returns=log_rets, source_closes=pd.Series(np.linspace(200, 170, n + 1)),
        realized_vol=0.80, baseline_vol=0.20, vol_expansion=4.0,
    )
    exits = ExitConfig(hold_to_expiry=True)
    defined = stress.replay_position(row, sc, naked=False, exits_cfg=exits, beta=1.3)
    undefined = stress.replay_position(row, sc, naked=True, exits_cfg=exits, beta=1.3)
    assert defined is not None and undefined is not None

    # the defined-risk loss is bounded by (width - credit) * qty * multiplier
    width = 190.0 - 185.0
    max_loss_dollars = (width - 0.80) * row["qty"] * CONTRACT_MULTIPLIER
    assert defined.pnl >= -max_loss_dollars - 1e-6

    # the naked variant has no such floor and must lose materially more on this path
    assert undefined.pnl < defined.pnl
    divergence = undefined.pnl - defined.pnl
    assert divergence < -1000  # a real, not a rounding-noise, divergence
    assert undefined.breached_short_strike
    assert defined.breached_short_strike


def test_run_stress_comparisons_report_divergence_and_worst_position():
    trades = pd.DataFrame([_trade_row(position_id="P1", dte_entry=40)])
    n = 25
    log_rets = np.full(n, np.log(0.85) / n)
    sc = stress.Scenario(
        name="2008-09", category="historical", description="crash", start=date(2008, 9, 1), end=date(2008, 10, 1),
        daily_log_returns=log_rets, source_closes=pd.Series(np.linspace(200, 170, n + 1)),
        realized_vol=0.80, baseline_vol=0.20, vol_expansion=4.0,
    )
    lib = stress.ScenarioLibrary(scenarios={"2008-09": sc}, skipped={})
    result = stress.run_stress(trades, lib, ExitConfig(hold_to_expiry=True), betas=(1.0, 1.3))

    assert not result.comparisons.empty
    row = result.comparisons.iloc[0]
    assert row["undefined_pnl"] < row["defined_pnl"]
    assert row["divergence"] < 0
    assert not result.beta_sensitivity.empty
    assert set(result.beta_sensitivity["beta"].unique()) == {1.0, 1.3}


# ======================================================================================
# long-window empirical distribution feed
# ======================================================================================


def test_run_backtest_with_long_history_reports_trade_and_pnl_deltas(tmp_path):
    from odds_lab.data.providers.synthetic import make_sample_store

    store = make_sample_store(
        tmp_path / "store", roots=("SPY",), start=date(2015, 1, 1), end=date(2015, 9, 1), seed=11,
    )
    entry = EntryConfig(
        strategy="put_credit_spread", strike_rule="delta", delta_min=0.15, delta_max=0.30,
        width_strikes=1, dte_min=21, dte_max=56, min_credit=0.10, expected_return_min=0.0,
        expected_return_max=1.0, market_filter="all", require_positive_edge=True,
        entry_schedule="weekly", entry_weekday=0, max_concurrent_per_root=2,
    )
    exits = ExitConfig(profit_target_pct=0.50, stop_loss_multiple=2.0, dte_exit=21, delta_breach=0.50, hold_to_expiry=False)
    risk = RiskConfig(starting_equity=100_000.0, risk_pct_per_trade=0.05, min_contracts=1, max_contracts=50)
    empirical = EmpiricalConfig(lookback_years=2.0, min_samples=20, seed=1)
    cfg = BacktestConfig(
        roots=("SPY",), start=date(2015, 3, 1), end=date(2015, 9, 1),
        entry=entry, exits=exits, risk=risk, costs=CostModel(), empirical=empirical,
        data=DataConfig(provider="synthetic", allow_synthetic=True),
    )

    # Long history: the store's own closes, prepended with a violent synthetic crash --
    # this should make `build_empirical`'s "Count" fatter-tailed than the short window.
    native = store.closes("SPY", end=cfg.end)
    native_idx = pd.DatetimeIndex([pd.Timestamp(d) for d in native.index])
    crash_days = pd.bdate_range(end=native_idx.min() - pd.Timedelta(days=1), periods=400)
    rng = np.random.default_rng(3)
    crash_rets = rng.normal(-0.01, 0.05, len(crash_days) - 1)
    crash_prices = float(native.iloc[0]) * np.exp(
        np.concatenate([np.cumsum(crash_rets[::-1])[::-1] * -1, [0.0]])
    )
    long_series = pd.concat(
        [pd.Series(crash_prices, index=crash_days), pd.Series(native.to_numpy(), index=native_idx)]
    ).sort_index()
    long_series = long_series[~long_series.index.duplicated(keep="last")]

    short_result, long_result, comparison = stress.run_backtest_with_long_history(cfg, store, {"SPY": long_series})

    assert "trades" in comparison["short_window"]
    assert "trades" in comparison["long_window_with_2008"]
    assert comparison["trade_count_delta"] == comparison["long_window_with_2008"]["trades"] - comparison["short_window"]["trades"]
    assert comparison["pnl_delta"] == pytest.approx(
        comparison["long_window_with_2008"]["total_pnl"] - comparison["short_window"]["total_pnl"]
    )


def test_long_history_store_delegates_everything_except_closes(tmp_path):
    from odds_lab.data.providers.synthetic import make_sample_store

    store = make_sample_store(tmp_path / "store", roots=("SPY",), start=date(2015, 1, 1), end=date(2015, 3, 1), seed=2)
    long_series = pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("1990-01-01", periods=3))
    wrapped = stress.LongHistoryStore(store, {"SPY": long_series})

    # trading_dates/expiries/chain/underlying pass straight through
    assert wrapped.trading_dates("SPY", date(2015, 1, 1), date(2015, 3, 1)) == store.trading_dates(
        "SPY", date(2015, 1, 1), date(2015, 3, 1)
    )
    # closes() is substituted for a root with a supplied long series
    c = wrapped.closes("SPY", date(1990, 1, 3))
    assert len(c) == 3
    # a root with no supplied long series falls back to the wrapped store
    c2 = wrapped.closes("QQQ", date(2015, 3, 1))
    assert c2.equals(store.closes("QQQ", date(2015, 3, 1)))


# ======================================================================================
# end-to-end CLI
# ======================================================================================


def test_cli_stress_end_to_end(tmp_path):
    from odds_lab.data.providers.synthetic import make_sample_store
    from odds_lab.engine.loop import run_backtest
    from odds_lab import cli

    store = make_sample_store(
        tmp_path / "store", roots=("SPY",), start=date(2015, 1, 1), end=date(2016, 6, 1), seed=7,
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
    cfg = BacktestConfig(
        roots=("SPY",), start=date(2015, 6, 1), end=date(2016, 6, 1),
        entry=entry, exits=exits, risk=risk, costs=CostModel(), empirical=empirical,
        data=DataConfig(provider="synthetic", allow_synthetic=True),
    )
    result = run_backtest(cfg, store)
    assert len(result.trades) > 0
    run_dir = tmp_path / "run"
    result.save(run_dir)
    from odds_lab.report.build import build_report

    build_report(result, run_dir / "report.html")

    history_path = _write_history_csv(tmp_path, _synthetic_long_history())

    rc = cli.main(
        [
            "stress",
            "--run", str(run_dir),
            "--underlying-history", str(history_path),
            "--scenarios", "2008-09,2008-09-worst-21d,2018-02",
        ]
    )
    assert rc == 0
    assert (run_dir / "stress" / "comparisons.parquet").exists()
    assert (run_dir / "stress" / "replays.parquet").exists()
    comparisons = pd.read_parquet(run_dir / "stress" / "comparisons.parquet")
    assert set(comparisons["scenario"]) == {"2008-09", "2008-09-worst-21d", "2018-02"}
    content = (run_dir / "report.html").read_text(encoding="utf-8")
    assert 'id="section-f"' in content
    assert "Crisis stress" in content
