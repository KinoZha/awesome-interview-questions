"""Tests for report/figures.py, report/build.py, and cli.py.

`engine.loop.run_backtest` / `engine.result.BacktestResult` are owned by another agent and
may not exist yet -- CLAUDE.md says keep going and re-run at the end. Everything that does
NOT need the real engine (every figure, build_report, the "missing store" CLI path) is
fully exercised here against hand-built, schema-valid fixtures. The one test that needs a
real end-to-end backtest run (`test_cli_backtest_success_with_synthetic_store`) uses
`pytest.importorskip` so it skips cleanly until engine/loop.py lands, instead of failing.
"""

from __future__ import annotations

import dataclasses
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from odds_lab import schema
from odds_lab.config import BacktestConfig
from odds_lab.report import figures as F
from odds_lab.report.build import build_report

# --------------------------------------------------------------------------------------
# Fixture generators (schema-valid, fabricated) -- kept in the test file only, per the
# task brief: "Do not put fake-data helpers in the shipped report package."
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class _FakeResult:
    """Stands in for engine.result.BacktestResult -- same field names/shapes."""

    config: BacktestConfig
    trades: pd.DataFrame
    equity: pd.DataFrame
    snapshots: pd.DataFrame
    manifest: dict


def _make_config(**overrides) -> BacktestConfig:
    cfg = BacktestConfig(roots=("SPY", "QQQ"), start=date(2018, 1, 1), end=date(2019, 6, 1))
    return dataclasses.replace(cfg, **overrides) if overrides else cfg


def _make_manifest(is_synthetic: bool) -> dict:
    return {
        "run_id": "abc123def456",
        "git_sha": "deadbeef",
        "generated_at": "2026-01-01T00:00:00Z",
        "source": "synthetic" if is_synthetic else "thetadata",
        "coverage": {"SPY": {"min_date": "2018-01-01", "max_date": "2019-06-01"}},
        "is_synthetic": is_synthetic,
        "rejected_trades": [{"reason": "min_credit not met", "date": "2018-03-01", "root": "SPY"}],
        "warnings": ["iv disagreement > 1e-3 on 12 rows"],
    }


def _make_trades(n: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    roots = ["SPY", "QQQ"]
    strategies = ["put_credit_spread", "call_credit_spread"]
    states = ["bullish", "neutral", "bearish"]
    reasons = list(schema.ExitReason)
    rows = []
    start = date(2018, 1, 8)
    for i in range(n):
        entry_date = start + timedelta(days=7 * i)
        dte = int(rng.integers(21, 57))
        exit_date = entry_date + timedelta(days=int(rng.integers(5, dte)))
        underlying_entry = float(300 + 5 * np.sin(i) + rng.normal(0, 2))
        short_delta = float(rng.uniform(0.15, 0.30))
        short_strike = round(underlying_entry * (1 - short_delta * 0.6), 1)
        width = float(rng.choice([1, 2, 3, 5]))
        long_strike = short_strike - width
        credit = round(float(rng.uniform(0.3, 1.2)), 2)
        # deliberately fat left tail: most trades win small, a few lose big
        if i % 5 == 0:
            pnl = -float(rng.uniform(200, 900))
        else:
            pnl = float(rng.uniform(20, 180))
        underlying_exit = underlying_entry + (pnl / 100.0)
        rows.append(
            dict(
                position_id=f"pos-{i:03d}",
                strategy=strategies[i % 2],
                root=roots[i % 2],
                entry_date=pd.Timestamp(entry_date),
                exit_date=pd.Timestamp(exit_date),
                expiry=pd.Timestamp(entry_date + timedelta(days=dte)),
                dte_entry=dte,
                qty=int(rng.integers(1, 5)),
                short_strike=short_strike,
                long_strike=long_strike,
                width=width,
                short_put_strike=short_strike,
                long_put_strike=long_strike,
                short_call_strike=float("nan"),
                long_call_strike=float("nan"),
                entry_credit=credit,
                exit_debit=round(max(credit - pnl / 100.0, 0.0), 2),
                max_loss=(width - credit) * 100,
                margin=(width - credit) * 100,
                pnl=pnl,
                pnl_pct_of_max_loss=pnl / max((width - credit) * 100, 1e-6),
                commission=1.3,
                fees=0.1,
                slippage=float(rng.uniform(1, 15)),
                exit_reason=reasons[i % len(reasons)].value,
                short_delta=short_delta,
                iv_entry=float(rng.uniform(0.14, 0.28)),
                iv_rank=float(rng.uniform(0, 1)),
                realized_vol=float(rng.uniform(0.1, 0.25)),
                market_state=states[i % 3],
                p_theo_loss=float(rng.uniform(0.1, 0.3)),
                p_actual_loss=float(rng.uniform(0.05, 0.25)),
                edge_prob=float(rng.uniform(-0.05, 0.1)),
                edge_ev=float(rng.uniform(-10, 40)),
                underlying_entry=underlying_entry,
                underlying_exit=underlying_exit,
            )
        )
    # Fabricated but bridge-consistent (STRATEGY.md §8A): costs + execution are set
    # directly, greek terms absorb the remainder, so `pnl == sum of the bridge columns`
    # holds exactly here too, same as a real run.
    for row in rows:
        row["pnl_costs"] = -(row["commission"] + row["fees"])
        row["pnl_entry_execution"] = round(row["pnl"] * 0.03, 6)
        row["pnl_exit_execution"] = round(row["pnl"] * 0.02, 6)
        remainder = (
            row["pnl"] - row["pnl_costs"] - row["pnl_entry_execution"] - row["pnl_exit_execution"]
        )
        row["pnl_delta"] = remainder * 0.4
        row["pnl_gamma"] = remainder * 0.05
        row["pnl_vega"] = remainder * 0.1
        row["pnl_theta"] = remainder * 0.5
        row["pnl_residual"] = remainder * -0.05
    df = pd.DataFrame(rows)
    return schema.validate_frame(df, schema.TRADE_DTYPES, "trades")


def _make_equity(trades: pd.DataFrame, start_equity: float = 100_000.0) -> pd.DataFrame:
    if trades.empty:
        d0, d1 = date(2018, 1, 1), date(2018, 1, 5)
    else:
        d0 = pd.Timestamp(trades["entry_date"].min()).date()
        d1 = pd.Timestamp(trades["exit_date"].max()).date() + timedelta(days=5)
    dates = pd.bdate_range(d0, d1)
    rng = np.random.default_rng(1)
    equity = start_equity + np.cumsum(rng.normal(30, 120, size=len(dates)))
    df = pd.DataFrame(
        {
            "date": dates,
            "cash": equity * 0.6,
            "positions_value": equity * 0.4,
            "equity": equity,
            "margin_used": np.abs(rng.normal(5000, 1000, size=len(dates))),
            "margin_pct": np.clip(rng.uniform(0.05, 0.5, size=len(dates)), 0, 1),
            "open_positions": rng.integers(0, 5, size=len(dates)),
            "net_delta": rng.normal(0, 5, size=len(dates)),
            "net_gamma": rng.normal(0, 0.5, size=len(dates)),
            "net_vega": rng.normal(0, 10, size=len(dates)),
            "net_theta": rng.normal(-5, 2, size=len(dates)),
        }
    )
    return schema.validate_frame(df, schema.EQUITY_DTYPES, "equity")


def _make_snapshots(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return schema.empty_frame(schema.SNAPSHOT_DTYPES)
    rng = np.random.default_rng(2)
    rows = []
    for _, t in trades.iterrows():
        days = pd.bdate_range(t["entry_date"], t["exit_date"])
        if len(days) == 0:
            days = pd.DatetimeIndex([t["entry_date"]])
        n = len(days)
        walk = np.cumsum(rng.normal(0, 1.5, size=n))
        price = t["underlying_entry"] + walk - walk[-1] + (t["underlying_exit"] - t["underlying_entry"])
        mtm = np.linspace(0, t["pnl"], n)
        short_delta_path = np.clip(t["short_delta"] + np.linspace(0, 0.1, n) * np.sign(t["pnl"] * -1 + 1e-9), 0.01, 0.9)
        for j, d in enumerate(days):
            rows.append(
                dict(
                    position_id=t["position_id"],
                    date=d,
                    underlying_price=float(price[j]),
                    mark=float(t["entry_credit"] - mtm[j] / 100.0),
                    mtm_pnl=float(mtm[j]),
                    dte=int(max(t["dte_entry"] - j, 0)),
                    short_delta=float(short_delta_path[j]),
                    net_delta=float(rng.normal(0, 2)),
                    net_gamma=float(rng.normal(0, 0.1)),
                    net_vega=float(rng.normal(0, 3)),
                    net_theta=float(rng.normal(-1, 0.3)),
                    iv_short=float(t["iv_entry"]),
                    d_delta=float(rng.normal(0, 2)),
                    d_gamma=float(rng.normal(0, 0.1)),
                    d_vega=float(rng.normal(0, 1)),
                    d_theta=float(rng.normal(0, 0.2)),
                    d_residual=float(rng.normal(0, 0.5)),
                )
            )
    df = pd.DataFrame(rows)
    return schema.validate_frame(df, schema.SNAPSHOT_DTYPES, "snapshots")


def _make_result(n_trades: int = 8, is_synthetic: bool = True, seed: int = 0) -> _FakeResult:
    trades = _make_trades(n_trades, seed=seed)
    equity = _make_equity(trades)
    snaps = _make_snapshots(trades)
    return _FakeResult(config=_make_config(), trades=trades, equity=equity, snapshots=snaps,
                        manifest=_make_manifest(is_synthetic))


def _make_empty_result() -> _FakeResult:
    return _FakeResult(
        config=_make_config(),
        trades=schema.empty_frame(schema.TRADE_DTYPES),
        equity=schema.empty_frame(schema.EQUITY_DTYPES),
        snapshots=schema.empty_frame(schema.SNAPSHOT_DTYPES),
        manifest=_make_manifest(is_synthetic=False),
    )


# --------------------------------------------------------------------------------------
# Figures build without error, on real fixtures and on empty data
# --------------------------------------------------------------------------------------

_NO_ARG_FIGURES = [
    F.fig_equity_curve,
    F.fig_drawdown,
    F.fig_monthly_heatmap,
    F.fig_attribution_stack,
    F.fig_edge_over_time,
    F.fig_win_rate_gauge,
    F.fig_pnl_distribution,
    F.fig_year_week_calendar,
]

_COHORTS = ["iv_rank", "market_state", "dte_entry", "short_delta", "year", "root"]


@pytest.mark.parametrize("fn", _NO_ARG_FIGURES)
def test_figure_builds_on_fabricated_data(fn):
    result = _make_result()
    fig = fn(result)
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize("fn", _NO_ARG_FIGURES)
def test_figure_builds_on_empty_data(fn):
    result = _make_empty_result()
    fig = fn(result)
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize("by", _COHORTS)
def test_fig_cohort_bars_builds_on_fabricated_data(by):
    result = _make_result()
    fig = F.fig_cohort_bars(result, by)
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize("by", _COHORTS)
def test_fig_cohort_bars_builds_on_empty_data(by):
    result = _make_empty_result()
    fig = F.fig_cohort_bars(result, by)
    assert isinstance(fig, go.Figure)


def test_fig_cohort_bars_rejects_unknown_dimension():
    result = _make_result()
    with pytest.raises(ValueError):
        F.fig_cohort_bars(result, "not_a_real_column")


def test_fig_trade_lifecycle_builds_for_a_real_trade():
    result = _make_result()
    pid = result.trades["position_id"].iloc[0]
    fig = F.fig_trade_lifecycle(result, pid)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_fig_trade_lifecycle_unknown_position_id_does_not_crash():
    result = _make_result()
    fig = F.fig_trade_lifecycle(result, "no-such-position")
    assert isinstance(fig, go.Figure)


def test_fig_trade_lifecycle_empty_trades_does_not_crash():
    result = _make_empty_result()
    fig = F.fig_trade_lifecycle(result, "whatever")
    assert isinstance(fig, go.Figure)


def test_fig_trade_table_html_on_fabricated_and_empty_data():
    html_full = F.fig_trade_table(_make_result())
    assert "<table" in html_full
    assert "pos-000" in html_full
    html_empty = F.fig_trade_table(_make_empty_result())
    assert "<table" not in html_empty  # explicit empty-state message instead


def test_fig_strategy_comparison_builds():
    results = {"run-a": _make_result(seed=0), "run-b": _make_result(seed=1)}
    fig = F.fig_strategy_comparison(results)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_fig_strategy_comparison_empty_dict_does_not_crash():
    fig = F.fig_strategy_comparison({})
    assert isinstance(fig, go.Figure)


# --------------------------------------------------------------------------------------
# The ODDS chart: theoretical vs empirical must agree on an exact lognormal sample
# --------------------------------------------------------------------------------------


def test_fig_odds_distribution_matches_exact_lognormal():
    rng = np.random.default_rng(42)
    sigma_annual = 0.20
    horizon = 21
    n_days = 3000
    sigma_daily = sigma_annual / np.sqrt(252)
    daily_log_returns = rng.normal(0.0, sigma_daily, size=n_days)
    prices = 100.0 * np.exp(np.cumsum(daily_log_returns))
    dates = pd.bdate_range("2010-01-04", periods=n_days)
    closes = pd.Series(prices, index=dates)
    asof = dates[-1] + pd.Timedelta(days=1)

    fig = F.fig_odds_distribution(closes, horizon, sigma_annual, asof)
    bar_trace = next(t for t in fig.data if isinstance(t, go.Bar))
    line_trace = next(t for t in fig.data if isinstance(t, go.Scatter) and "Theoretical" in (t.name or ""))

    bar_x = np.asarray(bar_trace.x, dtype=float)
    bar_y = np.asarray(bar_trace.y, dtype=float)
    line_x = np.asarray(line_trace.x, dtype=float)
    line_y = np.asarray(line_trace.y, dtype=float)

    theo_at_bars = np.interp(bar_x, line_x, line_y)
    # empirical density should track the theoretical curve reasonably closely across
    # the bulk of the distribution (generous tolerance -- this is a finite MC sample,
    # not an exact match, and the tails are thin by construction of a histogram).
    mask = np.abs(bar_x) < 3 * sigma_annual * np.sqrt(horizon / 252.0)
    assert mask.sum() >= 5
    err = np.abs(bar_y[mask] - theo_at_bars[mask])
    assert np.median(err) < 0.15 * theo_at_bars[mask].max()

    # first moment sanity: empirical horizon-return std should be close to sigma_T
    sigma_T = sigma_annual * np.sqrt(horizon / 252.0)
    csum = np.concatenate([[0.0], np.cumsum(daily_log_returns)])
    empirical_returns = csum[horizon:] - csum[:-horizon]
    assert abs(empirical_returns.std() - sigma_T) < 0.15 * sigma_T


def test_fig_odds_distribution_insufficient_history_does_not_crash():
    dates = pd.bdate_range("2020-01-01", periods=5)
    closes = pd.Series(np.linspace(100, 101, 5), index=dates)
    fig = F.fig_odds_distribution(closes, horizon=21, iv=0.2, asof=dates[-1] + pd.Timedelta(days=1))
    assert isinstance(fig, go.Figure)


def test_fig_odds_distribution_marks_strikes():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2015-01-01", periods=800)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=800)))
    closes = pd.Series(prices, index=dates)
    asof = dates[-1] + pd.Timedelta(days=1)
    fig = F.fig_odds_distribution(closes, 21, 0.18, asof, strikes=[closes.iloc[-1] * 0.95, closes.iloc[-1] * 1.05])
    shapes = fig.layout.shapes
    assert any(s.type == "line" and s.x0 == s.x1 for s in shapes)


# --------------------------------------------------------------------------------------
# build_report: self-contained, banner logic
# --------------------------------------------------------------------------------------


def _assert_no_remote_script_src(content: str) -> None:
    """No <script src="http(s)://..."> anywhere -- the page must open with no network.
    (plotly.js's own inlined source text legitimately contains the literal string
    'https://' as a default CDN-URL config value; that is not a network reference.)"""
    import re

    for m in re.finditer(r'<script\b[^>]*\bsrc\s*=\s*["\']([^"\']+)["\']', content, re.IGNORECASE):
        assert not m.group(1).startswith("http"), f"remote script src found: {m.group(1)}"


def test_build_report_is_self_contained_and_shows_synthetic_banner(tmp_path):
    result = _make_result(is_synthetic=True)
    out = build_report(result, tmp_path / "report.html")
    content = out.read_text(encoding="utf-8")
    _assert_no_remote_script_src(content)
    assert "SYNTHETIC" in content
    assert 'class="synthetic-banner"' in content


def test_build_report_no_banner_when_not_synthetic(tmp_path):
    result = _make_result(is_synthetic=False)
    out = build_report(result, tmp_path / "report.html")
    content = out.read_text(encoding="utf-8")
    _assert_no_remote_script_src(content)
    assert 'class="synthetic-banner"' not in content


def test_build_report_handles_zero_trades(tmp_path):
    result = _make_empty_result()
    out = build_report(result, tmp_path / "report_empty.html")
    content = out.read_text(encoding="utf-8")
    assert out.exists()
    _assert_no_remote_script_src(content)
    assert "No trades" in content or "odds-empty" in content


def test_build_report_includes_key_sections(tmp_path):
    result = _make_result()
    out = build_report(result, tmp_path / "report.html")
    content = out.read_text(encoding="utf-8")
    for anchor in ("section-a", "section-b", "section-c", "section-d", "section-e"):
        assert f'id="{anchor}"' in content
    assert "trade-table" in content


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def test_cli_backtest_missing_store(tmp_path, capsys):
    from odds_lab import cli

    rc = cli.main(
        [
            "backtest",
            "--store", str(tmp_path / "does-not-exist"),
            "--roots", "SPY",
            "--start", "2018-01-01",
            "--end", "2018-06-01",
            "--out", str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr()
    assert rc != 0
    assert "error" in captured.err.lower()
    assert "does-not-exist" in captured.err


def test_cli_backtest_success_with_synthetic_store(tmp_path):
    pytest.importorskip("odds_lab.engine.loop", reason="engine.loop not implemented yet")
    pytest.importorskip("odds_lab.engine.result", reason="engine.result not implemented yet")
    from odds_lab.data.providers.synthetic import make_sample_store
    from odds_lab import cli

    store_path = tmp_path / "store"
    make_sample_store(store_path, roots=("SPY",), start=date(2018, 1, 1), end=date(2018, 6, 1), seed=7)

    rc = cli.main(
        [
            "backtest",
            "--store", str(store_path),
            "--roots", "SPY",
            "--start", "2018-01-01",
            "--end", "2018-06-01",
            "--out", str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    assert (tmp_path / "out" / "report.html").exists()


def test_cli_report_missing_run_dir(tmp_path, capsys):
    from odds_lab import cli

    rc = cli.main(["report", "--run", str(tmp_path / "no-such-run")])
    captured = capsys.readouterr()
    assert rc != 0
    assert "error" in captured.err.lower()


def test_cli_no_args_is_nonzero():
    from odds_lab import cli

    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code != 0
