"""Self-contained HTML report builder. STRATEGY.md §8, ARCHITECTURE.md §5.

`build_report` writes ONE HTML file: plotly JS inlined once (`include_plotlyjs='inline'`
on the first figure, `False` after -- ARCHITECTURE.md §5, "must open offline"), a sticky
nav, a summary stat header, sections A-E, and a loud SYNTHETIC banner whenever
`result.manifest['is_synthetic']` is true. Never silently drops that banner.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from odds_lab.report import figures as F
from odds_lab.report.assets import CSS, TABLE_JS, THEME_JS

if TYPE_CHECKING:  # pragma: no cover
    from odds_lab.engine.result import BacktestResult

__all__ = ["build_report"]


# --------------------------------------------------------------------------------------
# Summary stats for the header
# --------------------------------------------------------------------------------------


def _summary_stats(result: "BacktestResult") -> dict:
    eq = result.equity
    trades = result.trades

    cagr = sharpe = max_dd = np.nan
    if not eq.empty and len(eq) > 1:
        eq_sorted = eq.sort_values("date")
        start_eq = float(eq_sorted["equity"].iloc[0])
        end_eq = float(eq_sorted["equity"].iloc[-1])
        days = (pd.Timestamp(eq_sorted["date"].iloc[-1]) - pd.Timestamp(eq_sorted["date"].iloc[0])).days
        years = days / 365.25 if days > 0 else np.nan
        if years and years > 0 and start_eq > 0:
            cagr = (end_eq / start_eq) ** (1.0 / years) - 1.0
        rets = eq_sorted["equity"].pct_change().dropna()
        if len(rets) > 1 and rets.std() > 0:
            sharpe = float(rets.mean() / rets.std() * np.sqrt(252))
        running_max = eq_sorted["equity"].cummax()
        max_dd = float(((eq_sorted["equity"] / running_max.replace(0, np.nan)) - 1.0).min())

    win_rate = profit_factor = avg_credit = avg_days_held = np.nan
    total_trades = 0
    if not trades.empty:
        total_trades = int(len(trades))
        win_rate = float((trades["pnl"] > 0).mean())
        gains = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
        losses = float(-trades.loc[trades["pnl"] < 0, "pnl"].sum())
        profit_factor = gains / losses if losses > 0 else float("inf") if gains > 0 else np.nan
        avg_credit = float(trades["entry_credit"].mean())
        held = (
            pd.to_datetime(trades["exit_date"].fillna(trades["entry_date"]))
            - pd.to_datetime(trades["entry_date"])
        ).dt.days
        avg_days_held = float(held.mean())

    return dict(
        cagr=cagr, sharpe=sharpe, max_dd=max_dd, win_rate=win_rate, profit_factor=profit_factor,
        total_trades=total_trades, avg_credit=avg_credit, avg_days_held=avg_days_held,
    )


def _fmt_pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.1f}%"


def _fmt_num(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "n/a" if not (isinstance(x, float) and np.isinf(x)) else "inf"
    return f"{x:.{digits}f}"


def _stat_card(label: str, value: str, good: bool | None = None) -> str:
    cls = "" if good is None else (" good" if good else " bad")
    return (
        f'<div class="stat-card"><div class="label">{html.escape(label)}</div>'
        f'<div class="value{cls}">{value}</div></div>'
    )


def _render_stat_header(stats: dict) -> str:
    cards = [
        _stat_card("CAGR", _fmt_pct(stats["cagr"]), good=(stats["cagr"] or 0) >= 0 if not np.isnan(stats["cagr"]) else None),
        _stat_card("Sharpe", _fmt_num(stats["sharpe"]), good=(stats["sharpe"] or 0) >= 1 if not np.isnan(stats["sharpe"]) else None),
        _stat_card("Max drawdown", _fmt_pct(stats["max_dd"]), good=False if not np.isnan(stats["max_dd"]) else None),
        _stat_card("Win rate", _fmt_pct(stats["win_rate"]), good=(stats["win_rate"] or 0) >= 0.5 if not np.isnan(stats["win_rate"]) else None),
        _stat_card("Profit factor", _fmt_num(stats["profit_factor"]), good=(stats["profit_factor"] or 0) >= 1 if not np.isnan(stats["profit_factor"]) else None),
        _stat_card("Total trades", str(stats["total_trades"])),
        _stat_card("Avg credit", f"${_fmt_num(stats['avg_credit'])}"),
        _stat_card("Avg days held", _fmt_num(stats["avg_days_held"], 1)),
    ]
    return f'<div class="stat-grid">{"".join(cards)}</div>'


# --------------------------------------------------------------------------------------
# Figure embedding
# --------------------------------------------------------------------------------------


class _JsBudget:
    """Ensures plotly.js is inlined exactly once; every later figure references it."""

    def __init__(self) -> None:
        self.used = False

    def embed(self, fig: go.Figure, *, div_id: str | None = None) -> str:
        include = "inline" if not self.used else False
        self.used = True
        return fig.to_html(
            full_html=False,
            include_plotlyjs=include,
            config={"responsive": True, "displaylogo": False},
            div_id=div_id,
        )


def _block(inner: str, heading: str | None = None) -> str:
    head = f"<h3>{html.escape(heading)}</h3>" if heading else ""
    return f'{head}<div class="fig-block">{inner}</div>'


# --------------------------------------------------------------------------------------
# Section builders
# --------------------------------------------------------------------------------------


def _select_lifecycle_positions(trades: pd.DataFrame, n_each: int = 2) -> list[str]:
    if trades.empty:
        return []
    ranked = trades.sort_values("pnl")
    worst = list(ranked["position_id"].head(n_each))
    best = list(ranked["position_id"].tail(n_each))
    ids: list[str] = []
    for pid in worst + best:
        if pid not in ids:
            ids.append(pid)
    return ids[:5]


def _reconstruct_closes(result: "BacktestResult", root: str, store=None) -> pd.Series | None:
    """Best-effort underlying close series for `root`, for the ODDS chart. Prefers a
    ChainStore if given; otherwise falls back to the underlying prices recorded in
    `result.snapshots` for that root (only covers dates when a position was open)."""
    if store is not None:
        try:
            end = pd.Timestamp(result.config.end)
            closes = store.closes(root, end)
            if closes is not None and len(closes) > 0:
                return closes
        except Exception:
            pass
    trades = result.trades
    snaps = result.snapshots
    if trades.empty or snaps.empty:
        return None
    root_positions = set(trades.loc[trades["root"] == root, "position_id"])
    sub = snaps[snaps["position_id"].isin(root_positions)]
    if sub.empty:
        return None
    px = sub.groupby("date")["underlying_price"].mean().sort_index()
    return px if len(px) > 5 else None


def _section_c_odds_charts(result: "BacktestResult", store=None) -> str:
    trades = result.trades
    if trades.empty:
        return '<p class="odds-empty">No trades -- nothing to plot.</p>'
    parts = []
    for root in sorted(trades["root"].dropna().unique()):
        sub = trades[trades["root"] == root]
        closes = _reconstruct_closes(result, root, store=store)
        if closes is None or closes.empty:
            parts.append(f'<p class="odds-empty">{html.escape(root)}: no price history available for the ODDS chart.</p>')
            continue
        horizon = int(round(sub["dte_entry"].median())) if sub["dte_entry"].notna().any() else 30
        iv = float(sub["iv_entry"].median()) if sub["iv_entry"].notna().any() else 0.20
        asof = closes.index.max()
        strikes = sorted(set(sub["short_strike"].dropna().tolist()))
        fig = F.fig_odds_distribution(closes, horizon, iv, asof, strikes=strikes)
        parts.append(("__FIG__", f"{root} — {horizon}d horizon, IV={iv:.0%}", fig))
    return parts  # figures embedded by caller (needs the shared JS budget)


def build_report(result: "BacktestResult", out_path, *, store=None, extra: dict | None = None) -> Path:
    """Build the single-file HTML report and write it to `out_path`. Returns the Path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extra = extra or {}

    manifest = result.manifest or {}
    is_synthetic = bool(manifest.get("is_synthetic", False))
    stats = _summary_stats(result)
    js = _JsBudget()

    banner = ""
    if is_synthetic:
        banner = (
            '<div class="synthetic-banner">⚠ SYNTHETIC DATA — this run used generated, '
            "not real, market data. Every number on this page is illustrative only and "
            "must not be treated as a real backtest result. ⚠</div>"
        )

    nav = """
    <nav class="odds-nav">
      <span class="brand">odds_lab</span>
      <a href="#section-a">A. Equity</a>
      <a href="#section-b">B. Trade lifecycle</a>
      <a href="#section-c">C. ODDS chart</a>
      <a href="#section-d">D. Diagnostics</a>
      <a href="#section-e">E. Manifest</a>
      <a href="javascript:oddsToggleTheme()">Toggle theme</a>
    </nav>
    """

    header = f"""
    <div class="container">
      <h1>odds_lab backtest report{" — " + html.escape(result.config.label) if getattr(result.config, "label", "") else ""}</h1>
      <p style="color:var(--text-dim)">
        {html.escape(", ".join(result.config.roots))} &middot;
        {result.config.start.isoformat()} &rarr; {result.config.end.isoformat()} &middot;
        run <code>{html.escape(manifest.get("run_id", "n/a"))}</code>
      </p>
      {_render_stat_header(stats)}
    </div>
    """

    # -- Section A --------------------------------------------------------------------
    a_parts = [
        _block(js.embed(F.fig_equity_curve(result)), "Equity curve"),
        _block(js.embed(F.fig_drawdown(result)), "Drawdown"),
        _block(js.embed(F.fig_monthly_heatmap(result)), "Monthly returns"),
        _block(js.embed(F.fig_attribution_stack(result)), "P&L attribution (cumulative)"),
    ]
    section_a = f"""
    <section id="section-a" class="odds-section"><div class="container">
      <h2>A. Equity &amp; attribution</h2>
      {"".join(a_parts)}
    </div></section>
    """

    # -- Section B --------------------------------------------------------------------
    lifecycle_ids = _select_lifecycle_positions(result.trades)
    b_parts = []
    if lifecycle_ids:
        for pid in lifecycle_ids:
            b_parts.append(_block(js.embed(F.fig_trade_lifecycle(result, pid, store=store)), f"Trade {pid}"))
    else:
        b_parts.append('<p class="odds-empty">No trades to show a lifecycle for.</p>')
    b_parts.append(_block(F.fig_trade_table(result), "All trades"))
    section_b = f"""
    <section id="section-b" class="odds-section"><div class="container">
      <h2>B. Trade lifecycle (the key screen)</h2>
      <p style="color:var(--text-dim)">Best and worst trades by P&amp;L, shown first; the full trade table is filterable/sortable below.</p>
      {"".join(b_parts)}
    </div></section>
    """

    # -- Section C --------------------------------------------------------------------
    odds_parts = _section_c_odds_charts(result, store=store)
    c_html_parts = []
    if isinstance(odds_parts, str):
        c_html_parts.append(odds_parts)
    else:
        for item in odds_parts:
            if isinstance(item, tuple):
                _, heading, fig = item
                c_html_parts.append(_block(js.embed(fig), heading))
            else:
                c_html_parts.append(item)
    c_html_parts.append(_block(js.embed(F.fig_edge_over_time(result)), "Edge over time vs realized P&L"))
    section_c = f"""
    <section id="section-c" class="odds-section"><div class="container">
      <h2>C. The ODDS chart (Casino Secret pp.52-63)</h2>
      {"".join(c_html_parts)}
    </div></section>
    """

    # -- Section D --------------------------------------------------------------------
    cohort_dims = ["iv_rank", "market_state", "dte_entry", "short_delta", "year", "root"]
    cohort_blocks = "".join(
        _block(js.embed(F.fig_cohort_bars(result, dim)), f"by {dim}") for dim in cohort_dims
    )
    d_parts = [
        _block(js.embed(F.fig_win_rate_gauge(result)), "Win rate"),
        _block(js.embed(F.fig_pnl_distribution(result)), "P&L distribution"),
        cohort_blocks,
        _block(js.embed(F.fig_year_week_calendar(result)), "Year x week calendar"),
    ]
    comparison = extra.get("comparison")
    if comparison:
        d_parts.append(_block(js.embed(F.fig_strategy_comparison(comparison)), "Strategy comparison"))
    section_d = f"""
    <section id="section-d" class="odds-section"><div class="container">
      <h2>D. Aggregate diagnostics</h2>
      {"".join(d_parts)}
    </div></section>
    """

    # -- Section E --------------------------------------------------------------------
    config_json = html.escape(json.dumps(result.config.to_dict(), indent=2, default=str))
    manifest_json = html.escape(json.dumps(manifest, indent=2, default=str))
    rejected = manifest.get("rejected_trades") or []
    warnings = manifest.get("warnings") or []
    rejected_html = (
        "<ul>" + "".join(f"<li>{html.escape(str(r))}</li>" for r in rejected[:200]) + "</ul>"
        if rejected else '<p class="odds-empty">No rejected trades recorded.</p>'
    )
    warnings_html = (
        "<ul>" + "".join(f"<li>{html.escape(str(w))}</li>" for w in warnings[:200]) + "</ul>"
        if warnings else '<p class="odds-empty">No warnings.</p>'
    )
    section_e = f"""
    <section id="section-e" class="odds-section"><div class="container">
      <h2>E. Run manifest</h2>
      <div class="manifest-grid">
        <div>
          <h3>Config</h3>
          <pre>{config_json}</pre>
        </div>
        <div>
          <h3>Manifest</h3>
          <pre>{manifest_json}</pre>
          <h3>Data-quality warnings</h3>
          {warnings_html}
          <h3>Rejected trades</h3>
          {rejected_html}
        </div>
      </div>
    </div></section>
    """

    footer = f"""
    <footer class="odds-footer">
      Generated by odds_lab &middot; git {html.escape(str(manifest.get("git_sha", "n/a")))} &middot;
      {html.escape(str(manifest.get("generated_at", "")))}
    </footer>
    """

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>odds_lab report — {html.escape(", ".join(result.config.roots))}</title>
<style>{CSS}</style>
<script>{THEME_JS}</script>
<script>{TABLE_JS}</script>
</head>
<body>
{banner}
{nav}
{header}
{section_a}
{section_b}
{section_c}
{section_d}
{section_e}
{footer}
</body>
</html>
"""
    out_path.write_text(doc, encoding="utf-8")
    return out_path
