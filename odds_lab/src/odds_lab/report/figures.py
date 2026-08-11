"""Plotly figures for the odds_lab HTML report. STRATEGY.md §8.

One function per figure. Every function must build without error on an empty (zero-trade)
frame -- filters rejecting every candidate is a real backtest outcome, not a bug -- and must
never crash on missing optional columns. All figures share one palette/template (`TEMPLATE`)
so nothing is hand-colored per figure (ARCHITECTURE.md §5).

`fig_trade_table` is the one exception to "returns a Figure": it returns a self-contained
HTML fragment (a sortable/filterable table), because a JS-searchable table is not a plotly
chart. `report/build.py` and `report/assets.py` know how to embed it.
"""

from __future__ import annotations

import calendar
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing engine at runtime
    from odds_lab.engine.result import BacktestResult

__all__ = [
    "COLORS",
    "TEMPLATE",
    "fig_equity_curve",
    "fig_drawdown",
    "fig_monthly_heatmap",
    "fig_attribution_stack",
    "fig_trade_lifecycle",
    "fig_trade_table",
    "fig_odds_distribution",
    "fig_edge_over_time",
    "fig_win_rate_gauge",
    "fig_pnl_distribution",
    "fig_cohort_bars",
    "fig_year_week_calendar",
    "fig_strategy_comparison",
    "fig_selection_funnel",
    "fig_stress_scenario_pnl",
    "fig_stress_worst_position",
    "fig_stress_odds_overlay",
]

# --------------------------------------------------------------------------------------
# Shared palette + template -- one clean look, applied everywhere (ARCHITECTURE.md §5).
# Backgrounds are transparent so the surrounding page (light or dark) shows through;
# every color below is chosen to read on both.
# --------------------------------------------------------------------------------------

COLORS: dict[str, str] = {
    "equity": "#4C78A8",
    "benchmark": "#9CA3AF",
    "positive": "#2CA858",
    "negative": "#D64545",
    "warning": "#E0A62B",
    "short_strike": "#D64545",
    "long_strike": "#4C78A8",
    "theo": "#4C78A8",
    "empirical": "#E0A62B",
    "accent": "#7C6FD1",
    "grid": "rgba(128,128,128,0.25)",
    "text": "#8892A0",
}

DIVERGING = [
    [0.0, "#B23A48"],
    [0.35, "#E0A62B"],
    [0.5, "rgba(150,150,150,0.15)"],
    [0.65, "#7FB57F"],
    [1.0, "#1E7A3D"],
]

TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, Segoe UI, Roboto, sans-serif", color=COLORS["text"], size=13),
        colorway=[
            COLORS["equity"],
            COLORS["warning"],
            COLORS["positive"],
            COLORS["negative"],
            COLORS["accent"],
            COLORS["benchmark"],
        ],
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"], linecolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"], linecolor=COLORS["grid"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=40, t=60, b=50),
        hovermode="x unified",
    )
)


def _new_fig(title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template=TEMPLATE, title=title)
    return fig


def _empty_fig(title: str, message: str = "No trades / no data for this view") -> go.Figure:
    """A figure that renders cleanly for the zero-trade / missing-data case."""
    fig = go.Figure()
    fig.update_layout(
        template=TEMPLATE,
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    fig.add_annotation(
        text=message,
        showarrow=False,
        font=dict(size=16, color=COLORS["text"]),
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
    )
    return fig


# --------------------------------------------------------------------------------------
# A. Equity & attribution
# --------------------------------------------------------------------------------------


def _benchmark_series(result: "BacktestResult") -> pd.Series | None:
    """Buy-and-hold benchmark, reconstructed from the underlying prices recorded in
    `snapshots` (the result carries no independent full price series -- ChainStore is
    optional). Equal-weighted mean across roots/positions open on each date, scaled to
    start at the same equity as the strategy. Returns None if there's nothing to build it
    from (e.g. zero trades)."""
    snaps = result.snapshots
    eq = result.equity
    if snaps.empty or eq.empty:
        return None
    px = snaps.groupby("date")["underlying_price"].mean().sort_index()
    dates = pd.DatetimeIndex(eq["date"])
    px = px.reindex(px.index.union(dates)).sort_index().ffill().bfill()
    px = px.reindex(dates).ffill().bfill()
    if px.isna().all():
        return None
    start_equity = float(eq["equity"].iloc[0])
    scaled = px / float(px.iloc[0]) * start_equity
    scaled.index = dates
    return scaled


def fig_equity_curve(result: "BacktestResult") -> go.Figure:
    """Strategy equity vs a buy-and-hold benchmark of the traded underlying(s)."""
    eq = result.equity
    title = "Equity curve"
    if eq.empty:
        return _empty_fig(title)
    fig = _new_fig(title)
    fig.add_trace(
        go.Scatter(x=eq["date"], y=eq["equity"], name="Strategy equity", mode="lines",
                   line=dict(color=COLORS["equity"], width=2.5))
    )
    bench = _benchmark_series(result)
    if bench is not None:
        fig.add_trace(
            go.Scatter(x=bench.index, y=bench.values, name="Buy & hold (underlying)", mode="lines",
                       line=dict(color=COLORS["benchmark"], width=1.5, dash="dot"))
        )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=1.0,
                y=1.15,
                showactive=True,
                buttons=[
                    dict(label="Linear", method="relayout", args=[{"yaxis.type": "linear"}]),
                    dict(label="Log", method="relayout", args=[{"yaxis.type": "log"}]),
                ],
            )
        ],
    )
    return fig


def fig_drawdown(result: "BacktestResult") -> go.Figure:
    eq = result.equity
    title = "Drawdown"
    if eq.empty:
        return _empty_fig(title)
    running_max = eq["equity"].cummax()
    dd = (eq["equity"] / running_max.replace(0, np.nan) - 1.0) * 100
    fig = _new_fig(title)
    fig.add_trace(
        go.Scatter(x=eq["date"], y=dd, name="Drawdown", mode="lines", fill="tozeroy",
                   line=dict(color=COLORS["negative"], width=1.5))
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Drawdown (%)")
    return fig


def fig_monthly_heatmap(result: "BacktestResult") -> go.Figure:
    """Year x month returns, diverging colorscale centered at 0 (STRATEGY.md §8A)."""
    eq = result.equity
    title = "Monthly returns"
    if eq.empty or len(eq) < 2:
        return _empty_fig(title)
    s = eq.set_index("date")["equity"].sort_index()
    monthly = s.resample("ME").last()
    monthly_ret = monthly.pct_change().dropna() * 100
    if monthly_ret.empty:
        return _empty_fig(title, "not enough history for a monthly return")
    df = pd.DataFrame({"ret": monthly_ret})
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot_table(index="year", columns="month", values="ret")
    pivot = pivot.reindex(columns=range(1, 13))
    z = pivot.to_numpy()
    finite = z[np.isfinite(z)]
    zmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    zmax = zmax or 1.0
    fig = _new_fig(title)
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=[calendar.month_abbr[m] for m in range(1, 13)],
            y=[str(y) for y in pivot.index],
            colorscale=DIVERGING,
            zmid=0,
            zmin=-zmax,
            zmax=zmax,
            colorbar=dict(title="%"),
            hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="Year")
    return fig


_BRIDGE_COMPONENTS: list[tuple[str, str]] = [
    ("pnl_delta", "delta"),
    ("pnl_gamma", "gamma"),
    ("pnl_vega", "vega"),
    ("pnl_theta", "theta"),
    ("pnl_residual", "residual"),
    ("pnl_entry_execution", "entry execution"),
    ("pnl_exit_execution", "exit execution"),
    ("pnl_costs", "commission + fees"),
]
"""Every term in the exact P&L bridge (schema.TRADE_DTYPES docstring, STRATEGY.md §8A):
`pnl == sum of these`, computed per trade from its own inputs -- greeks explain mark
movement between recorded snapshots only, execution/costs cover everything a greek can
never see (the entry fill, the exit fill or settlement, commissions and fees)."""


def fig_attribution_stack(result: "BacktestResult") -> go.Figure:
    """The full P&L bridge as a waterfall: greeks, then execution, then costs, summing
    exactly to the realized P&L bar at the end (STRATEGY.md §8A). Aggregated across every
    trade in the run so the bars are legible; `fig_trade_table` carries the per-trade
    numbers for anyone who needs to check a single position.

    This used to be a cumulative-over-time stack of the greek terms plus an approximate
    "costs" (commission+fees+slippage) that was never reconciled against `pnl` -- on a
    real run that silently left ~5/6 of the money unexplained (an entry fill happens
    before the position is ever snapshotted, and an exit fill/settlement happens after
    the last one, so pure greek attribution structurally cannot see either). The waterfall
    below is exact by construction: every bar is one column of `result.trades`, and the
    running total after the last bar equals `sum(pnl)` because that is the bridge identity,
    not a coincidence of the chart.
    """
    trades = result.trades
    title = "P&L bridge: greeks -> execution -> costs -> realized P&L"
    if trades.empty:
        return _empty_fig(title)
    totals = {comp: float(trades[comp].fillna(0).sum()) for comp, _ in _BRIDGE_COMPONENTS}
    total_pnl = float(trades["pnl"].fillna(0).sum())

    labels = [label for _, label in _BRIDGE_COMPONENTS] + ["realized P&L"]
    values = [totals[comp] for comp, _ in _BRIDGE_COMPONENTS] + [total_pnl]
    measures = ["relative"] * len(_BRIDGE_COMPONENTS) + ["total"]

    fig = _new_fig(title)
    fig.add_trace(
        go.Waterfall(
            x=labels,
            y=values,
            measure=measures,
            increasing=dict(marker=dict(color=COLORS["positive"])),
            decreasing=dict(marker=dict(color=COLORS["negative"])),
            totals=dict(marker=dict(color=COLORS["equity"])),
            connector=dict(line=dict(color=COLORS["grid"])),
            text=[f"${v:,.0f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis_title=None,
        yaxis_title="$ P&L",
        showlegend=False,
    )
    return fig


# --------------------------------------------------------------------------------------
# B. Trade lifecycle -- the key screen
# --------------------------------------------------------------------------------------


def fig_trade_lifecycle(result: "BacktestResult", position_id: str, store=None) -> go.Figure:
    """Underlying price path with strike bands, entry/exit markers, MTM P&L, and the
    short-strike delta path for one trade. STRATEGY.md §8B -- "the market went through my
    short strike on day 14" must be visible in one glance.

    `store` (a ChainStore) is optional and, if given, may be used by callers to prepend a
    few days of pre-entry price context; the figure itself is fully derivable from
    `result.snapshots` alone.
    """
    trades, snaps = result.trades, result.snapshots
    title = f"Trade lifecycle — {position_id}"
    if trades.empty:
        return _empty_fig(title, "No trades in this run")
    trow_df = trades[trades["position_id"] == position_id]
    if trow_df.empty:
        return _empty_fig(title, f"position_id '{position_id}' not found")
    trow = trow_df.iloc[0]
    tsnaps = snaps[snaps["position_id"] == position_id].sort_values("date") if not snaps.empty else snaps
    if tsnaps.empty:
        return _empty_fig(title, f"no daily snapshots recorded for '{position_id}'")

    strategy = str(trow["strategy"])

    def _k(col: str) -> float | None:
        v = trow.get(col)
        return float(v) if pd.notna(v) else None

    # Draw every side the structure actually has. A two-sided structure (iron condor,
    # short strangle) has a loss zone below the short put AND above the short call;
    # showing only one of them would hide half the risk on exactly the trades where
    # the risk matters most.
    sides: list[dict] = []
    sp, lp = _k("short_put_strike"), _k("long_put_strike")
    sc, lc = _k("short_call_strike"), _k("long_call_strike")
    if sp is not None:
        sides.append({"short": sp, "long": lp, "up": False, "label": "put"})
    if sc is not None:
        sides.append({"short": sc, "long": lc, "up": True, "label": "call"})
    if not sides:  # single-sided legacy rows without the per-side columns populated
        sk = _k("short_strike")
        if sk is None:
            return _empty_fig(title, f"trade '{position_id}' has no strike recorded")
        sides.append({"short": sk, "long": _k("long_strike"), "up": "call" in strategy,
                      "label": "call" if "call" in strategy else "put"})

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.08,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        subplot_titles=(
            f"{trow['root']} underlying, strikes & MTM P&L ({strategy})",
            "Short-strike |delta|",
        ),
    )

    x = tsnaps["date"]
    price = tsnaps["underlying_price"]
    band_vals = [v for side in sides for v in (side["short"], side["long"]) if v is not None]
    y_lo = float(min(price.min(), *band_vals)) * 0.985
    y_hi = float(max(price.max(), *band_vals)) * 1.015

    # Profit zone = between the short strikes (or on the safe side of the only one).
    lo_edge = max([s["short"] for s in sides if not s["up"]], default=y_lo)
    hi_edge = min([s["short"] for s in sides if s["up"]], default=y_hi)
    if hi_edge > lo_edge:
        fig.add_hrect(y0=lo_edge, y1=hi_edge, fillcolor=COLORS["positive"], opacity=0.05,
                      line_width=0, row=1, col=1)
    for side in sides:
        lo, hi = (side["short"], y_hi) if side["up"] else (y_lo, side["short"])
        fig.add_hrect(y0=lo, y1=hi, fillcolor=COLORS["negative"], opacity=0.10,
                      line_width=0, row=1, col=1)
        fig.add_hline(y=side["short"], line=dict(color=COLORS["short_strike"], width=1.5, dash="dash"),
                      annotation_text=f"short {side['label']} {side['short']:g}",
                      annotation_position="right", row=1, col=1)
        if side["long"] is not None:
            fig.add_hline(y=side["long"], line=dict(color=COLORS["long_strike"], width=1.5, dash="dot"),
                          annotation_text=f"long {side['label']} {side['long']:g}",
                          annotation_position="right", row=1, col=1)

    fig.add_trace(
        go.Scatter(x=x, y=price, mode="lines", name="Underlying", line=dict(color=COLORS["text"], width=2.2)),
        row=1, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=[trow["entry_date"]], y=[trow["underlying_entry"]], mode="markers", name="Entry",
                   marker=dict(color=COLORS["positive"], size=12, symbol="triangle-up",
                               line=dict(width=1, color="white"))),
        row=1, col=1, secondary_y=False,
    )
    exit_x = trow["exit_date"] if pd.notna(trow["exit_date"]) else x.iloc[-1]
    exit_y = trow["underlying_exit"] if pd.notna(trow["underlying_exit"]) else price.iloc[-1]
    fig.add_trace(
        go.Scatter(x=[exit_x], y=[exit_y], mode="markers", name="Exit",
                   marker=dict(color=COLORS["negative"], size=12, symbol="triangle-down",
                               line=dict(width=1, color="white"))),
        row=1, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=x, y=tsnaps["mtm_pnl"], mode="lines", name="MTM P&L ($)",
                   line=dict(color=COLORS["accent"], width=2, dash="dash")),
        row=1, col=1, secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=x, y=tsnaps["short_delta"].abs(), mode="lines", name="|short delta|",
                   line=dict(color=COLORS["warning"], width=2)),
        row=2, col=1,
    )
    fig.add_hline(y=0.5, line=dict(color=COLORS["negative"], width=1, dash="dot"), row=2, col=1,
                  annotation_text="delta breach 0.50")

    reason = trow["exit_reason"] if pd.notna(trow["exit_reason"]) else "open"
    fig.add_annotation(
        x=exit_x, y=exit_y, text=f"exit: {reason}", showarrow=True, arrowhead=2,
        ax=0, ay=-45, row=1, col=1, bgcolor="rgba(20,20,20,0.55)", font=dict(color="white"),
    )

    fig.update_yaxes(title_text="Underlying price ($)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="MTM P&L ($)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="|delta|", range=[0, 1], row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(template=TEMPLATE, height=680, legend=dict(orientation="h", y=1.18, x=0))
    return fig


def fig_trade_table(result: "BacktestResult") -> str:
    """Sortable/filterable HTML table of every trade with the key diagnostics.

    Returns an HTML fragment, not a plotly Figure -- a searchable table is not a chart.
    Sorting/filtering JS lives in `report/assets.py` (`TABLE_JS`); this function only emits
    markup + data attributes the JS operates on.
    """
    trades = result.trades
    if trades.empty:
        return '<p class="odds-empty">No trades.</p>'

    cols = [
        "position_id", "root", "strategy", "entry_date", "exit_date", "dte_entry",
        "short_strike", "long_strike", "width", "entry_credit", "exit_debit", "pnl",
        "pnl_pct_of_max_loss", "exit_reason", "short_delta", "iv_entry", "iv_rank",
        "market_state", "edge_prob", "edge_ev",
    ]
    df = trades[cols].copy()
    for c in ("entry_date", "exit_date"):
        df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d").fillna("")
    for c in ("short_strike", "long_strike", "width", "entry_credit", "exit_debit", "pnl",
              "pnl_pct_of_max_loss", "short_delta", "iv_entry", "iv_rank", "edge_prob", "edge_ev"):
        df[c] = df[c].round(4)

    header = "".join(f'<th onclick="oddsSortTable({i})">{c}</th>' for i, c in enumerate(df.columns))
    rows = []
    for _, r in df.iterrows():
        pnl_val = r["pnl"]
        cls = "pnl-neg" if isinstance(pnl_val, (int, float)) and pnl_val < 0 else "pnl-pos"
        cells = "".join(f"<td>{r[c]}</td>" for c in df.columns)
        rows.append(f'<tr class="{cls}">{cells}</tr>')

    return (
        '<input type="text" class="trade-filter" placeholder="Filter trades (any column)..." '
        'oninput="oddsFilterTable(this, \'trade-table\')">'
        f'<div class="table-scroll"><table class="trade-table" id="trade-table">'
        f'<thead><tr>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


# --------------------------------------------------------------------------------------
# C. The ODDS chart -- STRATEGY.md §8C, Casino Secret pp.52-63
# --------------------------------------------------------------------------------------


def fig_odds_distribution(
    closes: pd.Series,
    horizon: int,
    iv: float,
    asof,
    *,
    strikes: list[float] | None = None,
) -> go.Figure:
    """Realized T-day log-return histogram vs the theoretical lognormal density implied
    by `iv`. STRATEGY.md §8C / Casino Secret pp.52-63.

    Bars beyond +-1 std are colored red where the empirical density exceeds theoretical
    (the tail region where selling premium loses) and green where theoretical exceeds
    empirical (the seller's edge). The theoretical density uses a risk-neutral,
    zero-drift assumption (mu_T = -sigma_T^2/2, r=q=0) since this figure takes no rate
    inputs -- it is a visualization of shape/tail mass, not the edge_ev computation in
    quant/edge.py, which does use the full r/q/empirical machinery.
    """
    title = f"ODDS chart — {horizon}d realized vs theoretical (asof {pd.Timestamp(asof).date()})"
    idx = pd.DatetimeIndex(closes.index)
    usable = closes[idx < pd.Timestamp(asof)]
    if len(usable) < horizon + 2:
        return _empty_fig(title, "insufficient price history before asof")

    vals = usable.to_numpy(dtype=float)
    daily = np.diff(np.log(vals))
    n = len(daily)
    if n < horizon:
        return _empty_fig(title, "insufficient price history before asof")
    csum = np.concatenate([[0.0], np.cumsum(daily)])
    log_returns = csum[horizon:] - csum[:-horizon]
    if len(log_returns) < 5:
        return _empty_fig(title, "not enough overlapping windows")

    sigma_T = float(iv) * np.sqrt(horizon / 252.0)
    sigma_T = max(sigma_T, 1e-9)
    mu_T = -0.5 * sigma_T**2

    nbins = int(np.clip(np.sqrt(len(log_returns)) * 2, 15, 60))
    counts, edges = np.histogram(log_returns, bins=nbins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    theo_at_centers = norm.pdf(centers, loc=mu_T, scale=sigma_T)
    z = (centers - mu_T) / sigma_T

    bar_colors = []
    for c_val, t_val, zz in zip(counts, theo_at_centers, z):
        if abs(zz) < 1.0:
            bar_colors.append(COLORS["empirical"])
        elif c_val > t_val:
            bar_colors.append(COLORS["negative"])  # actual > theoretical: seller loses here
        else:
            bar_colors.append(COLORS["positive"])  # theoretical > actual: seller's edge

    fig = _new_fig(title)
    fig.add_trace(
        go.Bar(x=centers, y=counts, width=(edges[1] - edges[0]) * 0.95, marker=dict(color=bar_colors),
               name="Realized (empirical)", opacity=0.85)
    )

    x_lo = min(centers.min(), mu_T - 4.2 * sigma_T)
    x_hi = max(centers.max(), mu_T + 4.2 * sigma_T)
    xs = np.linspace(x_lo, x_hi, 400)
    fig.add_trace(
        go.Scatter(x=xs, y=norm.pdf(xs, loc=mu_T, scale=sigma_T), mode="lines",
                   name=f"Theoretical lognormal (IV={float(iv):.0%})",
                   line=dict(color=COLORS["theo"], width=2.5))
    )

    S0 = float(usable.iloc[-1])
    if strikes:
        for k in strikes:
            xk = float(np.log(float(k) / S0))
            fig.add_vline(x=xk, line=dict(color=COLORS["accent"], width=1.5, dash="dot"),
                          annotation_text=f"K={k:g}", annotation_position="top")

    tick_lr = np.linspace(x_lo, x_hi, 9)
    pct_labels = [f"{lr:.3f}<br>{(np.exp(lr) - 1) * 100:+.1f}%" for lr in tick_lr]
    fig.update_xaxes(
        title_text="log return  (% price change below)",
        tickmode="array",
        tickvals=tick_lr,
        ticktext=pct_labels,
        range=[x_lo, x_hi],
    )

    sd_ticks = np.arange(-4, 5, 1)
    sd_vals = mu_T + sd_ticks * sigma_T
    fig.update_layout(
        xaxis2=dict(
            overlaying="x",
            side="top",
            tickmode="array",
            tickvals=sd_vals,
            ticktext=[f"{s:+d}σ" for s in sd_ticks],
            range=[x_lo, x_hi],
            title="standard deviations",
        ),
        yaxis_title="density",
        barmode="overlay",
    )
    return fig


def fig_edge_over_time(result: "BacktestResult") -> go.Figure:
    """edge_prob and edge_ev per trade over calendar time, with realized trade P&L
    overlaid on a secondary axis (STRATEGY.md §8C)."""
    trades = result.trades
    title = "Edge over time vs realized P&L"
    if trades.empty:
        return _empty_fig(title)
    df = trades.sort_values("entry_date")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df["entry_date"], y=df["edge_prob"], mode="markers+lines", name="edge_prob",
                   line=dict(color=COLORS["accent"])),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df["entry_date"], y=df["edge_ev"], mode="markers+lines", name="edge_ev ($)",
                   line=dict(color=COLORS["warning"])),
        secondary_y=False,
    )
    pnl = df["pnl"].fillna(0)
    bar_colors = np.where(pnl >= 0, COLORS["positive"], COLORS["negative"])
    fig.add_trace(
        go.Bar(x=df["entry_date"], y=pnl, name="realized P&L ($)", marker=dict(color=bar_colors), opacity=0.5),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="edge_prob / edge_ev", secondary_y=False)
    fig.update_yaxes(title_text="trade P&L ($)", secondary_y=True)
    fig.update_layout(template=TEMPLATE, title=title, xaxis_title="Entry date")
    return fig


# --------------------------------------------------------------------------------------
# D. Aggregate diagnostics
# --------------------------------------------------------------------------------------


def fig_win_rate_gauge(result: "BacktestResult") -> go.Figure:
    """Actual win rate vs the "80%" claim (STRATEGY.md §8D, *How to Win 80%*)."""
    trades = result.trades
    title = "Win rate vs the 80% claim"
    wr = float((trades["pnl"] > 0).mean()) * 100 if not trades.empty else 0.0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=wr,
            number={"suffix": "%"},
            delta={"reference": 80, "increasing": {"color": COLORS["positive"]},
                   "decreasing": {"color": COLORS["negative"]}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": COLORS["equity"]},
                "threshold": {"line": {"color": COLORS["negative"], "width": 3}, "value": 80},
                "steps": [
                    {"range": [0, 50], "color": "rgba(214,69,69,0.15)"},
                    {"range": [50, 80], "color": "rgba(224,166,43,0.15)"},
                    {"range": [80, 100], "color": "rgba(44,168,88,0.15)"},
                ],
            },
            title={"text": f"{title} (n={len(trades)})"},
        )
    )
    fig.update_layout(template=TEMPLATE)
    return fig


def fig_pnl_distribution(result: "BacktestResult") -> go.Figure:
    """Histogram of per-trade P&L -- the fat left tail should be obvious."""
    trades = result.trades
    title = "Per-trade P&L distribution"
    if trades.empty:
        return _empty_fig(title)
    pnl = trades["pnl"].dropna()
    if pnl.empty:
        return _empty_fig(title)
    fig = _new_fig(title)
    fig.add_trace(go.Histogram(x=pnl, nbinsx=40, marker=dict(color=COLORS["accent"]), name="P&L"))
    fig.add_vline(x=float(pnl.mean()), line=dict(color=COLORS["warning"], dash="dash"),
                  annotation_text="mean", annotation_position="top")
    fig.add_vline(x=float(pnl.median()), line=dict(color=COLORS["equity"], dash="dot"),
                  annotation_text="median", annotation_position="bottom")
    p05 = float(pnl.quantile(0.05))
    if p05 < pnl.min() + 1e-9 or True:
        fig.add_vrect(x0=float(pnl.min()) - 1, x1=p05, fillcolor=COLORS["negative"], opacity=0.12,
                      line_width=0, annotation_text="fat left tail (<= 5th pct)")
    fig.update_layout(xaxis_title="Trade P&L ($)", yaxis_title="count")
    return fig


_COHORT_BINS: dict[str, tuple[list[float], list[str]]] = {
    "iv_rank": ([0.0, 0.2, 0.4, 0.6, 0.8, 1.0001], ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]),
    "short_delta": ([0.0, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0], ["<.15", ".15-.20", ".20-.25", ".25-.30", ".30-.50", ">.50"]),
    "dte_entry": ([0, 21, 28, 35, 42, 49, 56, 1000], ["<21", "21-28", "28-35", "35-42", "42-49", "49-56", ">56"]),
}
_COHORT_CATEGORICAL = {"market_state", "year", "root"}


def fig_cohort_bars(result: "BacktestResult", by: str) -> go.Figure:
    """P&L bucketed by a cohort dimension. STRATEGY.md §8D.

    `by` in {'iv_rank', 'market_state', 'dte_entry', 'short_delta', 'year', 'root'}.
    """
    valid = set(_COHORT_BINS) | _COHORT_CATEGORICAL
    if by not in valid:
        raise ValueError(f"fig_cohort_bars: by must be one of {sorted(valid)}, got {by!r}")
    trades = result.trades
    title = f"P&L by {by}"
    if trades.empty:
        return _empty_fig(title)
    df = trades.copy()

    if by == "year":
        key = pd.to_datetime(df["entry_date"]).dt.year.astype("Int64").astype(str)
    elif by in ("market_state", "root"):
        key = df[by].astype(str)
    else:
        bins, labels = _COHORT_BINS[by]
        col = df[by].abs() if by == "short_delta" else df[by]
        key = pd.cut(col, bins=bins, labels=labels, include_lowest=True)

    grouped = (
        df.assign(_cohort=key)
        .groupby("_cohort", observed=True)["pnl"]
        .agg(mean="mean", total="sum", n="count")
        .dropna(how="all")
    )
    if grouped.empty:
        return _empty_fig(title)
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in grouped["mean"]]
    fig = _new_fig(title)
    fig.add_trace(
        go.Bar(
            x=[str(i) for i in grouped.index],
            y=grouped["mean"],
            marker=dict(color=colors),
            text=[f"n={int(n)}" for n in grouped["n"]],
            textposition="outside",
            name="mean P&L",
        )
    )
    fig.update_layout(xaxis_title=by, yaxis_title="mean trade P&L ($)")
    return fig


def fig_year_week_calendar(result: "BacktestResult") -> go.Figure:
    """Year x ISO-week grid colored by that cohort's P&L (STRATEGY.md §8D -- explicit
    per-week-per-year granularity request)."""
    trades = result.trades
    title = "P&L by ISO week / year"
    if trades.empty:
        return _empty_fig(title)
    dates = pd.to_datetime(trades["exit_date"].fillna(trades["entry_date"]))
    iso = dates.dt.isocalendar()
    df = pd.DataFrame({"year": iso["year"], "week": iso["week"], "pnl": trades["pnl"].to_numpy()})
    pivot = df.groupby(["year", "week"])["pnl"].sum().unstack("week")
    if pivot.empty:
        return _empty_fig(title)
    years = sorted(pivot.index)
    weeks = list(range(1, 54))
    pivot = pivot.reindex(index=years, columns=weeks)
    z = pivot.to_numpy()
    finite = z[np.isfinite(z)]
    zmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    zmax = zmax or 1.0
    fig = _new_fig(title)
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=weeks,
            y=[str(y) for y in years],
            colorscale=DIVERGING,
            zmid=0,
            zmin=-zmax,
            zmax=zmax,
            colorbar=dict(title="$"),
            hovertemplate="year %{y} week %{x}: $%{z:.0f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="ISO week", yaxis_title="Year")
    return fig


def fig_strategy_comparison(results: dict[str, "BacktestResult"]) -> go.Figure:
    """Naive +-5% vs delta-selected vs ODDS-edge-filtered -- equity curves overlaid,
    each indexed to 100 at its own start so shapes compare regardless of starting capital."""
    title = "Strategy comparison"
    if not results:
        return _empty_fig(title, "no runs to compare")
    fig = _new_fig(title)
    plotted = 0
    for name, res in results.items():
        eq = res.equity
        if eq.empty:
            continue
        base = float(eq["equity"].iloc[0]) or 1.0
        norm_eq = eq["equity"] / base * 100.0
        fig.add_trace(go.Scatter(x=eq["date"], y=norm_eq, mode="lines", name=name))
        plotted += 1
    if plotted == 0:
        return _empty_fig(title, "no runs with equity history to compare")
    fig.update_layout(xaxis_title="Date", yaxis_title="Equity (indexed to 100 at start)")
    return fig


# --------------------------------------------------------------------------------------
# Selection funnel -- "why did propose_trade discard almost everything" (STRATEGY.md §8)
# --------------------------------------------------------------------------------------


def fig_stress_scenario_pnl(comparisons: pd.DataFrame) -> go.Figure:
    """Scenario P&L, defined-risk vs undefined-risk, side by side -- the headline output
    of `engine.stress` (STRATEGY.md §10.3 / §2.1)."""
    title = "Crisis stress: defined-risk vs undefined-risk P&L per scenario"
    if comparisons is None or comparisons.empty:
        return _empty_fig(title, "No stress scenarios were built -- see the manifest for why each was skipped")
    df = comparisons.sort_values("scenario")
    fig = _new_fig(title)
    fig.add_trace(
        go.Bar(x=df["scenario"], y=df["defined_pnl"], name="Defined-risk (spread)",
               marker=dict(color=COLORS["equity"]),
               text=[f"n={int(n)}" for n in df["n_positions_defined"]], textposition="outside")
    )
    fig.add_trace(
        go.Bar(x=df["scenario"], y=df["undefined_pnl"], name="Undefined-risk (naked)",
               marker=dict(color=COLORS["negative"]),
               text=[f"n={int(n)}" for n in df["n_positions_undefined"]], textposition="outside")
    )
    fig.update_layout(barmode="group", xaxis_title="scenario", yaxis_title="total replayed P&L ($)")
    return fig


def fig_stress_worst_position(comparisons: pd.DataFrame) -> go.Figure:
    """Worst single position's P&L per scenario, defined vs undefined -- the "how bad can
    one trade get" waterfall STRATEGY.md §10.3 asks for."""
    title = "Crisis stress: worst single position per scenario"
    if comparisons is None or comparisons.empty:
        return _empty_fig(title, "No stress scenarios were built")
    df = comparisons.sort_values("undefined_worst_position")
    labels = list(df["scenario"])
    fig = _new_fig(title)
    fig.add_trace(
        go.Waterfall(
            name="defined-risk worst position", x=[f"{s} (defined)" for s in labels], y=df["defined_worst_position"],
            measure=["relative"] * len(labels),
            increasing=dict(marker=dict(color=COLORS["positive"])),
            decreasing=dict(marker=dict(color=COLORS["warning"])),
            text=[f"${v:,.0f}" for v in df["defined_worst_position"]], textposition="outside",
        )
    )
    fig.add_trace(
        go.Waterfall(
            name="undefined-risk worst position", x=[f"{s} (undefined)" for s in labels], y=df["undefined_worst_position"],
            measure=["relative"] * len(labels),
            increasing=dict(marker=dict(color=COLORS["positive"])),
            decreasing=dict(marker=dict(color=COLORS["negative"])),
            text=[f"${v:,.0f}" for v in df["undefined_worst_position"]], textposition="outside",
        )
    )
    fig.update_layout(yaxis_title="worst single position P&L ($)", showlegend=True)
    return fig


def fig_stress_odds_overlay(
    closes: pd.Series, horizon: int, iv: float, asof, scenarios: dict, *, strikes: list[float] | None = None,
) -> go.Figure:
    """The ODDS chart (`fig_odds_distribution`) with each crisis scenario's `horizon`-day
    cumulative log return overlaid as a vertical marker, so the user can see where the
    crisis sits relative to the strikes they were selling (STRATEGY.md §10.5 / §8C)."""
    fig = fig_odds_distribution(closes, horizon, iv, asof, strikes=strikes)
    if not scenarios:
        return fig
    for name, sc in scenarios.items():
        n = min(horizon, sc.n_days)
        if n <= 0:
            continue
        lr = float(np.sum(sc.daily_log_returns[:n]))
        fig.add_vline(
            x=lr, line=dict(color=COLORS["negative"], width=2, dash="dashdot"),
            annotation_text=name, annotation_position="bottom", annotation_textangle=-90,
        )
    fig.update_layout(title=fig.layout.title.text + " + crisis scenario overlays")
    return fig


def fig_selection_funnel(result: "BacktestResult") -> go.Figure:
    """Horizontal bar chart of the selection funnel from `manifest['selection_funnel']`:
    every expiry candidate evaluated across all entry opportunities is either accepted
    (a trade taken) or rejected for exactly one named reason -- so the bars below are a
    partition of `candidates_evaluated` (accepted + every rejection reason, mutually
    exclusive by construction), sorted by how many candidates each reason discarded.
    `candidates_evaluated` is a finer count than `opportunities` (one (root, date) entry
    decision can evaluate several expiries, e.g. contributing one rejection per expiry)."""
    title = "Selection funnel: why candidates were discarded"
    funnel = (result.manifest or {}).get("selection_funnel") or {}
    opportunities = int(funnel.get("opportunities", 0))
    if opportunities == 0:
        return _empty_fig(title, "No entry opportunities were evaluated (empty date range or no roots).")

    accepted = int(funnel.get("accepted", 0))
    rejected: dict = funnel.get("rejected", {})
    total = int(funnel.get("candidates_evaluated", accepted + sum(rejected.values()))) or 1
    reasons = sorted(((r, int(n)) for r, n in rejected.items() if n > 0), key=lambda kv: kv[1])

    labels = ["trades taken"] + [r for r, _ in reasons]
    values = [accepted] + [n for _, n in reasons]
    colors = [COLORS["positive"]] + [COLORS["negative"]] * len(reasons)

    fig = _new_fig(title)
    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v} ({v / total * 100:.0f}%)" for v in values],
            textposition="outside",
            hovertemplate="%{y}: %{x} of " + str(total) + " candidates evaluated<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{title}<br><sup>{opportunities} entry opportunities -> {total} expiry candidates evaluated -> {accepted} accepted</sup>",
        xaxis_title=f"candidates (of {total} evaluated)",
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        margin=dict(l=180, r=60, t=80, b=50),
        height=max(320, 40 * (len(labels) + 2)),
    )
    return fig
