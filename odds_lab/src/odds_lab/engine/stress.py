"""Crisis stress module -- STRATEGY.md §10.

Answers: *what would the positions this backtest actually held have done, had the
underlying moved the way it did in a crisis the option data does not cover?*

The user's option-chain history starts ~2012-2013 (no 2008 chain data exists), but the
empirical distribution's "Count" step (STRATEGY.md §2 step 2, `quant/empirical.py`) is
built from UNDERLYING CLOSES ONLY -- free, and available back to 1993 for SPY / the
1950s for the S&P 500 index. This module supplies 2008-class scenarios from that cheap,
long-history data:

  1. A scenario library of REALIZED underlying paths (never invented shocks), built only
     from a caller-supplied close series -- a scenario whose window the series does not
     cover is skipped and reported, never fabricated.
  2. Position replay: re-price every position the backtest actually opened along a
     scenario's path from its own entry date, using `quant.bs` and the SAME exit policy
     the run used (`strategy.exits.evaluate_exit`, unmodified).
  3. The defined-vs-undefined-risk comparison STRATEGY.md §2.1 predicts diverges only
     under a crash jump -- computed on real crisis paths, not a stylized one.
  4. `run_backtest_with_long_history`: re-runs the whole backtest with the empirical
     distribution fed from a long underlying series (so 2015 decisions are gated by a
     "Count" that includes 2008), compared against the short-window default.

## The IV-mapping assumption (STRATEGY.md §10.2, the single biggest one in this module)

A scenario only tells us how the UNDERLYING moved. To re-price an option along that path
we need an implied vol at every step, and the market's IV during a real crisis is not
simply "whatever the trailing realized vol of that crisis path is" -- IV is the forward-
looking, panicked price, and it historically OVERSHOOTS realized vol (VIX far above
trailing SPX realized vol at the Oct 2008 peak, e.g. VIX ~80 vs trailing-30d RV ~65-70).

`scenario_iv` maps this as:

    vol_expansion  = scenario.realized_vol / scenario.baseline_vol   (crisis RV / calm RV,
                                                                       same underlying series)
    iv_scenario    = iv_entry * vol_expansion * beta

`beta` is the calibratable IV/RV overshoot multiplier, default `DEFAULT_IV_BETA = 1.3`
(30% overshoot). This is parameterised, not buried: every `run_stress` call also computes
`beta_sensitivity` across `IV_BETA_SENSITIVITY_RANGE = (0.8, 1.0, 1.3, 1.6, 2.0)` so the
report shows how much the headline P&L numbers move as this one assumption is varied.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from odds_lab import schema
from odds_lab.config import BacktestConfig, ExitConfig
from odds_lab.quant import bs as bs_mod
from odds_lab.schema import CONTRACT_MULTIPLIER
from odds_lab.strategy import exits as exits_mod

__all__ = [
    "Scenario",
    "ScenarioLibrary",
    "PositionReplay",
    "StressResult",
    "SCENARIO_WINDOWS",
    "DEFAULT_SCENARIO_NAMES",
    "DEFAULT_IV_BETA",
    "IV_BETA_SENSITIVITY_RANGE",
    "load_underlying_history",
    "build_scenarios",
    "scenario_iv",
    "replay_position",
    "run_stress",
    "LongHistoryStore",
    "run_backtest_with_long_history",
]


# ==========================================================================================
# 1. Historical scenario library
# ==========================================================================================

SCENARIO_WINDOWS: dict[str, tuple[date, date, str, str]] = {
    # name -> (start, end, category, description). "historical": out-of-sample crises the
    # user's option data cannot reach. "calibration": crises the user's data DOES cover
    # (2018-02, 2020-03), included so the replay's numbers can be sanity-checked against
    # what actually happened in the real backtest for the same period.
    "1987-10": (date(1987, 10, 1), date(1987, 10, 30), "historical", "Black Monday (Oct 19 1987 crash)"),
    "2000-02": (date(2000, 3, 1), date(2002, 10, 31), "historical", "dot-com bear market"),
    "2008-09": (date(2008, 9, 1), date(2009, 3, 31), "historical", "GFC crash (Lehman -> trough)"),
    "2011-08": (date(2011, 7, 15), date(2011, 10, 15), "historical", "US downgrade / euro-crisis air pocket"),
    "2018-02": (date(2018, 1, 26), date(2018, 2, 12), "calibration", "Feb 2018 \"volmageddon\" (in the user's data)"),
    "2020-03": (date(2020, 2, 15), date(2020, 4, 15), "calibration", "COVID crash (in the user's data)"),
}
_2008_SUBWINDOWS: tuple[int, ...] = (21, 30, 45)
"""STRATEGY.md §10: the full 2008-09 drawdown, plus its worst 21/30/45-trading-day
windows -- the horizons that actually matter for a 21-56 DTE credit spread."""

DEFAULT_SCENARIO_NAMES: tuple[str, ...] = tuple(SCENARIO_WINDOWS) + tuple(
    f"2008-09-worst-{w}d" for w in _2008_SUBWINDOWS
)

_BASELINE_WINDOW = 60
"""Trading days of pre-scenario history used as the 'calm' realized-vol baseline for
`vol_expansion` -- STRATEGY.md §10.2."""


@dataclass(frozen=True)
class Scenario:
    """A realized underlying path plus its realized-vol/IV context (STRATEGY.md §10.1)."""

    name: str
    category: str
    description: str
    start: date
    end: date
    daily_log_returns: np.ndarray
    source_closes: pd.Series
    realized_vol: float
    baseline_vol: float
    vol_expansion: float

    @property
    def n_days(self) -> int:
        return len(self.daily_log_returns)


@dataclass
class ScenarioLibrary:
    scenarios: dict[str, Scenario] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    """name -> reason it could NOT be built (never fabricated) -- e.g. 'underlying
    history does not cover this window'."""


def load_underlying_history(path: str | Path) -> pd.Series:
    """Read a long underlying-close history CSV for stress scenarios (STRATEGY.md §10.1).

    Free sources (this is the whole point -- 2008 costs nothing to obtain here, only the
    option chains do): SPY daily closes back to 1993 from Stooq
    (https://stooq.com/q/d/l/?s=spy.us&i=d) or Yahoo Finance; the S&P 500 index itself
    (^GSPC / SPX) back to the 1950s from Stooq (`^spx`) or a downloaded Yahoo/Cboe series
    -- this is exactly the series *Casino Secret* pp.49-65 charts.

    Expects `date` and `close` columns (case-insensitive; Stooq's `Date`/`Close` and
    Yahoo's `Date`/`Close`/`Adj Close` all match -- plain `close` is preferred over
    `adj close` when both are present, since dividend adjustment shifts the return
    series' level, not its crash *shape*, and everything downstream works in log
    returns only). Raises ValueError naming exactly what's missing -- never guesses.
    """
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    date_col = cols.get("date")
    close_col = cols.get("close") or cols.get("close*") or cols.get("adj close")
    if date_col is None or close_col is None:
        raise ValueError(
            f"load_underlying_history({path}): need 'date' and 'close' (or 'adj close') "
            f"columns, got {list(df.columns)}"
        )
    s = pd.Series(
        pd.to_numeric(df[close_col], errors="coerce").to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(df[date_col])),
        name="close",
    ).sort_index()
    s = s[~s.index.duplicated(keep="last")].dropna()
    if s.empty:
        raise ValueError(f"load_underlying_history({path}): no usable rows")
    return s


def _window_slice(closes: pd.Series, start: date, end: date) -> pd.Series:
    idx = pd.DatetimeIndex(closes.index)
    return closes[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))].sort_index()


def _annualized_vol(log_returns: np.ndarray) -> float:
    if len(log_returns) < 2:
        return float("nan")
    return float(np.std(log_returns, ddof=1) * np.sqrt(252.0))


def _baseline_vol(closes: pd.Series, before: date, window: int = _BASELINE_WINDOW) -> float | None:
    idx = pd.DatetimeIndex(closes.index)
    pre = closes[idx < pd.Timestamp(before)].sort_index()
    if len(pre) < window + 1:
        return None
    vals = pre.iloc[-(window + 1):].to_numpy(dtype=float)
    return _annualized_vol(np.diff(np.log(vals)))


def _scenario_from_slice(name: str, category: str, description: str, closes: pd.Series, sub: pd.Series) -> Scenario:
    vals = sub.to_numpy(dtype=float)
    log_rets = np.diff(np.log(vals))
    rv = _annualized_vol(log_rets)
    base = _baseline_vol(closes, sub.index.min().date()) or (rv if rv and np.isfinite(rv) and rv > 0 else 1.0)
    vol_expansion = rv / base if base and np.isfinite(rv) and base > 0 else float("nan")
    return Scenario(
        name=name, category=category, description=description,
        start=sub.index.min().date(), end=sub.index.max().date(),
        daily_log_returns=log_rets, source_closes=sub,
        realized_vol=rv, baseline_vol=base, vol_expansion=vol_expansion,
    )


def _build_window_scenario(closes: pd.Series, name: str, category: str, description: str, start: date, end: date) -> Scenario | None:
    sub = _window_slice(closes, start, end)
    if len(sub) < 3:
        return None
    return _scenario_from_slice(name, category, description, closes, sub)


def _build_worst_subwindow(closes: pd.Series, base_start: date, base_end: date, window: int, name: str) -> Scenario | None:
    """Worst (most negative cumulative log return) `window`-trading-day slice inside
    [base_start, base_end] -- STRATEGY.md §10.1."""
    sub = _window_slice(closes, base_start, base_end)
    if len(sub) < window + 1:
        return None
    vals = sub.to_numpy(dtype=float)
    log_rets = np.diff(np.log(vals))
    csum = np.concatenate([[0.0], np.cumsum(log_rets)])
    windows = csum[window:] - csum[:-window]
    worst_i = int(np.argmin(windows))
    sub_slice = sub.iloc[worst_i: worst_i + window + 1]
    return _scenario_from_slice(
        name, "historical", f"worst {window}-trading-day window inside {base_start}..{base_end}",
        closes, sub_slice,
    )


def build_scenarios(closes: pd.Series, names: Sequence[str] | None = None) -> ScenarioLibrary:
    """Build the historical scenario library from a supplied underlying-close series
    (STRATEGY.md §10.1). NEVER invents a scenario: a window the series does not cover
    (or covers too thinly to form log returns) is recorded in `.skipped`, with a reason,
    and simply excluded from `.scenarios` -- it is never approximated from a different
    window or from remembered/hardcoded numbers.
    """
    wanted = list(names) if names is not None else list(DEFAULT_SCENARIO_NAMES)
    out: dict[str, Scenario] = {}
    skipped: dict[str, str] = {}
    for name in wanted:
        if name in SCENARIO_WINDOWS:
            start, end, category, desc = SCENARIO_WINDOWS[name]
            sc = _build_window_scenario(closes, name, category, desc, start, end)
        elif name.startswith("2008-09-worst-") and name.endswith("d"):
            try:
                window = int(name[len("2008-09-worst-"):-1])
            except ValueError:
                skipped[name] = "unrecognized scenario name"
                continue
            base_start, base_end, _, _ = SCENARIO_WINDOWS["2008-09"]
            sc = _build_worst_subwindow(closes, base_start, base_end, window, name)
        else:
            skipped[name] = "unrecognized scenario name"
            continue
        if sc is None:
            skipped[name] = "underlying history does not cover this window (or covers it too thinly)"
        else:
            out[name] = sc
    return ScenarioLibrary(scenarios=out, skipped=skipped)


# ==========================================================================================
# 2. IV mapping (STRATEGY.md §10.2 -- see module docstring)
# ==========================================================================================

DEFAULT_IV_BETA = 1.3
IV_BETA_SENSITIVITY_RANGE: tuple[float, ...] = (0.8, 1.0, 1.3, 1.6, 2.0)


def scenario_iv(iv_entry: float, scenario: Scenario, beta: float = DEFAULT_IV_BETA) -> float:
    """Map a scenario's realized-vol expansion onto the position's entry IV -- the single
    biggest modelling assumption in this module. See the module docstring for the formula
    and rationale: `iv_entry * vol_expansion * beta`, floored at 1bp of vol so pricing
    never divides by ~0."""
    vol_expansion = scenario.vol_expansion
    if not np.isfinite(vol_expansion) or vol_expansion <= 0:
        vol_expansion = 1.0
    return float(max(iv_entry * vol_expansion * beta, 1e-4))


# ==========================================================================================
# 3. Position replay
# ==========================================================================================


def _legs_from_trade_row(row: pd.Series, *, naked: bool) -> list[schema.Leg]:
    """Reconstruct a position's legs from a saved `trades.parquet` row -- the inverse of
    `engine.loop._trade_row`'s `short_*_strike`/`long_*_strike` columns (TRADE_DTYPES does
    not carry raw `Leg` objects). `naked=True` drops every long (protective) leg, giving
    the undefined-risk sibling of the SAME short strike(s) the backtest actually sold --
    this is the S1/S2/S3-vs-S4/S5 comparison STRATEGY.md §2.1 calls for, built from the
    position actually taken, never a fresh selection."""
    root = str(row.get("root", ""))
    expiry = row["expiry"]
    legs: list[schema.Leg] = []
    for short_col, long_col, right in (
        ("short_put_strike", "long_put_strike", "P"),
        ("short_call_strike", "long_call_strike", "C"),
    ):
        sk = row.get(short_col)
        if sk is None or pd.isna(sk):
            continue
        legs.append(schema.Leg(root=root, expiry=expiry, strike=float(sk), right=right, ratio=-1))
        lk = row.get(long_col)
        if not naked and lk is not None and pd.notna(lk):
            legs.append(schema.Leg(root=root, expiry=expiry, strike=float(lk), right=right, ratio=1))
    if legs:
        return legs
    # legacy single-sided rows without the per-side columns populated
    sk = row.get("short_strike")
    if sk is None or pd.isna(sk):
        return []
    right = "P" if "put" in str(row.get("strategy", "")) else "C"
    legs.append(schema.Leg(root=root, expiry=expiry, strike=float(sk), right=right, ratio=-1))
    lk = row.get("long_strike")
    if not naked and lk is not None and pd.notna(lk):
        legs.append(schema.Leg(root=root, expiry=expiry, strike=float(lk), right=right, ratio=1))
    return legs


def _naked_reg_t_series(legs: list[schema.Leg], S_path: np.ndarray, credit: float) -> np.ndarray:
    """Reg-T margin per contract at every point of `S_path` -- STRATEGY.md §6, the same
    formula as `engine.portfolio.Portfolio.reg_t_margin`, evaluated vectorized along a
    whole path (not just at entry). Duplicated rather than imported: that method is an
    instance method of a stateful `Portfolio` and needs no state at all for this
    calculation; constructing one just to call it once per path point would be pure
    overhead. Both implementations must be kept in sync if STRATEGY.md §6 changes."""
    by_right: dict[str, list[schema.Leg]] = {"P": [], "C": []}
    for leg in legs:
        by_right.setdefault(leg.right, []).append(leg)
    side_reqs = []
    for right, side_legs in by_right.items():
        shorts = [x for x in side_legs if x.ratio < 0]
        if not shorts or any(x.ratio > 0 for x in side_legs):
            continue
        worst = np.zeros_like(S_path)
        for leg in shorts:
            otm = (
                np.maximum(S_path - leg.strike, 0.0)
                if right == "P"
                else np.maximum(leg.strike - S_path, 0.0)
            )
            worst = np.maximum(worst, np.maximum(0.20 * S_path - otm, 0.10 * S_path) * 100.0)
        side_reqs.append(worst)
    if not side_reqs:
        return np.zeros_like(S_path)
    return np.maximum(np.max(np.stack(side_reqs), axis=0) + credit * 100.0, 250.0)


@dataclass(frozen=True)
class PositionReplay:
    position_id: str
    scenario: str
    variant: str  # 'defined' | 'undefined'
    root: str
    entry_date: date
    qty: int
    pnl: float
    max_drawdown: float
    breached_short_strike: bool
    exit_day: int
    exit_reason: str
    margin_peak: float
    beta: float


def replay_position(
    row: pd.Series,
    scenario: Scenario,
    *,
    naked: bool,
    exits_cfg: ExitConfig,
    r: float = 0.02,
    q: float = 0.0,
    beta: float = DEFAULT_IV_BETA,
) -> PositionReplay | None:
    """Re-price ONE position the backtest actually opened along ONE scenario's realized
    path, from its own entry date, using `quant.bs` with entry IV scaled per
    `scenario_iv` (STRATEGY.md §10.2) and the SAME exit policy the run used
    (`strategy.exits.evaluate_exit`, unmodified). Returns None if the row lacks what's
    needed to reconstruct a priceable position -- never fabricates one.
    """
    legs = _legs_from_trade_row(row, naked=naked)
    if not legs:
        return None
    S0, iv0, credit, dte_entry = row.get("underlying_entry"), row.get("iv_entry"), row.get("entry_credit"), row.get("dte_entry")
    if any(v is None or (isinstance(v, float) and not np.isfinite(v)) for v in (S0, iv0, credit, dte_entry)):
        return None
    S0, iv0, credit = float(S0), float(iv0), float(credit)
    dte_entry = int(dte_entry)
    qty = int(row.get("qty", 1) or 1)
    if dte_entry <= 0 or iv0 <= 0 or S0 <= 0 or qty <= 0:
        return None

    iv_scen = scenario_iv(iv0, scenario, beta=beta)
    n = min(scenario.n_days, dte_entry)
    if n <= 0:
        return None
    cum = np.concatenate([[0.0], np.cumsum(scenario.daily_log_returns[:n])])
    S_path = S0 * np.exp(cum)  # length n+1, S_path[0] == S0

    max_loss = row.get("max_loss")
    if naked:
        margin_series = _naked_reg_t_series(legs, S_path, credit)
        margin_peak = float(np.max(margin_series)) * qty
    else:
        margin_peak = (
            float(max_loss) * qty * CONTRACT_MULTIPLIER
            if max_loss is not None and np.isfinite(max_loss)
            else float("nan")
        )

    fake_pos = schema.Position(
        position_id=str(row["position_id"]), strategy=str(row.get("strategy", "")), root=str(row.get("root", "")),
        entry_date=row["entry_date"], expiry=row["expiry"], legs=legs, qty=qty, entry_credit=credit,
    )

    running_peak = 0.0
    max_dd = 0.0
    breached = False
    exit_day = n
    exit_reason = "expiry"
    final_mark = 0.0

    for t in range(1, n + 1):
        S_t = float(S_path[t])
        dte_remaining = max(dte_entry - t, 0)
        T_years = dte_remaining / 365.0
        mark = 0.0
        short_delta = None
        for leg in legs:
            price = float(bs_mod.bs_price(S_t, leg.strike, T_years, r, q, iv_scen, leg.right))
            mark += (-leg.ratio) * price
            if leg.ratio < 0:
                itm = (S_t > leg.strike) if leg.right == "C" else (S_t < leg.strike)
                breached = breached or itm
                greeks = bs_mod.bs_greeks(S_t, leg.strike, T_years, r, q, iv_scen, leg.right)
                short_delta = float(greeks["delta"])
        final_mark = mark
        mtm = (credit - mark) * qty * CONTRACT_MULTIPLIER
        running_peak = max(running_peak, mtm)
        max_dd = min(max_dd, mtm - running_peak)

        if dte_remaining <= 0:
            exit_day, exit_reason = t, "expiry"
            break
        reason = exits_mod.evaluate_exit(fake_pos, {"mark": mark, "short_delta": short_delta, "dte": dte_remaining}, exits_cfg)
        if reason is not None:
            exit_day, exit_reason = t, reason.value
            break
    else:
        exit_reason = "expiry" if n >= dte_entry else "scenario_path_exhausted"

    pnl = (credit - final_mark) * qty * CONTRACT_MULTIPLIER
    return PositionReplay(
        position_id=str(row["position_id"]), scenario=scenario.name,
        variant="undefined" if naked else "defined", root=str(row.get("root", "")),
        entry_date=row["entry_date"], qty=qty, pnl=pnl, max_drawdown=max_dd,
        breached_short_strike=breached, exit_day=exit_day, exit_reason=exit_reason,
        margin_peak=margin_peak, beta=beta,
    )


# ==========================================================================================
# The comparison that matters + the aggregate result
# ==========================================================================================

_REPLAY_COLS = [
    "position_id", "scenario", "variant", "root", "entry_date", "qty", "pnl",
    "max_drawdown", "breached_short_strike", "exit_day", "exit_reason", "margin_peak", "beta",
]


def _replays_to_frame(replays: list[PositionReplay]) -> pd.DataFrame:
    if not replays:
        return pd.DataFrame(columns=_REPLAY_COLS)
    return pd.DataFrame([r.__dict__ for r in replays])[_REPLAY_COLS]


def _comparisons_from_replays(replays: pd.DataFrame, scenarios: ScenarioLibrary) -> pd.DataFrame:
    """STRATEGY.md §2.1's headline comparison: defined-risk vs undefined-risk P&L, per
    scenario, on the SAME positions and the SAME crisis path."""
    rows = []
    if not replays.empty:
        base = replays[np.isclose(replays["beta"], DEFAULT_IV_BETA)]
    else:
        base = replays
    for name, sc in scenarios.scenarios.items():
        sub = base[base["scenario"] == name] if not base.empty else base
        row: dict = {"scenario": name, "category": sc.category, "description": sc.description}
        for variant, prefix in (("defined", "defined"), ("undefined", "undefined")):
            vsub = sub[sub["variant"] == variant] if not sub.empty else sub
            n = int(len(vsub))
            row[f"n_positions_{prefix}"] = n
            row[f"{prefix}_pnl"] = float(vsub["pnl"].sum()) if n else 0.0
            row[f"{prefix}_max_drawdown"] = float(vsub["max_drawdown"].min()) if n else 0.0
            row[f"{prefix}_worst_position"] = float(vsub["pnl"].min()) if n else float("nan")
            finite_margin = vsub["margin_peak"].replace([np.inf, -np.inf], np.nan).dropna()
            row[f"{prefix}_margin_peak"] = float(finite_margin.sum()) if len(finite_margin) else float("nan")
            row[f"n_breached_short_{prefix}"] = int(vsub["breached_short_strike"].sum()) if n else 0
        row["divergence"] = row["undefined_pnl"] - row["defined_pnl"]
        rows.append(row)
    return pd.DataFrame(rows)


def _beta_sensitivity(replays: pd.DataFrame) -> pd.DataFrame:
    if replays.empty:
        return pd.DataFrame(columns=["scenario", "variant", "beta", "total_pnl"])
    g = replays.groupby(["scenario", "variant", "beta"], as_index=False)["pnl"].sum()
    return g.rename(columns={"pnl": "total_pnl"})


@dataclass
class StressResult:
    scenarios: ScenarioLibrary
    replays: pd.DataFrame
    comparisons: pd.DataFrame
    beta_sensitivity: pd.DataFrame
    long_history_edge: dict | None = None
    manifest: dict = field(default_factory=dict)

    def save(self, run_dir: str | Path) -> Path:
        out = Path(run_dir) / "stress"
        out.mkdir(parents=True, exist_ok=True)
        self.replays.to_parquet(out / "replays.parquet")
        self.comparisons.to_parquet(out / "comparisons.parquet")
        self.beta_sensitivity.to_parquet(out / "beta_sensitivity.parquet")
        import json

        (out / "manifest.json").write_text(
            json.dumps(
                {
                    "scenarios_built": sorted(self.scenarios.scenarios),
                    "scenarios_skipped": self.scenarios.skipped,
                    "long_history_edge": self.long_history_edge,
                    **self.manifest,
                },
                indent=2,
                default=str,
            )
        )
        return out


def run_stress(
    trades: pd.DataFrame,
    scenarios: ScenarioLibrary,
    exits_cfg: ExitConfig,
    *,
    r: float = 0.02,
    q: float = 0.0,
    betas: Sequence[float] = IV_BETA_SENSITIVITY_RANGE,
) -> StressResult:
    """Replay every trade in `trades` (a saved run's `trades.parquet`) through every
    scenario, defined-risk and undefined-risk, at every beta in `betas` (the IV-mapping
    sensitivity sweep) -- STRATEGY.md §10.3. `DEFAULT_IV_BETA` must be included in
    `betas` for the headline comparison table to be populated."""
    replays: list[PositionReplay] = []
    if not trades.empty:
        for _, row in trades.iterrows():
            for sc in scenarios.scenarios.values():
                for beta in betas:
                    for naked in (False, True):
                        rep = replay_position(row, sc, naked=naked, exits_cfg=exits_cfg, r=r, q=q, beta=beta)
                        if rep is not None:
                            replays.append(rep)
    replays_df = _replays_to_frame(replays)
    comparisons = _comparisons_from_replays(replays_df, scenarios)
    sensitivity = _beta_sensitivity(replays_df)
    manifest = {
        "n_trades_considered": int(len(trades)),
        "n_replays": int(len(replays_df)),
        "betas": list(betas),
        "default_beta": DEFAULT_IV_BETA,
    }
    return StressResult(
        scenarios=scenarios, replays=replays_df, comparisons=comparisons,
        beta_sensitivity=sensitivity, manifest=manifest,
    )


# ==========================================================================================
# 4. Feeding the long history into the edge itself (STRATEGY.md §10.4)
# ==========================================================================================


class LongHistoryStore:
    """Wraps a `ChainStore`, substituting a long-history closes() series for the "Count"
    step of the edge computation (`strategy.selector.propose_trade` -> `quant.empirical.
    build_empirical`, both reached only via `store.closes(root, end=asof)`), while every
    other read (`chain`, `expiries`, `underlying`, `trading_dates`, `contains_synthetic`,
    ...) passes straight through to the wrapped store unchanged. This is the only
    integration point needed and it does not touch `data/store.py` (owned by a
    concurrent change) or `strategy/selector.py`.
    """

    def __init__(self, store, long_closes: dict[str, pd.Series]):
        self._store = store
        self._long_closes = long_closes

    def closes(self, root: str, end: date) -> pd.Series:
        s = self._long_closes.get(root)
        if s is None:
            return self._store.closes(root, end)
        idx = pd.DatetimeIndex(s.index)
        out = s[idx <= pd.Timestamp(end)].sort_index()
        return pd.Series(out.to_numpy(dtype=float), index=pd.DatetimeIndex(out.index).date, name="close")

    def __getattr__(self, name):
        return getattr(self._store, name)


def run_backtest_with_long_history(cfg: BacktestConfig, store, long_closes: dict[str, pd.Series]):
    """Run the SAME backtest config twice against the SAME option chains: once with the
    store's native (short) underlying history feeding `build_empirical`, once with
    `long_closes` substituted in via `LongHistoryStore` -- so a 2015 entry decision is
    gated by an empirical distribution that includes 2008 (STRATEGY.md §2.1, §10.4).

    Returns `(short_result, long_result, comparison)`. Lazy-imports `engine.loop` (owned
    by a concurrent change -- CLAUDE.md's "keep going, import lazily" pattern, mirroring
    `cli.py`).
    """
    from odds_lab.engine.loop import run_backtest

    short_result = run_backtest(cfg, store)
    long_result = run_backtest(cfg, LongHistoryStore(store, long_closes))

    def _stats(result) -> dict:
        trades = result.trades
        funnel = result.funnel
        rejected_negative_edge = (
            int(funnel["rejected_negative_edge"].sum())
            if not funnel.empty and "rejected_negative_edge" in funnel.columns
            else 0
        )
        return {
            "trades": int(len(trades)),
            "total_pnl": float(trades["pnl"].sum()) if not trades.empty else 0.0,
            "rejected_negative_edge": rejected_negative_edge,
        }

    short_stats = _stats(short_result)
    long_stats = _stats(long_result)
    comparison = {
        "short_window": short_stats,
        "long_window_with_2008": long_stats,
        "trade_count_delta": long_stats["trades"] - short_stats["trades"],
        "pnl_delta": long_stats["total_pnl"] - short_stats["total_pnl"],
        "rejected_negative_edge_delta": long_stats["rejected_negative_edge"] - short_stats["rejected_negative_edge"],
    }
    return short_result, long_result, comparison
