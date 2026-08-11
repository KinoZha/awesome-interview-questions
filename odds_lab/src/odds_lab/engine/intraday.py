"""Intraday (1-minute quote) exit evaluation -- lifts the "no intraday entries in
v1" non-goal (ARCHITECTURE.md §9) for EXITS only, per this change's brief.
STRATEGY.md §5/§7.3.

Design (read before touching anything below):

  Entries stay EOD-only, always. A weekly-entry strategy (STRATEGY.md §4) does not
  need intraday entry timing, and `engine/loop.py`'s selector call is untouched.

  Only exit evaluation for the rules named in `IntradayConfig.exit_rules` gains
  minute resolution. `dte_exit` is daily by nature (STRATEGY.md §5) and is masked
  out here unconditionally -- it is still evaluated once/day by the ordinary EOD
  path in `engine/loop.py`, exactly as before this feature existed.

  No-lookahead (CLAUDE.md rule 1, STRATEGY.md §7.3): the trigger is detected on
  bar i; the fill happens at bar i+1's quotes, NEVER bar i's -- the same discipline
  the daily engine already applies, just at minute instead of day granularity. If
  the trigger fires on the day's LAST bar, there is no same-day i+1 to fill at;
  the caller (`engine/loop.py`) then falls back to the existing "trigger day D,
  fill day D+1's EOD quote" mechanism, so a late-day intraday trigger is exactly
  as lookahead-safe as the EOD-only path already was.

  Delta is not present in a quote-only export (bid/ask, no vendor greeks).
  `evaluate_intraday_day` recomputes it per bar via `quant.bs`: invert IV from the
  leg's own quote mid, holding the day's EOD underlying price CONSTANT through the
  day (a 1-minute OPTION QUOTE export carries no per-minute underlying print --
  getting a true intraday S would mean also buying underlying trade data, which is
  out of this change's scope). This is a documented approximation, not a silent
  one: it affects `delta_breach` only (profit_target/stop_loss depend only on the
  option's own quoted mark, not on S or delta, so they are exact given the quotes).

  THE FEEDBACK LOOP (task requirement -- must not be silently assumed to
  converge): intraday exits can move an exit date earlier or later than the EOD
  backtest would have, which changes concurrent-position counts and margin, which
  can admit or reject different LATER entries, which changes the held-contract
  set pass 2 needs bars for. `run_hybrid_backtest` below ITERATES the full
  backtest to a fixed point on that held-contract-set (not a one-shot "fetch a
  superset and hope"), capped at `IntradayConfig.iteration_cap`, and NEVER
  silently declares convergence it didn't check for -- see that function's
  docstring for why iteration (not a superset-and-prove-sufficient argument) was
  chosen, and how non-convergence is reported.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, replace
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd

from odds_lab.config import BacktestConfig
from odds_lab.engine import fills as fills_mod
from odds_lab.quant import bs as bs_mod
from odds_lab.schema import ExitReason, Fill, Position
from odds_lab.strategy import exits as exits_mod

__all__ = [
    "IntradayExitResult",
    "bars_for_position",
    "evaluate_intraday_day",
    "contract_day_keys_from_trades",
    "run_hybrid_backtest",
    "compare_eod_vs_intraday",
]


@dataclass
class IntradayExitResult:
    reason: ExitReason | None
    trigger_ts: pd.Timestamp | None
    fill_ts: pd.Timestamp | None
    """None means: a rule triggered but there was no same-day bar after it to fill
    at (last bar of the day) -- the caller must defer to the next day's EOD fill,
    exactly like the pre-existing daily-only path."""
    close_fills: list[Fill] | None
    mark_at_trigger: float | None


def bars_for_position(store, pos: Position, asof: date) -> pd.DataFrame | None:
    """Per-leg minute bars for `pos` on `asof`, inner-joined on a common minute
    timestamp (a minute missing a quote on ANY leg cannot mark the whole spread, so
    it is dropped rather than guessed). Returns `None` if ANY leg has zero intraday
    coverage for this day -- the caller treats that as "fall back to EOD for this
    position on this day" and records the fallback (task requirement)."""
    per_leg: dict[str, pd.DataFrame] = {}
    for leg in pos.legs:
        df = store.intraday(pos.root, leg.expiry, leg.strike, leg.right, asof)
        if df is None or df.empty:
            return None
        leg_id = _leg_id(leg)
        per_leg[leg_id] = df[["ts", "bid", "ask"]].rename(columns={"bid": f"bid_{leg_id}", "ask": f"ask_{leg_id}"})

    merged: pd.DataFrame | None = None
    for frame in per_leg.values():
        merged = frame if merged is None else merged.merge(frame, on="ts", how="inner")
    if merged is None or merged.empty:
        return None
    return merged.sort_values("ts").reset_index(drop=True)


def _leg_id(leg) -> str:
    return f"{leg.strike:g}_{leg.right}"


def _masked_exit_config(exits_cfg, intraday_cfg):
    """Only let rules named in `IntradayConfig.exit_rules` fire from the intraday
    path -- reuses `strategy.exits.evaluate_exit` unmodified rather than
    duplicating its trigger logic (CLAUDE.md rule 3 in spirit: one rule
    implementation, not two that can drift). `dte_exit` is masked unconditionally:
    it is daily by nature and always evaluated once/day by `engine/loop.py`
    itself, never here."""
    masked = exits_cfg
    if "profit_target" not in intraday_cfg.exit_rules:
        masked = replace(masked, profit_target_pct=None)
    if "stop_loss" not in intraday_cfg.exit_rules:
        masked = replace(masked, stop_loss_multiple=None)
    if "delta_breach" not in intraday_cfg.exit_rules:
        masked = replace(masked, delta_breach=None)
    return replace(masked, dte_exit=None)


def evaluate_intraday_day(
    pos: Position, merged: pd.DataFrame, cfg: BacktestConfig, S_eod: float | None, asof: date,
) -> IntradayExitResult:
    """Walk `merged`'s minute bars in order; fire the FIRST rule that triggers;
    fill at the NEXT bar (see module docstring). `S_eod` (held constant through the
    day) is only needed for `delta_breach`'s recomputed delta -- see module
    docstring's documented approximation."""
    exits_cfg = cfg.exits
    intraday_cfg = cfg.intraday
    if exits_cfg.hold_to_expiry:
        return IntradayExitResult(None, None, None, None, None)

    masked_cfg = _masked_exit_config(exits_cfg, intraday_cfg)
    legs = pos.legs
    leg_ids = [_leg_id(leg) for leg in legs]
    dte = (pos.expiry - asof).days
    r = cfg.risk_free_rate
    want_delta = masked_cfg.delta_breach is not None

    for i in range(len(merged)):
        row = merged.iloc[i]
        mark = 0.0
        short_delta = None
        for leg, lid in zip(legs, leg_ids):
            bid, ask = float(row[f"bid_{lid}"]), float(row[f"ask_{lid}"])
            mid = (bid + ask) / 2.0
            mark += (-leg.ratio) * mid
            if want_delta and leg.ratio < 0 and S_eod is not None and S_eod > 0:
                short_delta = _recompute_delta(mid, S_eod, leg.strike, dte, r, leg.right)

        snap = {"mark": mark, "short_delta": short_delta, "dte": dte, "underlying_price": S_eod}
        reason = exits_mod.evaluate_exit(pos, snap, masked_cfg)
        if reason is None:
            continue

        trigger_ts = pd.Timestamp(row["ts"])
        if i + 1 >= len(merged):
            return IntradayExitResult(reason, trigger_ts, None, None, mark)

        fill_row = merged.iloc[i + 1]
        fill_ts = pd.Timestamp(fill_row["ts"])
        quotes = pd.DataFrame(
            [
                {
                    "strike": leg.strike, "right": leg.right, "expiry": leg.expiry,
                    "bid": float(fill_row[f"bid_{lid}"]), "ask": float(fill_row[f"ask_{lid}"]),
                }
                for leg, lid in zip(legs, leg_ids)
            ]
        )
        side_map = {leg: ("BUY" if leg.ratio < 0 else "SELL") for leg in legs}
        close_fills = fills_mod.execute(legs, pos.qty, quotes, side_map, cfg.costs, asof)
        return IntradayExitResult(reason, trigger_ts, fill_ts, close_fills, mark)

    return IntradayExitResult(None, None, None, None, None)


def _recompute_delta(mid: float, S: float, K: float, dte: int, r: float, right: str) -> float | None:
    """IV-from-mid then delta-at-that-IV, per module docstring. Never raises --
    a bar with a degenerate quote (e.g. mid <= the no-arbitrage floor) just
    contributes no delta reading for that minute rather than aborting the walk."""
    T = max(dte, 0) / 365.0
    if T <= 0:
        return None
    try:
        iv = float(np.asarray(bs_mod.implied_vol(mid, S, K, T, r, 0.0, right)).reshape(-1)[0])
    except Exception:
        return None
    if not math.isfinite(iv) or iv <= 0:
        return None
    greeks = bs_mod.bs_greeks(S, K, T, r, 0.0, iv, right)
    delta = float(np.asarray(greeks["delta"]).reshape(-1)[0])
    return delta if math.isfinite(delta) else None


# ==========================================================================================
# Pass 1 -> held-contract set, and the pass-2 fixed-point iteration
# ==========================================================================================

_LEG_COLS = {
    "short_put_strike": "P", "long_put_strike": "P",
    "short_call_strike": "C", "long_call_strike": "C",
}


def contract_day_keys_from_trades(trades_df: pd.DataFrame) -> set[tuple[str, date, float, str, date]]:
    """`(root, expiry, strike, right, date)` for every day a leg in `trades_df` was
    actually held -- i.e. exactly the pass-2 fetch set the task brief describes
    ("~1200 trades x 4 legs x ~30 days"), computed from pass 1's own output rather
    than guessed up front."""
    keys: set[tuple[str, date, float, str, date]] = set()
    if trades_df is None or trades_df.empty:
        return keys
    for _, t in trades_df.iterrows():
        root = str(t["root"])
        expiry = pd.Timestamp(t["expiry"]).date()
        entry = pd.Timestamp(t["entry_date"]).date()
        exit_d = pd.Timestamp(t["exit_date"]).date() if pd.notna(t["exit_date"]) else expiry
        if exit_d < entry:
            exit_d = entry
        held_dates = [d.date() for d in pd.bdate_range(entry, exit_d)]
        for col, right in _LEG_COLS.items():
            strike = t.get(col)
            if strike is None or (isinstance(strike, float) and math.isnan(strike)):
                continue
            for d in held_dates:
                keys.add((root, expiry, float(strike), right, d))
    return keys


def _available_keys(store, needed: set[tuple[str, date, float, str, date]]) -> set[tuple[str, date, float, str, date]]:
    by_root_date: dict[tuple[str, date], set[tuple[date, float, str]]] = {}
    out: set[tuple[str, date, float, str, date]] = set()
    for root, expiry, strike, right, d in needed:
        cache_key = (root, d)
        if cache_key not in by_root_date:
            by_root_date[cache_key] = store.intraday_available_keys(root, d)
        if (expiry, strike, right) in by_root_date[cache_key]:
            out.add((root, expiry, strike, right, d))
    return out


def run_hybrid_backtest(
    cfg: BacktestConfig,
    store,
    *,
    fetch_fn: "Callable[[set[tuple[str, date, float, str, date]]], None] | None" = None,
    iteration_cap: int | None = None,
):
    """The two-pass hybrid backtest: pass 1 (EOD, unmodified engine) determines the
    held-contract set; pass 2 re-runs with intraday exit evaluation over the store's
    (possibly `fetch_fn`-augmented) 1-minute quote table.

    FEEDBACK LOOP DESIGN (task requirement -- pick one, do not pretend it
    converges silently): **iterate to a fixed point**, not a one-shot deliberate
    superset. Reasoning: intraday exits can move an exit date, which changes
    concurrent-position counts/margin, which can admit or reject different LATER
    entries, which changes the held-contract set -- so the "right" superset to
    fetch is itself an output of running the intraday backtest, not something
    knowable in advance. Iterating on the ACTUAL held-set the previous pass
    produced is exact by construction; the alternative (fetch a generously wide
    superset up front and argue it's "obviously" enough) would require bounding
    how far entries/exits can possibly drift, which this system does not attempt
    to prove and should not claim.

    Each iteration: run the full backtest with whatever intraday coverage `store`
    currently has -> read off the held-contract-day set from the resulting trades
    -> if any of it is missing from `store` and `fetch_fn` is given, call it to
    pull more in -> if the held-set is unchanged from the previous iteration AND
    fully covered, STOP (converged). Capped at `iteration_cap`
    (`cfg.intraday.iteration_cap` by default); hitting the cap without converging
    is reported, never hidden -- `result.manifest['intraday']['converged'] = False`.
    Whichever way it ends, the manifest also carries `engine/loop.py`'s own
    per-position bookkeeping of which contracts were actually evaluated intraday
    vs. fell back to EOD-only during the FINAL iteration's run -- a silently
    mixed-resolution backtest is worse than a consistent one (task requirement).
    """
    from odds_lab.engine.loop import run_backtest  # lazy: loop.py imports this module

    if not cfg.intraday.enabled:
        raise ValueError("run_hybrid_backtest requires cfg.intraday.enabled=True")
    cap = iteration_cap if iteration_cap is not None else cfg.intraday.iteration_cap
    if cap < 1:
        raise ValueError(f"iteration_cap must be >= 1, got {cap}")

    prev_needed: set | None = None
    result = None
    converged = False
    missing: set = set()
    iterations_run = 0

    for iteration in range(1, cap + 1):
        iterations_run = iteration
        result = run_backtest(cfg, store)
        needed = contract_day_keys_from_trades(result.trades)
        available = _available_keys(store, needed)
        missing = needed - available

        if missing and fetch_fn is not None:
            fetch_fn(missing)
            available = _available_keys(store, needed)
            missing = needed - available

        held_set_stable = prev_needed is not None and needed == prev_needed
        if held_set_stable and not missing:
            converged = True
            break
        if held_set_stable and missing:
            # The held-contract set itself has stopped changing, but data is still
            # missing and nothing (no fetch_fn, or fetch_fn couldn't supply it) is
            # going to make it appear -- further iterations cannot change the
            # outcome. Stop now rather than silently spend the rest of the cap.
            converged = False
            break
        prev_needed = needed

    assert result is not None
    result.manifest["intraday"]["hybrid"] = {
        "iteration_cap": cap,
        "iterations_run": iterations_run,
        "converged": converged,
        "held_contract_days_needed": len(needed),
        "held_contract_days_missing_data": len(missing),
        "contract_days_eod_fallback_due_to_missing_data": sorted(
            f"{root}/{right}{strike:g}/{expiry.isoformat()}/{d.isoformat()}"
            for (root, expiry, strike, right, d) in missing
        ),
    }
    return result


def compare_eod_vs_intraday(
    cfg: BacktestConfig,
    store,
    *,
    fetch_fn: "Callable[[set[tuple[str, date, float, str, date]]], None] | None" = None,
    iteration_cap: int | None = None,
) -> dict:
    """Run the SAME configuration both ways (EOD-only exits vs. the intraday
    hybrid) and report, per exit rule, how many exits changed, how the exit dates
    shifted, and the P&L difference -- this is the first-class deliverable (task
    brief: "that number is what justifies the data purchase, so make it a
    first-class output, not a footnote"), not a byproduct.

    Trades are matched between the two runs by a data-derived key (root,
    entry_date, expiry, short/long strikes per side) rather than `position_id`
    (which embeds a random per-run uuid and cannot be compared across runs). A
    trade with no counterpart in the other run (admitted in one, rejected in the
    other because of the margin/concurrency feedback loop the module docstring
    describes) is excluded from the per-rule diff and counted separately.
    """
    cfg_eod = dataclasses.replace(cfg, intraday=dataclasses.replace(cfg.intraday, enabled=False))
    from odds_lab.engine.loop import run_backtest

    eod_result = run_backtest(cfg_eod, store)

    cfg_hybrid = dataclasses.replace(cfg, intraday=dataclasses.replace(cfg.intraday, enabled=True))
    hybrid_result = run_hybrid_backtest(cfg_hybrid, store, fetch_fn=fetch_fn, iteration_cap=iteration_cap)

    eod_by_key = _index_trades_by_key(eod_result.trades)
    hyb_by_key = _index_trades_by_key(hybrid_result.trades)
    matched_keys = set(eod_by_key) & set(hyb_by_key)

    by_rule: dict[str, dict] = {}
    for rule in ("profit_target", "stop_loss", "delta_breach"):
        n_changed = 0
        exit_date_shift_days: list[int] = []
        pnl_diff = 0.0
        n_considered = 0
        for k in matched_keys:
            e, h = eod_by_key[k], hyb_by_key[k]
            if e["exit_reason"] != rule and h["exit_reason"] != rule:
                continue
            n_considered += 1
            changed = (e["exit_reason"] != h["exit_reason"]) or (e["exit_date"] != h["exit_date"])
            if changed:
                n_changed += 1
                exit_date_shift_days.append(
                    int((pd.Timestamp(h["exit_date"]) - pd.Timestamp(e["exit_date"])).days)
                )
                pnl_diff += float(h["pnl"] - e["pnl"])
        by_rule[rule] = {
            "trades_considered": n_considered,
            "exits_changed": n_changed,
            "mean_exit_date_shift_days": float(np.mean(exit_date_shift_days)) if exit_date_shift_days else 0.0,
            "pnl_difference_hybrid_minus_eod": pnl_diff,
        }

    overall = {
        "n_trades_eod": int(len(eod_result.trades)),
        "n_trades_hybrid": int(len(hybrid_result.trades)),
        "n_matched": len(matched_keys),
        "eod_total_pnl": float(eod_result.trades["pnl"].sum()) if not eod_result.trades.empty else 0.0,
        "hybrid_total_pnl": float(hybrid_result.trades["pnl"].sum()) if not hybrid_result.trades.empty else 0.0,
    }
    overall["pnl_difference_hybrid_minus_eod"] = overall["hybrid_total_pnl"] - overall["eod_total_pnl"]

    return {
        "overall": overall,
        "by_rule": by_rule,
        "eod_result": eod_result,
        "hybrid_result": hybrid_result,
    }


def _index_trades_by_key(trades_df: pd.DataFrame) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    if trades_df is None or trades_df.empty:
        return out
    for _, t in trades_df.iterrows():
        key = (
            str(t["root"]), pd.Timestamp(t["entry_date"]).date(), pd.Timestamp(t["expiry"]).date(),
            _nan_or(t.get("short_put_strike")), _nan_or(t.get("long_put_strike")),
            _nan_or(t.get("short_call_strike")), _nan_or(t.get("long_call_strike")),
        )
        out[key] = {
            "exit_reason": t["exit_reason"], "exit_date": pd.Timestamp(t["exit_date"]).date(), "pnl": float(t["pnl"]),
        }
    return out


def _nan_or(v):
    if v is None:
        return None
    try:
        if math.isnan(v):
            return None
    except TypeError:
        pass
    return round(float(v), 6)
