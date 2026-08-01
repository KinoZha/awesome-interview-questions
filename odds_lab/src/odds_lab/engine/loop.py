"""Daily event loop -- ARCHITECTURE.md §3.

Per trading date, in order: mark positions -> evaluate exits (trigger only) -> settle
expirations -> fill exits triggered on a PRIOR day (no lookahead, STRATEGY.md §7.3) ->
record snapshots -> run entries if scheduled.

Early-assignment policy (STRATEGY.md §5/§7, judgement call -- see docstring on
`_flag_early_assignment_risk`): DETECTED AND FLAGGED ONLY. Actual settlement always uses
the documented intrinsic-value rule at expiry (or the configured exit policy pre-expiry);
we do not simulate an assignment event landing before expiration. Flags are recorded in
`manifest['warnings']` and `position.meta['early_assignment_risk']`.
"""

from __future__ import annotations

import math
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from odds_lab import schema
from odds_lab.config import BacktestConfig
from odds_lab.engine import attribution as attribution_mod
from odds_lab.engine import fills as fills_mod
from odds_lab.engine.portfolio import Portfolio
from odds_lab.engine.result import BacktestResult
from odds_lab.schema import CONTRACT_MULTIPLIER, ExitReason, Position
from odds_lab.strategy import exits as exits_mod
from odds_lab.strategy import selector

__all__ = ["run_backtest"]


def _mark_position(pos: Position, chain: pd.DataFrame) -> dict | None:
    """Cost-to-close (mid) + position greeks for one open position on one day's chain.
    Returns None if none of the legs have a quote (e.g. data gap)."""
    if chain.empty:
        return None
    total_mark = 0.0
    net_delta = net_gamma = net_vega = net_theta = 0.0
    short_delta = None
    iv_short = None
    found_any = False
    for leg in pos.legs:
        row = chain[np.isclose(chain["strike"].astype(float), leg.strike) & (chain["right"] == leg.right)]
        if row.empty:
            continue
        r = row.iloc[0]
        found_any = True
        mid = (float(r["bid"]) + float(r["ask"])) / 2.0
        total_mark += (-leg.ratio) * mid
        net_delta += leg.ratio * float(r["delta"])
        net_gamma += leg.ratio * float(r["gamma"])
        net_vega += leg.ratio * float(r["vega"])
        net_theta += leg.ratio * float(r["theta"])
        if leg.ratio < 0:
            short_delta = float(r["delta"])
            iv_short = float(r["iv"])
    if not found_any:
        return None
    return {
        "mark": total_mark, "net_delta": net_delta, "net_gamma": net_gamma,
        "net_vega": net_vega, "net_theta": net_theta,
        "short_delta": short_delta, "iv_short": iv_short,
    }


def _flag_early_assignment_risk(pos: Position, chain: pd.DataFrame, S: float, asof: date, under_row) -> bool:
    """Detect (do not simulate) early-assignment risk on a short ITM call around an
    ex-dividend date -- STRATEGY.md §5/§7. A short call is at risk of early exercise when
    it is ITM and its time value (extrinsic) is smaller than the dividend the holder would
    capture by exercising just before the ex-date. We flag the position for one day and
    record it in `pos.meta['early_assignment_risk']`; the actual settlement mechanics are
    unchanged (see module docstring)."""
    if under_row is None:
        return False
    dividend = float(under_row.get("dividend", 0.0) or 0.0)
    if dividend <= 0:
        return False
    for leg in pos.legs:
        if leg.right != "C" or leg.ratio >= 0:
            continue
        intrinsic = max(S - leg.strike, 0.0)
        if intrinsic <= 0:
            continue
        row = chain[np.isclose(chain["strike"].astype(float), leg.strike) & (chain["right"] == "C")]
        if row.empty:
            continue
        r = row.iloc[0]
        mid = (float(r["bid"]) + float(r["ask"])) / 2.0
        extrinsic = mid - intrinsic
        if extrinsic < dividend:
            return True
    return False


def _settle_expiry(pos: Position, S_close: float) -> float:
    """Intrinsic settlement cash for the whole position (dollars, all qty), using the
    underlying's official close -- STRATEGY.md §7.7. American-style auto-exercise at
    $0.01 ITM per OCC rules is approximated as exact intrinsic settlement (no fee)."""
    total_per_share = 0.0
    for leg in pos.legs:
        intrinsic = max(S_close - leg.strike, 0.0) if leg.right == "C" else max(leg.strike - S_close, 0.0)
        total_per_share += leg.ratio * intrinsic
    return total_per_share * pos.qty * CONTRACT_MULTIPLIER


def _trade_row(pos: Position) -> dict:
    all_fills = pos.open_fills + pos.close_fills
    commission = sum(f.commission for f in all_fills)
    fees = sum(f.fees for f in all_fills)
    slippage = sum(f.spread_cost_vs_mid for f in all_fills)

    if pos.close_fills:
        pnl = pos.realized_pnl
        exit_debit = sum((f.price if f.side == "BUY" else -f.price) for f in pos.close_fills)
    else:
        settlement_cash = pos.meta.get("_settlement_cash", 0.0)
        pnl = sum(f.cash for f in pos.open_fills) + settlement_cash
        per_share = settlement_cash / (pos.qty * CONTRACT_MULTIPLIER) if pos.qty else 0.0
        exit_debit = -per_share

    max_loss_notional = pos.max_loss * pos.qty * CONTRACT_MULTIPLIER
    pnl_pct = pnl / max_loss_notional if math.isfinite(max_loss_notional) and max_loss_notional > 0 else float("nan")

    put_legs = [l for l in pos.legs if l.right == "P"]
    call_legs = [l for l in pos.legs if l.right == "C"]
    short_strike = next((l.strike for l in pos.legs if l.ratio < 0), float("nan"))
    long_strike = next((l.strike for l in pos.legs if l.ratio > 0), float("nan"))

    def _strike(side: list, short: bool) -> float:
        want = (lambda x: x.ratio < 0) if short else (lambda x: x.ratio > 0)
        return next((l.strike for l in side if want(l)), float("nan"))

    attrib = pos.meta.get("_attrib_totals", {})
    notional = pos.qty * CONTRACT_MULTIPLIER

    return {
        "position_id": pos.position_id,
        "strategy": pos.strategy,
        "root": pos.root,
        "entry_date": pos.entry_date,
        "exit_date": pos.exit_date,
        "expiry": pos.expiry,
        "dte_entry": pos.meta.get("dte_entry", 0),
        "qty": pos.qty,
        "short_strike": short_strike,
        "long_strike": long_strike,
        "width": abs(short_strike - long_strike) if (put_legs or call_legs) and not math.isnan(long_strike) else float("nan"),
        "short_put_strike": _strike(put_legs, short=True),
        "long_put_strike": _strike(put_legs, short=False),
        "short_call_strike": _strike(call_legs, short=True),
        "long_call_strike": _strike(call_legs, short=False),
        "entry_credit": pos.entry_credit,
        "exit_debit": exit_debit,
        "max_loss": pos.max_loss,
        "margin": pos.margin,
        "pnl": pnl,
        "pnl_pct_of_max_loss": pnl_pct,
        "commission": commission,
        "fees": fees,
        "slippage": slippage,
        "exit_reason": pos.exit_reason.value if pos.exit_reason else "",
        "short_delta": pos.meta.get("short_delta", float("nan")),
        "iv_entry": pos.meta.get("iv_entry", float("nan")),
        "iv_rank": pos.meta.get("iv_rank", float("nan")),
        "realized_vol": pos.meta.get("realized_vol", float("nan")),
        "market_state": pos.meta.get("market_state", ""),
        "p_theo_loss": pos.meta.get("p_theo_loss", float("nan")),
        "p_actual_loss": pos.meta.get("p_actual_loss", float("nan")),
        "edge_prob": pos.meta.get("edge_prob", float("nan")),
        "edge_ev": pos.meta.get("edge_ev", float("nan")),
        "underlying_entry": pos.meta.get("underlying_entry", float("nan")),
        "underlying_exit": pos.meta.get("_underlying_exit", float("nan")),
        # Snapshot attribution accumulates in dollars-per-share; `pnl` on this row is
        # position dollars. Scale so the attribution stack is comparable with P&L
        # instead of being three orders of magnitude smaller than it.
        "pnl_delta": attrib.get("d_delta", 0.0) * notional,
        "pnl_gamma": attrib.get("d_gamma", 0.0) * notional,
        "pnl_vega": attrib.get("d_vega", 0.0) * notional,
        "pnl_theta": attrib.get("d_theta", 0.0) * notional,
        "pnl_residual": attrib.get("d_residual", 0.0) * notional,
    }


def run_backtest(cfg: BacktestConfig, store) -> BacktestResult:
    portfolio = Portfolio(cfg.risk, risk_free_rate=cfg.risk_free_rate)
    dist_cache: dict = {}
    trades_rows: list[dict] = []
    equity_rows: list[dict] = []
    snapshot_rows: list[dict] = []
    warnings: list[str] = []
    position_counter = 0

    all_dates: set[date] = set()
    for root in cfg.roots:
        all_dates.update(store.trading_dates(root, cfg.start, cfg.end))
    trading_dates = sorted(all_dates)

    entry_weekday = cfg.entry.entry_weekday
    schedule = cfg.entry.entry_schedule

    def _is_entry_day(d: date) -> bool:
        if schedule == "daily":
            return True
        if schedule == "weekly":
            return d.weekday() == entry_weekday
        if schedule == "monthly_expiry":
            return d.day <= 7 and d.weekday() == entry_weekday
        return False

    for asof in trading_dates:
        # 1. mark existing positions to market (mid)
        marks: dict[str, float] = {}
        mark_details: dict[str, dict] = {}
        for pos in portfolio.positions:
            if not pos.is_open:
                continue
            chain = store.chain(pos.root, asof, expiry=pos.expiry)
            schema.assert_no_lookahead(chain, asof)
            detail = _mark_position(pos, chain)
            if detail is not None:
                marks[pos.position_id] = detail["mark"]
                mark_details[pos.position_id] = detail

        equity_val = portfolio.equity(marks)

        # 2. evaluate exits -> trigger only (fills on the NEXT trading day, no lookahead)
        for pos in portfolio.positions:
            if not pos.is_open or pos.meta.get("_pending_exit"):
                continue
            detail = mark_details.get(pos.position_id)
            if detail is None:
                continue
            snap = {
                "mark": detail["mark"], "short_delta": detail["short_delta"],
                "dte": (pos.expiry - asof).days, "underlying_price": None,
            }
            reason = exits_mod.evaluate_exit(pos, snap, cfg.exits)
            if reason is not None:
                pos.meta["_pending_exit"] = reason
                pos.meta["_pending_exit_trigger_date"] = asof

        # 3. settle expirations at today's official underlying close
        for pos in list(portfolio.positions):
            if pos.is_open and pos.expiry == asof:
                closes = store.closes(pos.root, end=asof)
                idx = pd.DatetimeIndex(closes.index)
                on_date = closes[idx == pd.Timestamp(asof)]
                S_close = float(on_date.iloc[-1]) if not on_date.empty else float(closes.iloc[-1])
                settlement_cash = _settle_expiry(pos, S_close)
                portfolio.cash += settlement_cash
                pos.meta["_settlement_cash"] = settlement_cash
                pos.meta["_underlying_exit"] = S_close
                any_itm_short = any(
                    (S_close > l.strike if l.right == "C" else S_close < l.strike) and l.ratio < 0
                    for l in pos.legs
                )
                pos.exit_date = asof
                pos.exit_reason = ExitReason.ASSIGNED if any_itm_short else ExitReason.EXPIRY

        # 4. fill exits that were TRIGGERED on a prior day
        for pos in list(portfolio.positions):
            if not pos.is_open or not pos.meta.get("_pending_exit"):
                continue
            if pos.meta.get("_pending_exit_trigger_date") == asof:
                continue  # triggered today, fills tomorrow
            chain = store.chain(pos.root, asof, expiry=pos.expiry)
            schema.assert_no_lookahead(chain, asof)
            if chain.empty:
                continue
            side_map = {leg: ("BUY" if leg.ratio < 0 else "SELL") for leg in pos.legs}
            close_fills = fills_mod.execute(pos.legs, pos.qty, chain, side_map, cfg.costs, asof)
            for f in close_fills:
                portfolio.cash += f.cash
            pos.close_fills = close_fills
            pos.exit_date = asof
            pos.exit_reason = pos.meta["_pending_exit"]
            pos.meta["_underlying_exit"] = float(chain["underlying_price"].iloc[0])

        # 4b. early-assignment risk flag (detect-and-flag only; see module docstring)
        for pos in portfolio.positions:
            if not pos.is_open:
                continue
            under = store.underlying(pos.root, start=asof, end=asof)
            under_row = under.iloc[0].to_dict() if under is not None and not under.empty else None
            detail = mark_details.get(pos.position_id)
            if detail is None or under_row is None:
                continue
            chain = store.chain(pos.root, asof, expiry=pos.expiry)
            S = float(chain["underlying_price"].iloc[0]) if not chain.empty else None
            if S is None:
                continue
            if _flag_early_assignment_risk(pos, chain, S, asof, under_row):
                pos.meta["early_assignment_risk"] = True
                warnings.append(f"early_assignment_risk: {pos.position_id} on {asof}")

        # 5. record snapshots + attribution for still-open positions
        for pos in portfolio.positions:
            if not pos.is_open:
                continue
            detail = mark_details.get(pos.position_id)
            if detail is None:
                continue
            chain = store.chain(pos.root, asof, expiry=pos.expiry)
            S = float(chain["underlying_price"].iloc[0]) if not chain.empty else float("nan")
            cur_snap = {
                "date": asof, "underlying_price": S, "mark": detail["mark"],
                "net_delta": detail["net_delta"], "net_gamma": detail["net_gamma"],
                "net_vega": detail["net_vega"], "net_theta": detail["net_theta"],
                "iv_short": detail["iv_short"],
            }
            prev_snap = pos.meta.get("_prev_snap")
            if prev_snap is not None:
                incr = attribution_mod.attribute(prev_snap, cur_snap)
            else:
                incr = {"d_delta": 0.0, "d_gamma": 0.0, "d_vega": 0.0, "d_theta": 0.0, "d_residual": 0.0}
            totals = pos.meta.setdefault(
                "_attrib_totals", {"d_delta": 0.0, "d_gamma": 0.0, "d_vega": 0.0, "d_theta": 0.0, "d_residual": 0.0}
            )
            for k, v in incr.items():
                totals[k] += v
            pos.meta["_prev_snap"] = cur_snap

            mtm_pnl = (pos.entry_credit - detail["mark"]) * pos.qty * CONTRACT_MULTIPLIER
            snapshot_rows.append({
                "position_id": pos.position_id, "date": asof, "underlying_price": S,
                "mark": detail["mark"], "mtm_pnl": mtm_pnl, "dte": (pos.expiry - asof).days,
                "short_delta": detail["short_delta"] if detail["short_delta"] is not None else float("nan"),
                "net_delta": detail["net_delta"], "net_gamma": detail["net_gamma"],
                "net_vega": detail["net_vega"], "net_theta": detail["net_theta"],
                "iv_short": detail["iv_short"] if detail["iv_short"] is not None else float("nan"),
                **incr,
            })

        # 6. finalize any positions that closed today -> trade row
        for pos in list(portfolio.positions):
            if not pos.is_open and pos.position_id not in {r["position_id"] for r in trades_rows}:
                trades_rows.append(_trade_row(pos))

        # Drop positions once their trade row has been recorded, so the open-position lists
        # walked on later days don't grow without bound (perf: 12y x 3-root run).
        recorded_ids = {r["position_id"] for r in trades_rows}
        portfolio.positions = [p for p in portfolio.positions if p.is_open or p.position_id not in recorded_ids]

        # 7. entries, if scheduled today
        if _is_entry_day(asof):
            for root in cfg.roots:
                open_count = sum(1 for p in portfolio.positions if p.is_open and p.root == root)
                if open_count >= cfg.entry.max_concurrent_per_root:
                    continue
                proposal = selector.propose_trade(store, root, asof, cfg, dist_cache)
                if proposal is None:
                    continue

                S = proposal.meta["underlying_entry"]
                if math.isfinite(proposal.max_loss):
                    margin_per_contract = proposal.max_loss * CONTRACT_MULTIPLIER
                else:
                    margin_per_contract = portfolio.reg_t_margin(proposal.legs, 1, S, proposal.credit)
                margin_per_share = margin_per_contract / CONTRACT_MULTIPLIER

                qty = portfolio.size(proposal.max_loss, margin_per_share, cfg.risk)
                if qty <= 0:
                    continue

                position_counter += 1
                position_id = f"{root}-{proposal.strategy}-{asof.isoformat()}-{position_counter}-{uuid.uuid4().hex[:6]}"
                position = Position(
                    position_id=position_id, strategy=proposal.strategy, root=root,
                    entry_date=asof, expiry=proposal.expiry, legs=proposal.legs, qty=qty,
                    entry_credit=proposal.credit, max_loss=proposal.max_loss,
                    margin=margin_per_contract * qty, meta=dict(proposal.meta),
                )

                accepted, _reason = portfolio.try_open(position, cfg.risk)
                if not accepted:
                    continue

                chain = store.chain(root, asof, expiry=proposal.expiry)
                side_map = {leg: ("SELL" if leg.ratio < 0 else "BUY") for leg in proposal.legs}
                open_fills = fills_mod.execute(proposal.legs, qty, chain, side_map, cfg.costs, asof)
                for f in open_fills:
                    portfolio.cash += f.cash
                position.open_fills = open_fills

        # 8. daily interest accrual + equity record
        portfolio.accrue_cash_interest(days=1.0)
        eq_marks = {
            p.position_id: mark_details[p.position_id]["mark"]
            for p in portfolio.positions if p.is_open and p.position_id in mark_details
        }
        eq_val = portfolio.equity(eq_marks)
        net_delta = sum(
            mark_details[p.position_id]["net_delta"] * p.qty
            for p in portfolio.positions if p.is_open and p.position_id in mark_details
        )
        net_gamma = sum(
            mark_details[p.position_id]["net_gamma"] * p.qty
            for p in portfolio.positions if p.is_open and p.position_id in mark_details
        )
        net_vega = sum(
            mark_details[p.position_id]["net_vega"] * p.qty
            for p in portfolio.positions if p.is_open and p.position_id in mark_details
        )
        net_theta = sum(
            mark_details[p.position_id]["net_theta"] * p.qty
            for p in portfolio.positions if p.is_open and p.position_id in mark_details
        )
        equity_rows.append({
            "date": asof, "cash": portfolio.cash,
            "positions_value": eq_val - portfolio.cash,
            "equity": eq_val, "margin_used": portfolio.margin_used(),
            "margin_pct": portfolio.margin_pct(),
            "open_positions": sum(1 for p in portfolio.positions if p.is_open),
            "net_delta": net_delta, "net_gamma": net_gamma,
            "net_vega": net_vega, "net_theta": net_theta,
        })

    trades_df = (
        schema.validate_frame(pd.DataFrame(trades_rows), schema.TRADE_DTYPES, "trades")
        if trades_rows else schema.empty_frame(schema.TRADE_DTYPES)
    )
    equity_df = (
        schema.validate_frame(pd.DataFrame(equity_rows), schema.EQUITY_DTYPES, "equity")
        if equity_rows else schema.empty_frame(schema.EQUITY_DTYPES)
    )
    snapshots_df = (
        schema.validate_frame(pd.DataFrame(snapshot_rows), schema.SNAPSHOT_DTYPES, "snapshots")
        if snapshot_rows else schema.empty_frame(schema.SNAPSHOT_DTYPES)
    )

    coverage = {}
    for root in cfg.roots:
        dates = store.trading_dates(root, cfg.start, cfg.end)
        coverage[root] = {
            "n_dates": len(dates),
            "start": min(dates).isoformat() if dates else None,
            "end": max(dates).isoformat() if dates else None,
        }

    manifest = {
        "run_id": cfg.run_id(),
        "git_sha": _git_sha(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": cfg.data.provider,
        "coverage": coverage,
        "is_synthetic": (
            cfg.data.provider == "synthetic"
            or cfg.data.allow_synthetic
            or store.contains_synthetic(cfg.roots)
        ),
        "rejected_trades": portfolio.rejected_trades,
        "warnings": warnings,
    }

    return BacktestResult(config=cfg, trades=trades_df, equity=equity_df, snapshots=snapshots_df, manifest=manifest)


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=Path(__file__).resolve().parent
        ).decode().strip()
    except Exception:
        return "unknown"
