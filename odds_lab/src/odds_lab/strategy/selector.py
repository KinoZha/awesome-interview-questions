"""Expiry + strike selection -- STRATEGY.md §4.1, §4.2, §4.3.

`propose_trade` is the single entry point the engine loop calls once per (root, scheduled
entry date): it selects an expiry, selects strikes under the configured `StrikeRule`,
prices the structure, applies every OPI filter, computes the empirical edge (STRATEGY.md
§2), and ranks surviving candidate expiries by `edge_ev` (highest wins) -- never by raw
expected return.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from odds_lab import schema
from odds_lab.config import BacktestConfig, CostModel, EntryConfig, MarketFilter, Strategy, StrikeRule
from odds_lab.quant.edge import naked_edge, spread_edge
from odds_lab.quant.empirical import EmpiricalDist, build_empirical
from odds_lab.schema import Leg, MarketState
from odds_lab.strategy import strategies, universe

__all__ = [
    "select_expiry", "select_strikes", "propose_trade", "TradeProposal",
    "FUNNEL_REASONS", "new_funnel",
]

_DEBIT_STRATEGIES = {"long_strangle"}
_DEFINED_RISK_2LEG = {"put_credit_spread", "call_credit_spread"}


# --------------------------------------------------------------------------------------
# Selection funnel -- every `return None` / `continue` below is attributed to exactly one
# of these named reasons (see docstring on `propose_trade`). No catch-all bucket.
# --------------------------------------------------------------------------------------

FUNNEL_REASONS: list[str] = [
    "no_expiry",                   # store.expiries() returned nothing in [dte_min, dte_max]
    "market_state",                # entry.market_filter blocks this strategy outright today
    "empty_chain",                 # no chain rows for this expiry (data gap)
    "no_atm_iv",                   # universe.atm_iv() couldn't compute an ATM IV
    "no_empirical_dist",           # build_empirical() had insufficient history
    "no_strike",                   # select_strikes(): no listed strike hit cfg.strike_rule's band/target
    "no_width_strike",             # select_strikes(): short leg found, but the long leg (width_strikes away) is missing/invalid
    "structure_economics_error",   # a selected leg has no quote in the chain (schema.SchemaError)
    "liquidity_no_quote",          # a leg has no quote row on this date's chain
    "liquidity_spread",            # (ask-bid)/mid exceeds costs.max_spread_pct_of_mid, or mid<=0
    "liquidity_min_bid",           # short leg bid below costs.min_bid
    "liquidity_oi",                # short leg open interest below costs.min_open_interest
    "min_credit",                  # net credit below entry.min_credit
    "expected_return_band",        # credit/(width-credit) outside (expected_return_min, expected_return_max]
    "negative_edge",               # entry.require_positive_edge and edge_ev <= 0
]


def new_funnel(root: str, asof: date) -> dict:
    """A fresh, zero-initialized selection-funnel record for one (root, asof) opportunity."""
    return {
        "root": root,
        "asof": asof,
        "expiries_considered": 0,
        "candidates": 0,
        "rejected": {reason: 0 for reason in FUNNEL_REASONS},
        "accepted": 0,
        "best_edge_ev": None,
        # Informational, NOT part of the `rejected` partition: a real ThetaData bulk CSV
        # export has no open_interest column at all, so providers emit the
        # OPEN_INTEREST_UNKNOWN sentinel (-1, never 0 -- see data/providers/csv_export.py).
        # `liquidity_oi_unknown` counts candidates where the OI filter was SKIPPED because
        # the data didn't have an answer -- it must not be conflated with `liquidity_oi`
        # (a real OI reading below the threshold), and it does not by itself reject
        # anything, so it can co-occur with `accepted`.
        "liquidity_oi_unknown": 0,
    }


@dataclass
class TradeProposal:
    strategy: str
    root: str
    expiry: date
    legs: list[Leg]
    credit: float
    max_loss: float
    width: float
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Expiry selection -- STRATEGY.md §4.2 (21 <= DTE <= 56)
# --------------------------------------------------------------------------------------


def select_expiry(store, root, asof: date, cfg: EntryConfig) -> date | None:
    """Pick the single expiry in [dte_min, dte_max] closest to the midpoint of that band."""
    expiries = store.expiries(root, asof, dte_min=cfg.dte_min, dte_max=cfg.dte_max)
    if not expiries:
        return None
    target_dte = (cfg.dte_min + cfg.dte_max) / 2.0
    return min(expiries, key=lambda e: abs((e - asof).days - target_dte))


# --------------------------------------------------------------------------------------
# Strike selection -- StrikeRule modes
# --------------------------------------------------------------------------------------


def _increment(strikes: np.ndarray) -> float:
    u = np.unique(np.asarray(strikes, dtype=float))
    if len(u) < 2:
        return 1.0
    diffs = np.diff(u)
    pos = diffs[diffs > 1e-9]
    return float(np.min(pos)) if len(pos) else 1.0


def _nearest_strike(strikes: np.ndarray, target: float) -> float:
    strikes = np.asarray(strikes, dtype=float)
    return float(strikes[np.argmin(np.abs(strikes - target))])


def _pick_by_delta(sub: pd.DataFrame, delta_min: float, delta_max: float) -> pd.Series | None:
    d = sub["delta"].abs()
    band = sub[(d >= delta_min) & (d <= delta_max)]
    if band.empty:
        return None
    mid = (delta_min + delta_max) / 2.0
    idx = (band["delta"].abs() - mid).abs().idxmin()
    return band.loc[idx]


def _pick_by_pct_otm(sub: pd.DataFrame, S: float, pct: float, right: str) -> pd.Series | None:
    if sub.empty:
        return None
    target = S * (1 - pct) if right == "P" else S * (1 + pct)
    k = _nearest_strike(sub["strike"].to_numpy(), target)
    row = sub[np.isclose(sub["strike"].astype(float), k)]
    return row.iloc[0] if not row.empty else None


def _pick_by_empirical(
    sub: pd.DataFrame, S: float, target_prob: float, right: str, dist: EmpiricalDist | None
) -> pd.Series | None:
    if dist is None:
        raise ValueError("select_strikes: strike_rule='empirical_prob' requires `dist`")
    if sub.empty:
        return None
    best_k, best_diff = None, None
    for k in sorted(sub["strike"].astype(float).unique()):
        x = math.log(k / S)
        p = dist.prob_below(x) if right == "P" else dist.prob_above(x)
        diff = abs(p - target_prob)
        if best_diff is None or diff < best_diff:
            best_diff, best_k = diff, k
    if best_k is None:
        return None
    row = sub[np.isclose(sub["strike"].astype(float), best_k)]
    return row.iloc[0] if not row.empty else None


def _pick(sub: pd.DataFrame, right: str, S: float, cfg: EntryConfig, dist: EmpiricalDist | None) -> pd.Series | None:
    if cfg.strike_rule == "delta":
        return _pick_by_delta(sub, cfg.delta_min, cfg.delta_max)
    if cfg.strike_rule == "pct_otm":
        return _pick_by_pct_otm(sub, S, cfg.pct_otm, right)
    if cfg.strike_rule == "empirical_prob":
        return _pick_by_empirical(sub, S, cfg.target_prob_itm, right, dist)
    raise ValueError(f"select_strikes: unknown strike_rule {cfg.strike_rule!r}")


def select_strikes(
    chain: pd.DataFrame, S: float, cfg: EntryConfig, *, dist: EmpiricalDist | None = None, **_
) -> dict | None:
    """Pick strikes for `cfg.strategy` under `cfg.strike_rule`. Returns None if no listed
    strike qualifies (StrikeRule miss, or a long leg would cross/equal the short leg)."""
    puts = chain[chain["right"] == "P"]
    calls = chain[chain["right"] == "C"]
    inc = _increment(chain["strike"].to_numpy()) if not chain.empty else 1.0

    out: dict = {}

    if cfg.strategy in ("put_credit_spread", "iron_condor", "short_strangle", "short_put"):
        row = _pick(puts, "P", S, cfg, dist)
        if row is None:
            return None
        out["short_put"] = float(row["strike"])
        out["short_put_delta"] = float(row["delta"])

    if cfg.strategy in ("call_credit_spread", "iron_condor", "short_strangle"):
        row = _pick(calls, "C", S, cfg, dist)
        if row is None:
            return None
        out["short_call"] = float(row["strike"])
        out["short_call_delta"] = float(row["delta"])

    if cfg.strategy in ("put_credit_spread", "iron_condor"):
        target = out["short_put"] - cfg.width_strikes * inc
        avail = puts["strike"].to_numpy()
        if len(avail) == 0:
            return None
        long_k = _nearest_strike(avail, target)
        if long_k >= out["short_put"]:
            return None
        out["long_put"] = long_k

    if cfg.strategy in ("call_credit_spread", "iron_condor"):
        target = out["short_call"] + cfg.width_strikes * inc
        avail = calls["strike"].to_numpy()
        if len(avail) == 0:
            return None
        long_k = _nearest_strike(avail, target)
        if long_k <= out["short_call"]:
            return None
        out["long_call"] = long_k

    if cfg.strategy == "long_strangle":
        prow = _pick(puts, "P", S, cfg, dist)
        crow = _pick(calls, "C", S, cfg, dist)
        if prow is None or crow is None:
            return None
        out["long_put"] = float(prow["strike"])
        out["long_call"] = float(crow["strike"])

    return out


# --------------------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------------------


def _liquidity_reason(chain: pd.DataFrame, legs: list[Leg], costs: CostModel) -> tuple[str | None, bool]:
    """Returns `(reason, oi_was_unknown)`: `reason` is the funnel reason the structure
    fails OPI §4.2 liquidity, or None if it passes. Checked leg by leg, first failure
    wins (matches the old short-circuit `_passes_liquidity` behaviour, but now
    attributed instead of collapsed to a bool).

    `open_interest < 0` is the `OPEN_INTEREST_UNKNOWN` sentinel (real ThetaData bulk CSV
    exports have no OI column at all -- see data/providers/csv_export.py). Unknown OI
    must never be treated as "0 contracts of interest": that would silently reject every
    row once a real (non-synthetic, non-REST) data source is used, exactly the class of
    zero-trade failure this funnel exists to catch. So the OI check is skipped (not
    failed) when OI is unknown, and `oi_was_unknown` is set so the caller can still make
    that visible -- a skipped filter is not the same as a passed one."""
    oi_was_unknown = False
    for leg in legs:
        row = chain[np.isclose(chain["strike"].astype(float), leg.strike) & (chain["right"] == leg.right)]
        if row.empty:
            return "liquidity_no_quote", oi_was_unknown
        r = row.iloc[0]
        bid, ask = float(r["bid"]), float(r["ask"])
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return "liquidity_spread", oi_was_unknown
        if (ask - bid) / mid > costs.max_spread_pct_of_mid:
            return "liquidity_spread", oi_was_unknown
        if leg.ratio < 0:  # the short (liquidity-sensitive) leg -- OPI §4.2
            if bid < costs.min_bid:
                return "liquidity_min_bid", oi_was_unknown
            oi = float(r["open_interest"])
            if oi < 0:
                oi_was_unknown = True
            elif oi < costs.min_open_interest:
                return "liquidity_oi", oi_was_unknown
    return None, oi_was_unknown


def _diagnose_strike_failure(chain: pd.DataFrame, S: float, cfg: EntryConfig, dist: EmpiricalDist | None) -> str:
    """Called only when `select_strikes` returned None -- re-walks the same branches to
    say *which* one failed: the short/debit leg pick itself (`no_strike`), or the
    defined-risk long leg placed `width_strikes` away (`no_width_strike`)."""
    puts = chain[chain["right"] == "P"]
    calls = chain[chain["right"] == "C"]

    if cfg.strategy in ("put_credit_spread", "iron_condor", "short_strangle", "short_put"):
        if _pick(puts, "P", S, cfg, dist) is None:
            return "no_strike"
    if cfg.strategy in ("call_credit_spread", "iron_condor", "short_strangle"):
        if _pick(calls, "C", S, cfg, dist) is None:
            return "no_strike"
    if cfg.strategy == "long_strangle":
        if _pick(puts, "P", S, cfg, dist) is None or _pick(calls, "C", S, cfg, dist) is None:
            return "no_strike"
    # Every leg pick succeeded -- select_strikes still returned None, so the failure is in
    # the width step (no listed strike `width_strikes` away, or it crosses the short leg).
    return "no_width_strike"


def _market_filter_ok(strategy: str, state: MarketState, market_filter: MarketFilter) -> bool:
    if market_filter == "all":
        return True
    if market_filter == "skip_bearish":
        return state != MarketState.BEARISH
    if market_filter == "skip_bullish":
        return state != MarketState.BULLISH
    if market_filter == "trend_aligned":
        if strategy == "long_strangle":
            return True
        if strategy in ("put_credit_spread", "short_put"):
            return state != MarketState.BEARISH
        if strategy == "call_credit_spread":
            return state != MarketState.BULLISH
        if strategy in ("iron_condor", "short_strangle"):
            return state == MarketState.NEUTRAL
    return True


def _dividend_yield(store, root: str, asof: date, S: float) -> float:
    try:
        under = store.underlying(root, end=asof)
    except Exception:
        return 0.0
    if under is None or under.empty or "dividend" not in under.columns:
        return 0.0
    idx = pd.DatetimeIndex(under["date"] if "date" in under.columns else under.index)
    one_year_ago = pd.Timestamp(asof) - pd.Timedelta(days=365)
    mask = (idx < pd.Timestamp(asof)) & (idx >= one_year_ago)
    total_div = float(under.loc[mask, "dividend"].sum()) if mask.any() else 0.0
    return total_div / S if S > 0 else 0.0


def _compute_edge(
    strategy: str,
    strikes: dict,
    credit: float,
    width: float,
    max_loss: float,
    chain: pd.DataFrame,
    S: float,
    T_years: float,
    r: float,
    q: float,
    dist: EmpiricalDist,
) -> dict:
    """Returns {p_theo_loss, p_actual_loss, edge_prob, edge_ev, short_delta, iv_entry}."""

    def _iv_of(strike: float, right: str) -> float:
        row = chain[np.isclose(chain["strike"].astype(float), strike) & (chain["right"] == right)]
        return float(row.iloc[0]["iv"]) if not row.empty else float("nan")

    if strategy in _DEFINED_RISK_2LEG:
        right = "P" if strategy == "put_credit_spread" else "C"
        short_k = strikes["short_put"] if right == "P" else strikes["short_call"]
        long_k = strikes["long_put"] if right == "P" else strikes["long_call"]
        iv = _iv_of(short_k, right)
        er = spread_edge(
            credit=credit, width=width, short_strike=short_k, long_strike=long_k,
            right=right, S=S, T_years=T_years, iv=iv, r=r, q=q, dist=dist, with_ci=False,
        )
        return {
            "p_theo_loss": er.p_theo_loss, "p_actual_loss": er.p_actual_loss,
            "edge_prob": er.edge_prob, "edge_ev": er.edge_ev,
            "short_delta": strikes.get("short_put_delta", strikes.get("short_call_delta")),
            "iv_entry": iv,
        }

    if strategy == "iron_condor":
        iv_p = _iv_of(strikes["short_put"], "P")
        iv_c = _iv_of(strikes["short_call"], "C")
        put_width = strikes["short_put"] - strikes["long_put"]
        call_width = strikes["long_call"] - strikes["short_call"]
        put_er = spread_edge(
            credit=credit / 2.0, width=put_width, short_strike=strikes["short_put"],
            long_strike=strikes["long_put"], right="P", S=S, T_years=T_years, iv=iv_p,
            r=r, q=q, dist=dist, with_ci=False,
        )
        call_er = spread_edge(
            credit=credit / 2.0, width=call_width, short_strike=strikes["short_call"],
            long_strike=strikes["long_call"], right="C", S=S, T_years=T_years, iv=iv_c,
            r=r, q=q, dist=dist, with_ci=False,
        )
        # Summing the two sides is EXACT here, not an approximation. The loss-onset
        # events {S_T < short_put} and {S_T > short_call} are mutually exclusive, so the
        # probabilities add; the EVs add by linearity of expectation. The even credit/2
        # split between the sides is arbitrary (index skew puts most of the credit on the
        # put side) but cancels in the sum, and neither probability depends on it.
        return {
            "p_theo_loss": put_er.p_theo_loss + call_er.p_theo_loss,
            "p_actual_loss": put_er.p_actual_loss + call_er.p_actual_loss,
            "edge_prob": (put_er.p_theo_loss + call_er.p_theo_loss)
            - (put_er.p_actual_loss + call_er.p_actual_loss),
            "edge_ev": put_er.edge_ev + call_er.edge_ev,
            "short_delta": strikes.get("short_put_delta"),
            "iv_entry": (iv_p + iv_c) / 2.0,
        }

    if strategy == "short_strangle":
        iv_p = _iv_of(strikes["short_put"], "P")
        iv_c = _iv_of(strikes["short_call"], "C")
        put_credit = max(credit / 2.0, 0.0)
        call_credit = max(credit / 2.0, 0.0)
        put_er = naked_edge(
            credit=put_credit, short_strike=strikes["short_put"], right="P", S=S,
            T_years=T_years, iv=iv_p, r=r, q=q, dist=dist, with_ci=False,
        )
        call_er = naked_edge(
            credit=call_credit, short_strike=strikes["short_call"], right="C", S=S,
            T_years=T_years, iv=iv_c, r=r, q=q, dist=dist, with_ci=False,
        )
        return {
            "p_theo_loss": put_er.p_theo_loss + call_er.p_theo_loss,
            "p_actual_loss": put_er.p_actual_loss + call_er.p_actual_loss,
            "edge_prob": (put_er.p_theo_loss + call_er.p_theo_loss)
            - (put_er.p_actual_loss + call_er.p_actual_loss),
            "edge_ev": put_er.edge_ev + call_er.edge_ev,
            "short_delta": strikes.get("short_put_delta"),
            "iv_entry": (iv_p + iv_c) / 2.0,
        }

    if strategy == "short_put":
        iv = _iv_of(strikes["short_put"], "P")
        er = naked_edge(
            credit=credit, short_strike=strikes["short_put"], right="P", S=S,
            T_years=T_years, iv=iv, r=r, q=q, dist=dist, with_ci=False,
        )
        return {
            "p_theo_loss": er.p_theo_loss, "p_actual_loss": er.p_actual_loss,
            "edge_prob": er.edge_prob, "edge_ev": er.edge_ev,
            "short_delta": strikes.get("short_put_delta"), "iv_entry": iv,
        }

    if strategy == "long_strangle":
        # S6 is the inverse trade (risk-comparison baseline); the edge/require_positive_edge
        # filter is a premium-SELLING concept and is waived here by design (judgement call).
        iv_p = _iv_of(strikes["long_put"], "P")
        iv_c = _iv_of(strikes["long_call"], "C")
        return {
            "p_theo_loss": float("nan"), "p_actual_loss": float("nan"),
            "edge_prob": float("nan"), "edge_ev": 0.0,
            "short_delta": float("nan"), "iv_entry": (iv_p + iv_c) / 2.0,
        }

    raise ValueError(f"_compute_edge: unknown strategy {strategy!r}")


def propose_trade(
    store, root: str, asof: date, cfg: BacktestConfig, dist_cache: dict | None = None
) -> tuple[TradeProposal | None, dict]:
    """Select expiry + strikes, price, filter, and rank -- STRATEGY.md §4.1-4.3, §2.

    Ranks across all expiries in [dte_min, dte_max] that survive every filter, by highest
    `edge_ev` (STRATEGY.md §4.2 'Ranking').

    Returns `(proposal, funnel)`: `proposal` is None if nothing qualifies; `funnel` is a
    structured account (see `FUNNEL_REASONS`/`new_funnel`) of what happened to every
    expiry considered for this (root, asof) opportunity -- every `return` and `continue`
    below increments exactly one named `funnel['rejected']` counter, so a caller can
    always answer "why did this opportunity produce zero trades?" without re-deriving it.
    """
    entry = cfg.entry
    if dist_cache is None:
        dist_cache = {}

    funnel = new_funnel(root, asof)

    expiries = store.expiries(root, asof, dte_min=entry.dte_min, dte_max=entry.dte_max)
    funnel["expiries_considered"] = len(expiries)
    if not expiries:
        funnel["rejected"]["no_expiry"] += 1
        return None, funnel

    closes = store.closes(root, end=asof)
    market_state_val = universe.market_state(closes, asof)
    if not _market_filter_ok(entry.strategy, market_state_val, entry.market_filter):
        funnel["rejected"]["market_state"] += len(expiries)
        return None, funnel
    realized_vol_val = universe.realized_vol(closes, asof) if len(closes[pd.DatetimeIndex(closes.index) < pd.Timestamp(asof)]) > 21 else float("nan")

    candidates: list[TradeProposal] = []

    for expiry in expiries:
        chain = store.chain(root, asof, expiry=expiry)
        if chain.empty:
            funnel["rejected"]["empty_chain"] += 1
            continue
        schema.assert_no_lookahead(chain, asof)
        S = float(chain["underlying_price"].iloc[0])

        try:
            atm_iv_val = universe.atm_iv(chain, expiry)
        except ValueError:
            funnel["rejected"]["no_atm_iv"] += 1
            continue

        iv_key = ("iv_hist", root)
        iv_hist: dict = dist_cache.setdefault(iv_key, {})
        iv_hist[pd.Timestamp(asof)] = atm_iv_val
        iv_series = pd.Series(iv_hist).sort_index()
        try:
            iv_rank_val = universe.iv_rank(iv_series, asof)
        except ValueError:
            iv_rank_val = float("nan")

        # The empirical dist is needed for edge computation regardless of strike_rule.
        dte_calendar = (expiry - asof).days
        try:
            horizon = len(store.trading_dates(root, asof, expiry)) - 1
        except Exception:
            horizon = max(int(round(dte_calendar * 5 / 7)), 1)
        horizon = max(horizon, 1)
        dist_key = ("empirical", root, horizon, asof)
        dist = dist_cache.get(dist_key)
        if dist is None:
            try:
                dist = build_empirical(closes, horizon, cfg.empirical, asof)
            except ValueError:
                funnel["rejected"]["no_empirical_dist"] += 1
                continue
            dist_cache[dist_key] = dist

        strikes = select_strikes(chain, S, entry, dist=dist)
        if strikes is None:
            funnel["rejected"][_diagnose_strike_failure(chain, S, entry, dist)] += 1
            continue

        legs = strategies.build_legs(
            entry.strategy, root, expiry,
            short_put=strikes.get("short_put"), long_put=strikes.get("long_put"),
            short_call=strikes.get("short_call"), long_call=strikes.get("long_call"),
        )

        try:
            econ = strategies.structure_economics(legs, chain)
        except schema.SchemaError:
            funnel["rejected"]["structure_economics_error"] += 1
            continue
        credit, max_loss, width = econ["credit"], econ["max_loss"], econ["width"]

        liq_reason, oi_was_unknown = _liquidity_reason(chain, legs, cfg.costs)
        if oi_was_unknown:
            funnel["liquidity_oi_unknown"] += 1
        if liq_reason is not None:
            funnel["rejected"][liq_reason] += 1
            continue

        is_debit = entry.strategy in _DEBIT_STRATEGIES
        if not is_debit:
            if credit < entry.min_credit:
                funnel["rejected"]["min_credit"] += 1
                continue
            if math.isfinite(max_loss) and max_loss > 0:
                er = credit / (width - credit) if (width - credit) > 0 else math.inf
                if not (entry.expected_return_min < er <= entry.expected_return_max):
                    funnel["rejected"]["expected_return_band"] += 1
                    continue

        r = cfg.risk_free_rate
        q = _dividend_yield(store, root, asof, S)
        T_years = max((expiry - asof).days, 0) / 365.0

        edge = _compute_edge(entry.strategy, strikes, credit, width, max_loss, chain, S, T_years, r, q, dist)

        if entry.require_positive_edge and not is_debit:
            if not (edge["edge_ev"] > 0):
                funnel["rejected"]["negative_edge"] += 1
                continue

        dte_entry = (expiry - asof).days
        meta = {
            "short_delta": edge["short_delta"],
            "iv_entry": edge["iv_entry"],
            "iv_rank": iv_rank_val,
            "realized_vol": realized_vol_val,
            "market_state": market_state_val.value,
            "p_theo_loss": edge["p_theo_loss"],
            "p_actual_loss": edge["p_actual_loss"],
            "edge_prob": edge["edge_prob"],
            "edge_ev": edge["edge_ev"],
            "underlying_entry": S,
            "dte_entry": dte_entry,
            "strikes": strikes,
        }
        candidates.append(
            TradeProposal(
                strategy=entry.strategy, root=root, expiry=expiry, legs=legs,
                credit=credit, max_loss=max_loss, width=width, meta=meta,
            )
        )

    funnel["candidates"] = len(candidates)
    if not candidates:
        return None, funnel

    finite_evs = [c.meta["edge_ev"] for c in candidates if math.isfinite(c.meta["edge_ev"])]
    funnel["accepted"] = 1
    funnel["best_edge_ev"] = max(finite_evs) if finite_evs else None

    best = max(candidates, key=lambda c: c.meta["edge_ev"] if math.isfinite(c.meta["edge_ev"]) else -math.inf)
    return best, funnel
