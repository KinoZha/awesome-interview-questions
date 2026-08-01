"""Canonical data contracts. Every dataframe crossing a module boundary conforms to these.

Add a column here first; never introduce one inline. See CLAUDE.md rule 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd

Right = Literal["C", "P"]
Side = Literal["BUY", "SELL"]
PriceKind = Literal["bid", "ask", "mid"]

CONTRACT_MULTIPLIER = 100
"""Applied exactly once, at the position layer (CLAUDE.md rule 6)."""


# --------------------------------------------------------------------------------------
# Option chain
# --------------------------------------------------------------------------------------

CHAIN_DTYPES: dict[str, str] = {
    "root": "string",
    "quote_date": "datetime64[ns]",
    "ms_of_day": "int32",
    "expiry": "datetime64[ns]",
    "strike": "float64",  # dollars per share
    "right": "string",  # 'C' | 'P'
    "bid": "float64",
    "ask": "float64",
    "bid_size": "int32",
    "ask_size": "int32",
    "last": "float64",
    "volume": "int64",
    "open_interest": "int64",
    "underlying_price": "float64",
    # recomputed by quant.bs -- authoritative
    "iv": "float64",
    "delta": "float64",
    "gamma": "float64",
    "theta": "float64",
    "vega": "float64",
    "rho": "float64",
    # vendor-supplied, kept only for data-quality comparison
    "iv_vendor": "float64",
    "delta_vendor": "float64",
    "source": "string",
    "is_synthetic": "bool",
}

CHAIN_COLUMNS: list[str] = list(CHAIN_DTYPES)

CHAIN_REQUIRED: list[str] = [
    "root",
    "quote_date",
    "expiry",
    "strike",
    "right",
    "bid",
    "ask",
    "underlying_price",
]
"""Columns a provider must supply. The rest are derived or nullable."""

UNDERLYING_DTYPES: dict[str, str] = {
    "root": "string",
    "date": "datetime64[ns]",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "int64",
    "dividend": "float64",  # cash dividend paid with ex-date == this date, else 0.0
}


class SchemaError(ValueError):
    """Raised when a dataframe violates its contract. Never downgrade to a warning."""


def validate_chain(df: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """Validate + coerce an option-chain frame to CHAIN_DTYPES. Raises SchemaError.

    strict=False allows derived columns (greeks) to be absent -- used by providers
    before quant.bs fills them in.
    """
    missing = [c for c in CHAIN_REQUIRED if c not in df.columns]
    if missing:
        raise SchemaError(f"chain missing required columns: {missing}")

    wanted = CHAIN_COLUMNS if strict else [c for c in CHAIN_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in CHAIN_COLUMNS]
    if extra:
        raise SchemaError(f"chain has unknown columns: {extra}")
    if strict:
        absent = [c for c in wanted if c not in df.columns]
        if absent:
            raise SchemaError(f"chain missing derived columns: {absent}")

    out = df.copy()
    for col in wanted:
        out[col] = out[col].astype(CHAIN_DTYPES[col])

    bad_right = ~out["right"].isin(["C", "P"])
    if bad_right.any():
        raise SchemaError(f"{int(bad_right.sum())} rows with right not in C/P")
    if (out["strike"] <= 0).any():
        raise SchemaError("non-positive strike")
    if (out["bid"] < 0).any() or (out["ask"] < 0).any():
        raise SchemaError("negative bid/ask")
    crossed = out["bid"] > out["ask"]
    if crossed.any():
        raise SchemaError(f"{int(crossed.sum())} crossed quotes (bid > ask)")
    if (out["expiry"] < out["quote_date"]).any():
        raise SchemaError("expiry before quote_date")
    return out[wanted]


# --------------------------------------------------------------------------------------
# Trades / positions
# --------------------------------------------------------------------------------------


class ExitReason(str, Enum):
    EXPIRY = "expiry"
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    DTE = "dte"
    DELTA_BREACH = "delta_breach"
    ASSIGNED = "assigned"
    END_OF_BACKTEST = "end_of_backtest"


class MarketState(str, Enum):
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"


@dataclass(frozen=True, slots=True)
class Leg:
    """One option contract in a structure. `ratio` is signed: +1 long, -1 short."""

    root: str
    expiry: date
    strike: float
    right: Right
    ratio: int

    @property
    def key(self) -> tuple[str, date, float, str]:
        return (self.root, self.expiry, self.strike, self.right)


@dataclass(frozen=True, slots=True)
class Fill:
    """A single executed leg. price_kind records which side of the book we crossed."""

    leg: Leg
    qty: int  # contracts, always positive
    side: Side
    price: float  # dollars per share
    price_kind: PriceKind
    commission: float
    fees: float
    mid: float
    ts: date

    @property
    def spread_cost_vs_mid(self) -> float:
        """Dollars given up to the bid/ask spread on this fill (>= 0)."""
        signed = (self.price - self.mid) if self.side == "BUY" else (self.mid - self.price)
        return signed * self.qty * CONTRACT_MULTIPLIER

    @property
    def cash(self) -> float:
        """Signed cash impact including costs."""
        gross = -self.price * self.qty * CONTRACT_MULTIPLIER
        if self.side == "SELL":
            gross = -gross
        return gross - self.commission - self.fees


@dataclass
class Position:
    """An open multi-leg structure, tracked as a unit."""

    position_id: str
    strategy: str
    root: str
    entry_date: date
    expiry: date
    legs: list[Leg]
    qty: int
    open_fills: list[Fill] = field(default_factory=list)
    close_fills: list[Fill] = field(default_factory=list)
    entry_credit: float = 0.0  # per spread, dollars/share, >0 for credit structures
    max_loss: float = 0.0  # per spread, dollars/share; inf for undefined risk
    margin: float = 0.0  # dollars, total for the position
    exit_date: date | None = None
    exit_reason: ExitReason | None = None
    meta: dict = field(default_factory=dict)  # selection diagnostics: iv_rank, edge_ev, ...

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    @property
    def realized_pnl(self) -> float:
        return sum(f.cash for f in self.open_fills) + sum(f.cash for f in self.close_fills)


TRADE_DTYPES: dict[str, str] = {
    "position_id": "string",
    "strategy": "string",
    "root": "string",
    "entry_date": "datetime64[ns]",
    "exit_date": "datetime64[ns]",
    "expiry": "datetime64[ns]",
    "dte_entry": "int32",
    "qty": "int32",
    "short_strike": "float64",
    "long_strike": "float64",
    "width": "float64",
    # Per-side strikes. A two-sided structure (iron condor, strangle) has no single
    # short/long pair, and the lifecycle chart must draw both bands or it silently
    # hides half the risk. NaN on the side a structure does not use.
    "short_put_strike": "float64",
    "long_put_strike": "float64",
    "short_call_strike": "float64",
    "long_call_strike": "float64",
    "entry_credit": "float64",
    "exit_debit": "float64",
    "max_loss": "float64",
    "margin": "float64",
    "pnl": "float64",
    "pnl_pct_of_max_loss": "float64",
    "commission": "float64",
    "fees": "float64",
    "slippage": "float64",
    "exit_reason": "string",
    # selection-time diagnostics -- these are what the cohort charts slice on
    "short_delta": "float64",
    "iv_entry": "float64",
    "iv_rank": "float64",
    "realized_vol": "float64",
    "market_state": "string",
    "p_theo_loss": "float64",
    "p_actual_loss": "float64",
    "edge_prob": "float64",
    "edge_ev": "float64",
    "underlying_entry": "float64",
    "underlying_exit": "float64",
    # attribution, summed over the life of the trade
    "pnl_delta": "float64",
    "pnl_gamma": "float64",
    "pnl_vega": "float64",
    "pnl_theta": "float64",
    "pnl_residual": "float64",
}

TRADE_COLUMNS: list[str] = list(TRADE_DTYPES)

EQUITY_DTYPES: dict[str, str] = {
    "date": "datetime64[ns]",
    "cash": "float64",
    "positions_value": "float64",
    "equity": "float64",
    "margin_used": "float64",
    "margin_pct": "float64",
    "open_positions": "int32",
    "net_delta": "float64",
    "net_gamma": "float64",
    "net_vega": "float64",
    "net_theta": "float64",
}


SNAPSHOT_DTYPES: dict[str, str] = {
    "position_id": "string",
    "date": "datetime64[ns]",
    "underlying_price": "float64",
    "mark": "float64",  # per spread, dollars/share, positive = cost to close
    "mtm_pnl": "float64",  # dollars, position total, mark-to-mid
    "dte": "int32",
    "short_delta": "float64",
    "net_delta": "float64",
    "net_gamma": "float64",
    "net_vega": "float64",
    "net_theta": "float64",
    "iv_short": "float64",
    # per-day attribution increments, ARCHITECTURE.md §4
    "d_delta": "float64",
    "d_gamma": "float64",
    "d_vega": "float64",
    "d_theta": "float64",
    "d_residual": "float64",
}
"""Per-position, per-day marks. Drives the trade-lifecycle view (STRATEGY.md §8B)."""


def empty_frame(dtypes: dict[str, str]) -> pd.DataFrame:
    """An empty frame with the right columns and dtypes, so downstream code never
    special-cases 'no trades'."""
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in dtypes.items()})


def validate_frame(df: pd.DataFrame, dtypes: dict[str, str], name: str) -> pd.DataFrame:
    """Coerce and check a frame against a dtype map."""
    missing = [c for c in dtypes if c not in df.columns]
    if missing:
        raise SchemaError(f"{name} missing columns: {missing}")
    extra = [c for c in df.columns if c not in dtypes]
    if extra:
        raise SchemaError(f"{name} has unknown columns: {extra}")
    out = df.copy()
    for col, dt in dtypes.items():
        if dt.startswith("int") and out[col].isna().any():
            raise SchemaError(f"{name}.{col} is integer-typed but contains NaN")
        out[col] = out[col].astype(dt)
    return out[list(dtypes)]


def assert_no_lookahead(df: pd.DataFrame, asof, col: str = "quote_date") -> None:
    """CLAUDE.md rule 1. Call this in every selection/exit path."""
    if df.empty:
        return
    latest = pd.Timestamp(df[col].max())
    if latest > pd.Timestamp(asof):
        raise SchemaError(f"lookahead: {col} max {latest} > asof {pd.Timestamp(asof)}")


__all__ = [
    "CHAIN_COLUMNS",
    "CHAIN_DTYPES",
    "CHAIN_REQUIRED",
    "CONTRACT_MULTIPLIER",
    "EQUITY_DTYPES",
    "ExitReason",
    "Fill",
    "Leg",
    "MarketState",
    "Position",
    "PriceKind",
    "Right",
    "SchemaError",
    "SNAPSHOT_DTYPES",
    "Side",
    "TRADE_COLUMNS",
    "TRADE_DTYPES",
    "UNDERLYING_DTYPES",
    "assert_no_lookahead",
    "empty_frame",
    "validate_chain",
    "validate_frame",
]
