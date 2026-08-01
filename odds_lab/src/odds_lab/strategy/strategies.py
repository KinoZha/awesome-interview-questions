"""S1..S6 structure constructors -- STRATEGY.md §3.

`build_legs` turns a chosen set of strikes into `Leg` objects; `structure_economics` prices
the resulting structure at bid/ask (sell crosses bid, buy crosses ask -- CLAUDE.md rule 2)
and derives credit/max_loss/width/breakevens.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from odds_lab.config import Strategy
from odds_lab.schema import Leg, SchemaError

__all__ = ["build_legs", "structure_economics"]


def _require(*strikes: float | None) -> None:
    if any(s is None for s in strikes):
        raise ValueError("build_legs: missing required strike(s) for this strategy")


def build_legs(
    strategy: Strategy,
    root: str,
    expiry,
    *,
    short_put: float | None = None,
    long_put: float | None = None,
    short_call: float | None = None,
    long_call: float | None = None,
) -> list[Leg]:
    """Build the leg list for `strategy`. STRATEGY.md §3 (S1-S6)."""
    if strategy == "put_credit_spread":
        _require(short_put, long_put)
        return [
            Leg(root, expiry, short_put, "P", -1),
            Leg(root, expiry, long_put, "P", +1),
        ]
    if strategy == "call_credit_spread":
        _require(short_call, long_call)
        return [
            Leg(root, expiry, short_call, "C", -1),
            Leg(root, expiry, long_call, "C", +1),
        ]
    if strategy == "iron_condor":
        _require(short_put, long_put, short_call, long_call)
        return [
            Leg(root, expiry, short_put, "P", -1),
            Leg(root, expiry, long_put, "P", +1),
            Leg(root, expiry, short_call, "C", -1),
            Leg(root, expiry, long_call, "C", +1),
        ]
    if strategy == "short_strangle":
        _require(short_put, short_call)
        return [
            Leg(root, expiry, short_put, "P", -1),
            Leg(root, expiry, short_call, "C", -1),
        ]
    if strategy == "short_put":
        _require(short_put)
        return [Leg(root, expiry, short_put, "P", -1)]
    if strategy == "long_strangle":
        _require(long_put, long_call)
        return [
            Leg(root, expiry, long_put, "P", +1),
            Leg(root, expiry, long_call, "C", +1),
        ]
    raise ValueError(f"build_legs: unknown strategy {strategy!r}")


def structure_economics(legs: list[Leg], quotes: pd.DataFrame) -> dict:
    """Credit, max_loss, width, breakevens for an arbitrary leg list, priced at bid/ask.

    `quotes` must contain a row per leg (matched on strike + right); raises SchemaError
    if a leg has no quote -- never fabricate a price (CLAUDE.md rule 4/fail-loud).

    Sign convention: `credit` is dollars/share received (positive) or paid (negative, for
    a debit structure like `long_strangle`). `max_loss` is dollars/share, `inf` for
    undefined-risk (naked) structures -- STRATEGY.md deliverable list.
    """
    credit = 0.0
    for leg in legs:
        row = quotes[
            np.isclose(quotes["strike"].astype(float), leg.strike) & (quotes["right"] == leg.right)
        ]
        if row.empty:
            raise SchemaError(f"structure_economics: missing quote for leg {leg.key}")
        r = row.iloc[0]
        price = float(r["bid"]) if leg.ratio < 0 else float(r["ask"])
        credit += -leg.ratio * price

    puts = sorted((l for l in legs if l.right == "P"), key=lambda l: l.strike)
    calls = sorted((l for l in legs if l.right == "C"), key=lambda l: l.strike)
    short_puts = [l for l in puts if l.ratio < 0]
    long_puts = [l for l in puts if l.ratio > 0]
    short_calls = [l for l in calls if l.ratio < 0]
    long_calls = [l for l in calls if l.ratio > 0]

    put_width = short_puts[0].strike - long_puts[0].strike if (short_puts and long_puts) else None
    call_width = long_calls[0].strike - short_calls[0].strike if (short_calls and long_calls) else None

    if short_puts and not long_puts and not calls:
        # naked short put (S5)
        max_loss = math.inf
        width = math.nan
        breakevens = [short_puts[0].strike - credit]
    elif short_puts and short_calls and not long_puts and not long_calls:
        # short strangle (S4), naked both sides
        max_loss = math.inf
        width = math.nan
        breakevens = [short_puts[0].strike - credit, short_calls[0].strike + credit]
    elif long_puts and long_calls and not short_puts and not short_calls:
        # long strangle (S6), debit structure
        max_loss = -credit
        width = math.nan
        breakevens = [long_puts[0].strike - max_loss, long_calls[0].strike + max_loss]
    elif put_width is not None and call_width is not None:
        # iron condor (S3): only one side can be max-loss at a time
        width = max(put_width, call_width)
        max_loss = width - credit
        breakevens = [short_puts[0].strike - credit, short_calls[0].strike + credit]
    elif put_width is not None:
        # put credit spread (S1)
        width = put_width
        max_loss = width - credit
        breakevens = [short_puts[0].strike - credit]
    elif call_width is not None:
        # call credit spread (S2)
        width = call_width
        max_loss = width - credit
        breakevens = [short_calls[0].strike + credit]
    else:
        raise ValueError("structure_economics: cannot classify leg structure")

    return {"credit": credit, "max_loss": max_loss, "width": width, "breakevens": breakevens}
