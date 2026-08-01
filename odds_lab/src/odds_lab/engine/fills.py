"""Per-leg bid/ask fills + commissions/fees -- STRATEGY.md §7.1-7.2, CLAUDE.md rule 2.

Sell crosses to bid, buy crosses to ask for `fill_kind='bid_ask'` (the default and only
realistic mode); `mid` and `mid_plus_frac` are reporting/sensitivity variants.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from odds_lab.config import CostModel
from odds_lab.schema import Fill, Leg, PriceKind, Side

__all__ = ["fill_price", "execute"]


def fill_price(bid: float, ask: float, side: Side, costs: CostModel) -> tuple[float, PriceKind]:
    """Price + which side of the book was crossed, per `costs.fill_kind`."""
    if bid < 0 or ask < 0:
        raise ValueError(f"fill_price: negative quote bid={bid} ask={ask}")
    if bid > ask:
        raise ValueError(f"fill_price: crossed quote bid={bid} > ask={ask}")
    mid = (bid + ask) / 2.0

    if costs.fill_kind == "bid_ask":
        return (bid, "bid") if side == "SELL" else (ask, "ask")
    if costs.fill_kind == "mid":
        return (mid, "mid")
    if costs.fill_kind == "mid_plus_frac":
        half_spread = (ask - bid) / 2.0
        frac = costs.mid_frac
        price = mid - frac * half_spread if side == "SELL" else mid + frac * half_spread
        return (price, "mid")
    raise ValueError(f"fill_price: unknown fill_kind {costs.fill_kind!r}")


def execute(
    legs: list[Leg],
    qty: int,
    quotes: pd.DataFrame,
    side_map: dict,
    costs: CostModel,
    ts: date,
) -> list[Fill]:
    """Fill every leg in `legs` at `qty` contracts. `side_map` maps a `Leg` (or its `.key`)
    to a `Side` ('BUY' opens/closes long, 'SELL' opens/closes short). Raises ValueError on
    a missing quote or missing side -- never fabricate a price (CLAUDE.md rule 4)."""
    fills: list[Fill] = []
    for leg in legs:
        side = side_map.get(leg, side_map.get(leg.key))
        if side is None:
            raise ValueError(f"execute: no side given for leg {leg.key}")

        row = quotes[
            np.isclose(quotes["strike"].astype(float), leg.strike)
            & (quotes["right"] == leg.right)
            & (pd.to_datetime(quotes["expiry"]) == pd.Timestamp(leg.expiry))
        ]
        if row.empty:
            row = quotes[
                np.isclose(quotes["strike"].astype(float), leg.strike) & (quotes["right"] == leg.right)
            ]
        if row.empty:
            raise ValueError(f"execute: missing quote for leg {leg.key} at ts={ts}")

        r = row.iloc[0]
        bid, ask = float(r["bid"]), float(r["ask"])
        price, kind = fill_price(bid, ask, side, costs)
        mid = (bid + ask) / 2.0

        commission = costs.commission_per_contract * qty
        fees = costs.fees_per_contract_sell * qty if side == "SELL" else 0.0

        fills.append(
            Fill(
                leg=leg, qty=qty, side=side, price=price, price_kind=kind,
                commission=commission, fees=fees, mid=mid, ts=ts,
            )
        )
    return fills
