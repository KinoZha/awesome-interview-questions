"""P&L attribution -- ARCHITECTURE.md §4.

`dP = delta*dS + 1/2*gamma*dS^2 + vega*dIV + theta*dt + residual`, computed with the
PREVIOUS snapshot's greeks and reconciled against the actual mark-to-mid change. The
residual is always reported, never absorbed into another term.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["attribute"]


def attribute(prev: dict, cur: dict) -> dict:
    """`prev`/`cur` are snapshot-shaped dicts with at least: date, underlying_price,
    iv_short, mark, net_delta, net_gamma, net_vega, net_theta. `mark` is dollars/share,
    cost-to-close (positive=cost) -- so the actual position-value change is
    `prev.mark - cur.mark` (a falling cost to close is a gain for a credit seller).

    Returns {d_delta, d_gamma, d_vega, d_theta, d_residual} that sum to that actual change.
    """
    dS = float(cur["underlying_price"]) - float(prev["underlying_price"])

    prev_iv = prev.get("iv_short")
    cur_iv = cur.get("iv_short")
    prev_iv = 0.0 if prev_iv is None or pd.isna(prev_iv) else float(prev_iv)
    cur_iv = 0.0 if cur_iv is None or pd.isna(cur_iv) else float(cur_iv)
    d_iv_points = (cur_iv - prev_iv) * 100.0  # vega convention: per 1 vol POINT (bs.py)

    dt_days = float((pd.Timestamp(cur["date"]) - pd.Timestamp(prev["date"])).days)

    d_delta = float(prev["net_delta"]) * dS
    d_gamma = 0.5 * float(prev["net_gamma"]) * dS * dS
    d_vega = float(prev["net_vega"]) * d_iv_points
    d_theta = float(prev["net_theta"]) * dt_days

    actual_dP = float(prev["mark"]) - float(cur["mark"])
    d_residual = actual_dP - (d_delta + d_gamma + d_vega + d_theta)

    return {
        "d_delta": d_delta,
        "d_gamma": d_gamma,
        "d_vega": d_vega,
        "d_theta": d_theta,
        "d_residual": d_residual,
    }
