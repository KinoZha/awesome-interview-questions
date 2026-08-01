"""Cash, positions, equity, margin accounting, sizing -- STRATEGY.md §6.

Money is float dollars **per share** everywhere except at the boundary of this module,
which multiplies by `CONTRACT_MULTIPLIER` exactly once (CLAUDE.md rule 6). Refusals
(margin/style caps breached) are recorded on `self.rejected_trades` -- they must be visible
in the report, never silently dropped (per the task spec).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from odds_lab.config import RiskConfig
from odds_lab.schema import CONTRACT_MULTIPLIER, Leg, Position

__all__ = ["Portfolio"]


class Portfolio:
    def __init__(self, cfg: RiskConfig, risk_free_rate: float = 0.02):
        self.cfg = cfg
        self.risk_free_rate = risk_free_rate
        self.cash: float = cfg.starting_equity
        self.positions: list[Position] = []
        self.rejected_trades: list[dict] = []
        self._current_equity: float = cfg.starting_equity

    # -- equity / margin bookkeeping -----------------------------------------------------

    def equity(self, marks: dict[str, float]) -> float:
        """`marks`: position_id -> current cost-to-close, dollars/share. Total equity =
        cash + sum((entry_credit - mark) * qty * 100) over open positions."""
        val = self.cash
        for p in self.positions:
            if not p.is_open:
                continue
            mark = marks.get(p.position_id, 0.0)
            val += (p.entry_credit - mark) * p.qty * CONTRACT_MULTIPLIER
        self._current_equity = val
        return val

    def margin_used(self) -> float:
        return sum(p.margin for p in self.positions if p.is_open)

    def margin_pct(self) -> float:
        eq = self._current_equity
        return self.margin_used() / eq if eq else 0.0

    def style_exposure(self, style: str) -> float:
        return sum(p.margin for p in self.positions if p.is_open and p.strategy == style)

    def accrue_cash_interest(self, days: float = 1.0) -> None:
        """Cash earns the risk-free rate daily -- STRATEGY.md §6/§7.5."""
        self.cash *= (1.0 + self.risk_free_rate) ** (days / 365.0)

    # -- sizing ---------------------------------------------------------------------------

    def size(self, max_loss_per_spread: float, margin_per_spread: float, cfg: RiskConfig) -> int:
        """Contract count = floor(equity * risk_pct / risk_dollars_per_contract).
        `max_loss_per_spread` and `margin_per_spread` are both dollars/SHARE (pre-multiplier).
        For undefined-risk structures `max_loss_per_spread` is inf; size off margin instead.
        """
        equity = self._current_equity
        risk_basis = (
            max_loss_per_spread
            if math.isfinite(max_loss_per_spread) and max_loss_per_spread > 0
            else margin_per_spread
        )
        if not math.isfinite(risk_basis) or risk_basis <= 0:
            return 0
        budget = equity * cfg.risk_pct_per_trade
        qty = int(budget // (risk_basis * CONTRACT_MULTIPLIER))
        qty = min(max(qty, 0), cfg.max_contracts)
        if qty < cfg.min_contracts:
            return 0
        return qty

    def reg_t_margin(self, legs: list[Leg], qty: int, S: float, credit: float) -> float:
        """Reg-T approximation for NAKED short legs only -- STRATEGY.md §6:
        `max(20%*S - OTM_amount, 10%*S) * 100 + premium`, floor $250/contract.
        Defined-risk structures use `max_loss * qty * 100` directly (computed by the
        caller from `structure_economics`); this function is not meaningful for those.
        Sums independently across naked legs (put + call of a strangle) -- an approximation,
        real brokers usually take the greater single-side requirement plus the other side's
        premium; documented as a judgement call for the reviewer.
        """
        naked_shorts = [l for l in legs if l.ratio < 0]
        by_right = {"P": [l for l in legs if l.right == "P"], "C": [l for l in legs if l.right == "C"]}
        total_per_contract = 0.0
        for leg in naked_shorts:
            same_right = by_right[leg.right]
            has_long_cover = any(x.ratio > 0 for x in same_right)
            if has_long_cover:
                continue  # defined risk on this side; not part of the naked formula
            otm_amount = max(S - leg.strike, 0.0) if leg.right == "P" else max(leg.strike - S, 0.0)
            total_per_contract += max(0.20 * S - otm_amount, 0.10 * S) * 100.0
        if total_per_contract <= 0.0:
            return 0.0
        per_contract = max(total_per_contract + credit * 100.0, 250.0)
        return per_contract * qty

    # -- opening ---------------------------------------------------------------------------

    def try_open(self, position: Position, cfg: RiskConfig) -> tuple[bool, str | None]:
        """Enforce max_pct_per_style and max_margin_pct (STRATEGY.md §6). Registers and
        returns (True, None) on success; else (False, reason) and logs the refusal."""
        eq = self._current_equity
        if eq <= 0:
            reason = "non_positive_equity"
            self.rejected_trades.append(self._refusal(position, reason))
            return False, reason

        new_margin_total = self.margin_used() + position.margin
        if new_margin_total / eq > cfg.max_margin_pct:
            reason = "max_margin_pct_breach"
            self.rejected_trades.append(self._refusal(position, reason))
            return False, reason

        new_style_total = self.style_exposure(position.strategy) + position.margin
        if new_style_total / eq > cfg.max_pct_per_style:
            reason = "max_pct_per_style_breach"
            self.rejected_trades.append(self._refusal(position, reason))
            return False, reason

        self.positions.append(position)
        return True, None

    def _refusal(self, position: Position, reason: str) -> dict:
        return {
            "reason": reason,
            "root": position.root,
            "strategy": position.strategy,
            "entry_date": position.entry_date,
            "expiry": position.expiry,
            "qty": position.qty,
            "margin": position.margin,
            "equity": self._current_equity,
        }
