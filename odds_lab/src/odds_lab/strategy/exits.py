"""Exit policies -- STRATEGY.md §5. Pure functions, no state.

Priority order (judgement call -- STRATEGY.md lists the four rules but not an explicit
tie-break order; reviewer should confirm this is the intended production behavior):
    1. profit_target
    2. stop_loss
    3. delta_breach
    4. dte_exit
First trigger wins. `hold_to_expiry=True` disables every early-exit rule.

The TRIGGER is evaluated here on day D's snapshot; the caller (engine/loop.py) is
responsible for filling the close on day D+1's quotes -- no lookahead (STRATEGY.md §7.3).
"""

from __future__ import annotations

from odds_lab.config import ExitConfig
from odds_lab.schema import ExitReason, Position

__all__ = ["evaluate_exit"]


def evaluate_exit(position: Position, snapshot: dict, cfg: ExitConfig) -> ExitReason | None:
    """`snapshot` carries at least: mark, short_delta, dte, underlying_price (per
    STRATEGY.md §5). `mark` is the current cost-to-close, dollars/share (positive=cost),
    matching `Position.entry_credit`'s sign convention."""
    if cfg.hold_to_expiry:
        return None

    credit = position.entry_credit
    mark = snapshot.get("mark")

    if cfg.profit_target_pct is not None and credit > 0 and mark is not None:
        captured = credit - mark
        if captured >= cfg.profit_target_pct * credit:
            return ExitReason.PROFIT_TARGET

    if cfg.stop_loss_multiple is not None and credit > 0 and mark is not None:
        loss = mark - credit
        if loss >= cfg.stop_loss_multiple * credit:
            return ExitReason.STOP_LOSS

    if cfg.delta_breach is not None:
        short_delta = snapshot.get("short_delta")
        if short_delta is not None and abs(short_delta) >= cfg.delta_breach:
            return ExitReason.DELTA_BREACH

    if cfg.dte_exit is not None:
        dte = snapshot.get("dte")
        if dte is not None and dte <= cfg.dte_exit:
            return ExitReason.DTE

    return None
