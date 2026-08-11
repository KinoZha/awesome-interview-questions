"""Run configuration. Every knob the backtest exposes lives here and nowhere else.

A run is fully identified by `BacktestConfig.run_id()`; re-running the same config must
reproduce byte-identical trades (ARCHITECTURE.md §6).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

Strategy = Literal[
    "put_credit_spread",
    "call_credit_spread",
    "iron_condor",
    "short_strangle",
    "short_put",
    "long_strangle",
]

StrikeRule = Literal["delta", "pct_otm", "empirical_prob"]
"""delta: |Δ| band (OPI). pct_otm: fixed % OTM (the ±5% baseline). empirical_prob:
strike whose *empirical* P(ITM) hits a target -- the ODDS rule."""

MarketFilter = Literal["all", "skip_bearish", "skip_bullish", "trend_aligned"]


@dataclass(frozen=True)
class CostModel:
    """STRATEGY.md §7.1-7.2. Defaults match a typical retail options account."""

    commission_per_contract: float = 0.65
    fees_per_contract_sell: float = 0.05  # exchange + regulatory, sells only
    fill_kind: Literal["bid_ask", "mid", "mid_plus_frac"] = "bid_ask"
    mid_frac: float = 0.5
    """For fill_kind='mid_plus_frac': fraction of the half-spread paid. 0=mid, 1=bid/ask."""
    max_spread_pct_of_mid: float = 0.15
    """Reject a contract whose quoted spread exceeds this fraction of mid."""
    min_bid: float = 0.05
    min_open_interest: int = 100


@dataclass(frozen=True)
class EmpiricalConfig:
    """STRATEGY.md §2 step 2 -- the 'Count' half of the edge."""

    lookback_years: float = 10.0
    """Trailing window of underlying closes used to build the realized distribution.
    Uses only data strictly before the decision date."""
    symmetrize: bool = True
    min_samples: int = 250
    bootstrap_samples: int = 500
    bootstrap_mean_block: int | None = None
    """Stationary-bootstrap mean block length; None => equal to the horizon in days."""
    seed: int = 20120601


@dataclass(frozen=True)
class EntryConfig:
    """STRATEGY.md §4."""

    strategy: Strategy = "put_credit_spread"
    strike_rule: StrikeRule = "delta"
    # strike_rule='delta'
    delta_min: float = 0.15
    delta_max: float = 0.30
    # strike_rule='pct_otm'
    pct_otm: float = 0.05
    # strike_rule='empirical_prob'
    target_prob_itm: float = 0.20
    width_strikes: int = 2
    """Spread width in strike increments; the sweep varies this over {1,2,3,5}.

    Default changed from 1 -> 2 (STRATEGY.md §4, "Defaults on $1-spaced ETF strikes")
    after the selection funnel showed width=1 combined with the OPI $0.30 min_credit
    produces ZERO trades on SPY/QQQ/IWM: at width=1 the expected-return band
    (credit <= width/3 for ER<=0.50) already caps an achievable credit near $0.33, and
    real bid/ask spreads push most candidates out on liquidity before min_credit even
    applies. width=2 leaves enough room in both the ER band and the credit itself for
    the OPI $0.30 threshold to mean something on these underlyings, without changing
    that threshold's dollar value. See STRATEGY.md §4.2 for the measured funnel."""
    dte_min: int = 21
    dte_max: int = 56
    min_credit: float = 0.30
    expected_return_min: float = 0.0
    expected_return_max: float = 0.50
    market_filter: MarketFilter = "all"
    require_positive_edge: bool = True
    """Reject trades whose EV under the *empirical* distribution is <= 0. This is the
    filter that distinguishes the ODDS system from naive premium selling."""
    entry_schedule: Literal["weekly", "monthly_expiry", "daily"] = "weekly"
    entry_weekday: int = 0  # Monday, used when entry_schedule='weekly'
    max_concurrent_per_root: int = 4


@dataclass(frozen=True)
class ExitConfig:
    """STRATEGY.md §5. All enabled rules are evaluated; the first to trigger wins."""

    profit_target_pct: float | None = 0.50  # of credit received
    stop_loss_multiple: float | None = 2.0  # of credit received
    dte_exit: int | None = 21
    delta_breach: float | None = 0.50
    hold_to_expiry: bool = False


@dataclass(frozen=True)
class RiskConfig:
    """STRATEGY.md §6."""

    starting_equity: float = 100_000.0
    risk_pct_per_trade: float = 0.05
    max_pct_per_style: float = 0.50
    max_margin_pct: float = 0.60
    min_contracts: int = 1
    max_contracts: int = 100


@dataclass(frozen=True)
class IntradayConfig:
    """Hybrid two-pass intraday exit evaluation -- ARCHITECTURE.md §9 (lifts the
    "no intraday in v1" non-goal), STRATEGY.md §5/§7.3. OFF by default so every
    existing config/run is byte-identical to before this feature existed
    (CLAUDE.md rule 5).

    Entries stay EOD-only always (STRATEGY.md §4's weekly-entry schedule does not
    need intraday timing, per this change's brief). Only exit evaluation for the
    rules named in `exit_rules` gains minute resolution; `dte_exit` is excluded on
    purpose (it is daily by nature, STRATEGY.md §5) and is always evaluated once/day
    regardless of what's listed here -- see `engine/intraday.py`.
    """

    enabled: bool = False
    provider: Literal["parquet", "csv_export"] = "parquet"
    """Where 1-minute quotes come from for pass 2. 'parquet' reads whatever is
    already in the store's intraday table (`data/store.py::ChainStore.intraday`);
    'csv_export' additionally allows `engine.intraday.run_hybrid_backtest`'s
    `fetch_fn` hook to ingest more via `data.providers.csv_export.read_quote_1m_csv`."""
    source_path: str | None = None
    """Directory of real ThetaData 1-minute QUOTE CSV exports, when provider='csv_export'."""
    bar_interval_minutes: int = 1
    exit_rules: tuple[str, ...] = ("profit_target", "stop_loss", "delta_breach")
    """Which of ExitConfig's early-exit rules get intraday evaluation. `dte_exit`
    must never appear here (engine/intraday.py masks it out even if it does)."""
    iteration_cap: int = 3
    """Max fixed-point iterations `engine.intraday.run_hybrid_backtest` will run
    before giving up and reporting non-convergence (task requirement -- never
    silently assume the held-contract-set feedback loop converged)."""


@dataclass(frozen=True)
class DataConfig:
    provider: Literal["thetadata", "parquet", "synthetic"] = "parquet"
    theta_base_url: str = "http://127.0.0.1:25503"
    theta_api_version: Literal["v3", "v2"] = "v3"
    store_root: str = "data"
    allow_synthetic: bool = False
    """Must be explicitly True to run on generated data; every report is then stamped."""


@dataclass(frozen=True)
class BacktestConfig:
    roots: tuple[str, ...] = ("SPY", "QQQ", "IWM")
    start: date = date(2012, 6, 1)
    end: date = date(2026, 1, 1)
    entry: EntryConfig = field(default_factory=EntryConfig)
    exits: ExitConfig = field(default_factory=ExitConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostModel = field(default_factory=CostModel)
    empirical: EmpiricalConfig = field(default_factory=EmpiricalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    intraday: IntradayConfig = field(default_factory=IntradayConfig)
    risk_free_rate: float = 0.02
    """Fallback flat rate; overridden per-date when a rate curve is present in the store."""
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), default=str))

    def run_id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha1(payload).hexdigest()[:12]

    def run_dir(self, root: str | Path = "runs") -> Path:
        return Path(root) / f"{self.run_id()}-{self.end.isoformat()}"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BacktestConfig:
        def _date(v: Any) -> date:
            return v if isinstance(v, date) else date.fromisoformat(str(v))

        return cls(
            roots=tuple(d.get("roots", ("SPY", "QQQ", "IWM"))),
            start=_date(d.get("start", date(2012, 6, 1))),
            end=_date(d.get("end", date(2026, 1, 1))),
            entry=EntryConfig(**d.get("entry", {})),
            exits=ExitConfig(**d.get("exits", {})),
            risk=RiskConfig(**d.get("risk", {})),
            costs=CostModel(**d.get("costs", {})),
            empirical=EmpiricalConfig(**d.get("empirical", {})),
            data=DataConfig(**d.get("data", {})),
            intraday=IntradayConfig(**d.get("intraday", {})),
            risk_free_rate=float(d.get("risk_free_rate", 0.02)),
            label=str(d.get("label", "")),
        )


__all__ = [
    "BacktestConfig",
    "CostModel",
    "DataConfig",
    "EmpiricalConfig",
    "EntryConfig",
    "ExitConfig",
    "IntradayConfig",
    "MarketFilter",
    "RiskConfig",
    "Strategy",
    "StrikeRule",
]
