"""`BacktestResult` -- FROZEN shape, the report agent codes against this. ARCHITECTURE.md §6."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from odds_lab.config import BacktestConfig

__all__ = ["BacktestResult"]


@dataclass
class BacktestResult:
    config: BacktestConfig
    trades: pd.DataFrame
    equity: pd.DataFrame
    snapshots: pd.DataFrame
    manifest: dict

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.trades.to_parquet(out / "trades.parquet")
        self.equity.to_parquet(out / "equity.parquet")
        self.snapshots.to_parquet(out / "snapshots.parquet")
        (out / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2, sort_keys=True))
        (out / "manifest.json").write_text(json.dumps(self.manifest, indent=2, sort_keys=True, default=str))
        return out

    @classmethod
    def load(cls, path: str | Path) -> "BacktestResult":
        src = Path(path)
        trades = pd.read_parquet(src / "trades.parquet")
        equity = pd.read_parquet(src / "equity.parquet")
        snapshots = pd.read_parquet(src / "snapshots.parquet")
        config = BacktestConfig.from_dict(json.loads((src / "config.json").read_text()))
        manifest = json.loads((src / "manifest.json").read_text())
        return cls(config=config, trades=trades, equity=equity, snapshots=snapshots, manifest=manifest)
