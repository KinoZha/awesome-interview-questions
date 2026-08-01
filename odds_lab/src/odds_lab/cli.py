"""`odds-lab` command line entry point.

Subcommands: ingest | probe | backtest | sweep | report.

Downstream modules (`engine.loop`, `engine.result`, `data.probe`, `data.providers.thetadata`)
are being developed concurrently and may not exist yet at import time -- every command
imports them lazily, inside the function body, and turns a missing module into a readable
one-line error (never a traceback) so this CLI is usable/testable before those land.
"""

from __future__ import annotations

import argparse
import dataclasses
import html
import itertools
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from odds_lab.config import BacktestConfig

__all__ = ["main"]


# --------------------------------------------------------------------------------------
# Config resolution: config.yaml + flag overrides
# --------------------------------------------------------------------------------------


def _split_roots(raw: str) -> tuple[str, ...]:
    return tuple(r.strip().upper() for r in raw.split(",") if r.strip())


def _load_config(args: argparse.Namespace) -> BacktestConfig:
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"config file not found: {cfg_path}")
        raw = yaml.safe_load(cfg_path.read_text()) or {}
        cfg = BacktestConfig.from_dict(raw)
    else:
        cfg = BacktestConfig()

    entry_overrides: dict[str, Any] = {}
    if getattr(args, "strategy", None):
        entry_overrides["strategy"] = args.strategy
    if getattr(args, "strike_rule", None):
        entry_overrides["strike_rule"] = args.strike_rule
    if getattr(args, "delta_min", None) is not None:
        entry_overrides["delta_min"] = args.delta_min
    if getattr(args, "delta_max", None) is not None:
        entry_overrides["delta_max"] = args.delta_max
    if getattr(args, "width", None) is not None:
        entry_overrides["width_strikes"] = args.width
    if getattr(args, "dte_min", None) is not None:
        entry_overrides["dte_min"] = args.dte_min
    if getattr(args, "dte_max", None) is not None:
        entry_overrides["dte_max"] = args.dte_max

    exit_overrides: dict[str, Any] = {}
    if getattr(args, "profit_target", None) is not None:
        exit_overrides["profit_target_pct"] = args.profit_target
    if getattr(args, "stop_loss", None) is not None:
        exit_overrides["stop_loss_multiple"] = args.stop_loss

    top_overrides: dict[str, Any] = {}
    if getattr(args, "roots", None):
        top_overrides["roots"] = _split_roots(args.roots)
    if getattr(args, "start", None):
        top_overrides["start"] = date.fromisoformat(args.start)
    if getattr(args, "end", None):
        top_overrides["end"] = date.fromisoformat(args.end)

    if entry_overrides:
        cfg = dataclasses.replace(cfg, entry=dataclasses.replace(cfg.entry, **entry_overrides))
    if exit_overrides:
        cfg = dataclasses.replace(cfg, exits=dataclasses.replace(cfg.exits, **exit_overrides))
    if top_overrides:
        cfg = dataclasses.replace(cfg, **top_overrides)
    return cfg


def _apply_dotted_overrides(cfg: BacktestConfig, overrides: dict[str, Any]) -> BacktestConfig:
    """Apply {'entry.width_strikes': 2, 'roots': (...)}-style overrides. One level of
    dotted nesting only -- matches config.py's shape (top-level dataclass fields, each
    possibly a nested frozen dataclass)."""
    top_changes: dict[str, Any] = {}
    nested_changes: dict[str, dict[str, Any]] = {}
    for key, value in overrides.items():
        if "." in key:
            section, field_name = key.split(".", 1)
            nested_changes.setdefault(section, {})[field_name] = value
        else:
            top_changes[key] = value
    for section, changes in nested_changes.items():
        current = getattr(cfg, section)
        top_changes[section] = dataclasses.replace(current, **changes)
    return dataclasses.replace(cfg, **top_changes)


def _cartesian(grid: dict[str, Any]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    value_lists = [v if isinstance(v, list) else [v] for v in grid.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


# --------------------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> int:
    roots = _split_roots(args.roots)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    from odds_lab.data.store import ChainStore

    store = ChainStore(args.store)

    if args.provider == "synthetic":
        from odds_lab.data.providers.synthetic import SyntheticProvider

        provider = SyntheticProvider(roots=roots, start=start, end=end)
    else:
        try:
            from odds_lab.data.providers.thetadata import ThetaDataProvider
        except ImportError as e:
            print(f"error: thetadata provider not available yet: {e}", file=sys.stderr)
            return 1
        provider = ThetaDataProvider(base_url=args.base_url, version=args.version)

    from odds_lab.data.ingest import ingest

    manifest = ingest(provider, store, roots, start, end)
    written = sum(v.get("rows_written", 0) for v in manifest.get("roots", {}).values())
    print(f"wrote {written} rows to {args.store}")
    return 0


# --------------------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------------------


def cmd_probe(args: argparse.Namespace) -> int:
    try:
        from odds_lab.data.probe import probe
    except ImportError as e:
        print(f"error: probe not available yet: {e}", file=sys.stderr)
        return 1
    import json

    result = probe(args.base_url, args.version, args.root, date.fromisoformat(args.date))
    print(json.dumps(result, indent=2, default=str))
    return 0


# --------------------------------------------------------------------------------------
# backtest
# --------------------------------------------------------------------------------------


def cmd_backtest(args: argparse.Namespace) -> int:
    store_path = Path(args.store)
    if not store_path.exists():
        print(f"error: data store not found at '{store_path}' -- run 'odds-lab ingest' first", file=sys.stderr)
        return 1

    try:
        cfg = _load_config(args)
    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"error: invalid config: {e}", file=sys.stderr)
        return 1

    try:
        from odds_lab.data.store import ChainStore
        from odds_lab.engine.loop import run_backtest
    except ImportError as e:
        print(f"error: engine not available yet: {e}", file=sys.stderr)
        return 1

    store = ChainStore(store_path)
    try:
        result = run_backtest(cfg, store)
    except Exception as e:  # engine/data errors -> readable message, not a traceback
        print(f"error: backtest failed: {e}", file=sys.stderr)
        return 1

    out_dir = Path(args.out) if args.out else cfg.run_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    save = getattr(result, "save", None)
    if callable(save):
        try:
            save(out_dir)
        except Exception as e:
            print(f"warning: result.save() failed, continuing to report only: {e}", file=sys.stderr)

    from odds_lab.report.build import build_report

    report_path = build_report(result, out_dir / "report.html", store=store)
    print(f"wrote {report_path}")
    return 0


# --------------------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------------------


def _run_quick_stats(result: Any) -> dict[str, Any]:
    trades = result.trades
    eq = result.equity
    win_rate = float((trades["pnl"] > 0).mean()) if not trades.empty else float("nan")
    final_equity = float(eq["equity"].iloc[-1]) if not eq.empty else float("nan")
    return {"total_trades": int(len(trades)), "win_rate": win_rate, "final_equity": final_equity}


def _write_comparison_report(results: dict[str, Any], path: Path) -> None:
    from odds_lab.report import figures as F
    from odds_lab.report.assets import CSS

    fig = F.fig_strategy_comparison(results)
    fig_html = fig.to_html(full_html=False, include_plotlyjs="inline", config={"responsive": True})
    rows = []
    for name, res in results.items():
        s = _run_quick_stats(res)
        wr = "n/a" if s["win_rate"] != s["win_rate"] else f"{s['win_rate'] * 100:.1f}%"
        eqv = "n/a" if s["final_equity"] != s["final_equity"] else f"${s['final_equity']:,.0f}"
        rows.append(
            f"<tr><td>{html.escape(str(name))}</td><td>{s['total_trades']}</td>"
            f"<td>{wr}</td><td>{eqv}</td></tr>"
        )
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>odds_lab sweep comparison</title>
<style>{CSS}</style></head><body><div class="container">
<h1>Sweep comparison</h1>
<div class="fig-block">{fig_html}</div>
<table class="trade-table"><thead><tr><th>run</th><th>trades</th><th>win rate</th><th>final equity</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</div></body></html>"""
    path.write_text(body, encoding="utf-8")


def cmd_sweep(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config)
    grid_path = Path(args.grid)
    if not cfg_path.exists():
        print(f"error: config file not found: {cfg_path}", file=sys.stderr)
        return 1
    if not grid_path.exists():
        print(f"error: grid file not found: {grid_path}", file=sys.stderr)
        return 1

    base_cfg = BacktestConfig.from_dict(yaml.safe_load(cfg_path.read_text()) or {})
    grid_raw = yaml.safe_load(grid_path.read_text()) or {}
    combos = _cartesian(grid_raw)

    store_path = Path(args.store) if args.store else Path(base_cfg.data.store_root)
    if not store_path.exists():
        print(f"error: data store not found at '{store_path}' -- run 'odds-lab ingest' first", file=sys.stderr)
        return 1

    try:
        from odds_lab.data.store import ChainStore
        from odds_lab.engine.loop import run_backtest
    except ImportError as e:
        print(f"error: engine not available yet: {e}", file=sys.stderr)
        return 1

    store = ChainStore(store_path)
    results: dict[str, Any] = {}
    total = len(combos)
    for i, overrides in enumerate(combos, 1):
        cfg = _apply_dotted_overrides(base_cfg, overrides)
        label = cfg.label or (", ".join(f"{k}={v}" for k, v in overrides.items()) or f"run{i}")
        print(f"[{i}/{total}] running {label} ...", flush=True)
        try:
            results[label] = run_backtest(cfg, store)
        except Exception as e:
            print(f"warning: run '{label}' failed: {e}", file=sys.stderr)

    if not results:
        print("error: every sweep run failed -- no comparison report to write", file=sys.stderr)
        return 1

    out_dir = Path(args.out) if getattr(args, "out", None) else Path("runs") / f"sweep-{base_cfg.run_id()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "comparison.html"
    _write_comparison_report(results, report_path)
    print(f"wrote {report_path}")
    return 0


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"error: run directory not found: {run_dir}", file=sys.stderr)
        return 1
    try:
        from odds_lab.engine.result import BacktestResult
    except ImportError as e:
        print(f"error: engine not available yet: {e}", file=sys.stderr)
        return 1
    try:
        result = BacktestResult.load(run_dir)
    except Exception as e:
        print(f"error: failed to load result from '{run_dir}': {e}", file=sys.stderr)
        return 1

    from odds_lab.report.build import build_report

    report_path = build_report(result, run_dir / "report.html")
    print(f"wrote {report_path}")
    return 0


# --------------------------------------------------------------------------------------
# argparse wiring
# --------------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="odds-lab", description="ODDS options backtester")
    sub = p.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="pull chain data into the local parquet store")
    p_ingest.add_argument("--provider", choices=["thetadata", "synthetic"], required=True)
    p_ingest.add_argument("--roots", required=True, help="comma-separated, e.g. SPY,QQQ,IWM")
    p_ingest.add_argument("--start", required=True, help="YYYY-MM-DD")
    p_ingest.add_argument("--end", required=True, help="YYYY-MM-DD")
    p_ingest.add_argument("--store", required=True)
    p_ingest.add_argument("--base-url", default="http://127.0.0.1:25503")
    p_ingest.add_argument("--version", default="v3", choices=["v2", "v3"])
    p_ingest.set_defaults(func=cmd_ingest)

    p_probe = sub.add_parser("probe", help="dump/diff the live provider schema")
    p_probe.add_argument("--base-url", required=True)
    p_probe.add_argument("--version", default="v3", choices=["v2", "v3"])
    p_probe.add_argument("--root", required=True)
    p_probe.add_argument("--date", required=True, help="YYYY-MM-DD")
    p_probe.set_defaults(func=cmd_probe)

    p_bt = sub.add_parser("backtest", help="run a backtest and write runs/<run_id>/")
    p_bt.add_argument("--config", default=None, help="config.yaml; flags below override it")
    p_bt.add_argument("--roots", default=None, help="comma-separated, e.g. SPY,QQQ,IWM")
    p_bt.add_argument("--start", default=None, help="YYYY-MM-DD")
    p_bt.add_argument("--end", default=None, help="YYYY-MM-DD")
    p_bt.add_argument("--strategy", default=None)
    p_bt.add_argument("--strike-rule", dest="strike_rule", default=None)
    p_bt.add_argument("--delta-min", dest="delta_min", type=float, default=None)
    p_bt.add_argument("--delta-max", dest="delta_max", type=float, default=None)
    p_bt.add_argument("--width", type=int, default=None)
    p_bt.add_argument("--dte-min", dest="dte_min", type=int, default=None)
    p_bt.add_argument("--dte-max", dest="dte_max", type=int, default=None)
    p_bt.add_argument("--profit-target", dest="profit_target", type=float, default=None)
    p_bt.add_argument("--stop-loss", dest="stop_loss", type=float, default=None)
    p_bt.add_argument("--store", required=True)
    p_bt.add_argument("--out", default=None)
    p_bt.set_defaults(func=cmd_backtest)

    p_sweep = sub.add_parser("sweep", help="cartesian-product parameter sweep + comparison report")
    p_sweep.add_argument("--config", required=True)
    p_sweep.add_argument("--grid", required=True)
    p_sweep.add_argument("--store", default=None)
    p_sweep.add_argument("--out", default=None)
    p_sweep.set_defaults(func=cmd_sweep)

    p_report = sub.add_parser("report", help="rebuild report.html from a saved run")
    p_report.add_argument("--run", required=True)
    p_report.set_defaults(func=cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError, PermissionError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
