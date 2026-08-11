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
        from odds_lab.data import probe as probe_mod
    except ImportError as e:
        print(f"error: probe not available yet: {e}", file=sys.stderr)
        return 1

    if args.csv and args.base_url:
        print(
            "error: --csv and --base-url are mutually exclusive -- --csv profiles a local "
            "export directory, --base-url talks to a live Theta Terminal; pick one",
            file=sys.stderr,
        )
        return 1

    if args.coverage:
        if not args.roots:
            print("error: --coverage requires --roots, e.g. --roots SPY,QQQ,IWM", file=sys.stderr)
            return 1
        roots = _split_roots(args.roots)
        floor = date.fromisoformat(args.floor)
        ceiling = date.fromisoformat(args.ceiling) if args.ceiling else None
        if args.csv:
            path = probe_mod.probe_coverage_csv(args.csv, list(roots))
        else:
            path = probe_mod.probe_coverage_rest(
                args.base_url,
                args.version,
                list(roots),
                floor=floor,
                ceiling=ceiling,
                sample_dates=args.sample_dates,
                rate_limit_per_min=args.rate_limit,
            )
        print(f"wrote {path}")
        return 0

    if args.csv:
        path = probe_mod.probe_csv(
            args.csv, head_rows=args.head_rows, tail_rows=args.tail_rows, sample_out=args.sample_out
        )
        print(f"wrote {path}")
        if args.sample_out:
            print(f"schema sample archive: {args.sample_out} -- send this if a shape came back 'unknown'")
        return 0

    if not args.root or not args.date:
        print(
            "error: --root and --date are required for the REST schema probe "
            "(or pass --csv <dir> to profile local exports instead)",
            file=sys.stderr,
        )
        return 1
    import json

    result = probe_mod.probe(args.base_url, args.version, args.root, date.fromisoformat(args.date))
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
# intraday-compare -- STRATEGY.md §5.1 / ARCHITECTURE.md §3.1
# --------------------------------------------------------------------------------------


def cmd_intraday_compare(args: argparse.Namespace) -> int:
    """Run the same config EOD-only vs. intraday-hybrid and report the bias,
    per exit rule -- the first-class deliverable this feature exists to produce
    (STRATEGY.md §5.1), not a footnote on a regular backtest."""
    import dataclasses
    import json as json_mod

    store_path = Path(args.store)
    if not store_path.exists():
        print(f"error: data store not found at '{store_path}' -- run 'odds-lab ingest' first", file=sys.stderr)
        return 1

    try:
        cfg = _load_config(args)
    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"error: invalid config: {e}", file=sys.stderr)
        return 1

    intraday_overrides: dict[str, Any] = {"enabled": True}
    if getattr(args, "iteration_cap", None) is not None:
        intraday_overrides["iteration_cap"] = args.iteration_cap
    cfg = dataclasses.replace(cfg, intraday=dataclasses.replace(cfg.intraday, **intraday_overrides))

    try:
        from odds_lab.data.store import ChainStore
        from odds_lab.engine.intraday import compare_eod_vs_intraday
    except ImportError as e:
        print(f"error: engine not available yet: {e}", file=sys.stderr)
        return 1

    store = ChainStore(store_path)
    try:
        comparison = compare_eod_vs_intraday(cfg, store, iteration_cap=args.iteration_cap)
    except Exception as e:
        print(f"error: intraday comparison failed: {e}", file=sys.stderr)
        return 1

    out_dir = Path(args.out) if args.out else cfg.run_dir() / "intraday-compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"overall": comparison["overall"], "by_rule": comparison["by_rule"]}
    (out_dir / "comparison.json").write_text(json_mod.dumps(summary, indent=2, sort_keys=True, default=str))
    comparison["eod_result"].save(out_dir / "eod")
    comparison["hybrid_result"].save(out_dir / "hybrid")

    print(json_mod.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"wrote {out_dir}")
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
# stress
# --------------------------------------------------------------------------------------


def cmd_stress(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"error: run directory not found: {run_dir}", file=sys.stderr)
        return 1
    history_path = Path(args.underlying_history)
    if not history_path.exists():
        print(f"error: underlying history file not found: {history_path}", file=sys.stderr)
        return 1

    try:
        from odds_lab.engine.result import BacktestResult
        from odds_lab.engine import stress as stress_mod
    except ImportError as e:
        print(f"error: engine not available yet: {e}", file=sys.stderr)
        return 1

    try:
        result = BacktestResult.load(run_dir)
    except Exception as e:
        print(f"error: failed to load run from '{run_dir}': {e}", file=sys.stderr)
        return 1

    try:
        closes = stress_mod.load_underlying_history(history_path)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    names = [s.strip() for s in args.scenarios.split(",") if s.strip()] if args.scenarios else None
    scenarios = stress_mod.build_scenarios(closes, names)
    if not scenarios.scenarios:
        print(
            "warning: no scenarios could be built from the supplied history "
            f"(skipped: {scenarios.skipped})", file=sys.stderr,
        )

    stress_result = stress_mod.run_stress(result.trades, scenarios, result.config.exits)
    stress_out = stress_result.save(run_dir)
    print(f"wrote {stress_out}")

    long_history_edge = None
    if args.store:
        store_path = Path(args.store)
        if not store_path.exists():
            print(f"warning: --store '{store_path}' not found -- skipping the long-window edge comparison", file=sys.stderr)
        else:
            from odds_lab.data.store import ChainStore

            store = ChainStore(store_path)
            long_closes = {root: closes for root in result.config.roots}
            try:
                _short, _long, long_history_edge = stress_mod.run_backtest_with_long_history(
                    result.config, store, long_closes
                )
            except Exception as e:
                print(f"warning: long-window edge comparison failed: {e}", file=sys.stderr)

    odds_overlay_figs = []
    from odds_lab.report import figures as F

    trades = result.trades
    if not trades.empty and scenarios.scenarios:
        for root in sorted(trades["root"].dropna().unique()):
            sub = trades[trades["root"] == root]
            horizon = int(round(sub["dte_entry"].median())) if sub["dte_entry"].notna().any() else 30
            iv = float(sub["iv_entry"].median()) if sub["iv_entry"].notna().any() else 0.20
            strikes = sorted(set(sub["short_strike"].dropna().tolist()))
            asof = closes.index.max()
            fig = F.fig_stress_odds_overlay(closes, horizon, iv, asof, scenarios.scenarios, strikes=strikes)
            odds_overlay_figs.append((f"{root} ODDS chart with crisis overlays", fig))

    from odds_lab.report.build import build_report

    report_path = build_report(
        result, run_dir / "report.html",
        extra={
            "stress": {
                "comparisons": stress_result.comparisons,
                "beta_sensitivity": stress_result.beta_sensitivity,
                "skipped": scenarios.skipped,
                "long_history_edge": long_history_edge,
                "odds_overlay_figs": odds_overlay_figs,
            }
        },
    )
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

    p_probe = sub.add_parser("probe", help="dump/diff the live provider schema, or profile local CSV exports")
    p_probe.add_argument("--base-url", default=None, help="live Theta Terminal base URL (REST probe / --coverage)")
    p_probe.add_argument("--version", default="v3", choices=["v2", "v3"])
    p_probe.add_argument("--root", default=None, help="REST schema probe only")
    p_probe.add_argument("--date", default=None, help="YYYY-MM-DD; REST schema probe only")
    p_probe.add_argument(
        "--csv", default=None, metavar="DIR",
        help="profile a directory of ThetaData CSV bulk exports instead of a live REST probe "
        "(mutually exclusive with --base-url)",
    )
    p_probe.add_argument("--sample-out", default=None, metavar="PATH", help="with --csv: write a small schema-sample .tar.gz to PATH")
    p_probe.add_argument("--head-rows", type=int, default=200, help="with --csv: data rows sampled from the start of each file")
    p_probe.add_argument("--tail-rows", type=int, default=5, help="with --csv: data rows sampled from the end of each file")
    p_probe.add_argument(
        "--coverage", action="store_true",
        help="binary-search the earliest usable date per (root, data kind) instead of a schema dump "
        "(works with --base-url or --csv)",
    )
    p_probe.add_argument("--roots", default=None, help="--coverage: comma-separated roots, e.g. SPY,QQQ,IWM")
    p_probe.add_argument("--floor", default="2010-01-01", help="--coverage (REST): earliest date to search from")
    p_probe.add_argument("--ceiling", default=None, help="--coverage (REST): latest known-good date to search to (default: today-5d)")
    p_probe.add_argument("--sample-dates", type=int, default=5, help="--coverage (REST): sample dates for the density table")
    p_probe.add_argument("--rate-limit", type=int, default=20, help="--coverage (REST): max requests/min against the terminal")
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

    p_ic = sub.add_parser(
        "intraday-compare",
        help="run a config EOD-only vs. intraday-hybrid and report the exit-timing/P&L bias per rule",
    )
    p_ic.add_argument("--config", default=None, help="config.yaml; flags below override it")
    p_ic.add_argument("--roots", default=None, help="comma-separated, e.g. SPY,QQQ,IWM")
    p_ic.add_argument("--start", default=None, help="YYYY-MM-DD")
    p_ic.add_argument("--end", default=None, help="YYYY-MM-DD")
    p_ic.add_argument("--strategy", default=None)
    p_ic.add_argument("--strike-rule", dest="strike_rule", default=None)
    p_ic.add_argument("--delta-min", dest="delta_min", type=float, default=None)
    p_ic.add_argument("--delta-max", dest="delta_max", type=float, default=None)
    p_ic.add_argument("--width", type=int, default=None)
    p_ic.add_argument("--dte-min", dest="dte_min", type=int, default=None)
    p_ic.add_argument("--dte-max", dest="dte_max", type=int, default=None)
    p_ic.add_argument("--profit-target", dest="profit_target", type=float, default=None)
    p_ic.add_argument("--stop-loss", dest="stop_loss", type=float, default=None)
    p_ic.add_argument("--iteration-cap", dest="iteration_cap", type=int, default=None, help="feedback-loop fixed-point cap (default: config's IntradayConfig.iteration_cap)")
    p_ic.add_argument("--store", required=True, help="store containing BOTH the EOD chains and data/intraday/ coverage")
    p_ic.add_argument("--out", default=None)
    p_ic.set_defaults(func=cmd_intraday_compare)

    p_sweep = sub.add_parser("sweep", help="cartesian-product parameter sweep + comparison report")
    p_sweep.add_argument("--config", required=True)
    p_sweep.add_argument("--grid", required=True)
    p_sweep.add_argument("--store", default=None)
    p_sweep.add_argument("--out", default=None)
    p_sweep.set_defaults(func=cmd_sweep)

    p_stress = sub.add_parser(
        "stress", help="crisis stress-test a saved run's positions against realized underlying paths"
    )
    p_stress.add_argument("--run", required=True, help="run directory (as written by 'odds-lab backtest')")
    p_stress.add_argument(
        "--underlying-history", dest="underlying_history", required=True,
        help="CSV with 'date' and 'close' columns, long enough to cover the scenarios requested "
        "(see docs/STRATEGY.md §10.1 for free sources)",
    )
    p_stress.add_argument(
        "--scenarios", default=None,
        help="comma-separated scenario names (default: the full library, STRATEGY.md §10.1)",
    )
    p_stress.add_argument(
        "--store", default=None,
        help="optional data store, to also re-run the backtest with the empirical distribution "
        "fed from --underlying-history (STRATEGY.md §10.4); skipped if omitted",
    )
    p_stress.set_defaults(func=cmd_stress)

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
