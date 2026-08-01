"""`odds-lab probe` -- confirm or correct `providers.thetadata`'s UNVERIFIED
assumptions against a live Theta Terminal. ARCHITECTURE.md §0 "Schema is
probed, not assumed."

Hits every endpoint `ThetaDataProvider` uses for one (root, quote_date), dumps
the raw `header.format` array from each, diffs it against the field names
`thetadata.py` looks for, and writes a markdown report to
`runs/probe-<date>.md`. Run this once after getting a subscription and before
trusting a real `ingest()`; read the report, and if it flags a mismatch, fix
`providers/thetadata.py`'s `_V3_PATHS`/`_V2_PATHS`/`_FIELD_ALIASES` accordingly.

Degrades gracefully: if the terminal isn't running, writes a short report
saying so instead of raising.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from odds_lab.data.providers import thetadata as td

__all__ = ["probe"]


def _endpoint_check(provider: td.ThetaDataProvider, path: str, params: dict) -> dict:
    """Call one endpoint and summarize what came back, never raising."""
    result: dict = {"path": path, "params": params}
    try:
        payload = provider._get(path, params)
    except td.ThetaNotRunning as exc:
        result["error"] = f"ThetaNotRunning: {exc}"
        return result
    except td.ThetaError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    except Exception as exc:  # noqa: BLE001 -- probe must never crash on a bad payload
        result["error"] = f"unexpected {type(exc).__name__}: {exc}"
        return result

    try:
        df, fmt = td._rows_to_frame(payload)
    except td.ThetaError as exc:
        result["error"] = f"could not parse rows: {exc}"
        result["raw_keys"] = list(payload) if isinstance(payload, dict) else None
        return result

    result["format"] = fmt
    result["n_rows"] = len(df)
    result["first_rows"] = df.head(3).to_dict(orient="records")
    return result


def _diff(expected_canonical: list[str], fmt: list[str] | None) -> tuple[list[str], list[str]]:
    """(fields we need and didn't find, fields present that we ignore)."""
    if fmt is None:
        return list(expected_canonical), []
    found_aliases = set()
    missing = []
    for canon in expected_canonical:
        alias = td._find_alias(fmt, canon)
        if alias is None:
            missing.append(canon)
        else:
            found_aliases.add(alias)
    ignored = [f for f in fmt if f not in found_aliases]
    return missing, ignored


_EOD_FIELDS = ["root", "expiration", "strike", "right", "date", "ms_of_day", "bid", "ask",
               "bid_size", "ask_size", "last", "volume", "open_interest"]
_GREEKS_FIELDS = ["root", "expiration", "strike", "right", "delta", "gamma", "theta", "vega", "rho", "iv"]
_STOCK_FIELDS = ["date", "open", "high", "low", "close", "volume"]


def probe(base_url: str | None, version: str, root: str, quote_date: date) -> Path:
    """Run the live schema probe and write `runs/probe-<date>.md`. Returns the
    report path (always -- even a "terminal not running" report is written)."""
    provider = td.ThetaDataProvider(base_url=base_url, version=version)
    paths = provider._paths

    # pick a plausible expiry to probe the option endpoints with: try to list
    # real expirations first, fall back to a synthetic guess so option_eod/greeks
    # still get *some* response (and a useful error) even if list/expirations itself
    # is broken.
    exp_check = _endpoint_check(provider, paths["expirations"], {"root": root})
    expiry = None
    if "format" in exp_check and exp_check["n_rows"] > 0:
        try:
            expiry = provider.expirations(root, quote_date)[0]
        except Exception:
            expiry = None
    if expiry is None:
        expiry = date(quote_date.year, quote_date.month, 28)

    option_params = {
        "root": root,
        "exp": expiry.strftime("%Y%m%d"),
        "start_date": quote_date.strftime("%Y%m%d"),
        "end_date": quote_date.strftime("%Y%m%d"),
        "strike": "*",
        "right": "*",
    }
    stock_params = {
        "root": root,
        "start_date": quote_date.strftime("%Y%m%d"),
        "end_date": quote_date.strftime("%Y%m%d"),
    }

    checks = {
        "expirations": exp_check,
        "strikes": _endpoint_check(provider, paths["strikes"], {"root": root, "exp": expiry.strftime("%Y%m%d")}),
        "option_eod": _endpoint_check(provider, paths["option_eod"], option_params),
        "option_greeks": _endpoint_check(provider, paths["option_greeks"], option_params),
        "stock_eod": _endpoint_check(provider, paths["stock_eod"], stock_params),
    }

    terminal_unreachable = all("ThetaNotRunning" in c.get("error", "") for c in checks.values())

    lines: list[str] = []
    lines.append(f"# ThetaData live probe -- {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"- base_url: `{provider.base_url}`")
    lines.append(f"- version: `{version}`")
    lines.append(f"- root: `{root}`  quote_date: `{quote_date.isoformat()}`  probed expiry: `{expiry.isoformat()}`")
    lines.append("")

    if terminal_unreachable:
        lines.append("## Terminal unreachable")
        lines.append("")
        lines.append(
            "Every endpoint returned a connection error. Launch ThetaTerminal.jar "
            "and confirm it is listening on the base_url/version above, then re-run "
            "`odds-lab probe`."
        )
        lines.append("")
        lines.append("```")
        for name, c in checks.items():
            lines.append(f"{name}: {c.get('error')}")
        lines.append("```")
        report = "\n".join(lines) + "\n"
        return _write(report, quote_date)

    expected = {
        "expirations": [],
        "strikes": [],
        "option_eod": _EOD_FIELDS,
        "option_greeks": _GREEKS_FIELDS,
        "stock_eod": _STOCK_FIELDS,
    }

    for name, check in checks.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- path: `{check['path']}`")
        lines.append(f"- params: `{check['params']}`")
        if "error" in check:
            lines.append(f"- **error**: {check['error']}")
            lines.append("")
            continue
        fmt = check.get("format")
        lines.append(f"- header.format: `{fmt}`")
        lines.append(f"- n_rows: {check.get('n_rows')}")

        missing, ignored = _diff(expected.get(name, []), fmt)
        if missing:
            lines.append(f"- **fields we need and did NOT find**: {missing}")
        else:
            lines.append("- fields we need and did not find: none")
        lines.append(f"- fields present we ignore: {ignored}")

        if name == "option_eod" and fmt and check.get("first_rows"):
            strikes = [r.get(td._find_alias(fmt, "strike")) for r in check["first_rows"]]
            strikes = [s for s in strikes if s is not None]
            if strikes:
                med = sorted(strikes)[len(strikes) // 2]
                lines.append(f"- sample raw strikes: {strikes} (median-ish {med})")
                lines.append(
                    "- detected strike scaling: "
                    + ("TENTHS-OF-CENT (>20x typical spot, will /1000)" if med > 2000 else "DOLLARS (no scaling)")
                )
            date_col = td._find_alias(fmt, "date")
            if date_col and check["first_rows"]:
                raw_date = check["first_rows"][0].get(date_col)
                kind = "YYYY-MM-DD string" if isinstance(raw_date, str) and "-" in raw_date else "YYYYMMDD int"
                lines.append(f"- detected date encoding: {kind} (raw value: {raw_date!r})")

        if check.get("first_rows"):
            lines.append("- first 3 raw rows:")
            lines.append("")
            lines.append("```")
            for row in check["first_rows"]:
                lines.append(str(row))
            lines.append("```")
        lines.append("")

    report = "\n".join(lines) + "\n"
    return _write(report, quote_date)


def _write(report: str, quote_date: date) -> Path:
    out_dir = Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"probe-{quote_date.isoformat()}.md"
    path.write_text(report)
    return path
