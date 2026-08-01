# odds_lab — Architecture & Decisions

## 0. Decisions (with reasons)

**Build our own engine. Do not adopt LEAN.**
Surveyed: LEAN, optopsy, backtrader, nautilus_trader, vectorbt, optionlab,
lambdaclass/options_backtester, OptionSuite.
- LEAN is the only OSS engine with combo orders + assignment + combo margin, but it forces
  the LEAN proprietary data format, a Docker/.NET runtime, and its own data licensing. For
  three ETFs and one strategy family that is a large tax for features we can write in ~1k lines.
- backtrader: dead since Apr 2023, no options. vectorbt OSS: no options. nautilus: options
  backtesting immature. optionlab: single-snapshot payoff tool, not a backtester.
- optopsy (AGPL) and lambdaclass/options_backtester (MIT) are the right *references*:
  we mirror lambdaclass's parquet-native store and `MarketAtBidAsk` per-leg fill idea, and
  optopsy's strategy-template + filter-pipeline shape. Neither models margin, assignment,
  or rolling, and neither has the empirical-probability engine that is the entire point of
  this system (§2 of STRATEGY.md). AGPL also makes optopsy a poor base for a private repo.
- **Adopted as libraries**: numpy/scipy (own vectorized BS — `py_vollib_vectorized` is
  pinned to Python <3.10 and cannot be used), pandas + pyarrow + DuckDB (store),
  plotly (report). QuantLib only if American early-exercise pricing proves to matter.

**Provider-agnostic data layer.**
ThetaData's own docs give conflicting history floors: UTP tape (QQQ) from 2012-06-01, CTA
tape (SPY, IWM) from either 2017-01-01 or 2020-01-01 depending on the page. That is an
unresolved risk to the 12-year requirement and it is not resolvable from this sandbox
(outbound network is blocked by policy). So the chain source sits behind a `ChainProvider`
interface with adapters for ThetaData, Polygon flat files, ORATS, and plain CSV/Parquet.
Swapping providers must never touch strategy or engine code.

**Schema is probed, not assumed.**
There are two distinct ThetaData surfaces and they do not agree. The **CSV export** is now
ground truth — real files are committed as fixtures and the parser is pinned to them (§0.5).
The **REST/JSON** path remains unverified: responses carry a `header.format` array naming the
columns, so the parser is driven by that array at runtime rather than by hardcoded positions,
and `odds-lab probe` dumps the live format arrays and diffs them against our expectations.
Two documented REST scaling traps are asserted, not trusted: `strike` may be in tenths of a
cent (140000 == $140.00) while OHLC/bid/ask are in dollars — note the CSV export uses plain
dollars, so this is detected rather than assumed — and `vega`/`rho` need dividing by 100.

**Synthetic data so the pipeline is testable now.**
`providers/synthetic.py` generates a full arbitrage-consistent chain (underlying by a
regime-switching jump-diffusion, chain by BS with a fitted smile + a realistic bid/ask
spread and OI profile). It exists so every module has tests and the report renders today,
without a ThetaData subscription. It is clearly labelled and never silently substituted
for real data — a run on synthetic data stamps `SYNTHETIC` on every report page.

## 0.5 Real ThetaData CSV export ground truth

The ThetaData adapter was originally written against guessed/documented behavior
(no outbound network in this sandbox). Real ThetaData CSV bulk exports are now
committed as fixtures and are authoritative over any earlier guess:

```
data/samples/thetadata_spy_eod_20250819.csv.gz
data/samples/thetadata_spy_ohlc_1m_20250819_exp20251219.csv.gz
```

These are a different artifact from `providers/thetadata.py`'s REST/JSON client --
a Terminal user's direct CSV download, not the JSON API this sandbox could never
reach. `providers/csv_export.py::CsvExportProvider` reads directories of files
shaped like these; it is the path that matters operationally, since a real user's
chain history is hundreds of MB to GB of exactly this on their own machine and
never touches this sandbox. The REST/JSON path in `thetadata.py` remains
unverified and is confirmed/corrected only by `odds-lab probe` against a live
terminal.

**EOD chain export header** (exact, in order):
```
symbol,expiration,strike,right,created,last_trade,open,high,low,close,volume,count,
bid_size,bid_exchange,bid,bid_condition,ask_size,ask_exchange,ask,ask_condition
```

**1-minute OHLC export header** (exact, in order):
```
symbol,expiration,strike,right,timestamp,open,high,low,close,volume,count,vwap
```

Facts confirmed by measuring the fixtures, and what they mean for every consumer:

- **`strike` is DOLLARS**, not tenths-of-a-cent, in this export mode (690.000, not
  690000). `thetadata.py`'s REST client has a median>20x auto-detect heuristic for
  the tenths-of-cent case; it correctly leaves real dollar strikes alone (asserted
  by `tests/test_data.py::test_thetadata_strike_autodetect_leaves_real_dollars_alone`
  against the real fixture), but do not "fix" it into always assuming dollars --
  the REST/JSON scaling behavior is still unverified independently of this.
- **`right` is the string `"CALL"`/`"PUT"`**, mapped explicitly to `'C'`/`'P'`
  (`csv_export._map_right`).
- **`expiration` is an ISO date string; `created`/`last_trade` are ISO
  timestamps.** No `ms_of_day` int, no `YYYYMMDD` int, in this export mode -- it is
  a whole-day-at-once export, not an intraday snapshot. `CsvExportProvider` writes
  `ms_of_day=0` for every row purely to satisfy `schema.validate_chain(strict=True)`;
  it is not a real timestamp and must not be read as one.
- **`close`/`last` is NOT a mark.** 4107 of 8768 EOD fixture rows have `close == 0`
  while carrying a live bid/ask -- no trade happened that day, but the contract is
  still quoted and tradeable. Any code that marks a position off `close`/`last`
  silently zeroes out roughly half of a real backtest's positions on any given day.
  Marking MUST use the bid/ask mid (`quant.bs.enrich_chain` already does this
  correctly -- `mid = (bid + ask) / 2`, never `close`). Asserted directly in
  `tests/test_data.py::test_eod_fixture_close_is_not_a_mark` and
  `::test_csv_export_provider_mark_comes_from_mid_not_close`.
- **500 rows have `bid <= 0`** (worthless, far-OTM contracts nobody is bidding on).
- **The 1-minute export's `vwap` is not a trustworthy per-bar price.** 97.2% of
  bars are all-zero (open=high=low=close=volume=0, i.e. no trade printed that
  minute), and 43645 of those all-zero bars still carry `vwap > 0` -- a stale or
  carried value, not a real trade. `csv_export.read_ohlc_1m_csv` computes a `price`
  column that is `close` on a real bar and `NaN` on an all-zero bar, deliberately
  ignoring `vwap` as a fallback; anyone computing intraday returns off this export
  must use that `price` column, never `close` or `vwap` directly.
- There is **no `underlying_price`, no `open_interest`, no `iv`, and no greeks
  column** in the EOD chain export -- they come from other endpoints/exports.
  `CsvExportProvider.underlying_price` is resolved by joining a separate
  underlying-EOD source (`underlying_path`, or discovered via `underlying_glob` in
  the same directory); if it cannot resolve one it raises `CsvExportError` naming
  exactly what it looked for -- it never fills NaN and never silently substitutes 0.
  An approximate put-call-parity fallback exists (`infer_underlying_from_parity=True`)
  but is off by default and stamps `source="csv_export+parity"` when used so it can
  never be mistaken for a real quote.

**Open-interest-unknown policy.** The EOD export has no `open_interest` column at
all. `CostModel.min_open_interest` (default 100, `strategy/selector.py`) rejects
the short leg whenever `open_interest < min_open_interest` -- if a provider
defaulted missing OI to 0, every row would fail that check and the backtest would
silently place zero trades with no error anywhere. Both `CsvExportProvider` and
`thetadata.py` (whose REST EOD payload also may not include OI -- it's a separate
endpoint there too) now emit `providers.csv_export.OPEN_INTEREST_UNKNOWN` (`-1`,
never a valid real OI) instead of 0 when OI is absent. **This sentinel alone does
not yet make `strategy/selector.py`'s filter unknown-aware** -- that module is out
of this change's scope (owned by a concurrent change) and still compares
`open_interest < min_open_interest` as before, which currently treats "unknown"
(-1) the same as "known and too illiquid." Flagged here as a required follow-up:
`selector.py`'s liquidity check should special-case `open_interest < 0` as "unknown,
do not use for liquidity filtering" rather than folding it into that comparison.

## 1. Layout

```
src/odds_lab/
  schema.py            column contracts + dataclasses; the cross-module ABI
  config.py            BacktestConfig / StrategyConfig / CostModel, YAML load
  data/
    providers/base.py       ChainProvider ABC
    providers/thetadata.py  Theta Terminal REST (v3 primary, v2 fallback)
    providers/synthetic.py  generator for dev/tests
    providers/parquet.py    read from the local store
    ingest.py               provider -> normalized -> partitioned parquet
    store.py                DuckDB query layer; asof-safe accessors
    probe.py                live schema probe / diff
  quant/
    bs.py                vectorized BS price, greeks, IV inversion
    empirical.py         overlapping N-day log-return distribution + block bootstrap
    edge.py              P_theo vs P_actual, edge_prob, edge_ev
  strategy/
    universe.py          market-state classifier, IV rank, realized vol
    selector.py          expiry + strike selection (STRATEGY.md §4)
    strategies.py        S1..S6 spread constructors
    exits.py             exit policies (STRATEGY.md §5)
  engine/
    fills.py             per-leg bid/ask fills + commissions/fees
    portfolio.py         cash, positions, equity, Reg-T margin approximation
    attribution.py       greek P&L decomposition per snapshot
    loop.py              daily event loop
  report/
    figures.py           plotly figures
    build.py             self-contained HTML report
  cli.py                 ingest | probe | backtest | report | sweep
```

## 2. Storage

Partitioned parquet, queried in place by DuckDB (no server, partition pruning):

```
data/chains/underlying=SPY/year=2015/month=03/part.parquet
data/underlying/SPY.parquet
data/rates/dgs3mo.parquet
```

Estimated ~30–55M rows / ~3–8 GB compressed for SPY+QQQ+IWM daily EOD chains 2012–2026.
Partition by year+month (not day — too many small files). Row group size 128 MB.
Columns are stored `float32` for prices/greeks, `int32` for date/strike-in-cents, dictionary
encoding on `root`/`right`.

Canonical chain row (`schema.CHAIN_COLUMNS`):
`root, expiry(date32), strike(float64 dollars), right('C'|'P'), quote_date(date32),
ms_of_day(int32), bid, ask, bid_size, ask_size, last, volume, open_interest,
iv, delta, gamma, theta, vega, rho, underlying_price, source, is_synthetic`

Greeks/IV: prefer the vendor's when present, but **always recompute with `quant/bs.py`** and
store the vendor value as `iv_vendor` etc. Report disagreement > 1e-3 as a data-quality metric.

## 3. Engine model

Daily loop over trading dates. At each date:
1. Load that date's chain slice for the active roots (DuckDB, columns pushed down).
2. Mark existing positions to market (per-leg, at mid for MTM; at bid/ask only for fills).
3. Evaluate exit policies → close triggered positions at the **next** snapshot (no lookahead).
4. Handle expirations: intrinsic settlement, assignment, spread exercise.
5. If an entry is scheduled for this date, run the selector → construct legs → size → fill.
6. Record equity, margin usage, greek exposure, and a per-position attribution slice.

Every fill produces a `Fill` record with `leg, qty, side, price, price_kind('bid'|'ask'|'mid'),
commission, fees, spread_cost_vs_mid`. Total slippage is therefore always decomposable.

Position sizing per STRATEGY.md §6. Margin: defined-risk = max loss; undefined-risk =
Reg-T approximation, tracked as a time series because that is the real failure mode.

## 4. Attribution

Between consecutive snapshots, position P&L is decomposed by re-pricing under frozen factors:
`dP = delta·dS + ½gamma·dS² + vega·dIV + theta·dt + residual`, computed per leg with the
snapshot's own greeks, then reconciled against the actual mark change. The residual is
reported, never hidden — a large residual means the greeks or the marks are wrong.

## 5. Report

One self-contained HTML file per run (plotly inlined, no CDN — must open offline).
Sections map 1:1 to STRATEGY.md §8:
- A. equity / drawdown / monthly-return heatmap / attribution stack
- B. trade explorer: per-trade underlying path with strike bands, MTM path, delta path,
  exit-trigger marker
- C. the ODDS chart: theoretical lognormal vs empirical realized-return histogram, per year
  and per regime, with traded strikes marked
- D. diagnostics: win rate, P&L distribution, cohorts by IV rank / market state / DTE /
  delta / VIX / year, and the year × week calendar grid
- E. run manifest: config hash, data source, date coverage, data-quality counters,
  `SYNTHETIC` banner when applicable

## 6. Reproducibility

Every run writes `runs/<run_id>/` containing: resolved config JSON, git SHA, data manifest
(row counts + min/max dates per root), `trades.parquet`, `equity.parquet`, `report.html`.
`run_id = sha1(config)[:12] + '-' + date`. Re-running the same config must reproduce
byte-identical trades.parquet.
