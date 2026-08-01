# odds_lab — agent rules

Options backtester for Fishback/ODDS premium-selling strategies on SPY/QQQ/IWM, 2012→now.
Read `docs/STRATEGY.md` (what to implement) and `docs/ARCHITECTURE.md` (how) before coding.

## Hard rules
1. **No lookahead.** Any function that selects/decides takes an `asof` timestamp and must
   not read a row with `quote_ts > asof`. Every new selection/exit function needs a test
   that fails if this is violated.
2. **Fill at bid/ask, not mid.** Sell→bid, buy→ask, per leg. Mid is a reporting variant only.
3. **Schema is contract.** All dataframes crossing a module boundary conform to the dataclasses
   in `src/odds_lab/schema.py`. Add a field there first, never inline.
4. **No network at import time.** Data fetching lives only in `src/odds_lab/data/thetadata.py`.
   Everything else reads Parquet through `data/store.py`.
5. **Deterministic.** Seed every RNG. Same inputs ⇒ byte-identical outputs.
6. **Money is float dollars per share**; multiply by `contract_multiplier` (100) exactly once,
   at the position layer. Never store cents.

## Layout
```
src/odds_lab/
  schema.py        dataclasses + pandera-style column contracts
  data/            thetadata client, parquet store, sample-data loader
  quant/           BS pricing, vectorized IV/greeks, empirical distribution ("count")
  strategy/        strike selection, entry filters, exit policies
  engine/          portfolio, fills, margin, event loop
  report/          plotly figures + static HTML report builder
tests/             pytest; fixtures built from data/samples/
```

## Testing
- `uv run pytest -q` must pass before any commit.
- Every quant function needs a closed-form or published-value check
  (e.g. BS greeks vs known values; IV round-trip within 1e-6).
- Engine tests use the tiny sample chain in `tests/fixtures/`, never live network.

## Style
- Python 3.11, `uv` for deps (`uv add`, `uv run`). No conda, no pip.
- Type hints on all public functions. numpy/pandas vectorized; no row loops over chains.
- Docstrings: one line saying what + the source rule it implements
  (e.g. `# STRATEGY.md §4.2`). No decorative comments.
- Fail loud: raise on missing/ambiguous data. Never silently `fillna(0)` a price.

## Don't
- Don't add a dependency without checking it supports py3.11 (`py_vollib_vectorized` does NOT).
- Don't commit data files (`data/` is gitignored except `data/samples/`).
- Don't write new .md files; extend the two existing docs instead.
