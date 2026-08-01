# odds_lab

Backtester for Don Fishback's ODDS premium-selling methodology on SPY / QQQ / IWM,
built to answer one question honestly: **why did a strategy make money, and why did it
lose money.**

- `docs/STRATEGY.md` — the methodology, distilled from the source PDFs into testable rules.
- `docs/ARCHITECTURE.md` — how it is built, and why each dependency was adopted or rejected.
- `CLAUDE.md` — rules for anyone (human or agent) editing this code.

## What it does

Sells defined-risk premium (put/call credit spreads, iron condors) and undefined-risk
premium (short strangles/puts) on ETF options, and measures the edge the way the source
material describes it — *Calculate, Count, Compare*:

1. **Calculate** the market's risk-neutral probability that the short strike finishes ITM.
2. **Count** how often the underlying actually moved that far, from overlapping realized
   log returns over a trailing window.
3. **Compare**. A trade is eligible only when its expected value, computed by integrating
   the real payoff over the *empirical* distribution, is positive.

That third step is the whole system. Nothing in the OSS ecosystem implements it, which is
why this is a purpose-built engine rather than a fork.

## Quick start

```bash
uv sync

# 1. Generate a synthetic store so you can see the whole pipeline work with no data.
uv run python -c "
from datetime import date
from odds_lab.data.providers.synthetic import make_sample_store
make_sample_store('data/sample_store', start=date(2015,1,1), end=date(2017,1,1))"

# 2. Run a backtest and open the report.
uv run odds-lab backtest --store data/sample_store --roots SPY,QQQ,IWM \
  --start 2015-06-01 --end 2016-12-30 --strategy put_credit_spread --strike-rule delta
```

Synthetic runs stamp `SYNTHETIC` on every page of the report. The results are not real and
the report says so loudly.

## Using your own data

Your data never has to leave your machine — GB-scale chain history is ingested locally.

```bash
# ThetaData CSV exports (a directory of files; .csv and .csv.gz both work)
uv run odds-lab ingest --provider csv --source /path/to/exports --roots SPY,QQQ,IWM

# Or straight from a running Theta Terminal
uv run odds-lab probe --root SPY --date 2024-01-05   # confirm the live schema first
uv run odds-lab ingest --provider thetadata --roots SPY,QQQ,IWM --start 2013-01-01
```

Two things to get right before a long backtest:

- **Pull underlying closes further back than the chains.** The empirical distribution needs
  a 10-year trailing window, so a backtest starting in 2013 wants underlying history from
  ~2003. Closes are free; chains are not. Skip this and the engine rejects nearly every
  entry with `no_empirical_dist` — visibly, in the funnel, but you will have wasted the run.
- **Verify your provider's history floor.** ThetaData's docs conflict on when CTA-tape
  coverage begins: QQQ is confirmed from 2012-06, but SPY and IWM are documented as either
  2017 or 2020 depending on the page. If yours starts late, the data layer is
  provider-agnostic — ORATS reaches back to 2007 for EOD — and swapping sources touches no
  strategy or engine code.

## Reading the report

Beyond the equity curve, three sections carry the actual answers:

- **Trade lifecycle** — the underlying's path with the short/long strikes drawn as bands,
  the mark-to-market path, and the exit trigger marked. This is where "the market cut
  through my short strike on day 14" becomes obvious.
- **The ODDS chart** — theoretical lognormal density against the histogram of realized
  returns, per year, with the traded strikes marked. The source material's "see your edge"
  picture, reproduced.
- **The selection funnel** — every entry opportunity, and the named reason each rejected
  one was rejected. A run that takes no trades says so unmissably instead of rendering an
  empty report that looks successful.

## Two results worth knowing before you trust any premium-selling backtest

**Fat tails alone do not hurt a premium seller.** A 15–30 delta strike is only ~1.1σ out,
and a variance-matched leptokurtic distribution is *more* peaked near the money, so it puts
less mass beyond the short strike. What destroys a naked seller is a crash jump combined
with undefined risk: at matched variance and nearly identical P(ITM), a naked short put's EV
collapses to −0.44 while the same trade as a defined-risk spread stays at +0.23. See
STRATEGY.md §2.1 for the numbers.

**Execution cost is not a rounding error.** On a 15-trade sample run, the greek P&L summed
to −1,630 while execution and commissions summed to −9,330 — 85% of the loss. Theta captured
was +1,310. Every backtest here fills at the bid/ask, never at mid, and the report's P&L
bridge closes exactly, so this is visible rather than absorbed into an unexplained residual.

## Development

```bash
uv run pytest -q      # must be green before any commit
```

Everything crossing a module boundary conforms to `src/odds_lab/schema.py`. Selection and
exit code takes an `asof` timestamp and is tested against lookahead. See CLAUDE.md.
