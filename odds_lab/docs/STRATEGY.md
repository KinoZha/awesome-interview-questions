# ODDS Methodology — Formal Spec

Distilled from the five source PDFs (Don Fishback: *How to Win 80%*, *The Casino Secret*,
*Quick Start Guide*, *OPI Selection Process*, *Options Trading As A Business*).
This file is the **single source of truth** for what the backtester must implement.
Everything here is stated as a testable rule, not prose.

---

## 1. The thesis

Options are a zero-sum, probability-priced game. The market prices options off an
assumed **lognormal (Black-Scholes) distribution**. Realized returns are *not*
lognormal — they are leptokurtic, regime-dependent, and skewed.

> **Edge = P_theoretical(move) − P_actual(move)**

When the market's implied probability of a large move exceeds the empirically
observed frequency of that move, OTM options are overpriced → **sell premium**.
When the reverse holds (crisis regimes, 2008/2020-style), the seller's edge is
negative → **stand down**. The system must *measure* this, not assume it.

Source: *Casino Secret* pp. 20–21, 49–65 ("Calculate, Count, Compare"), which
explicitly documents that the actual distribution matched theory in 1995–2000,
was flatter/left-skewed in 2000–2002, and was wildly fat-tailed in 2008.

## 2. The three-step edge computation ("Calculate, Count, Compare")

For an underlying `S`, horizon `T` trading days, and a target price `K`:

1. **Calculate** (theoretical, from option prices):
   - `x = ln(K / S)`
   - `sigma_T = IV * sqrt(T_calendar / 365)`   (IV of the option being traded)
   - `z = (x - (r - q - IV^2/2) * T_cal/365) / sigma_T`
   - `P_theo(S_T >= K) = 1 - N(z)` — i.e. risk-neutral P(ITM) = `N(d2)` for calls.
   - Equivalently, `|delta|` is the market's cheap proxy for P(ITM). We compute
     both and store both; delta is used for *strike selection*, N(d2) for *edge*.

2. **Count** (empirical, from realized returns):
   - Take the trailing window of `S` closes (configurable: 10y default, also
     "all history" and "trailing 2y" variants).
   - Form all overlapping `T`-trading-day log changes: `r_i = ln(S_{i+T}/S_i)`.
   - `P_actual(S_T >= K) = #{r_i >= x} / N`.
   - **Symmetrization** (*Casino Secret* p. 59): under no-arbitrage there can be no
     persistent directional bias, so the reference distribution is also computed
     on the symmetrized sample `{r_i} ∪ {-r_i}` (centered on the drift). Both the
     raw and symmetrized estimates are stored; symmetrized is the default.
   - Overlapping windows ⇒ autocorrelated samples. Confidence intervals must use a
     block bootstrap (stationary bootstrap, mean block = T), never the naive
     binomial CI. This is a hard requirement — the naive CI will lie by ~3x.

3. **Compare**:
   - `edge_prob = P_theo(loss) - P_actual(loss)` for the short strike.
   - `edge_ev  = credit * P_actual(win) - max_loss * P_actual(loss)` (spread), i.e.
     expected value re-priced with empirical probabilities.
   - A trade is **eligible only if `edge_ev > 0`** under the empirical distribution.
     This is the single filter that separates this system from naive premium selling.

### 2.1 A result that shapes the whole design

Verified numerically with an independent reference implementation (S=400, IV=16%, 30d,
short put 380 / long put 375, 1e6 draws):

| terminal distribution (variance matched to IV) | P(short strike ITM) | naked short put EV | 380/375 spread EV |
|---|---|---|---|
| lognormal | 0.1345 | +0.003 | +0.002 |
| Student-t(3) | 0.0770 | +0.175 | +0.226 |
| 3% chance of a −4σ crash jump | 0.0757 | **−0.444** | **+0.229** |

Two things follow, and both are load-bearing:

1. **"Fat tails" alone do not hurt a premium seller at a 15–30 delta strike.** That strike
   is only ~1.1σ out. A variance-matched leptokurtic distribution is *more* peaked near the
   money, so it puts *less* mass beyond the short strike. This is why naive premium selling
   looks so good for years at a time. Any test asserting "fat tails ⇒ seller loses" is wrong.
2. **What kills the seller is a crash jump combined with undefined risk.** Same variance,
   same P(ITM) — the naked put's EV collapses while the defined-risk spread's does not,
   because the spread's loss is capped at `width − credit`. This is the Barings / Sep-2008
   argument from the source material, quantified.

So the backtester must (a) measure the edge with a payoff integral over the empirical
distribution rather than a two-point win/lose approximation, and (b) always report the naked
variant alongside the defined-risk one, because the difference between them only shows up in
a handful of months out of twelve years.

## 3. Strategies to implement

Ordered by priority. All are defined on ETF underlyings (SPY, QQQ, IWM).

| # | Name | Legs | Directional view | Source |
|---|------|------|------------------|--------|
| S1 | **Put credit spread** | short put @ short strike, long put 1+ strikes lower | not-down | Quick Start p.33, OPI |
| S2 | **Call credit spread** | short call, long call further OTM | not-up | Quick Start p.30 |
| S3 | **Iron condor** (= S1 + S2) | both credit spreads, same expiry | range-bound | *How to Win 80%* p.6-7 |
| S4 | **Short strangle** | short OTM put + short OTM call, naked | range-bound | *How to Win 80%* p.3 |
| S5 | **Cash-secured short put** | short OTM put | not-down | OPI Selection Process |
| S6 | **Long straddle / strangle** | the inverse trade | volatility expansion | Course syllabus |

S4 is included **only as a risk-comparison baseline** — the source material's central
argument (Barings, Sep/Oct 2008) is that undefined risk eventually ruins the account.
The backtest must be able to show this explicitly.

## 4. Entry rules

### 4.1 The "±5% / 80%" baseline rule (*How to Win 80%*)
- Once per expiry cycle, ~1 month (21 trading days) before expiration:
  - Upper strike = nearest listed strike to `S * 1.05`
  - Lower strike = nearest listed strike to `S * 0.95`
  - Sell those; buy the next strike further OTM for the spread version.
- This is the **naive baseline** the ODDS-filtered version must beat.

### 4.2 The ODDS / OPI rule (production rule)
Filters, from `OPI_Selection_Process.pdf` (adapted from single stocks to ETFs —
the industry-momentum stock screen is dropped, the option-level filters are kept):

| Parameter | Value |
|---|---|
| DTE at entry | 21 ≤ DTE ≤ 56 |
| Short-strike delta | 0.15 ≤ \|Δ\| ≤ 0.30 |
| Min net credit | $0.30 per spread |
| Spread width | 1, 2, 3, 5 strikes (sweep) |
| Expected return | 0% < ER ≤ 50%, where `ER = credit / (width - credit)` |
| Ranking | highest **empirical-probability** expected return (§2), not highest ER |
| Market-state filter | see §4.3 |
| Liquidity | short-leg bid > 0, spread(bid,ask)/mid ≤ configurable (default 15%), OI ≥ 100 |

`min_open_interest`'s default (100) is unchanged, but its semantics were clarified: a
real ThetaData bulk CSV chain export has no open-interest column at all, so
`data/providers/csv_export.py` (and `thetadata.py`) emit `OPEN_INTEREST_UNKNOWN` (-1,
never 0) when OI is absent. `strategy/selector.py`'s liquidity filter treats
`open_interest < 0` as "unknown -- skip this check", never as "0 contracts" -- defaulting
unknown OI to 0 would fail every row from that data source and reproduce the exact
silent-zero-trade failure mode this funnel exists to catch, just triggered by the data
source instead of a threshold. The funnel records this separately as
`liquidity_oi_unknown` (informational; it does not by itself reject a candidate, so it
is not part of the `rejected` partition) so a report on real CSV-export data visibly
says "OI was unavailable for N candidates, the filter was skipped" instead of silently
looking identical to a passing OI check.

#### 4.2.1 Selection funnel + measured defaults on $1-spaced ETF strikes

`propose_trade` records a structured selection funnel (`engine/loop.py`
`manifest['selection_funnel']`, per-opportunity detail in `BacktestResult.funnel`) so a
zero-trade or near-zero-trade run is diagnosable instead of silently reporting "success."
Running it on `data/sample_store` (SPY/QQQ/IWM, 2015-06-01→2016-12-30, weekly entries,
`put_credit_spread`, `strike_rule=delta`) gave:

| strike_rule | width_strikes | min_credit | trades | binding rejection reason |
|---|---|---|---|---|
| delta | 1 | 0.30 (old default) | **0** | `min_credit` / `liquidity_spread` |
| delta | 2 | 0.30 | 15 | `liquidity_spread` |
| delta | 3 | 0.30 | 18 | `liquidity_spread` |
| delta | 5 | 0.30 | 10 | `liquidity_spread` |
| delta | 1 | 0.20 | **0** | `min_credit` |
| delta | 1 | 0.10 | 12 | `liquidity_spread` |
| delta | 1 | 0.05 | 19 | `liquidity_spread` |
| pct_otm (±5% baseline) | 1 | 0.30 | **0** | `liquidity_spread` |

Root cause: at `width_strikes=1` the expected-return band (`ER = credit/(width-credit)
≤ 0.50`) already caps the *achievable* credit near `width/3 ≈ $0.33` before `min_credit`
is even checked, and real bid/ask spreads on a 1-strike-wide $180 ETF put spread push
most remaining candidates out on `liquidity_spread` (the OPI $0.30 threshold was written
for $20-40 single stocks with $2.50-5 strike spacing, where a comparable spread is
several strikes wide in dollar terms). Lowering `min_credit` alone (0.30→0.20) does not
fix it -- `width_strikes=1` stays at 0 trades even then; only widening the spread does,
because it gives both the ER band and the liquidity check room to work with.

**Default changed:** `EntryConfig.width_strikes` 1 → **2**. `min_credit` stays at $0.30
-- the dollar figure itself is not wrong, it is only infeasible when combined with a
1-strike width on $1-spaced strikes. This keeps the OPI-sourced numbers intact and fixes
the defect by picking a different point in OPI's own explicitly-stated width sweep
(`{1, 2, 3, 5}`) rather than inventing a new threshold. If a caller sets
`width_strikes=1` explicitly, they should also lower `min_credit` (roughly 10-15% of
`width_strikes * 100 * increment` reproduces the pre-existing $0.30-at-width-3ish
strictness) -- expressed as a fraction of spread width, not a bare dollar amount, since
that is the scale-free quantity that actually transfers across strike spacings; `min_credit`
itself is left as a dollar figure (unchanged semantics/type) so this is a documented
rule of thumb for callers, not a code behavior change.

`no_empirical_dist` also dominates the *first* year of any run on `data/sample_store`
regardless of `width_strikes`/`min_credit`: the store only has 2015-01-01→2016-12-30 of
history, and `EmpiricalConfig.min_samples=250` overlapping horizon-length windows are not
available until roughly a year in. This is a data-availability artifact of the small
sample store, not a defaults problem, and is out of scope for this change (`EmpiricalConfig`
defaults were not touched) -- flagged here so it isn't mistaken for another dead filter.

### 4.3 Market-state filter
`OPI` note: *"evaluate whether or not you want to take the trade if the Market State
is Bearish or Neutral."* Implemented as a 3-state classifier on the underlying,
computed **only from data available at entry** (strictly no lookahead):
- `bullish` : close > SMA200 and 6-month momentum > 0
- `bearish` : close < SMA200 and 6-month momentum < 0
- `neutral` : otherwise

Filter variants to backtest as a grid: `{take all, skip bearish, put-side only when
not bearish, call-side only when not bullish}`.

Additional regime input: `IV rank` / `IV percentile` of the underlying's ATM IV over
the trailing 252 days, and the `IV − realized vol` spread (the direct edge proxy).

## 5. Exit rules (each is a switchable policy, all must be backtested)

| Policy | Rule | Source |
|---|---|---|
| `hold_to_expiry` | settle at intrinsic on expiration | baseline |
| `profit_target` | buy back at X% of credit captured (X ∈ {25, 50, 75}) | *Two Ways to Exit*, class 5/20/19 |
| `stop_loss` | close when loss = N × credit (N ∈ {1, 2, 3}) or at short-strike touch | *Quick Start* p.20 (sell-stop at 50%) |
| `dte_exit` | close at fixed DTE remaining (7, 14, 21) | gamma-risk management |
| `delta_breach` | close when short-strike \|Δ\| ≥ 0.50 | |
| combined | profit_target OR stop_loss OR dte_exit, whichever first | production default |

Assignment: American-style ETF options. If the short leg is ITM at expiry it is
assigned; spread legs are exercised/assigned together (auto-exercise if ITM by
$0.01 per OCC rules). Early assignment must be modeled around ex-dividend dates for
short ITM calls (SPY/QQQ/IWM all pay quarterly dividends) — see §7.

## 6. Position sizing & risk (*Casino Secret* p. 72–73)

- **5%–10% of portfolio equity risked per trade** (default 5%; sweepable).
- **Never more than 50% of the portfolio allocated to a single style of trade.**
- Contract count = `floor(equity * risk_pct / max_loss_per_contract)`.
- For undefined-risk trades (S4/S5), size by broker margin (Reg-T approximation:
  `max(20% * S - OTM_amount, 10% * S) * 100 + premium`, floor $250/contract) —
  and the report must display the margin-utilization time series, because that is
  where naked selling actually dies.

## 7. Realism requirements (non-negotiable — these decide whether the result is real)

1. **Fill at bid/ask, never at mid.** Sell at bid, buy at ask, per leg. Then also
   report a mid-fill and a `mid ± k×(spread/2)` sweep so slippage sensitivity is visible.
2. **Commissions**: `$0.65/contract` default, both open and close, sweepable, plus
   exchange/regulatory fees on sells (~$0.05/contract).
3. **No lookahead**: entry decisions use the quote snapshot at the entry timestamp
   (EOD close quote for daily granularity); exits are evaluated on the *next*
   available quote after the trigger condition, not the triggering quote itself.
4. **No survivorship / no strike-existence hindsight**: only strikes that actually
   have quotes on the entry date may be selected.
5. **Dividends & interest**: `r` from a daily risk-free curve, `q` from the ETF's
   actual trailing dividend yield. Cash balance earns `r`. Both matter over 12 years.
6. **Corporate actions**: none of SPY/QQQ/IWM split in the window (verify at load).
   The loader must still assert that the option multiplier is 100 and flag any
   non-standard (adjusted) contracts, which must be excluded.
7. **Expiration settlement**: PM-settled at the close of the expiration Friday for
   these ETFs. Use the underlying's official close.

## 8. What the visualization must answer

The user's core requirement is *"why did it make money, why did it lose money"*.
Every backtest run must produce, per underlying and per strategy:

**A. Equity & attribution**
- Equity curve, drawdown curve, monthly/yearly return heatmap.
- P&L attribution decomposition per trade: `theta captured`, `delta P&L`,
  `vega P&L`, `gamma P&L`, `slippage/commission`. (Computed by re-pricing the
  position under each frozen-factor scenario between snapshots — a discrete
  Greek attribution that reconciles to actual P&L with a residual term.)

**B. The trade lifecycle view (the key screen)**
For any selected trade: underlying price path with the short/long strike bands
drawn as horizontal zones, the credit received, the mark-to-market P&L path,
short-strike delta path, and the exit trigger marked. This makes "the market went
through my short strike on day 14" visually obvious.

**C. The edge view (the ODDS chart)**
Reproduction of the *Casino Secret* pp. 52–63 chart: theoretical lognormal density
(line) vs the empirical histogram of realized T-day log returns (bars), per period.
Rendered per calendar year and per regime, with the traded strikes marked on the
x-axis. This is literally the "see your edge" picture and it must be reproduced.

**D. Aggregate diagnostics**
- Win rate vs the promised 80%; distribution of P&L per trade (the fat left tail).
- P&L bucketed by: entry IV rank, market state, DTE, delta, VIX level, year.
- Calendar view: every week × every year grid, colored by that cohort's return —
  the user explicitly asked for per-week-per-year granularity.
- Comparison panel: naive ±5% rule vs delta-selected vs ODDS-edge-filtered.

## 9. Explicit non-goals

- No single-stock universe / industry momentum screen (OPI §1) in v1 — ETFs only.
- No live trading, no broker integration.
- No intraday entries in v1 (EOD granularity); the data layer must not preclude it.
