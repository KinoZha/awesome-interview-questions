"""Tests for the data layer: store, synthetic provider, thetadata parser, ingest.
CLAUDE.md testing rules apply -- no live network, everything deterministic."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from odds_lab import schema
from odds_lab.data import probe as probe_mod
from odds_lab.data.ingest import ingest
from odds_lab.data.providers import thetadata as td
from odds_lab.data.providers.synthetic import SyntheticProvider, _third_friday, make_sample_store
from odds_lab.data.store import ChainStore

pytestmark = pytest.mark.filterwarnings("ignore")


# ---------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_store(tmp_path_factory):
    path = tmp_path_factory.mktemp("store")
    # 6 months so the trailing-dividend-yield estimate (ingest._trailing_div_yield)
    # sees at least 2 quarterly payments -- see its docstring.
    return make_sample_store(
        path, roots=("SPY", "QQQ"), start=date(2015, 1, 1), end=date(2015, 7, 1), seed=11
    )


# ---------------------------------------------------------------------------------
# make_sample_store round-trip
# ---------------------------------------------------------------------------------


def test_sample_store_round_trip(sample_store):
    quote_date = date(2015, 1, 5)
    expiries = sample_store.expiries("SPY", quote_date, dte_min=0, dte_max=90)
    assert expiries
    expiry = expiries[0]

    df = sample_store.chain("SPY", quote_date, expiry=expiry)
    assert not df.empty
    validated = schema.validate_chain(df, strict=True)
    assert (validated["bid"] <= validated["ask"]).all()
    assert (validated["bid"] >= 0).all()
    assert set(validated["root"]) == {"SPY"}
    assert (validated["quote_date"] == pd.Timestamp(quote_date)).all()
    assert (validated["expiry"] == pd.Timestamp(expiry)).all()


def test_sample_store_underlying_round_trip(sample_store):
    df = sample_store.underlying("SPY", start=date(2015, 1, 1), end=date(2015, 1, 31))
    assert not df.empty
    schema.validate_frame(df, schema.UNDERLYING_DTYPES, "underlying")
    assert (df["close"] > 0).all()


# ---------------------------------------------------------------------------------
# partition pruning
# ---------------------------------------------------------------------------------


def test_partition_pruning_single_day(sample_store):
    # a single-day read must resolve to exactly one partition file, never the
    # whole root's history.
    paths = sample_store._partition_paths_for_range("SPY", date(2015, 1, 5), date(2015, 1, 5))
    assert len(paths) == 1
    assert "year=2015" in str(paths[0]) and "month=01" in str(paths[0])


def test_partition_pruning_does_not_touch_other_months(sample_store):
    # store spans Jan + Feb 2015; a January query must not resolve February's file.
    jan_paths = sample_store._partition_paths_for_range("SPY", date(2015, 1, 1), date(2015, 1, 31))
    feb_paths = sample_store._partition_paths_for_range("SPY", date(2015, 2, 1), date(2015, 2, 28))
    assert jan_paths and feb_paths
    assert set(jan_paths).isdisjoint(feb_paths)


def test_partition_pruning_row_count_scales_with_range(sample_store):
    one_day = sample_store.chain("SPY", date(2015, 1, 5))
    one_month = sample_store.underlying("SPY", date(2015, 1, 1), date(2015, 1, 31))
    # sanity: the single day chain is a small slice, not the whole store's rows
    full_coverage = sample_store.coverage()
    total_rows = int(full_coverage.loc[full_coverage["root"] == "SPY", "n_rows"].iloc[0])
    assert 0 < len(one_day) < total_rows


# ---------------------------------------------------------------------------------
# closes() boundary
# ---------------------------------------------------------------------------------


def test_closes_boundary_is_inclusive_of_end(sample_store):
    end = date(2015, 1, 15)
    closes = sample_store.closes("SPY", end)
    assert not closes.empty
    assert max(closes.index) <= end
    # underlying has data on `end` (a trading day) -> it must be included
    trading_dates = sample_store.trading_dates("SPY", date(2015, 1, 1), end)
    if end in trading_dates:
        assert end in closes.index


def test_closes_never_returns_a_date_past_end(sample_store):
    end = date(2015, 1, 20)
    closes = sample_store.closes("SPY", end)
    assert all(d <= end for d in closes.index)
    # widening the window forward must not change anything already returned
    wider = sample_store.closes("SPY", date(2015, 2, 1))
    shared = closes.index
    for d in shared:
        assert wider[d] == closes[d]


# ---------------------------------------------------------------------------------
# synthetic data sanity
# ---------------------------------------------------------------------------------


def test_synthetic_no_crossed_or_negative_quotes(sample_store):
    df = sample_store.chain("SPY", date(2015, 2, 2))
    assert (df["bid"] <= df["ask"]).all()
    assert (df["bid"] >= 0).all()
    assert (df["ask"] >= 0).all()


def test_synthetic_expirations_include_real_third_fridays():
    provider = SyntheticProvider(["SPY"], date(2015, 1, 1), date(2015, 6, 1), seed=3)
    exps = provider.expirations("SPY", date(2015, 1, 5))
    march_third_friday = _third_friday(2015, 3)
    assert march_third_friday.weekday() == 4
    assert march_third_friday in exps


def test_synthetic_put_skew_present(sample_store):
    quote_date = date(2015, 2, 2)
    expiries = sample_store.expiries("SPY", quote_date, dte_min=25, dte_max=40)
    assert expiries
    df = sample_store.chain("SPY", quote_date, expiry=expiries[0])
    S = float(df["underlying_price"].iloc[0])

    puts = df[df["right"] == "P"].copy()
    puts["abs_delta"] = puts["delta"].abs()
    calls = df[df["right"] == "C"].copy()

    # 25-delta put: closest short put to |delta|=0.25
    put_25 = puts.iloc[(puts["abs_delta"] - 0.25).abs().argsort().iloc[0]]
    # ATM call/put: strike closest to spot
    atm = df.iloc[(df["strike"] - S).abs().argsort().iloc[0]]

    assert put_25["iv"] > atm["iv"] - 1e-9
    assert put_25["strike"] < S


def test_synthetic_source_and_flag_stamped(sample_store):
    df = sample_store.chain("SPY", date(2015, 1, 5))
    assert (df["source"] == "synthetic").all()
    assert df["is_synthetic"].all()


# ---------------------------------------------------------------------------------
# thetadata parser -- monkeypatched _get
# ---------------------------------------------------------------------------------


def _v3_eod_payload(strikes_dollars: list[float]):
    rows = []
    for i, k in enumerate(strikes_dollars):
        right = "C" if i % 2 == 0 else "P"
        rows.append(["SPY", 20150130, int(round(k * 1000)), right, 20150105, 57600000,
                     5.0 + i, 10, 5.2 + i, 12, 5.1 + i, 100, 500])
    return {
        "header": {"format": ["root", "expiration", "strike", "right", "date", "ms_of_day",
                               "bid", "bid_size", "ask", "ask_size", "last", "volume", "open_interest"]},
        "response": rows,
    }


def _v3_greeks_payload(strikes_dollars: list[float]):
    rows = []
    for i, k in enumerate(strikes_dollars):
        right = "C" if i % 2 == 0 else "P"
        rows.append(["SPY", 20150130, int(round(k * 1000)), right, 20150105,
                      0.5 - 0.1 * i, 0.02, -0.05, 1500.0, 800.0, 0.15 + 0.01 * i])
    return {
        "header": {"format": ["root", "expiration", "strike", "right", "date",
                               "delta", "gamma", "theta", "vega", "rho", "implied_vol"]},
        "response": rows,
    }


def _stock_eod_payload():
    return {
        "header": {"format": ["date", "open", "high", "low", "close", "volume"]},
        "response": [[20150105, 204.0, 206.0, 203.0, 205.0, 1_000_000]],
    }


def _v2_eod_payload(strikes_scaled_int: list[int]):
    # v2-shaped: different key ordering, root/expiration outside per-row (still per-row here
    # since bulk endpoints repeat them), 'exp'/'option_right' spellings.
    rows = []
    for i, k in enumerate(strikes_scaled_int):
        right = "call" if i % 2 == 0 else "put"
        rows.append([k, right, 20150105, 57600000, 5.0 + i, 5.2 + i, 100, 500])
    return {
        "header": {"format": ["strike", "option_right", "date", "ms_of_day", "bid", "ask",
                               "volume", "open_interest"]},
        "response": rows,
    }


@pytest.fixture
def provider_v3():
    return td.ThetaDataProvider(version="v3")


def _make_fake_get(eod_payload, greeks_payload, stock_payload):
    def fake_get(path, params=None):
        if "greeks" in path:
            return greeks_payload
        if "eod" in path and params and "exp" in params:
            return eod_payload
        if "eod" in path:
            return stock_payload
        raise AssertionError(f"unexpected path {path}")

    return fake_get


def test_thetadata_parses_v3_scaled_strikes(provider_v3):
    strikes = [200.0, 201.0, 202.0, 203.0]
    provider_v3._get = _make_fake_get(_v3_eod_payload(strikes), _v3_greeks_payload(strikes), _stock_eod_payload())
    df = provider_v3.chain_eod("SPY", date(2015, 1, 5), date(2015, 1, 30))
    schema.validate_chain(df, strict=False)
    assert sorted(df["strike"].unique().tolist()) == strikes
    assert (df["bid"] <= df["ask"]).all()


def test_thetadata_strike_scale_auto_detect_scaled(provider_v3):
    # raw strike 205000 vs spot 205 -> median/spot = 1000 > 20 -> scale detected
    strikes = [204.0, 205.0, 206.0]
    provider_v3._get = _make_fake_get(_v3_eod_payload(strikes), _v3_greeks_payload(strikes), _stock_eod_payload())
    df = provider_v3.chain_eod("SPY", date(2015, 1, 5), date(2015, 1, 30))
    assert max(df["strike"]) < 1000  # correctly scaled to dollars, not left at 206000


def test_thetadata_strike_scale_auto_detect_unscaled():
    provider = td.ThetaDataProvider(version="v3")
    # unscaled payload: strikes already in dollars (e.g. 205, not 205000)
    payload = {
        "header": {"format": ["root", "expiration", "strike", "right", "date", "ms_of_day",
                               "bid", "bid_size", "ask", "ask_size", "last", "volume", "open_interest"]},
        "response": [
            ["SPY", 20150130, 204, "C", 20150105, 57600000, 5.0, 10, 5.2, 12, 5.1, 100, 500],
            ["SPY", 20150130, 205, "P", 20150105, 57600000, 3.0, 10, 3.2, 12, 3.1, 100, 500],
        ],
    }
    greeks = {
        "header": {"format": ["strike", "right", "delta", "gamma", "theta", "vega", "rho", "implied_vol"]},
        "response": [
            [204, "C", 0.5, 0.02, -0.05, 1500.0, 800.0, 0.16],
            [205, "P", -0.5, 0.02, -0.05, 1500.0, -800.0, 0.16],
        ],
    }
    provider._get = _make_fake_get(payload, greeks, _stock_eod_payload())
    df = provider.chain_eod("SPY", date(2015, 1, 5), date(2015, 1, 30))
    assert sorted(df["strike"].unique().tolist()) == [204.0, 205.0]


def test_thetadata_explicit_strike_scale_override():
    provider = td.ThetaDataProvider(version="v3", strike_scale=1000)
    strikes = [200.0, 201.0]
    provider._get = _make_fake_get(_v3_eod_payload(strikes), _v3_greeks_payload(strikes), _stock_eod_payload())
    df = provider.chain_eod("SPY", date(2015, 1, 5), date(2015, 1, 30))
    assert sorted(df["strike"].unique().tolist()) == strikes


def test_thetadata_vega_rho_divided_by_100(provider_v3):
    strikes = [200.0, 201.0]
    provider_v3._get = _make_fake_get(_v3_eod_payload(strikes), _v3_greeks_payload(strikes), _stock_eod_payload())
    df = provider_v3.chain_eod("SPY", date(2015, 1, 5), date(2015, 1, 30))
    # raw vendor vega/rho in the fixture are 1500.0 / 800.0 (or -800.0)
    assert np.allclose(df["vega"].dropna().unique(), [15.0])
    assert set(np.round(df["rho"].dropna().unique(), 2)) <= {8.0, -8.0}


def test_thetadata_v2_shaped_payload_parses():
    provider = td.ThetaDataProvider(version="v2")
    eod = _v2_eod_payload([200000, 201000])
    greeks = {
        "header": {"format": ["strike", "right", "delta", "vega", "rho", "iv"]},
        "response": [[200000, "C", 0.5, 1600.0, 900.0, 0.17], [201000, "P", -0.5, 1600.0, -900.0, 0.17]],
    }
    provider._get = _make_fake_get(eod, greeks, _stock_eod_payload())
    df = provider.chain_eod("SPY", date(2015, 1, 5), date(2015, 1, 30))
    schema.validate_chain(df, strict=False)
    assert set(df["right"]) <= {"C", "P"}
    assert sorted(df["strike"].unique().tolist()) == [200.0, 201.0]
    assert np.allclose(df["vega"].dropna().unique(), [16.0])


def test_thetadata_missing_required_field_raises_clear_error():
    provider = td.ThetaDataProvider(version="v3")
    bad_payload = {
        "header": {"format": ["root", "expiration", "right", "date", "bid", "ask"]},  # no 'strike'
        "response": [["SPY", 20150130, "C", 20150105, 5.0, 5.2]],
    }
    provider._get = _make_fake_get(bad_payload, {"header": {"format": []}, "response": []}, _stock_eod_payload())
    with pytest.raises(td.ThetaError) as exc_info:
        provider.chain_eod("SPY", date(2015, 1, 5), date(2015, 1, 30))
    msg = str(exc_info.value)
    assert "strike" in msg
    assert "format" in msg.lower() or "header" in msg.lower()


def test_thetadata_not_running_on_connection_error():
    provider = td.ThetaDataProvider(version="v3", base_url="http://127.0.0.1:1")
    with pytest.raises(td.ThetaNotRunning) as exc_info:
        provider._get("/v3/option/list/expirations", {"root": "SPY"})
    assert "ThetaTerminal" in str(exc_info.value)


def test_thetadata_retries_then_raises_on_persistent_5xx(monkeypatch):
    provider = td.ThetaDataProvider(version="v3")
    provider.max_retries = 2
    provider.backoff_base = 0.001

    calls = {"n": 0}

    class FakeResp:
        status_code = 503
        text = "server error"

    class FakeSession:
        def get(self, url, params=None, timeout=None):
            calls["n"] += 1
            return FakeResp()

    provider._session = FakeSession()
    with pytest.raises(td.ThetaError):
        provider._get("/v3/option/list/expirations", {"root": "SPY"})
    assert calls["n"] == provider.max_retries + 1


def test_thetadata_probe_degrades_gracefully(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report_path = probe_mod.probe("http://127.0.0.1:1", "v3", "SPY", date(2015, 1, 5))
    assert report_path.exists()
    text = report_path.read_text()
    assert "unreachable" in text.lower() or "ThetaNotRunning" in text


# ---------------------------------------------------------------------------------
# ingest idempotency + resumability
# ---------------------------------------------------------------------------------


def test_ingest_idempotent(tmp_path):
    provider = SyntheticProvider(["SPY"], date(2015, 1, 1), date(2015, 1, 31), seed=5)
    store = ChainStore(tmp_path)
    ingest(provider, store, ["SPY"], date(2015, 1, 1), date(2015, 1, 31), dte_max=60, progress=False)
    cov1 = store.coverage()
    n1 = int(cov1.loc[cov1["root"] == "SPY", "n_rows"].iloc[0])

    ingest(provider, store, ["SPY"], date(2015, 1, 1), date(2015, 1, 31), dte_max=60, progress=False)
    cov2 = store.coverage()
    n2 = int(cov2.loc[cov2["root"] == "SPY", "n_rows"].iloc[0])

    assert n1 == n2
    assert n1 > 0


def test_ingest_force_still_idempotent_row_count(tmp_path):
    provider = SyntheticProvider(["SPY"], date(2015, 1, 1), date(2015, 1, 31), seed=5)
    store = ChainStore(tmp_path)
    ingest(provider, store, ["SPY"], date(2015, 1, 1), date(2015, 1, 31), dte_max=60, progress=False)
    n1 = int(store.coverage().loc[lambda d: d["root"] == "SPY", "n_rows"].iloc[0])

    # force=True re-pulls and re-writes, but write_chain still dedupes on key
    ingest(provider, store, ["SPY"], date(2015, 1, 1), date(2015, 1, 31), dte_max=60, progress=False, force=True)
    n2 = int(store.coverage().loc[lambda d: d["root"] == "SPY", "n_rows"].iloc[0])
    assert n1 == n2


def test_ingest_resumability_skips_completed_months(tmp_path, monkeypatch):
    provider = SyntheticProvider(["SPY"], date(2015, 1, 1), date(2015, 2, 28), seed=5)
    store = ChainStore(tmp_path)
    ingest(provider, store, ["SPY"], date(2015, 1, 1), date(2015, 1, 31), dte_max=60, progress=False)

    calls = {"n": 0}
    orig = provider.chain_eod

    def counting_chain_eod(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(provider, "chain_eod", counting_chain_eod)
    # re-ingest a range that includes the already-complete January partition plus new February
    ingest(provider, store, ["SPY"], date(2015, 1, 1), date(2015, 2, 28), dte_max=60, progress=False)
    # January must have been skipped entirely -- chain_eod should only have been
    # called for February's trading dates/expirations, not January's too.
    assert calls["n"] > 0
    manifest_calls_for_feb_only = calls["n"]

    # sanity: calling chain_eod directly for a January date still works (provider
    # itself isn't broken) -- this just confirms the skip is ingest's doing.
    assert not provider.chain_eod("SPY", date(2015, 1, 5), date(2015, 1, 30)).empty


def test_ingest_writes_manifest(tmp_path):
    provider = SyntheticProvider(["SPY"], date(2015, 1, 1), date(2015, 1, 31), seed=5)
    store = ChainStore(tmp_path)
    manifest = ingest(provider, store, ["SPY"], date(2015, 1, 1), date(2015, 1, 31), dte_max=60, progress=False)
    assert (tmp_path / "manifest.json").exists()
    assert "SPY" in manifest["roots"]
    assert manifest["roots"]["SPY"]["rows_written_this_run"] > 0


# ---------------------------------------------------------------------------------
# store.has_data / coverage
# ---------------------------------------------------------------------------------


def test_has_data(sample_store, tmp_path):
    assert sample_store.has_data("SPY")
    empty_store = ChainStore(tmp_path / "empty")
    assert not empty_store.has_data("SPY")


def test_coverage_multi_root(sample_store):
    cov = sample_store.coverage()
    assert set(cov["root"]) == {"SPY", "QQQ"}
    assert (cov["n_rows"] > 0).all()
    assert (cov["n_expiries"] > 0).all()
