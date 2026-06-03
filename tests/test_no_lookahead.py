"""
tests/test_no_lookahead.py — 룩어헤드 방지 단언

두 가지를 직접 단언:
(a) 미래 교란 불변성:
    t+k(k≥1) 입력(w_target)을 임의로 바꿔도 t 이전 체결 비중 w_t가 불변.
(b) 체결 lag:
    t일 gross return = w_held[t−1] × r_equity[t] + (1 − w_held[t−1]) × rf[t].
    즉 t일 수익은 t−1 정보로 확정된 비중으로 계산된다.

엔드투엔드: ConstantWeightIndicator → backtest.run() → metrics.summary() 흐름 확인.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import run
from src.benchmarks import buy_and_hold, equal_exposure
from src.indicators.base import ConstantWeightIndicator
from src.metrics import avg_exposure, summary


# ── 공통 픽스처 ───────────────────────────────────────────────────────────────

def _base_inputs(n: int = 100):
    idx = pd.bdate_range("2005-01-03", periods=n)
    r_eq = pd.Series(0.001, index=idx)
    rf_pct = pd.Series(4.0, index=idx)       # 연율 4% → 일간 4/100/252
    cfg = {"rebalance_band": 0.0, "cost_bps": 0.0, "w_max": 1.0}
    return idx, r_eq, rf_pct, cfg


# ── (a) 미래 교란 불변성 ──────────────────────────────────────────────────────

def test_future_perturbation_invariant():
    """
    w_target[50:] 를 0.3으로 바꿔도 weights[:49] 가 변하지 않아야 한다.
    백테스트 엔진이 t 이전 체결을 t 이후 입력과 독립적으로 계산함을 검증.
    """
    idx, r_eq, rf_pct, cfg = _base_inputs(100)
    w_orig = pd.Series(0.7, index=idx)

    result_orig = run(w_orig, r_eq, rf_pct, cfg)

    w_pert = w_orig.copy()
    w_pert.iloc[50:] = 0.3
    result_pert = run(w_pert, r_eq, rf_pct, cfg)

    pd.testing.assert_series_equal(
        result_orig["weights"].iloc[:50],
        result_pert["weights"].iloc[:50],
        check_names=False,
    )


def test_future_return_perturbation_invariant():
    """
    r_equity[70:] 를 ±0.1 크게 교란해도 weights[:69] 가 불변.
    (수익이 바뀌면 드리프트가 바뀌지만, 과거 구간은 무관해야 한다.)
    """
    idx, r_eq, rf_pct, cfg = _base_inputs(100)
    w = pd.Series(0.6, index=idx)

    result_orig = run(w, r_eq, rf_pct, cfg)

    r_pert = r_eq.copy()
    rng = np.random.default_rng(99)
    r_pert.iloc[70:] += rng.uniform(-0.1, 0.1, 30)
    result_pert = run(w, r_pert, rf_pct, cfg)

    pd.testing.assert_series_equal(
        result_orig["weights"].iloc[:70],
        result_pert["weights"].iloc[:70],
        check_names=False,
    )


# ── (b) 체결 lag ──────────────────────────────────────────────────────────────

def test_execution_lag_gross_return():
    """
    day t의 r_gross[t] = weights[t−1] × r_equity[t] + (1 − weights[t−1]) × rf_daily[t].
    cost=0·band=0이므로 weights[t] = w_target[t] 항등.
    """
    n = 30
    idx, r_eq, rf_pct, cfg = _base_inputs(n)

    # 교대 비중으로 변화를 만든다
    vals = [0.8 if i % 2 == 0 else 0.2 for i in range(n)]
    w_target = pd.Series(vals, index=idx)

    result = run(w_target, r_eq, rf_pct, cfg)
    weights = result["weights"]
    r_gross = result["returns_gross"]
    rf_daily = rf_pct / 100.0 / 252.0

    for t in range(1, n):
        expected = (
            weights.iloc[t - 1] * r_eq.iloc[t]
            + (1.0 - weights.iloc[t - 1]) * rf_daily.iloc[t]
        )
        assert r_gross.iloc[t] == pytest.approx(expected, abs=1e-12), (
            f"day {t}: expected {expected:.8f}, got {r_gross.iloc[t]:.8f}"
        )


def test_execution_lag_with_band_and_cost():
    """
    band>0·cost>0에서도 t일 gross return이 t−1 확정 비중으로 결정된다.
    (cost는 별도 차감이고 gross 자체는 t−1 비중 의존)
    """
    n = 50
    idx = pd.bdate_range("2005-01-03", periods=n)
    rng = np.random.default_rng(7)
    r_eq = pd.Series(rng.normal(0.001, 0.01, n), index=idx)
    rf_pct = pd.Series(4.0, index=idx)
    cfg = {"rebalance_band": 0.05, "cost_bps": 2.0, "w_max": 1.0}

    w_target = pd.Series(
        [0.8 if i % 3 != 0 else 0.2 for i in range(n)], index=idx
    )

    result = run(w_target, r_eq, rf_pct, cfg)
    weights = result["weights"]
    r_gross = result["returns_gross"]
    rf_daily = rf_pct / 100.0 / 252.0

    for t in range(1, n):
        expected = (
            weights.iloc[t - 1] * r_eq.iloc[t]
            + (1.0 - weights.iloc[t - 1]) * rf_daily.iloc[t]
        )
        assert r_gross.iloc[t] == pytest.approx(expected, abs=1e-12), (
            f"day {t}: expected {expected:.8f}, got {r_gross.iloc[t]:.8f}"
        )


# ── 엔드투엔드: 더미 신호 → 엔진 → 지표 → 대조군 ────────────────────────────

def test_e2e_dummy_constant_weight():
    """
    ConstantWeightIndicator(0.7) → backtest → metrics.summary 전 파이프라인 동작.
    gross ≥ net, avg_exposure ≈ 0.7.
    """
    n = 252 * 5
    idx = pd.bdate_range("2005-01-03", periods=n)
    rng = np.random.default_rng(42)

    data = {
        "sp500tr": pd.Series(rng.normal(0.0004, 0.01, n), index=idx),
        "rf":      pd.Series(4.0, index=idx),  # 연율 %
    }
    cfg = {
        "rebalance_band": 0.05, "cost_bps": 2.0, "w_max": 1.0,
        "borrow_spread_bps": 50, "leverage_max": 2.0,
    }

    indicator = ConstantWeightIndicator(0.7)
    w_target = indicator.signal(data, cfg)

    result = run(w_target, data["sp500tr"], data["rf"], cfg)

    # gross ≥ net (비용 차감)
    assert (result["equity_gross"].iloc[-1] >= result["equity_net"].iloc[-1])

    # 상수 비중 → avg_exposure가 0.7 근방 (밴드·드리프트로 약간 차이)
    avg_w = avg_exposure(result["weights"])
    assert avg_w == pytest.approx(0.7, abs=0.05)

    # metrics.summary 호출 가능
    bench_ret = data["sp500tr"]
    rf_daily = data["rf"] / 100.0 / 252.0
    m = summary(
        result["equity_net"],
        result["returns_net"],
        bench_ret,
        rf_daily,
        result["weights"],
    )
    assert set(m.keys()) == {
        "cagr", "annual_vol", "sharpe", "sortino",
        "mdd", "calmar", "up_capture", "down_capture",
        "annual_turnover", "avg_exposure",
    }
    assert not np.isnan(m["cagr"])
    assert m["mdd"] <= 0.0


def test_e2e_benchmarks():
    """buy_and_hold·equal_exposure 대조군이 정상 반환값을 돌려주는지."""
    n = 252 * 3
    idx = pd.bdate_range("2005-01-03", periods=n)
    rng = np.random.default_rng(0)
    r_eq = pd.Series(rng.normal(0.0004, 0.01, n), index=idx)
    rf_pct = pd.Series(4.0, index=idx)
    cfg = {"rebalance_band": 0.05, "cost_bps": 2.0, "w_max": 1.0,
           "borrow_spread_bps": 50}

    bnh = buy_and_hold(r_eq, rf_pct, cfg)
    ee = equal_exposure(0.6, r_eq, rf_pct, cfg)

    # 두 대조군 모두 올바른 키 반환
    for key in ("equity_gross", "equity_net", "returns_gross",
                "returns_net", "weights", "turnover"):
        assert key in bnh
        assert key in ee

    # buy&hold는 평균 노출 ≈ 1.0
    assert avg_exposure(bnh["weights"]) == pytest.approx(1.0, abs=0.01)

    # equal_exposure는 평균 노출 ≈ mean_w (band 드리프트로 약간 차이)
    assert avg_exposure(ee["weights"]) == pytest.approx(0.6, abs=0.05)
