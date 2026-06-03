"""
tests/test_no_lookahead.py — 룩어헤드 방지 단언

두 가지를 직접 단언:
(a) 미래 교란 불변성:
    t+k(k≥1) 입력(w_target)을 임의로 바꿔도 t 이전 체결 비중 w_t가 불변.
(b) 체결 lag:
    t일 gross return = w_held[t−1] × r_equity[t] + (1 − w_held[t−1]) × rf[t].
    즉 t일 수익은 t−1 정보로 확정된 비중으로 계산된다.

엔드투엔드: ConstantWeightIndicator → backtest.run() → metrics.summary() 흐름 확인.

M3-A 추가: VolatilityIndicator(더미 아닌 진짜 신호)로 두 단언 실제 통과 검증.
  (a-vix)  VIX 모드 미래 교란 불변
  (a-real) realized 모드 미래 교란 불변
  (b-vol)  VIX 모드 체결 lag

M3-A 추가 규약 단언:
  (c) _assert_realized_alignment: 시프트된 σ 인덱스를 감지해 ValueError 발생.
  (d) equal_exposure mean_w = avg_exposure(result["weights"])  (실현 비중 평균, 목표비중 평균 아님).
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
        result["turnover"],
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
    assert avg_exposure(ee["weights"]) == pytest.approx(0.6, abs=0.002)


# ── M3-A: VolatilityIndicator 룩어헤드 단언 ───────────────────────────────────

def _make_vol_data(n: int, seed: int) -> tuple[dict, pd.DatetimeIndex]:
    idx = pd.bdate_range("2005-01-03", periods=n)
    rng = np.random.default_rng(seed)
    sp_prices = 1000.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n))
    data = {
        "vix":     pd.Series(rng.uniform(10.0, 40.0, n), index=idx),
        "sp500tr": pd.Series(sp_prices, index=idx),
    }
    return data, idx


def test_vol_vix_future_perturbation_invariant():
    """
    (a-vix) VIX 모드: t+k VIX를 극단값으로 교란해도 t<40 signal 불변.
    VIX 신호는 각 t에서 VIX[t]만 사용 → 미래 교란 전파 없음.
    """
    from src.indicators.volatility import VolatilityIndicator
    data, _ = _make_vol_data(80, seed=42)
    cfg = {"vol_estimator": "vix", "sigma_target": 0.12, "w_max": 1.0}

    ind = VolatilityIndicator()
    sig_orig = ind.signal(data, cfg)

    data_pert = {k: v.copy() for k, v in data.items()}
    data_pert["vix"].iloc[40:] = 200.0          # 미래 극단 VIX
    sig_pert = ind.signal(data_pert, cfg)

    pd.testing.assert_series_equal(
        sig_orig.iloc[:40], sig_pert.iloc[:40], check_names=False
    )


def test_vol_realized_future_perturbation_invariant():
    """
    (a-real) realized 모드: t+k sp500tr 교란이 t<50 signal에 영향 없음.
    rolling(N) std at t → sp_ret[t-N+1:t] causal 구간만 사용.
    """
    from src.indicators.volatility import VolatilityIndicator
    N = 21
    data, _ = _make_vol_data(100, seed=43)
    cfg = {"vol_estimator": "realized", "realized_lookback": N,
           "sigma_target": 0.12, "w_max": 1.0}

    ind = VolatilityIndicator()
    sig_orig = ind.signal(data, cfg)

    # sp500tr[50:]을 100배 교란 → sp_ret[50]~가 바뀌고 sigma[50+]도 바뀜
    data_pert = {k: v.copy() for k, v in data.items()}
    data_pert["sp500tr"].iloc[50:] *= 100.0
    sig_pert = ind.signal(data_pert, cfg)

    # sigma[t<50] 은 sp_ret[t-N+1..t], 모두 50 미만 인덱스 → 불변
    pd.testing.assert_series_equal(
        sig_orig.iloc[:50], sig_pert.iloc[:50], check_names=False
    )


def test_vol_signal_execution_lag():
    """
    (b-vol) VolatilityIndicator(vix 모드): 체결 lag 단언.
    band=0·cost=0 → weights[t] = w_target[t] exactly.
    r_gross[t] = weights[t-1] × r_eq[t] + (1 − weights[t-1]) × rf[t].
    """
    from src.indicators.volatility import VolatilityIndicator
    n = 60
    data, idx = _make_vol_data(n, seed=7)
    cfg = {
        "vol_estimator": "vix", "sigma_target": 0.12, "w_max": 1.0,
        "rebalance_band": 0.0, "cost_bps": 0.0, "borrow_spread_bps": 50,
    }

    w_target = VolatilityIndicator().signal(data, cfg)
    r_eq    = data["sp500tr"].pct_change().fillna(0.0)
    rf_pct  = pd.Series(4.0, index=idx)

    result  = run(w_target, r_eq, rf_pct, cfg)
    weights = result["weights"]
    r_gross = result["returns_gross"]
    rf_d    = rf_pct / 100.0 / 252.0

    for t in range(1, n):
        expected = (
            weights.iloc[t - 1] * r_eq.iloc[t]
            + (1.0 - weights.iloc[t - 1]) * rf_d.iloc[t]
        )
        assert r_gross.iloc[t] == pytest.approx(expected, abs=1e-12), (
            f"day {t}: expected {expected:.10f}, got {r_gross.iloc[t]:.10f}"
        )


# ── (c) _assert_realized_alignment: 시프트 감지 ───────────────────────────────

def test_realized_alignment_catches_shifted_index():
    """
    (c) _assert_realized_alignment: σ 인덱스를 1일 앞당기면 ValueError(룩어헤드).
    정상(동일 인덱스)은 통과.
    """
    from src.indicators.volatility import _assert_realized_alignment

    n = 60
    idx = pd.bdate_range("2010-01-04", periods=n)
    rng = np.random.default_rng(5)
    sp_prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n))
    sp_ret = pd.Series(sp_prices, index=idx).pct_change()

    # 정상: rolling → 동일 인덱스 → 통과
    sigma_ok = sp_ret.rolling(5).std() * np.sqrt(252)
    _assert_realized_alignment(sp_ret, sigma_ok)  # 예외 없음

    # 비정상: σ 인덱스를 1일 앞당김 (길이 같지만 인덱스 다름)
    idx_shifted = idx - pd.tseries.offsets.BusinessDay(1)
    sigma_shifted = pd.Series(sigma_ok.values, index=idx_shifted)

    with pytest.raises(ValueError, match="룩어헤드"):
        _assert_realized_alignment(sp_ret, sigma_shifted)


# ── (d) equal_exposure: 실현 비중 평균 사용 패턴 ──────────────────────────────

def test_equal_exposure_uses_realized_weights():
    """
    (d) equal_exposure mean_w = avg_exposure(result["weights"]).
    목표비중 평균(w_target.mean()) 아님.

    올바른 패턴:
      result = run(w_target, ...)
      mean_w = avg_exposure(result["weights"])   ← 실현 비중 평균
      ee = equal_exposure(mean_w, ...)

    이 패턴으로 실행한 equal_exposure의 avg_exposure ≈ 전략의 avg_exposure.
    """
    n = 252 * 2
    idx = pd.bdate_range("2010-01-04", periods=n)
    rng = np.random.default_rng(77)
    r_eq = pd.Series(rng.normal(0.0004, 0.012, n), index=idx)
    rf_pct = pd.Series(4.0, index=idx)
    cfg = {
        "rebalance_band": 0.05, "cost_bps": 2.0, "w_max": 1.0,
        "borrow_spread_bps": 50,
    }

    # VolatilityIndicator vix 모드로 실제 신호 생성
    from src.indicators.volatility import VolatilityIndicator
    data_mock = {
        "vix":     pd.Series(rng.uniform(10.0, 40.0, n), index=idx),
        "sp500tr": pd.Series(
            100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n)), index=idx
        ),
    }
    w_target = VolatilityIndicator().signal(
        data_mock, {"vol_estimator": "vix", "sigma_target": 0.12, **cfg}
    )

    result = run(w_target, r_eq, rf_pct, cfg)

    # 실현 비중 평균 (올바른 방법)
    realized_mean_w = avg_exposure(result["weights"])

    # equal_exposure를 realized_mean_w로 실행
    ee = equal_exposure(realized_mean_w, r_eq, rf_pct, cfg)

    # 검증: 대조군 평균노출 ≈ 전략 실현 평균노출
    assert avg_exposure(ee["weights"]) == pytest.approx(realized_mean_w, abs=0.002), (
        f"대조군 avg_exposure({avg_exposure(ee['weights']):.4f}) ≠ "
        f"전략 realized_mean_w({realized_mean_w:.4f})"
    )


# ── M3-B: CreditIndicator 룩어헤드 단언 ──────────────────────────────────────

def _make_credit_data(n: int, seed: int) -> dict:
    """n 거래일 합성 baa10y 데이터 생성."""
    idx = pd.bdate_range("2000-01-03", periods=n)
    rng = np.random.default_rng(seed)
    # baa10y: 1~5% 사이 랜덤워크로 현실적 스프레드 모사
    spread = 2.0 + np.cumsum(rng.normal(0, 0.02, n))
    spread = np.clip(spread, 0.5, 8.0)
    sp_prices = 1000.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n))
    return {
        "baa10y":  pd.Series(spread, index=idx),
        "sp500tr": pd.Series(sp_prices, index=idx),
        "rf":      pd.Series(4.0, index=idx),
    }


def test_credit_future_perturbation_invariant():
    """
    (a-credit) baa10y t+k 교란이 t<300 신호에 영향 없음.

    p_t = trailing 252일 백분위: [t-251, t] 구간만 사용.
    t=300 이후 baa10y를 극단값으로 바꿔도 t<300 신호는 불변이어야 한다.
    (300은 워밍업 252 + 여유 48)
    """
    from src.indicators.credit import CreditIndicator
    n = 500
    data = _make_credit_data(n, seed=10)
    cfg  = {"percentile_window": 252, "theta_low_pct": 0.5,
            "theta_high_pct": 0.9, "w_max": 1.0}

    ind      = CreditIndicator()
    sig_orig = ind.signal(data, cfg)

    data_pert = {k: v.copy() for k, v in data.items()}
    data_pert["baa10y"].iloc[300:] = 99.0   # 극단 스트레스로 교란

    sig_pert = ind.signal(data_pert, cfg)

    # t < 300 신호는 [t-251, t] 구간만 사용 → 불변
    pd.testing.assert_series_equal(
        sig_orig.iloc[:300], sig_pert.iloc[:300], check_names=False
    )


def test_credit_signal_execution_lag():
    """
    (b-credit) CreditIndicator: 체결 lag 단언.
    band=0·cost=0 → weights[t] = w_target[t] exactly.
    r_gross[t] = weights[t-1] × r_eq[t] + (1 − weights[t-1]) × rf[t].
    """
    from src.indicators.credit import CreditIndicator
    n   = 400
    data = _make_credit_data(n, seed=11)
    cfg  = {
        "percentile_window": 252, "theta_low_pct": 0.5,
        "theta_high_pct": 0.9, "w_max": 1.0,
        "rebalance_band": 0.0, "cost_bps": 0.0, "borrow_spread_bps": 50,
    }

    ind      = CreditIndicator()
    w_target = ind.signal(data, cfg)

    # run()은 dropna() 인덱스 기준 → 워밍업 제외
    valid_idx = w_target.dropna().index
    r_eq  = data["sp500tr"].pct_change().reindex(valid_idx).fillna(0.0)
    rf    = data["rf"].reindex(valid_idx)
    rf_d  = rf / 100.0 / 252.0

    result  = run(w_target.reindex(valid_idx), r_eq, rf, cfg)
    weights = result["weights"]
    r_gross = result["returns_gross"]
    n_valid = len(valid_idx)

    for t in range(1, n_valid):
        expected = (
            weights.iloc[t - 1] * r_eq.iloc[t]
            + (1.0 - weights.iloc[t - 1]) * rf_d.iloc[t]
        )
        assert r_gross.iloc[t] == pytest.approx(expected, abs=1e-12), (
            f"day {t}: expected {expected:.10f}, got {r_gross.iloc[t]:.10f}"
        )


def test_credit_percentile_alignment_catches_shifted_index():
    """
    (c-credit) _assert_percentile_alignment: baa10y 인덱스를 시프트하면 ValueError.
    정상(동일 인덱스)은 통과.
    """
    from src.indicators.credit import _assert_percentile_alignment, _percentile_rank

    n   = 300
    idx = pd.bdate_range("2000-01-03", periods=n)
    rng = np.random.default_rng(99)
    baa = pd.Series(2.0 + np.cumsum(rng.normal(0, 0.02, n)), index=idx).clip(0.5, 8.0)

    # 정상: rolling → 동일 인덱스 → 통과
    p_ok = _percentile_rank(baa, 21)
    _assert_percentile_alignment(baa, p_ok)  # 예외 없음

    # 비정상: p 인덱스를 1일 앞당김 → 시프트 감지
    idx_shifted = idx - pd.tseries.offsets.BusinessDay(1)
    p_shifted   = pd.Series(p_ok.values, index=idx_shifted)

    with pytest.raises(ValueError):
        _assert_percentile_alignment(baa, p_shifted)
