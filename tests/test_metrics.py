"""
tests/test_metrics.py — 지표 기대값 단위 테스트

변별력 원칙: 수식이 틀리면 다른 값이 나와야 한다.
equity 규약: (1+r).cumprod() — equity[0]=1+r_0, 암시적 초기=1.0, N=len(equity)거래일.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    annual_turnover,
    annual_vol,
    avg_exposure,
    calmar,
    capture_ratios,
    cagr,
    max_drawdown,
    sharpe,
    sortino,
)

# ── 공통 헬퍼 ─────────────────────────────────────────────────────────────────

def _equity(returns: pd.Series) -> pd.Series:
    """backtest 규약: (1+r).cumprod(), equity[0]=1+r_0."""
    return (1.0 + returns).cumprod()


def _pos_mean_returns(
    rng: np.random.Generator,
    mean: float,
    std: float,
    n: int,
    start: str = "2000-01-03",
) -> pd.Series:
    """평균을 exactly mean으로 고정한 수익률 시리즈."""
    raw = rng.normal(0.0, std, n)
    adj = raw - raw.mean() + mean
    return pd.Series(adj, index=pd.bdate_range(start, periods=n))


# ── CAGR ──────────────────────────────────────────────────────────────────────

def test_cagr_varied_handcalc():
    """
    변동 있는 시계열 + 손계산으로 수식 변별.

    equity 규약: equity[-1] = 총수익배수 = ∏(1+r_i), 암시적 초기=1.0.
    CAGR = equity[-1]^(252/N) − 1, N = 5.

    잘못된 수식 (equity[-1]/equity[0])^(252/4) 은 약 −29% 를 반환 — 완전히 다름.
    """
    r = pd.Series(
        [0.01, -0.01, 0.02, -0.02, 0.005],
        index=pd.bdate_range("2020-01-02", periods=5),
    )
    eq = _equity(r)

    # 손 계산: 총수익 = 1.01×0.99×1.02×0.98×1.005
    total = 1.01 * 0.99 * 1.02 * 0.98 * 1.005   # ≈ 1.004498
    n = 5
    expected = total ** (252.0 / n) - 1.0         # ≈ +25.4%

    assert eq.iloc[-1] == pytest.approx(total, rel=1e-9)
    assert cagr(eq) == pytest.approx(expected, rel=1e-6)

    # 변별: 잘못된 (equity[-1]/equity[0])^(252/4) 은 크게 다름
    wrong = (eq.iloc[-1] / eq.iloc[0]) ** (252.0 / (n - 1)) - 1.0  # ≈ −29%
    assert abs(cagr(eq) - wrong) > 0.10  # 10% 이상 차이


def test_cagr_zero_return():
    """전 기간 수익 0 → CAGR = 0."""
    r = pd.Series(0.0, index=pd.bdate_range("2000-01-03", periods=252))
    eq = _equity(r)
    assert cagr(eq) == pytest.approx(0.0, abs=1e-10)


def test_cagr_exact_annual():
    """
    N=252 거래일, 일간수익 r → CAGR = (1+r)^252 − 1 (닫힌형).
    equity[-1] = (1+r)^252, CAGR = equity[-1]^(252/252) − 1 = (1+r)^252 − 1.
    """
    r_val = 0.0004
    n = 252
    r = pd.Series(r_val, index=pd.bdate_range("2000-01-03", periods=n))
    eq = _equity(r)
    expected = (1.0 + r_val) ** 252 - 1.0
    assert cagr(eq) == pytest.approx(expected, rel=1e-8)


# ── 연율 변동성 ───────────────────────────────────────────────────────────────

def test_annual_vol_constant_zero():
    r = pd.Series(0.001, index=pd.bdate_range("2000-01-03", periods=252))
    assert annual_vol(r) == pytest.approx(0.0, abs=1e-10)


def test_annual_vol_known():
    """std = s → annual_vol = s × √252"""
    rng = np.random.default_rng(42)
    arr = rng.normal(0, 0.01, 504)
    r = pd.Series(arr, index=pd.bdate_range("2000-01-03", periods=504))
    expected = float(np.std(arr, ddof=1) * np.sqrt(252))
    assert annual_vol(r) == pytest.approx(expected, rel=1e-6)


# ── Sharpe ────────────────────────────────────────────────────────────────────

def test_sharpe_near_zero_vol_returns_nan():
    """
    상수 수익률 시리즈는 std()≈2e-19 (부동소수점 잔차).
    _EPS 임계치로 vol<_EPS 판단 → nan 반환.
    """
    idx = pd.bdate_range("2000-01-03", periods=252)
    returns = pd.Series(0.001, index=idx)
    rf = pd.Series(0.001, index=idx)
    assert np.isnan(sharpe(returns, rf))


def test_sharpe_positive_excess():
    idx = pd.bdate_range("2000-01-03", periods=252 * 5)
    rng = np.random.default_rng(0)
    returns = _pos_mean_returns(rng, mean=0.001, std=0.01, n=len(idx))
    rf = pd.Series(0.0001, index=idx)
    assert sharpe(returns, rf) > 0.0


# ── MDD ───────────────────────────────────────────────────────────────────────

def test_mdd_planted():
    """고점 110 → 저점 60 → MDD = 60/110 − 1"""
    eq = pd.Series(
        [100.0, 110.0, 100.0, 90.0, 80.0, 70.0, 60.0, 70.0, 90.0, 110.0],
        index=pd.bdate_range("2000-01-03", periods=10),
    )
    assert max_drawdown(eq) == pytest.approx(60.0 / 110.0 - 1.0, rel=1e-6)


def test_mdd_monotone_up():
    r = pd.Series(0.001, index=pd.bdate_range("2000-01-03", periods=252))
    assert max_drawdown(_equity(r)) == pytest.approx(0.0, abs=1e-8)


# ── Calmar ────────────────────────────────────────────────────────────────────

def test_calmar_consistent():
    eq = pd.Series(
        [100.0, 110.0, 80.0, 90.0, 120.0],
        index=pd.bdate_range("2000-01-03", periods=5),
    )
    assert calmar(eq) == pytest.approx(cagr(eq) / abs(max_drawdown(eq)), rel=1e-6)


# ── Sortino ───────────────────────────────────────────────────────────────────

def test_sortino_positive():
    idx = pd.bdate_range("2000-01-03", periods=252 * 5)
    rng = np.random.default_rng(1)
    returns = _pos_mean_returns(rng, mean=0.001, std=0.01, n=len(idx))
    rf = pd.Series(0.0001, index=idx)
    assert sortino(returns, rf) > 0.0


def test_sortino_ge_sharpe_when_positive():
    """
    excess > 0, MAR=0 → 하방편차 DD ≤ full std → Sortino ≥ Sharpe.
    """
    idx = pd.bdate_range("2000-01-03", periods=252 * 5)
    rng = np.random.default_rng(2)
    returns = _pos_mean_returns(rng, mean=0.001, std=0.01, n=len(idx))
    rf = pd.Series(0.0001, index=idx)
    assert sortino(returns, rf) >= sharpe(returns, rf)


# ── 연 회전율 ─────────────────────────────────────────────────────────────────

def test_annual_turnover_alternating():
    """
    홀짝 0.51/0.49 교대 → |Δw|=0.02 매 거래일.
    연 회전율 = 0.02 × (n−1) / (n/252)
    """
    idx = pd.bdate_range("2000-01-03", periods=252 * 5)
    w = pd.Series(0.51, index=idx)
    w.iloc[1::2] = 0.49
    expected = 0.02 * (len(idx) - 1) / 5.0
    assert annual_turnover(w) == pytest.approx(expected, rel=1e-4)


# ── 평균 노출 ─────────────────────────────────────────────────────────────────

def test_avg_exposure():
    idx = pd.bdate_range("2000-01-03", periods=100)
    w = pd.Series(0.65, index=idx)
    assert avg_exposure(w) == pytest.approx(0.65, rel=1e-6)


# ── 상/하방 캡처 ──────────────────────────────────────────────────────────────

def test_capture_ratios_closed_form():
    """
    상승달 1개(벤치 +10%, 전략 +20%), 하락달 1개(벤치 −5%, 전략 −2.5%).
    각 월에 거래일 1개씩 → 월간 복리수익 = 일간 수익과 동일 (복리 모호성 없음).

    up_cap   = 0.20 / 0.10 = 2.0  (닫힌형)
    down_cap = (−0.025) / (−0.05) = 0.5
    연율화 없음 — 연율화 공식이면 다른 값이 나온다.
    """
    # 서로 다른 달에 1거래일씩
    idx = pd.DatetimeIndex(["2020-01-15", "2020-02-15"])
    bench = pd.Series([0.10, -0.05],  index=idx)
    strat = pd.Series([0.20, -0.025], index=idx)

    up_cap, down_cap = capture_ratios(strat, bench)
    assert up_cap   == pytest.approx(2.0, rel=1e-6)
    assert down_cap == pytest.approx(0.5, rel=1e-6)


def test_capture_ratios_symmetry():
    """전략 = 벤치마크 → up_cap = down_cap = 1.0"""
    idx = pd.bdate_range("2000-01-03", periods=252 * 2)
    rng = np.random.default_rng(99)
    bench = pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx)
    up_cap, down_cap = capture_ratios(bench, bench)
    assert up_cap   == pytest.approx(1.0, rel=1e-4)
    assert down_cap == pytest.approx(1.0, rel=1e-4)
