"""
tests/test_metrics.py — 지표 기대값 단위 테스트

합성 시계열으로 닫힌형(closed-form) 기대값과 비교.
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

# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def _make_returns(r: float, n: int, start: str = "2000-01-03") -> pd.Series:
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(r, index=idx)


def _equity_from_returns(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod()


def _equity_from_const(r: float, n: int, start: str = "2000-01-03") -> pd.Series:
    return _equity_from_returns(_make_returns(r, n, start))


def _pos_mean_returns(
    rng: np.random.Generator,
    mean: float,
    std: float,
    n: int,
    start: str = "2000-01-03",
) -> pd.Series:
    """평균을 exactly `mean`으로 고정한 수익률 시리즈 (std ≈ std)."""
    raw = rng.normal(0.0, std, n)
    adj = raw - raw.mean() + mean
    return pd.Series(adj, index=pd.bdate_range(start, periods=n))


# ── CAGR ──────────────────────────────────────────────────────────────────────

def test_cagr_constant_return():
    """
    일정 일간수익률 r → CAGR = (1+r)^252 − 1 (닫힌형).

    equity = (1+r).cumprod() → n 관측, n−1 기간.
    ratio = equity[-1]/equity[0] = (1+r)^(n−1).
    CAGR = ratio^(252/(n−1)) = (1+r)^252. ✓
    """
    r = 0.0003
    n = 252 * 10
    equity = _equity_from_const(r, n)
    expected = (1.0 + r) ** 252 - 1.0
    assert cagr(equity) == pytest.approx(expected, rel=1e-6)


def test_cagr_zero():
    equity = _equity_from_const(0.0, 252)
    assert cagr(equity) == pytest.approx(0.0, abs=1e-10)


# ── 연율 변동성 ───────────────────────────────────────────────────────────────

def test_annual_vol_constant_zero():
    """일정 수익률 → 변동성 0"""
    returns = _make_returns(0.001, 252)
    assert annual_vol(returns) == pytest.approx(0.0, abs=1e-10)


def test_annual_vol_known():
    """std = s → annual_vol = s × √252"""
    rng = np.random.default_rng(42)
    r_arr = rng.normal(0, 0.01, 504)
    returns = pd.Series(r_arr, index=pd.bdate_range("2000-01-03", periods=504))
    expected = float(np.std(r_arr, ddof=1) * np.sqrt(252))
    assert annual_vol(returns) == pytest.approx(expected, rel=1e-6)


# ── Sharpe ────────────────────────────────────────────────────────────────────

def test_sharpe_zero_excess_returns_zero():
    """
    returns = rf → excess = 0 → Sharpe = 0 (분자 0, 분모 > 0).
    상수 시리즈의 std()는 부동소수점 잔차(≈ 2e-19)로 0 미도달 → 0.0/vol = 0.0.
    """
    idx = pd.bdate_range("2000-01-03", periods=252)
    returns = pd.Series(0.001, index=idx)
    rf = pd.Series(0.001, index=idx)
    assert sharpe(returns, rf) == pytest.approx(0.0, abs=1e-6)


def test_sharpe_positive_excess():
    idx = pd.bdate_range("2000-01-03", periods=252 * 5)
    rng = np.random.default_rng(0)
    returns = _pos_mean_returns(rng, mean=0.001, std=0.01, n=len(idx))
    rf = pd.Series(0.0001, index=idx)
    assert sharpe(returns, rf) > 0.0


# ── MDD ───────────────────────────────────────────────────────────────────────

def test_mdd_planted_drawdown():
    """고점 110 → 저점 60 → MDD = (60/110 − 1)"""
    equity = pd.Series(
        [100.0, 110.0, 100.0, 90.0, 80.0, 70.0, 60.0, 70.0, 90.0, 110.0],
        index=pd.bdate_range("2000-01-03", periods=10),
    )
    expected = 60.0 / 110.0 - 1.0
    assert max_drawdown(equity) == pytest.approx(expected, rel=1e-6)


def test_mdd_monotone_up():
    """순상승 → MDD = 0"""
    equity = _equity_from_const(0.001, 252)
    assert max_drawdown(equity) == pytest.approx(0.0, abs=1e-8)


# ── Calmar ────────────────────────────────────────────────────────────────────

def test_calmar_consistent():
    equity = pd.Series(
        [100.0, 110.0, 80.0, 90.0, 120.0],
        index=pd.bdate_range("2000-01-03", periods=5),
    )
    c = cagr(equity)
    m = max_drawdown(equity)
    assert calmar(equity) == pytest.approx(c / abs(m), rel=1e-6)


# ── Sortino ───────────────────────────────────────────────────────────────────

def test_sortino_positive():
    """
    평균을 정확히 0.001로 고정 → excess > 0 → Sortino > 0.
    """
    idx = pd.bdate_range("2000-01-03", periods=252 * 5)
    rng = np.random.default_rng(1)
    returns = _pos_mean_returns(rng, mean=0.001, std=0.01, n=len(idx))
    rf = pd.Series(0.0001, index=idx)
    assert sortino(returns, rf) > 0.0


def test_sortino_ge_sharpe_when_positive():
    """
    excess > 0 일 때 Sortino ≥ Sharpe.
    이유: N(μ,σ²) 에서 dd = sqrt(E[min(r,0)²]) ≤ std(r).
    (하방 편차는 음수 편차만 집계 → 전체 std 이하)
    """
    idx = pd.bdate_range("2000-01-03", periods=252 * 5)
    rng = np.random.default_rng(2)
    returns = _pos_mean_returns(rng, mean=0.001, std=0.01, n=len(idx))
    rf = pd.Series(0.0001, index=idx)
    assert sortino(returns, rf) >= sharpe(returns, rf)


# ── 연 회전율 ─────────────────────────────────────────────────────────────────

def test_annual_turnover_alternating():
    """
    252×5=1260일, 홀짝 0.51/0.49 교대 → |Δw|=0.02 매 거래일.
    연 회전율 = 0.02 × (n−1) / 5
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

def test_capture_ratios_double_up():
    """
    전략 일간수익 = 벤치마크 2배인 상승달, 동일한 하락달.

    소폭 수익률(0.0001/day)을 써야 복리 비선형성이 작아 up_cap ≈ 2.0.
    annualized capture = s_ann/b_ann = ((1+2b)^12−1)/((1+b)^12−1) ≈ 2 for small b.
    허용 오차 rel=0.05 (5%).
    """
    idx = pd.bdate_range("2000-01-03", periods=252 * 3)
    month = idx.month

    b_up   =  0.0001   # 상승달 벤치마크 일간수익
    b_down = -0.00005  # 하락달 벤치마크 일간수익

    bench = pd.Series(0.0, index=idx)
    strat = pd.Series(0.0, index=idx)

    odd  = pd.Series(month % 2 == 1, index=idx)
    even = ~odd

    bench[odd]  = b_up
    bench[even] = b_down
    strat[odd]  = 2.0 * b_up    # 상승달 2배
    strat[even] = b_down        # 하락달 동일

    up_cap, down_cap = capture_ratios(strat, bench)

    assert up_cap   == pytest.approx(2.0, rel=0.05), f"up_cap={up_cap:.4f}"
    assert down_cap == pytest.approx(1.0, rel=0.05), f"down_cap={down_cap:.4f}"


def test_capture_ratios_symmetry():
    """strat = bench → up_cap = down_cap = 1.0"""
    idx = pd.bdate_range("2000-01-03", periods=252 * 2)
    rng = np.random.default_rng(99)
    bench = pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx)
    up_cap, down_cap = capture_ratios(bench, bench)
    assert up_cap   == pytest.approx(1.0, rel=1e-4)
    assert down_cap == pytest.approx(1.0, rel=1e-4)
