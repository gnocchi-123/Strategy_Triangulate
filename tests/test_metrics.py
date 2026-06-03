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
    annual_turnover는 이제 실제 체결 크기(turn_arr)를 입력받는다.
    홀짝 0.51/0.49 교대 → 매 거래일 체결 크기 0.02.
    연 회전율 = 0.02 × n / (n/252) = 0.02 × 252 (n→∞ 근사)
    n 거래일 기준: 0.02 × n / (n/252) = 0.02 × 252 / 1 = 5.04 (정확히 5yr 평균)
    """
    idx = pd.bdate_range("2000-01-03", periods=252 * 5)
    # turn_arr: 매 거래일 0.02씩 체결 (실제 체결 크기 시뮬레이션)
    n = len(idx)
    turn_arr = pd.Series(0.02, index=idx)
    expected = 0.02 * n / (n / 252)   # = 0.02 × 252 = 5.04
    assert annual_turnover(turn_arr) == pytest.approx(expected, rel=1e-4)


# ── 평균 노출 ─────────────────────────────────────────────────────────────────

def test_avg_exposure():
    idx = pd.bdate_range("2000-01-03", periods=100)
    w = pd.Series(0.65, index=idx)
    assert avg_exposure(w) == pytest.approx(0.65, rel=1e-6)


# ── 상/하방 캡처 ──────────────────────────────────────────────────────────────

def test_capture_ratios_closed_form():
    """
    연율화 복리수익 비율 닫힌형 검증 (DEFINITIONS 4절, Morningstar 표준).
    ann = (∏(1+r_m))^(12/k) − 1, k = 해당 달 수.

    상승달 12개: bench +2%/월, strat +1%/월 (각 1거래일)
      b_up_ann = 1.02^12 − 1 = 0.268242
      s_up_ann = 1.01^12 − 1 = 0.126825
      UpCap    = 0.126825 / 0.268242 ≈ 0.47280

    하락달 12개: bench −1%/월, strat −0.5%/월 (각 1거래일)
      b_dn_ann = 0.99^12 − 1 = −0.113615
      s_dn_ann = 0.995^12 − 1 = −0.058207
      DnCap    = −0.058207 / −0.113615 ≈ 0.51228

    비연율화(k=12이면 두 공식이 우연히 동치)이므로 아래 변별 테스트로 보완.
    """
    # 월별 1거래일: 상승 12개월(2020), 하락 12개월(2021)
    up_idx = pd.date_range("2020-01-15", periods=12, freq="MS") + pd.offsets.Day(14)
    dn_idx = pd.date_range("2021-01-15", periods=12, freq="MS") + pd.offsets.Day(14)

    bench = pd.concat([pd.Series([0.02]*12, index=up_idx),
                       pd.Series([-0.01]*12, index=dn_idx)])
    strat = pd.concat([pd.Series([0.01]*12, index=up_idx),
                       pd.Series([-0.005]*12, index=dn_idx)])

    up_cap, down_cap = capture_ratios(strat, bench)

    exp_up = (1.01**12 - 1) / (1.02**12 - 1)    # ≈ 0.47280
    exp_dn = (0.995**12 - 1) / (0.99**12 - 1)   # ≈ 0.51228
    assert up_cap   == pytest.approx(exp_up, rel=1e-6)
    assert down_cap == pytest.approx(exp_dn, rel=1e-6)


def test_capture_ratios_scale_invariance():
    """
    Scale-invariance: 같은 월수익 패턴을 k=12/120/240 으로 반복해도
    캡처 값이 동일해야 한다 (연율화 공식의 핵심 성질).

    균등 수익이면 (r^k)^(12/k) = r^12 이므로 k에 무관하다.
    비연율화(∏-1) 방식은 k=12→0.47, k=120→0.24, k=240→0.09 로 붕괴한다.
    """
    r_b_up, r_s_up = 0.02, 0.01
    r_b_dn, r_s_dn = -0.01, -0.005
    theory_up = (1.01**12 - 1) / (1.02**12 - 1)   # ≈ 0.47280
    theory_dn = (0.995**12 - 1) / (0.99**12 - 1)  # ≈ 0.51382

    for k in [12, 120, 240]:
        up_idx = pd.date_range("2000-01-01", periods=k, freq="MS") + pd.offsets.Day(14)
        dn_idx = pd.date_range(up_idx[-1] + pd.offsets.MonthBegin(1),
                               periods=k, freq="MS") + pd.offsets.Day(14)
        bench = pd.concat([pd.Series([r_b_up]*k, index=up_idx),
                           pd.Series([r_b_dn]*k, index=dn_idx)])
        strat = pd.concat([pd.Series([r_s_up]*k, index=up_idx),
                           pd.Series([r_s_dn]*k, index=dn_idx)])

        up_cap, dn_cap = capture_ratios(strat, bench)
        assert up_cap == pytest.approx(theory_up, rel=1e-8), \
            f"k={k}: UpCap={up_cap:.8f} ≠ {theory_up:.8f} (scale-invariance 위반)"
        assert dn_cap == pytest.approx(theory_dn, rel=1e-8), \
            f"k={k}: DnCap={dn_cap:.8f} ≠ {theory_dn:.8f} (scale-invariance 위반)"


def test_capture_ratios_long_series_sanity():
    """
    20년·65% 고정 노출 전략 → UpCap ≈ 60%, DnCap ≈ 69% (seed=42 기준).

    연율화 공식은 65% 노출에 상응하는 값을 반환.
    비연율화(∏-1) 방식이면 117개 상승달에서 ≈0%로 수렴한다.
    실제값: UpCap=59.99%, DnCap=69.08% (seed=42, 20yr, N(0.0004,0.01)).
    """
    n = 252 * 20
    rng = np.random.default_rng(42)
    bench_d = pd.Series(rng.normal(0.0004, 0.01, n),
                        index=pd.bdate_range("2000-01-03", periods=n))
    strat_d = bench_d * 0.65

    up_cap, down_cap = capture_ratios(strat_d, bench_d)

    # 65% 고정 노출 → 캡처가 노출 근처(50~80%)여야 함
    assert 0.50 <= up_cap   <= 0.80, f"UpCap={up_cap:.4f} 이탈 (기대 0.5~0.8)"
    assert 0.55 <= down_cap <= 0.85, f"DnCap={down_cap:.4f} 이탈 (기대 0.55~0.85)"


def test_capture_ratios_symmetry():
    """전략 = 벤치마크 → up_cap = down_cap = 1.0"""
    idx = pd.bdate_range("2000-01-03", periods=252 * 2)
    rng = np.random.default_rng(99)
    bench = pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx)
    up_cap, down_cap = capture_ratios(bench, bench)
    assert up_cap   == pytest.approx(1.0, rel=1e-4)
    assert down_cap == pytest.approx(1.0, rel=1e-4)
