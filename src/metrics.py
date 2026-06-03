"""
src/metrics.py — 성과 지표 (DEFINITIONS_AND_CONVENTIONS.md 4절)

모두 net 기준, 일간수익률 기반.
표기: r_t=전략 일간수익률, b_t=벤치마크 일간수익률,
      rf_t=무위험 일간수익률, N=거래일 수, 연율화=252.

─── equity 규약 ───────────────────────────────────────────────────────────────
backtest.run()이 반환하는 equity 시리즈는 (1+r).cumprod() 형태다.
  equity[0] = 1 + r_net[0]  (첫 거래일 수익 반영, ≠ 1.0)
  암시적 초기자산 = 1.0 (backtest 시작 전, 시리즈에 포함되지 않음)
  N = len(equity) = 거래일 수

CAGR = (최종/초기)^(252/N) − 1 = equity[-1]^(252/N) − 1
(초기 = 1.0, equity[-1]/equity[0] 형태를 쓰면 첫날 수익이 분모로 묻혀 N-1기간만 반영됨)
───────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_ANNUAL = 252
_EPS = 1e-10  # 부동소수점 잔차 방지용 임계치


def cagr(equity: pd.Series) -> float:
    """
    DEFINITIONS 4절: (최종/초기)^(252/N) − 1, N = 거래일 수.
    equity 규약: cumprod(1+r), 암시적 초기 = 1.0, N = len(equity).
    → equity[-1] ** (252/N) − 1.
    """
    n = len(equity)
    if n < 2:
        return float("nan")
    return float(equity.iloc[-1] ** (_ANNUAL / n) - 1)


def annual_vol(returns: pd.Series) -> float:
    """std(r_t) × √252"""
    return float(returns.std() * np.sqrt(_ANNUAL))


def sharpe(returns: pd.Series, rf: pd.Series) -> float:
    """
    [mean(r_t − rf_t) × 252] / [std(r_t) × √252]
    vol < _EPS이면 nan (부동소수점 잔차에 의한 0-나눗셈 방지).
    """
    excess = returns - rf.reindex(returns.index).fillna(0.0)
    vol = annual_vol(returns)
    if vol < _EPS:
        return float("nan")
    return float(excess.mean() * _ANNUAL / vol)


def sortino(returns: pd.Series, rf: pd.Series, mar: float = 0.0) -> float:
    """
    [mean(r_t − rf_t) × 252] / [DD × √252]
    DD = sqrt(mean(min(r_t − MAR, 0)^2)), MAR=0 기본.
    DD < _EPS이면 nan.
    """
    excess = returns - rf.reindex(returns.index).fillna(0.0)
    downside = np.minimum(returns.values - mar, 0.0)
    dd = float(np.sqrt(np.mean(downside ** 2)))
    if dd < _EPS:
        return float("nan")
    return float(excess.mean() * _ANNUAL / (dd * np.sqrt(_ANNUAL)))


def max_drawdown(equity: pd.Series) -> float:
    """min_t(equity_t / running_max_t − 1)  (음수)"""
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def calmar(equity: pd.Series) -> float:
    """CAGR / |MDD|"""
    mdd = max_drawdown(equity)
    if abs(mdd) < _EPS:
        return float("nan")
    return float(cagr(equity) / abs(mdd))


def capture_ratios(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> tuple[float, float]:
    """
    상/하방 캡처 (DEFINITIONS 4절 — Morningstar 표준, 연율화 복리수익 비율).

    상방 = 전략_ann(상승달) / 벤치_ann(상승달)
    하방 = 전략_ann(하락달) / 벤치_ann(하락달)
    연율화 복리수익 ann = (∏(1+r_m))^(12/k) − 1,  k = 해당 집합의 달 수.

    전체 누적곱(∏(1+r_m)−1) 방식은 k가 수백이면 분모가 수천 배로 폭발해
    비율이 0에 수렴하는 가짜값을 만든다 (예: 289 상승달 → 3.27%).
    연율화하면 k에 독립적인 "월평균 기하수익의 연율 환산"이 되어
    경제적으로 해석 가능한 값이 나온다.
    """
    strat_m = (1.0 + strategy_returns).resample("ME").prod() - 1.0
    bench_m = (1.0 + benchmark_returns).resample("ME").prod() - 1.0

    common = strat_m.index.intersection(bench_m.index)
    strat_m = strat_m.loc[common]
    bench_m = bench_m.loc[common]

    def _ann(vals: pd.Series) -> float:
        k = len(vals)
        if k == 0:
            return float("nan")
        return float((1.0 + vals).prod() ** (12.0 / k) - 1.0)

    def _ratio(s_vals: pd.Series, b_vals: pd.Series) -> float:
        if len(b_vals) == 0:
            return float("nan")
        b_ann = _ann(b_vals)
        if abs(b_ann) < _EPS:
            return float("nan")
        return _ann(s_vals) / b_ann

    up_mask   = bench_m > 0.0
    down_mask = bench_m < 0.0
    return _ratio(strat_m[up_mask], bench_m[up_mask]), _ratio(strat_m[down_mask], bench_m[down_mask])


def annual_turnover(weights: pd.Series) -> float:
    """one-way |Δw| 합 / 연 수 (연 회전율)"""
    if len(weights) < 2:
        return float("nan")
    delta = weights.diff().abs().dropna()
    n_years = len(weights) / _ANNUAL
    if n_years == 0.0:
        return float("nan")
    return float(delta.sum() / n_years)


def avg_exposure(weights: pd.Series) -> float:
    """mean(w_t) — 시장체류시간"""
    return float(weights.mean())


def summary(
    equity_net: pd.Series,
    returns_net: pd.Series,
    benchmark_returns: pd.Series,
    rf: pd.Series,
    weights: pd.Series,
) -> dict:
    """전체 지표 dict 반환 (모두 net 기준)."""
    up_cap, down_cap = capture_ratios(returns_net, benchmark_returns)
    return {
        "cagr":            cagr(equity_net),
        "annual_vol":      annual_vol(returns_net),
        "sharpe":          sharpe(returns_net, rf),
        "sortino":         sortino(returns_net, rf),
        "mdd":             max_drawdown(equity_net),
        "calmar":          calmar(equity_net),
        "up_capture":      up_cap,
        "down_capture":    down_cap,
        "annual_turnover": annual_turnover(weights),
        "avg_exposure":    avg_exposure(weights),
    }
