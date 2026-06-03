"""
src/metrics.py — 성과 지표 (DEFINITIONS_AND_CONVENTIONS.md 4절)

모두 net 기준, 일간수익률 기반.
표기: r_t=전략 일간수익률, b_t=벤치마크 일간수익률,
      rf_t=무위험 일간수익률, N=거래일 수, 연율화=252.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_ANNUAL = 252


def cagr(equity: pd.Series) -> float:
    """
    (최종자산/초기자산)^(252/N) − 1, N = 수익 기간 수 = len(equity) − 1.
    equity는 cumprod() 형태: equity[0] = 1+r_0, equity[N] = 누적곱.
    두 관측 사이 기간 수 = N−1 이므로 (len−1)로 연율화.
    """
    n = len(equity)
    if n < 2:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (_ANNUAL / (n - 1)) - 1)


def annual_vol(returns: pd.Series) -> float:
    """std(r_t) × √252"""
    return float(returns.std() * np.sqrt(_ANNUAL))


def sharpe(returns: pd.Series, rf: pd.Series) -> float:
    """[mean(r_t − rf_t) × 252] / [std(r_t) × √252]"""
    excess = returns - rf.reindex(returns.index).fillna(0.0)
    vol = annual_vol(returns)
    if vol == 0.0:
        return float("nan")
    return float(excess.mean() * _ANNUAL / vol)


def sortino(returns: pd.Series, rf: pd.Series, mar: float = 0.0) -> float:
    """
    [mean(r_t − rf_t) × 252] / [DD × √252]
    DD = sqrt(mean(min(r_t − MAR, 0)^2)), MAR=0 기본.
    """
    excess = returns - rf.reindex(returns.index).fillna(0.0)
    downside = np.minimum(returns.values - mar, 0.0)
    dd = float(np.sqrt(np.mean(downside ** 2)))
    if dd == 0.0:
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
    if mdd == 0.0:
        return float("nan")
    return float(cagr(equity) / abs(mdd))


def capture_ratios(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> tuple[float, float]:
    """
    상/하방 캡처 (월간 복리수익 기반, DEFINITIONS 4절).
    반환: (up_capture, down_capture)
    """
    strat_m = (1.0 + strategy_returns).resample("ME").prod() - 1.0
    bench_m = (1.0 + benchmark_returns).resample("ME").prod() - 1.0

    common = strat_m.index.intersection(bench_m.index)
    strat_m = strat_m.loc[common]
    bench_m = bench_m.loc[common]

    def _annualized_ratio(s_vals: pd.Series, b_vals: pd.Series) -> float:
        n = len(b_vals)
        if n == 0:
            return float("nan")
        s_ann = float((1.0 + s_vals).prod() ** (12.0 / n) - 1.0)
        b_ann = float((1.0 + b_vals).prod() ** (12.0 / n) - 1.0)
        if b_ann == 0.0:
            return float("nan")
        return s_ann / b_ann

    up_mask = bench_m > 0.0
    down_mask = bench_m < 0.0
    up_cap = _annualized_ratio(strat_m[up_mask], bench_m[up_mask])
    down_cap = _annualized_ratio(strat_m[down_mask], bench_m[down_mask])
    return up_cap, down_cap


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
