"""
src/benchmarks.py — 공통 대조군

1. buy_and_hold    : 100% 주식, 전액 주식 유지 (표류 없음, 비용 없음).
2. equal_exposure  : 전략의 실현 mean(w_t)를 고정비중으로 받아
                     동일 엔진·동일 비용 모델로 constant-mix 운용.
                     드리프트 교정 회전율과 비용도 전략과 동일하게 처리.

두 대조군 모두 backtest.run() 엔진을 재사용해 공정 비교를 보장한다.
"""
from __future__ import annotations

import pandas as pd

from src.backtest import run


def buy_and_hold(
    equity_returns: pd.Series,
    rf_annual_pct: pd.Series,
    cfg: dict,
) -> dict[str, pd.Series]:
    """
    100% 주식 buy&hold.
    w_target = 1.0 전 기간 → 표류 없음(all-equity엔 현금 없어 drift=0),
    단 시작일 현금→1.0 전환 비용 1회 발생 (다기간에서 무시 가능 수준).
    """
    idx = equity_returns.reindex(equity_returns.dropna().index).index
    w_target = pd.Series(1.0, index=idx, name="w_bnh")
    return run(w_target, equity_returns, rf_annual_pct, cfg)


def equal_exposure(
    mean_w: float,
    equity_returns: pd.Series,
    rf_annual_pct: pd.Series,
    cfg: dict,
) -> dict[str, pd.Series]:
    """
    동일노출 정적 배분 (constant-mix).

    Parameters
    ----------
    mean_w : 전략의 실현 평균 주식노출 avg_exposure(strategy_weights).
             엔진을 먼저 돌린 뒤 이 값을 사후 주입한다.

    전략과 동일한 rebalance_band·cost_bps로 드리프트 교정 회전율·비용까지
    동일하게 처리한다. 이 대조군을 위험조정 기준으로 이겨야
    신호의 알파(타이밍 가치)가 인정된다.
    """
    idx = equity_returns.reindex(equity_returns.dropna().index).index
    w_target = pd.Series(float(mean_w), index=idx, name="w_static")
    return run(w_target, equity_returns, rf_annual_pct, cfg)
