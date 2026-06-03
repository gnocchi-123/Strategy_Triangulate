"""
src/backtest.py — 공통 백테스트 엔진

체결 순서 (매 거래일 t):
  1. gross_t = w_prev × r_equity_t + (1 − w_prev) × rf_t
               [레버리지 시: w_eff = w_prev × L, 초과분(w_eff−1)에 차입비용 추가 차감]
  2. w_drifted_t = 수익 반영 후 자연 표류 비중
  3. 밴드 검사: |w_target_t − w_drifted_t| > band?
       예 → 거래 실행, turnover_t = |w_target_t − w_drifted_t|
            cost_t = turnover_t × cost_bps / 10_000  (당일 수익에서 차감)
            w_next = w_target_t
       아니오 → cost_t = 0, w_next = w_drifted_t
  4. net_t = gross_t − cost_t
  5. 다음 날 적용 비중 = w_next

비중 정의: w_held[t] = t일 종가 직후 확정 비중 → t+1일에 적용.
rf 입력: 연율 % (DTB3 기준). 내부에서 일간화(÷100÷252).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def run(
    w_target: pd.Series,
    equity_returns: pd.Series,
    rf_annual_pct: pd.Series,
    cfg: dict,
    leverage: float = 1.0,
) -> dict[str, pd.Series]:
    """
    Parameters
    ----------
    w_target        : t일 신호 출력, ∈ [0, w_max].
    equity_returns  : sp500tr 일간수익률.
    rf_annual_pct   : DTB3 연율 % (예: 5.0 = 5%). 내부 일간화.
    cfg             : base.yaml 파라미터.
    leverage        : 레버리지 배율 (기본 1.0, Phase 3에서 1 초과).

    Returns
    -------
    dict with keys:
        equity_gross, equity_net  : 자산곡선 (기준값=1)
        returns_gross, returns_net: 일간수익률
        weights                   : w_held[t] — t+1에 적용되는 비중
        turnover                  : 일간 one-way 회전율
    """
    band = float(cfg.get("rebalance_band", 0.05))
    cost_bps = float(cfg.get("cost_bps", 2.0))
    cost_rate = cost_bps / 10_000.0
    w_max = float(cfg.get("w_max", 1.0))
    borrow_spread_bps = float(cfg.get("borrow_spread_bps", 50.0))
    # 차입비용: 연율 bp → 일간 소수
    borrow_daily = borrow_spread_bps / 10_000.0 / 252.0

    # 공통 인덱스로 정렬
    idx = w_target.dropna().index
    r_eq = equity_returns.reindex(idx).fillna(0.0).values
    # rf: 연율 % → 일간 소수
    rf_d = (rf_annual_pct.reindex(idx).fillna(0.0) / 100.0 / 252.0).values
    w_arr = w_target.reindex(idx).fillna(0.0).clip(0.0, w_max).values

    n = len(idx)
    gross_arr = np.empty(n)
    net_arr = np.empty(n)
    w_held_arr = np.empty(n)
    turn_arr = np.zeros(n)

    # 시작: 전액 현금 (w_prev=0). 첫날은 rf만 획득.
    w_prev = 0.0

    for t in range(n):
        tgt = min(w_arr[t], w_max)
        w_eff = w_prev * leverage  # 레버리지 적용 (기본=1, 변화 없음)

        # ── Step 1: 당일 gross return ──────────────────────────────────────
        gross_t = w_eff * r_eq[t] + (1.0 - w_eff) * rf_d[t]
        # 레버리지 시 차입비용 차감 (w_eff > 1인 부분)
        if w_eff > 1.0:
            gross_t -= (w_eff - 1.0) * (rf_d[t] + borrow_daily)

        # ── Step 2: 수익 반영 후 자연 표류 비중 ───────────────────────────
        V_eq = w_prev * (1.0 + r_eq[t])
        V_cash = (1.0 - w_prev) * (1.0 + rf_d[t])
        V_total = V_eq + V_cash
        w_drifted = V_eq / V_total if V_total > 0.0 else w_prev

        # ── Step 3: 밴드 검사 & 거래 결정 ────────────────────────────────
        delta = abs(tgt - w_drifted)
        if delta > band:
            cost_t = delta * cost_rate
            turn_arr[t] = delta
            w_next = tgt
        else:
            cost_t = 0.0
            w_next = w_drifted

        # ── Step 4: net return ────────────────────────────────────────────
        net_arr[t] = gross_t - cost_t
        gross_arr[t] = gross_t

        # ── Step 5: 다음날 적용 비중 확정 ────────────────────────────────
        w_held_arr[t] = w_next
        w_prev = w_next

    r_gross = pd.Series(gross_arr, index=idx, name="r_gross")
    r_net = pd.Series(net_arr, index=idx, name="r_net")
    eq_gross = (1.0 + r_gross).cumprod()
    eq_net = (1.0 + r_net).cumprod()
    eq_gross.name = "equity_gross"
    eq_net.name = "equity_net"

    return {
        "equity_gross":  eq_gross,
        "equity_net":    eq_net,
        "returns_gross": r_gross,
        "returns_net":   r_net,
        "weights":       pd.Series(w_held_arr, index=idx, name="weight"),
        "turnover":      pd.Series(turn_arr,   index=idx, name="turnover"),
    }
