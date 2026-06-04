"""
src/indicators/trend.py — 추세 신호 (DEFINITIONS 1.3)

DEFINITIONS 1.3 이진 규칙 (연속 스케일 변형 없음):
  ma200:   w = w_max if P_t > MA200_t else w_floor
  tsmom12: w = w_max if (P_t / P_{t-252} − 1) > 0 else w_floor

두 규칙 모두 sp500tr 가격에서만 계산. 외부 데이터·NFCI류 일절 사용 안 함.
config['rule'] 로 분기. w_floor=0.0 고정 (변형 w_floor=0.5는 M4 robustness 전용).

─── 룩어헤드 규약 ──────────────────────────────────────────────────────────────
ma200  : MA200_t = P[t-199 : t+1].mean() — rolling(200) closed='right'.
         t 위치에서 P[t-199]~P[t] 까지 200개 가격의 단순이동평균.
         _assert_ma200_alignment() 이 causal 정렬 단언.

tsmom12: w_t = 1 if P_t / P[t-252] > 1 else 0.
         iloc 기반 후방 참조: idx_pos - 252 (정확히 252 거래일 전).
         첫 252개 거래일은 t-252 인덱스가 없으므로 NaN (워밍업).
         _assert_tsmom_alignment() 이 참조 날짜를 단언.

두 규칙 모두 t일 종가 정보로 산출 → t+1 체결 (엔진 담당).

─── 워밍업 규약 ────────────────────────────────────────────────────────────────
ma200  : 첫 199일 NaN (rolling(200) min_periods=200).
         eval_start ≈ 1988-10 (source_start 1988-01-01 기준 200거래일 후).
tsmom12: 첫 252일 NaN (iloc t-252 미존재).
         eval_start ≈ 1989-01 (source_start 1988-01-01 기준 252거래일 후).
두 규칙 eval_start가 다른 것은 같은 정보원 내 워밍업 차이 — 정상.
(volatility.py 에서 vix=당일 / realized=21일 차이와 동일 구조)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.base import BaseIndicator


class TrendIndicator(BaseIndicator):
    """DEFINITIONS 1.3 추세 신호 (ma200 | tsmom12)."""

    def signal(self, data: dict[str, pd.Series], cfg: dict) -> pd.Series:
        """
        Parameters
        ----------
        data : {"sp500tr": pd.Series, ...}   — 가격 시계열
        cfg  : base.yaml + trend.yaml 병합 dict.
               필수: rule, w_floor, w_max.

        Returns
        -------
        pd.Series  w_target ∈ {0, w_max}, 워밍업 구간 NaN.
        """
        rule    = cfg.get("rule", "ma200")
        w_floor = float(cfg.get("w_floor", 0.0))
        w_max   = float(cfg.get("w_max",   1.0))

        prices = data["sp500tr"].dropna()

        if rule == "ma200":
            w = _ma200_signal(prices, w_floor, w_max)
        elif rule == "tsmom12":
            w = _tsmom12_signal(prices, w_floor, w_max)
        else:
            raise ValueError(f"알 수 없는 trend rule: {rule!r}. 'ma200' | 'tsmom12'")

        w.name = "w_trend"
        return w


# ── 신호 계산 함수 ─────────────────────────────────────────────────────────────

def _ma200_signal(prices: pd.Series, w_floor: float, w_max: float) -> pd.Series:
    """
    MA200 이진 신호.

    MA200_t = simple mean of P[t-199 : t+1]  (rolling(200, closed='right')).
    t일 종가 기준 산출 → t+1 체결.

    w_t = w_max if P_t > MA200_t else w_floor.
    첫 199일은 rolling NaN → w_target NaN (워밍업).
    """
    ma200 = prices.rolling(200, min_periods=200).mean()
    _assert_ma200_alignment(prices, ma200)

    # 이진 매핑: P_t > MA200_t → w_max, else → w_floor
    above = prices > ma200
    w = above.map({True: w_max, False: w_floor}).astype(float)
    # 워밍업(첫 199일): ma200=NaN → above=False (rolling NaN은 비교 시 False)
    # 명시적으로 ma200 NaN 구간을 w=NaN으로 처리
    w[ma200.isna()] = np.nan
    return w


def _tsmom12_signal(prices: pd.Series, w_floor: float, w_max: float) -> pd.Series:
    """
    12개월 TSMOM 이진 신호.

    r12_t = P_t / P_{t-252} − 1  (정확히 252 거래일 전 가격 참조).
    w_t = w_max if r12_t > 0 else w_floor.
    첫 252 거래일은 t-252 인덱스 미존재 → w_target NaN (워밍업).

    iloc 기반 후방 참조로 t-252 를 정확히 지정 (off-by-one 방지).
    """
    n = len(prices)
    lag = 252

    w_vals = np.full(n, np.nan)
    for i in range(lag, n):
        r12 = prices.iloc[i] / prices.iloc[i - lag] - 1.0
        w_vals[i] = w_max if r12 > 0.0 else w_floor

    w = pd.Series(w_vals, index=prices.index)
    _assert_tsmom_alignment(prices, w)
    return w


# ── 내부 정렬 단언 ─────────────────────────────────────────────────────────────

def _assert_ma200_alignment(prices: pd.Series, ma200: pd.Series) -> None:
    """
    MA200_t 가 t일 이후 가격을 사용하지 않음을 단언.

    검증:
    (1) 인덱스 동일성: ma200.index == prices.index.
        rolling(200, closed='right') 은 입력과 동일 인덱스 반환 → t 위치에서
        prices[t-199:t+1] 사용. 인덱스가 같으면 미래값 미포함 보장.
    (2) 마지막 유효일 일치.
    """
    if prices.dropna().empty or ma200.dropna().empty:
        return

    if not prices.index.equals(ma200.index):
        raise ValueError(
            f"MA200 인덱스({len(ma200)})가 prices 인덱스({len(prices)})와 다름 — "
            "시프트 또는 길이 불일치: 룩어헤드 위험"
        )

    last_p  = prices.dropna().index[-1]
    last_ma = ma200.dropna().index[-1]
    if last_p != last_ma:
        raise ValueError(
            f"MA200 마지막 유효일({last_ma.date()}) ≠ "
            f"prices 마지막 유효일({last_p.date()}): 룩어헤드 위험"
        )


def _assert_tsmom_alignment(prices: pd.Series, w: pd.Series) -> None:
    """
    TSMOM w_t 산출 시 t-252 인덱스만 참조함을 단언.

    검증:
    (1) 인덱스 동일성: w.index == prices.index.
    (2) 워밍업 경계: w.iloc[:252] 가 모두 NaN, w.iloc[252] 가 non-NaN.
    (3) 마지막 유효일 일치.
    """
    if prices.dropna().empty or w.dropna().empty:
        return

    if not prices.index.equals(w.index):
        raise ValueError(
            f"TSMOM w 인덱스({len(w)})가 prices 인덱스({len(prices)})와 다름 — "
            "시프트 또는 길이 불일치: 룩어헤드 위험"
        )

    # 워밍업 경계: 첫 252개 → NaN, 252번째 → non-NaN
    if len(w) > 252:
        warmup = w.iloc[:252]
        if not warmup.isna().all():
            raise ValueError(
                f"TSMOM 워밍업 구간(첫 252일)에 non-NaN 값이 있음 — "
                "룩어헤드 위험 (t-252 미존재 구간을 계산에 사용)"
            )
        if np.isnan(w.iloc[252]):
            raise ValueError(
                "TSMOM w.iloc[252] 가 NaN — 252일 이후 신호가 생성되지 않음"
            )

    last_p = prices.dropna().index[-1]
    last_w = w.dropna().index[-1]
    if last_p != last_w:
        raise ValueError(
            f"TSMOM 마지막 유효일({last_w.date()}) ≠ "
            f"prices 마지막 유효일({last_p.date()}): 룩어헤드 위험"
        )
