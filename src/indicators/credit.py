"""
src/indicators/credit.py — 신용 신호 (DEFINITIONS 1.2·1.4)

대표 신호: BAA10Y (Moody's Baa − 10Y Treasury, 일간, 1986~)
신호 흐름:
  s_t  = baa10y[t]          (스트레스 지표, 높을수록 위험)
  p_t  = trailing W일 백분위  (0~1, 현재값이 과거 대비 얼마나 높은가)
  w_t  = 단조 감소 매핑       (p_t → w ∈ [0, w_max])

─── 룩어헤드 규약 ──────────────────────────────────────────────────────────────
BAA10Y는 일간 공표 시리즈(발표 시차 없음).
baa10y[t] = t일 종가 확정값 → t+1 체결 안전.

percentile: trailing rolling(W) → [t-W+1, t] 구간만 사용 (causal).
pandas rolling(W).apply() 는 기본 closed='right' 이므로 t 위치에서
baa10y[t-W+1 : t+1] 이 포함됨 — 미래값 미포함 ✓.
_assert_percentile_alignment()이 causal 정렬을 단언.

NFCI·STLFSI4·HY OAS 는 M4 robustness 전용. 이 파일에서 사용 금지.

─── 워밍업 규약 ────────────────────────────────────────────────────────────────
rolling(W) → 첫 W−1일은 NaN. w_target = NaN으로 반환.
eval_start = w_target.dropna().index[0] (신호 첫 유효일 ≈ 1987-01).
volatility.py와 동일 규약.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.base import BaseIndicator


class CreditIndicator(BaseIndicator):
    """DEFINITIONS 1.2·1.4 신용 신호 (BAA10Y 대표)."""

    def signal(self, data: dict[str, pd.Series], cfg: dict) -> pd.Series:
        """
        Parameters
        ----------
        data : {"baa10y": pd.Series, ...}
        cfg  : base.yaml + credit.yaml 병합 dict.
               필수: percentile_window, theta_low_pct, theta_high_pct, w_max.

        Returns
        -------
        pd.Series  w_target ∈ [0, w_max], 워밍업(첫 W-1일) NaN.
        """
        baa10y = data["baa10y"].dropna()

        W          = int(cfg.get("percentile_window", 252))
        theta_low  = float(cfg.get("theta_low_pct",  0.5))
        theta_high = float(cfg.get("theta_high_pct", 0.9))
        w_max      = float(cfg.get("w_max", 1.0))

        p_t = _percentile_rank(baa10y, W)
        _assert_percentile_alignment(baa10y, p_t)

        w = _monotone_map(p_t, theta_low, theta_high, w_max)
        w.name = "w_credit"
        return w


# ── 내부 함수 ─────────────────────────────────────────────────────────────────

def _percentile_rank(s: pd.Series, window: int) -> pd.Series:
    """
    trailing W일 백분위 (0~1).

    t 위치: baa10y[t-W+1 : t+1] 구간에서 s[t]가 차지하는 순위.
    = (해당 윈도우에서 s[t] 이하인 값의 수) / W.
    rolling(W, closed='right') 기본 → [t-W+1, t] causal 구간 ✓.
    첫 W-1일은 NaN (워밍업).
    """
    def _rank_last(arr: np.ndarray) -> float:
        return float((arr <= arr[-1]).sum()) / len(arr)

    return s.rolling(window, min_periods=window).apply(_rank_last, raw=True)


def _monotone_map(
    p: pd.Series,
    theta_low: float,
    theta_high: float,
    w_max: float,
) -> pd.Series:
    """
    DEFINITIONS 1.4 단조 감소 매핑 (임계 방식):
      p < theta_low  → w_max  (정상 여건, 풀 노출)
      p > theta_high → 0      (극단 스트레스, 완전 방어)
      사이           → 선형 보간: w = w_max × (theta_high − p) / (theta_high − theta_low)
    """
    span = theta_high - theta_low
    w_vals = np.where(
        p.isna(), np.nan,
        np.where(
            p < theta_low, w_max,
            np.where(
                p > theta_high, 0.0,
                w_max * (theta_high - p.values) / span,
            ),
        ),
    )
    return pd.Series(w_vals, index=p.index)


def _assert_percentile_alignment(baa10y: pd.Series, p: pd.Series) -> None:
    """
    percentile p_t 가 t일 baa10y 이후를 사용하지 않음을 단언.

    검증 항목:
    (1) 인덱스 동일성: p.index == baa10y.index
        rolling(W).apply()는 입력과 같은 인덱스를 반환하므로,
        인덱스가 같으면 t 위치의 p_t가 baa10y[t]까지만 참조함이 보장된다.
    (2) 마지막 유효일 일치: last valid p == last valid baa10y.

    인덱스가 다르거나 시프트됐으면 룩어헤드 위험 → ValueError.
    """
    if baa10y.dropna().empty or p.dropna().empty:
        return

    # (1) 인덱스 동일성
    if not baa10y.index.equals(p.index):
        raise ValueError(
            f"percentile p 인덱스({len(p)})가 baa10y 인덱스({len(baa10y)})와 다름 — "
            "시프트 또는 길이 불일치: 룩어헤드 위험"
        )

    # (2) 유효값 끝날 일치
    last_baa = baa10y.dropna().index[-1]
    last_p   = p.dropna().index[-1]
    if last_baa != last_p:
        raise ValueError(
            f"percentile 마지막 유효일({last_p.date()}) ≠ "
            f"baa10y 마지막 유효일({last_baa.date()}): 룩어헤드 위험"
        )
