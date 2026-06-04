"""
src/combine.py — Phase 2 신호 결합 (DEFINITIONS 2절)

Gate 1(M4) 통과 신호만 대상: 변동성(vix) + 추세(ma200·tsmom 평균).
신호 가중치 데이터 최적화 절대 금지.

NaN 정책:
  결합 입력은 공통기간(1990-01-31~)으로 자른 뒤 전달한다.
  공통기간 내 NaN은 정합 오류이므로 assert로 중단 — 조용한 현금처리 금지.
"""
from __future__ import annotations

import pandas as pd


def combine_equal_weight(signals: dict[str, pd.Series], cfg: dict) -> pd.Series:
    """
    등가중 평균: w_comb = mean_i( w^(i) )

    입력 시리즈는 반드시 NaN-free여야 한다. NaN이 있으면 AssertionError로 중단.
    인덱스는 inner join으로 정렬 — 길이 불일치 시 자동 정합.

    Parameters
    ----------
    signals : {"vol": w_vol_series, "trend": w_trend_series, ...}
        Gate 1 통과 신호. 공통기간 자른 뒤 전달.
    cfg : 미사용 (인터페이스 일관성 유지)
    """
    if not signals:
        raise ValueError("signals가 비어 있음")

    df = pd.concat(signals, axis=1).sort_index()

    # 공통기간 자른 이후에도 NaN이 있으면 정합 오류
    for name, col in df.items():
        nan_count = col.isna().sum()
        assert nan_count == 0, (
            f"결합 입력 '{name}'에 NaN {nan_count}개 — "
            "공통기간 자르기 전 신호를 전달했거나 신호 계산 오류. "
            "COMMON_START 이후로 자른 뒤 전달할 것."
        )

    w_comb = df.mean(axis=1)
    w_comb.name = "w_comb"
    return w_comb


def combine_vote(
    signals: dict[str, pd.Series],
    cfg: dict,
    threshold: float = 0.5,
) -> pd.Series:
    """
    투표 기반 결합: 방어 신호 비율이 threshold 초과 시 노출 0, 미만 시 w_max.

    "방어 신호" = w^(i) < w_max × 0.5 (중간값 미만).
    combine_equal_weight의 보조 확인용 — Gate 2 주 판정은 등가중 평균으로 한다.

    입력 NaN-free 조건은 combine_equal_weight와 동일.
    """
    if not signals:
        raise ValueError("signals가 비어 있음")

    w_max = float(cfg.get("w_max", 1.0))
    df = pd.concat(signals, axis=1).sort_index()

    for name, col in df.items():
        nan_count = col.isna().sum()
        assert nan_count == 0, (
            f"결합 입력 '{name}'에 NaN {nan_count}개 — 공통기간 자른 뒤 전달할 것."
        )

    midpoint = w_max * 0.5
    defensive_ratio = (df < midpoint).mean(axis=1)
    w_vote = defensive_ratio.apply(lambda r: 0.0 if r > threshold else w_max)
    w_vote.name = "w_vote"
    return w_vote
