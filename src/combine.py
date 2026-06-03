"""
src/combine.py — Phase 2 신호 결합 (스텁, M5에서 완성)

Gate 1(M4) 통과 신호만 대상.
결합 규칙: 등가중 평균 또는 투표. 신호 가중치 데이터 최적화 금지.
"""
from __future__ import annotations

import pandas as pd


def combine_equal_weight(signals: dict[str, pd.Series], cfg: dict) -> pd.Series:
    """등가중 평균: w_comb = mean_i(w^(i)). M5에서 구현."""
    raise NotImplementedError("M5에서 구현")


def combine_vote(
    signals: dict[str, pd.Series],
    cfg: dict,
    threshold: float = 0.5,
) -> pd.Series:
    """투표 기반 결합 (방어 신호 과반 시 노출 축소). M5에서 구현."""
    raise NotImplementedError("M5에서 구현")
