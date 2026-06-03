"""
src/indicators/base.py — 공통 신호 인터페이스

계약:
  - 입력  data : dict[str, pd.Series], 거래일 인덱스 정합 완료.
  - 출력  w_target : pd.Series, 값 ∈ [0, w_max], 거래일 인덱스.
  - 시점  : t일 종가 정보로 산출 → 백테스트 엔진이 t+1 체결. 룩어헤드 금지.
  - 주간 지표 입력은 data_loader.apply_weekly_lag() 적용 후 전달.
  - 신호별 파라미터는 cfg dict로 주입 (w_max, sigma_target 등).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseIndicator(ABC):
    """모든 신호 모듈이 구현해야 하는 공통 인터페이스."""

    @abstractmethod
    def signal(self, data: dict[str, pd.Series], cfg: dict) -> pd.Series:
        """
        Parameters
        ----------
        data : dict[str, pd.Series]
            거래일 인덱스로 정합된 시계열 (sp500tr, vix, baa10y 등).
        cfg : dict
            신호별 파라미터 (w_max, sigma_target, rebalance_band 등).

        Returns
        -------
        pd.Series
            w_target : 목표 주식노출 ∈ [0, w_max], 거래일 인덱스.
            t일 종가 기준 산출 — 백테스트 엔진이 t+1 체결.
        """
        ...


class ConstantWeightIndicator(BaseIndicator):
    """
    더미 신호: 항상 고정 비중 반환.
    엔드투엔드 파이프라인 검증(M2)과 단위 테스트에만 사용.
    """

    def __init__(self, weight: float = 1.0) -> None:
        self._w = weight

    def signal(self, data: dict[str, pd.Series], cfg: dict) -> pd.Series:
        ref = next(iter(data.values()))
        w = min(self._w, cfg.get("w_max", 1.0))
        return pd.Series(w, index=ref.index, name="w_dummy")
