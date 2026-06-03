"""
src/indicators/base.py — 공통 신호 인터페이스

계약:
  - 입력  data : dict[str, pd.Series], 거래일 인덱스 정합 완료.
  - 출력  w_target : pd.Series, 값 ∈ [0, w_max], 거래일 인덱스.
  - 시점  : t일 종가 정보로 산출 → 백테스트 엔진이 t+1 체결. 룩어헤드 금지.
  - 주간 지표 입력은 data_loader.apply_weekly_lag() 적용 후 전달.
  - 신호별 파라미터는 cfg dict로 주입 (w_max, sigma_target 등).

─── 단독 평가(M3) eval_start 규약 [규약 A] ───────────────────────────────────
CLAUDE.md: "단독=각자 최대 기간" — 이는 정보원 간 비대칭을 허용하는 것이지,
한 정보원 내 추정기 변형끼리 시작일을 다르게 하라는 뜻이 아님.

규약:
  1. 각 정보원에는 등록 시작일(source_start)이 있다:
       - 변동성: "1990-01-01"  (VIX 신방식 1990~, VXO 금지)
       - 신용:   "1986-01-01"  (BAA10Y 1986~)
       - 추세:   "1988-01-01"  (sp500tr 가용일 기준)
  2. 동일 정보원의 모든 추정기 변형은 source_start로 데이터를 자른 뒤
     각자 워밍업(rolling NaN)을 적용해 eval_start를 결정한다.
     → vix·realized·blend 모두 1990-01-01 입력 기준,
       eval_start는 vix=1990-01-02, realized=1990-01-31, blend=1990-01-31.
  3. period.start(base.yaml) 슬라이싱은 비교·결합(M4~, 공통 기간)에서만 사용.

헬퍼 standalone_data() 가 이 규약을 강제한다.
M3-B·C도 이 헬퍼에 해당 source_start를 넘겨 변형끼리 자동 통일.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


# ── 정보원별 등록 시작일 (규약 A) ─────────────────────────────────────────────
SOURCE_STARTS: dict[str, str] = {
    "volatility": "1990-01-01",   # VIX 신방식 1990~, VXO 금지
    "credit":     "1986-01-01",   # BAA10Y(Moody's Baa−10Y) 1986~
    "trend":      "1988-01-01",   # sp500tr 가용일 기준
}


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


def standalone_data(
    data_raw: dict[str, pd.Series],
    source: str,
) -> dict[str, pd.Series]:
    """
    단독 평가용 데이터 준비 — 규약 A.

    SOURCE_STARTS[source] 로 data_raw 를 슬라이싱한다.
    동일 정보원의 모든 추정기 변형이 이 함수를 통해 동일한 입력을 받으므로,
    변형별 시작일이 불일치하는 것을 방지한다.

    Parameters
    ----------
    data_raw : load_all() 반환값 (period.start 슬라이싱 없는 원시 데이터).
    source   : "volatility" | "credit" | "trend"

    Returns
    -------
    dict[str, pd.Series]  source_start 이후 구간만 남긴 데이터.
    """
    if source not in SOURCE_STARTS:
        raise ValueError(f"알 수 없는 source: {source!r}. 등록된 값: {list(SOURCE_STARTS)}")
    start = pd.Timestamp(SOURCE_STARTS[source])
    return {k: v.loc[start:] for k, v in data_raw.items()}


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
