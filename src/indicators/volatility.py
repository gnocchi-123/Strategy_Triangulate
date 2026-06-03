"""
src/indicators/volatility.py — 변동성 신호 (DEFINITIONS 1.1)

σ_t 추정기: vix | realized | blend  (config)
목표비중:   w = clip(σ_target / σ_t, 0, w_max)

─── 룩어헤드 규약 ──────────────────────────────────────────────────────────────
vix    : σ_t = VIX_t / 100. VIX_t = t일 종가 확정 → t+1 체결 안전.
realized: sp500tr.pct_change() rolling(N).std() × √252.
          sp_ret[t] = (P_t / P_{t-1}) − 1 (t일 종가 기준).
          rolling(N) at t → sp_ret[t-N+1 : t] 포함 (causal). ✓
          _assert_realized_alignment()이 인덱스 정렬 어긋남을 감지.
blend  : vix σ + realized σ 가중 평균. 두 추정기 모두 위 규약 준수.

─── 워밍업 규약 ────────────────────────────────────────────────────────────────
NaN 구간(워밍업)은 w_target = NaN으로 반환.
backtest 엔진(backtest.run)이 fillna(0)으로 현금 유지 처리.
평가 시작일 = signal.dropna().index[0] (신호 첫 유효일).
노트북에서 해당 날부터 백테스트 입력을 트림한다.
이 규약은 credit·trend 신호에도 동일 적용.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.base import BaseIndicator


class VolatilityIndicator(BaseIndicator):
    """DEFINITIONS 1.1 변동성 신호."""

    def signal(self, data: dict[str, pd.Series], cfg: dict) -> pd.Series:
        """
        Parameters
        ----------
        data : {"vix": pd.Series, "sp500tr": pd.Series, ...}
        cfg  : base.yaml + volatility.yaml 병합 dict.
               필수: sigma_target, w_max, vol_estimator.

        Returns
        -------
        pd.Series  w_target ∈ [0, w_max], 워밍업 구간 NaN.
        """
        vol_est = cfg.get("vol_estimator", "vix")
        sigma_target = float(cfg.get("sigma_target", 0.12))
        w_max = float(cfg.get("w_max", 1.0))

        if vol_est == "vix":
            sigma_t = _sigma_vix(data)
        elif vol_est == "realized":
            sigma_t = _sigma_realized(data, cfg)
        elif vol_est == "blend":
            sigma_t = _sigma_blend(data, cfg)
        else:
            raise ValueError(f"알 수 없는 vol_estimator: {vol_est!r}")

        # σ=0 방지 (0/0 → NaN 유지)
        sigma_t = sigma_t.where(sigma_t > 0)

        w = (sigma_target / sigma_t).clip(0.0, w_max)
        w.name = "w_vol"
        return w


# ── σ_t 추정기 ────────────────────────────────────────────────────────────────

def _sigma_vix(data: dict[str, pd.Series]) -> pd.Series:
    """VIX_t / 100. VIX는 t일 종가 기준 → 룩어헤드 없음."""
    return data["vix"] / 100.0


def _sigma_realized(data: dict[str, pd.Series], cfg: dict) -> pd.Series:
    """
    N일 rolling std × √252.
    sp_ret[t] = t일 종가 기준 수익률. rolling은 [t-N+1, t] causal 구간.
    """
    lookback = int(cfg.get("realized_lookback", 21))
    sp_ret = data["sp500tr"].pct_change()           # r_t: t일 종가 기준 ✓
    sigma = sp_ret.rolling(lookback).std() * np.sqrt(252)
    _assert_realized_alignment(sp_ret, sigma)       # 인덱스 정렬 단언
    return sigma


def _sigma_blend(data: dict[str, pd.Series], cfg: dict) -> pd.Series:
    """VIX + realized 가중 평균."""
    bw = cfg.get("blend_weights", {"vix": 0.5, "realized": 0.5})
    w_vix  = float(bw.get("vix",      0.5))
    w_real = float(bw.get("realized", 0.5))
    sig_v = _sigma_vix(data)
    sig_r = _sigma_realized(data, cfg)
    return w_vix * sig_v + w_real * sig_r


def _assert_realized_alignment(sp_ret: pd.Series, sigma: pd.Series) -> None:
    """
    realized σ_t이 t일 수익률(=t일 종가 기준)까지만 사용함을 단언.

    두 가지를 검증:
    (1) 인덱스 동일성: σ 인덱스 == sp_ret 인덱스 (시프트 미적용 보장).
        rolling(N).std()는 pandas 기본값이 closed='right'(우폐구간)이므로
        t 위치의 σ_t에 sp_ret[t]까지만 들어감. 인덱스가 같으면 σ_t의
        마지막 수익률이 t일 수익률을 절대 넘지 않는다.
    (2) 유효값 끝날 일치: last valid σ == last valid sp_ret.
    """
    if sp_ret.dropna().empty or sigma.dropna().empty:
        return

    # (1) 인덱스 동일성 — 시프트/길이 불일치 모두 감지
    if not sp_ret.index.equals(sigma.index):
        raise ValueError(
            f"realized σ 인덱스({len(sigma)})가 sp_ret 인덱스({len(sp_ret)})와 다름 — "
            "σ에 인위적 시프트 또는 길이 불일치: 룩어헤드 위험"
        )

    # (2) 유효값 끝날 일치
    last_ret = sp_ret.dropna().index[-1]
    last_sig = sigma.dropna().index[-1]
    if last_ret != last_sig:
        raise ValueError(
            f"realized σ 마지막 유효일({last_sig.date()}) ≠ "
            f"수익률 마지막 유효일({last_ret.date()}): "
            "인덱스 어긋남 — 룩어헤드 위험"
        )
