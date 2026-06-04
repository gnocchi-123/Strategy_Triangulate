# Code Reference — src/ 모듈 인덱스 + 핵심 모듈 전문

> **이 문서는 코드가 바뀌면 갱신 대상이다.** 코드를 수정할 때마다 해당 섹션을 함께 업데이트하라.

| 항목 | 내용 |
|---|---|
| 기준 커밋 | `55a19e2` |
| 날짜 | 2026-06-04 |
| 브랜치 | main |

---

## [1] 모듈 인덱스

### `src/data_loader.py`

| 함수/상수 | 시그니처 | 설명 | 입력 → 출력 |
|---|---|---|---|
| `CACHE_DIR` | `Path` | parquet 캐시 저장 경로 (`data/`) | — |
| `PRICES_CACHE` | `Path` | 시세 캐시 (`data/raw_prices.parquet`) | — |
| `FRED_CACHE` | `Path` | FRED 캐시 (`data/raw_fred.parquet`) | — |
| `_YFINANCE_TICKERS` | `dict[str, str]` | `sp500tr=^SP500TR, vix=^VIX, vix3m=^VIX3M` | — |
| `_FRED_IDS` | `dict[str, str]` | `rf=DTB3, baa10y=BAA10Y, hy_oas=BAMLH0A0HYM2, nfci=NFCI, stlfsi=STLFSI4` | — |
| `_extract_close` | `(raw: DataFrame, ticker: str) -> Series` | MultiIndex 포함 yfinance 반환값에서 Close 안전 추출 | DataFrame → Series |
| `load_prices` | `(cfg, force_refresh=False) -> DataFrame` | yfinance에서 sp500tr·vix·vix3m 수집. ffill 금지. 캐시 우선. | cfg → `DataFrame[sp500tr, vix, vix3m]` |
| `load_fred` | `(cfg, force_refresh=False) -> DataFrame` | FRED에서 rf·baa10y·hy_oas·nfci·stlfsi raw 수집. 캐시 우선. | cfg → `DataFrame[rf, baa10y, hy_oas, nfci, stlfsi]` |
| `apply_weekly_lag` | `(s, lag_bdays, trading_index) -> Series` | 주간 시리즈 발표 시차 시프트 → 거래일 reindex → ffill. BDay over-shift 설계. | Series → Series (거래일 정합) |
| `validate` | `(series_dict) -> dict` | 시리즈별 시작·종료·결측·중복·이상치 검증. baa10y 음수는 오류 중단 아닌 리포트. | `dict[str, Series]` → 검증 report dict |
| `load_all` | `(cfg, force_refresh=False) -> dict[str, Series]` | 전체 파이프라인. inner join 없음. 각자 최대 기간 보존. 주간 시차 적용 후 반환. | cfg → `dict[str, Series]` |

---

### `src/backtest.py`

| 함수 | 시그니처 | 설명 | 입력 → 출력 |
|---|---|---|---|
| `run` | `(w_target, equity_returns, rf_annual_pct, cfg, leverage=1.0) -> dict` | 단일 자산(주식/현금) 백테스트 엔진. t+1 체결, 밴드 리밸런스, 거래비용, 레버리지 차입비용. | Series×3, cfg → `dict[str, Series]` |

**`run()` 반환 키**: `equity_gross`, `equity_net`, `returns_gross`, `returns_net`, `weights`, `turnover`

**체결 순서 (매 거래일 t)**:
1. gross_t = w_prev(레버리지 적용) × r_eq[t] + (1−w_eff) × rf_d[t]
2. w_drifted = V_eq / (V_eq + V_cash) (수익 반영 표류)
3. |w_target − w_drifted| > band → 거래, cost = Δw × cost_bps/10000
4. net_t = gross_t − cost_t
5. w_held[t] = w_next (t+1 적용 비중)

**파라미터 (cfg)**:

| 키 | 기본값 | 설명 |
|---|---|---|
| `rebalance_band` | 0.05 | 불감대 (5%) |
| `cost_bps` | 2.0 | 편도 거래비용 bp |
| `borrow_spread_bps` | 50 | 레버리지 차입 스프레드 bp |
| `w_max` | 1.0 | 최대 주식노출 |

---

### `src/benchmarks.py`

| 함수 | 시그니처 | 설명 | 입력 → 출력 |
|---|---|---|---|
| `buy_and_hold` | `(equity_returns, rf_annual_pct, cfg) -> dict` | 100% 주식 buy&hold. w_target=1.0 전 기간. `backtest.run` 재사용. | Series×2, cfg → dict |
| `equal_exposure` | `(mean_w, equity_returns, rf_annual_pct, cfg) -> dict` | 전략 실현 평균 노출을 고정비중으로 상수 배분. **cfg의 rebalance_band를 무시하고 band=0 고정.** | float, Series×2, cfg → dict |

> `equal_exposure`의 `mean_w`는 반드시 `avg_exposure(result["weights"])` (실현 비중 평균)을 사용. `w_target.mean()` 금지.
>
> **band=0 고정 이유**: band > 0이면 강세장 drift로 realized mean > target 편향 발생(예: +0.015). "동일 평균노출" 비교 취지에 어긋나므로 band=0으로 매일 즉시 교정. 드리프트 교정 비용 ~0.86bp/yr은 net CAGR에 반영됨.

---

### `src/metrics.py`

| 함수 | 시그니처 | 설명 | 수식 |
|---|---|---|---|
| `cagr` | `(equity) -> float` | 연복리 수익률 | `equity[-1]^(252/N) − 1` |
| `annual_vol` | `(returns) -> float` | 연율 변동성 | `std(r) × √252` |
| `sharpe` | `(returns, rf) -> float` | 샤프 비율. vol < ε → nan. | `mean(r−rf)×252 / std(r)×√252` |
| `sortino` | `(returns, rf, mar=0.0) -> float` | 소르티노. DD < ε → nan. | `mean(r−rf)×252 / DD×√252` |
| `max_drawdown` | `(equity) -> float` | 최대 낙폭 (음수) | `min(equity/cummax − 1)` |
| `calmar` | `(equity) -> float` | 칼마 비율 | `CAGR / |MDD|` |
| `capture_ratios` | `(strategy_returns, benchmark_returns) -> tuple[float, float]` | 상/하방 캡처. Morningstar 표준 연율화. | `ann = (∏(1+r_m))^(12/k) − 1` |
| `annual_turnover` | `(turnover_arr) -> float` | 연 회전율. **`result["turnover"]`(turn_arr) 입력.** w_held.diff 기준 폐기. | `Σturn_arr / 연 수` |
| `avg_exposure` | `(weights) -> float` | 평균 주식노출 | `mean(w_t)` |
| `summary` | `(equity_net, returns_net, benchmark_returns, rf, weights, turnover_arr) -> dict` | 위 전 지표를 dict로 반환. **`turnover_arr` 필수** — None이면 ValueError. | — |

> **`annual_turnover` 변경 이유**: w_held.diff()는 band=0 equal_exposure에서 w_held=상수라 0을 반환해 실제 체결(0.43/yr)을 숨기는 표시 오류가 있었음. turn_arr를 직접 사용해 전략·ee가 동일 기준으로 집계됨.

---

### `src/combine.py`

| 함수 | 시그니처 | 설명 |
|---|---|---|
| `combine_equal_weight` | `(signals: dict[str, Series], cfg) -> Series` | 등가중 평균 `w_comb = mean_i(w^(i))`. **스텁, M5에서 구현.** |
| `combine_vote` | `(signals, cfg, threshold=0.5) -> Series` | 투표 기반 결합 (방어 신호 과반 시 노출 축소). **스텁, M5에서 구현.** |

---

### `notebooks/03_independence.ipynb` (M4 — 독립성 분석 · Gate 1)

**공통기간:** 1990-01-31~ (N=9148 거래일, realized·blend 워밍업 바닥)

| 섹션 | 내용 |
|---|---|
| §1 공통기간 재백테스트 | 6개 신호 공통기간 성과 재산출 → `results/common1990_metrics.csv` |
| §2 신호 상관 (i) | `w_target` 시리즈 간 Pearson 상관 → `results/independence_signal_corr.csv` |
| §3 전략 수익 상관 | 총수익 상관 행렬 (참고) → `results/independence_return_corr.csv` |
| §4 능동수익 정의 | 전략−B&H(구) vs **전략−ee(A, 기준)** — 공통 주식베타 제거 목적 |
| §5 신용 교차검증 | credit 5쌍 (i)·(A) 병기표 + 영역 분류 |
| §6 Gate 1 최종 판정 | 기준1(Sharpe/Sortino/Calmar vs ee, 3개 중 2개 이상) 적용 |
| §7 히트맵 저장 | (i)·(B)·(A) 3패널 → `results/independence_corr_heatmap.png` |
| §8 Gate 1 해석 노트 | 신용 판정 (i)·(A) 병기 근거 + M5 재검토 단서 기록 |

**산출물 파일:**

| 파일 | 설명 |
|---|---|
| `results/common1990_metrics.csv` | 공통기간 전 신호 성과표 |
| `results/independence_signal_corr.csv` | (i) 신호 w_t 상관 행렬 |
| `results/independence_active_corr.csv` | (A) 전략-ee 수익 상관 행렬 |
| `results/independence_active_bh_corr.csv` | (B) 전략-B&H 수익 상관 (참고용) |
| `results/independence_return_corr.csv` | 전략 총수익 상관 행렬 |
| `results/independence_corr_heatmap.png` | (i)·(B)·(A) 3패널 히트맵 |

**Gate 1 사전 등록 임계 (코드 Cell 2, 불변):**
```python
CORR_LOW  = 0.30   # |ρ| < 0.30  → 저상관
CORR_HIGH = 0.50   # |ρ| > 0.50  → 고상관
# 0.30 ≤ |ρ| ≤ 0.50 → 회색지대: (i) 신호상관 병기 판단
```

**Gate 1 포함/제외 명단:**
- 변동성 INCLUDE (3변형 모두 기준1 통과, 대표 변형은 M5 이월)
- 추세 INCLUDE (ma200·tsmom12 기준1 통과)
- 신용 EXCLUDE (기준1 실패 + 변동성 (A) 고상관 + (i)·(A) 병기 종합)

**M5 재검토 단서 (기록 전용):** credit↔tsmom12 (i)=0.288(저상관) — 볼+추세 결합 분산이득이 불충분 시 신용을 추세 보완 원소로 조건부 재검토 가능.

---

### `src/indicators/base.py`

| 항목 | 시그니처 | 설명 |
|---|---|---|
| `SOURCE_STARTS` | `dict[str, str]` | 정보원별 등록 시작일: `volatility=1990-01-01, credit=1986-01-01, trend=1988-01-01` |
| `BaseIndicator` | `ABC` | 모든 신호 모듈의 공통 인터페이스 |
| `BaseIndicator.signal` | `(data, cfg) -> Series` | 추상 메서드. t일 종가 기준 `w_target ∈ [0, w_max]` 반환 |
| `standalone_data` | `(data_raw, source: str) -> dict` | 규약 A 헬퍼: `SOURCE_STARTS[source]`로 슬라이싱. 동일 정보원 변형끼리 동일 입력 보장 |
| `ConstantWeightIndicator` | `class(BaseIndicator)` | 항상 고정 비중 반환하는 더미 신호 (테스트·M2 엔드투엔드 전용) |

---

### `src/indicators/credit.py`

| 항목 | 시그니처 | 설명 |
|---|---|---|
| `CreditIndicator` | `class(BaseIndicator)` | 신용 신호 (DEFINITIONS 1.2·1.4, BAA10Y 대표) |
| `CreditIndicator.signal` | `(data, cfg) -> Series` | baa10y → trailing 백분위 → 단조 감소 매핑. 워밍업(W-1일) NaN 반환. |
| `_percentile_rank` | `(s, window) -> Series` | trailing W일 causal 백분위. `(arr <= arr[-1]).sum() / len(arr)`. 첫 W-1일 NaN. |
| `_monotone_map` | `(p, theta_low, theta_high, w_max) -> Series` | p<θ_low→w_max, p>θ_high→0, 사이 선형 보간 (DEFINITIONS 1.4 임계 방식). |
| `_assert_percentile_alignment` | `(baa10y, p) -> None` | p 인덱스 정합 단언. 시프트 감지 시 ValueError. `_assert_realized_alignment`와 동급. |

**cfg 파라미터**: `percentile_window=252`, `theta_low_pct=0.5`, `theta_high_pct=0.9`, `w_max=1.0`
eval_start: `standalone_data(data_raw, "credit")` → 1986-01-01 입력 → sp500tr inner join으로 1988-01-04부터 시작 → rolling(252) 워밍업 후 **1988-12-29**

---

### `src/indicators/trend.py`

| 항목 | 시그니처 | 설명 |
|---|---|---|
| `TrendIndicator` | `class(BaseIndicator)` | 추세 신호 (DEFINITIONS 1.3, ma200 \| tsmom12) |
| `TrendIndicator.signal` | `(data, cfg) -> Series` | cfg `rule`로 ma200/tsmom12 분기. 워밍업 NaN 반환. sp500tr 가격만 사용. |
| `_ma200_signal` | `(prices, w_floor, w_max) -> Series` | rolling(200).mean() = MA200_t. `P_t > MA200_t` → w_max, else → w_floor. 워밍업 199일 NaN. |
| `_tsmom12_signal` | `(prices, w_floor, w_max) -> Series` | iloc 기반 `P[i]/P[i-252]-1`. r12>0 → w_max, else → w_floor. 워밍업 252일 NaN. |
| `_assert_ma200_alignment` | `(prices, ma200) -> None` | MA200 인덱스 정합 단언. 시프트·길이 불일치 → ValueError("룩어헤드"). |
| `_assert_tsmom_alignment` | `(prices, w) -> None` | TSMOM 워밍업 경계(첫 252일 NaN, iloc[252] non-NaN) 단언. 위반 → ValueError("룩어헤드"). |

**`TrendIndicator.signal` cfg 파라미터**:

| 키 | 값 | 설명 |
|---|---|---|
| `rule` | `"ma200"` \| `"tsmom12"` | 신호 규칙 선택 |
| `w_floor` | 0.0 | 방어 시 최소 노출 (변형 0.5는 M4 전용) |
| `w_max` | 1.0 | 최대 주식노출 |

eval_start: `standalone_data(data_raw, "trend")` → 1988-01-01 입력 (sp500tr inner join → 1988-01-04)
- ma200: rolling(200) 워밍업 후 **1988-10-14**
- tsmom12: iloc lag=252 워밍업 후 **1988-12-30**

> w_floor=0.5 변형·연속 스케일 변형은 추가하지 않음 (M4 robustness 전용).
> 외부 데이터 없음 — sp500tr 가격에서만 계산.

---

### `src/indicators/volatility.py`

| 항목 | 시그니처 | 설명 |
|---|---|---|
| `VolatilityIndicator` | `class(BaseIndicator)` | 변동성 신호 (DEFINITIONS 1.1) |
| `VolatilityIndicator.signal` | `(data, cfg) -> Series` | `vol_estimator` cfg로 vix/realized/blend 선택. 워밍업 NaN 반환. |
| `_sigma_vix` | `(data) -> Series` | `VIX_t / 100`. t일 종가 기준, 룩어헤드 없음. |
| `_sigma_realized` | `(data, cfg) -> Series` | rolling(N).std() × √252. `_assert_realized_alignment` 단언 호출. |
| `_sigma_blend` | `(data, cfg) -> Series` | `w_vix × σ_vix + w_real × σ_realized`. blend_weights cfg로 가중치 조정. |
| `_assert_realized_alignment` | `(sp_ret, sigma) -> None` | realized σ 인덱스 정합 단언. 시프트 감지 시 ValueError("룩어헤드"). |

**`VolatilityIndicator.signal` cfg 파라미터**:

| 키 | 값 | 설명 |
|---|---|---|
| `vol_estimator` | `"vix"` \| `"realized"` \| `"blend"` | 추정기 선택 |
| `sigma_target` | 0.12 | 연율 목표변동성 |
| `w_max` | 1.0 | 최대 주식노출 |
| `realized_lookback` | 21 | realized rolling 윈도우(영업일) |
| `blend_weights` | `{vix: 0.5, realized: 0.5}` | blend 가중치 |

---

## [2] 핵심 모듈 전문

### `src/metrics.py`

```python
"""
src/metrics.py — 성과 지표 (DEFINITIONS_AND_CONVENTIONS.md 4절)

모두 net 기준, 일간수익률 기반.
표기: r_t=전략 일간수익률, b_t=벤치마크 일간수익률,
      rf_t=무위험 일간수익률, N=거래일 수, 연율화=252.

─── equity 규약 ───────────────────────────────────────────────────────────────
backtest.run()이 반환하는 equity 시리즈는 (1+r).cumprod() 형태다.
  equity[0] = 1 + r_net[0]  (첫 거래일 수익 반영, ≠ 1.0)
  암시적 초기자산 = 1.0 (backtest 시작 전, 시리즈에 포함되지 않음)
  N = len(equity) = 거래일 수

CAGR = (최종/초기)^(252/N) − 1 = equity[-1]^(252/N) − 1
(초기 = 1.0, equity[-1]/equity[0] 형태를 쓰면 첫날 수익이 분모로 묻혀 N-1기간만 반영됨)
───────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_ANNUAL = 252
_EPS = 1e-10  # 부동소수점 잔차 방지용 임계치


def cagr(equity: pd.Series) -> float:
    """
    DEFINITIONS 4절: (최종/초기)^(252/N) − 1, N = 거래일 수.
    equity 규약: cumprod(1+r), 암시적 초기 = 1.0, N = len(equity).
    → equity[-1] ** (252/N) − 1.
    """
    n = len(equity)
    if n < 2:
        return float("nan")
    return float(equity.iloc[-1] ** (_ANNUAL / n) - 1)


def annual_vol(returns: pd.Series) -> float:
    """std(r_t) × √252"""
    return float(returns.std() * np.sqrt(_ANNUAL))


def sharpe(returns: pd.Series, rf: pd.Series) -> float:
    """
    [mean(r_t − rf_t) × 252] / [std(r_t) × √252]
    vol < _EPS이면 nan (부동소수점 잔차에 의한 0-나눗셈 방지).
    """
    excess = returns - rf.reindex(returns.index).fillna(0.0)
    vol = annual_vol(returns)
    if vol < _EPS:
        return float("nan")
    return float(excess.mean() * _ANNUAL / vol)


def sortino(returns: pd.Series, rf: pd.Series, mar: float = 0.0) -> float:
    """
    [mean(r_t − rf_t) × 252] / [DD × √252]
    DD = sqrt(mean(min(r_t − MAR, 0)^2)), MAR=0 기본.
    DD < _EPS이면 nan.
    """
    excess = returns - rf.reindex(returns.index).fillna(0.0)
    downside = np.minimum(returns.values - mar, 0.0)
    dd = float(np.sqrt(np.mean(downside ** 2)))
    if dd < _EPS:
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
    if abs(mdd) < _EPS:
        return float("nan")
    return float(cagr(equity) / abs(mdd))


def capture_ratios(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> tuple[float, float]:
    """
    상/하방 캡처 (DEFINITIONS 4절 — Morningstar 표준, 연율화 복리수익 비율).

    상방 = 전략_ann(상승달) / 벤치_ann(상승달)
    하방 = 전략_ann(하락달) / 벤치_ann(하락달)
    연율화 복리수익 ann = (∏(1+r_m))^(12/k) − 1,  k = 해당 집합의 달 수.

    전체 누적곱(∏(1+r_m)−1) 방식은 k가 수백이면 분모가 수천 배로 폭발해
    비율이 0에 수렴하는 가짜값을 만든다 (예: 289 상승달 → 3.27%).
    연율화하면 k에 독립적인 "월평균 기하수익의 연율 환산"이 되어
    경제적으로 해석 가능한 값이 나온다.
    """
    strat_m = (1.0 + strategy_returns).resample("ME").prod() - 1.0
    bench_m = (1.0 + benchmark_returns).resample("ME").prod() - 1.0

    common = strat_m.index.intersection(bench_m.index)
    strat_m = strat_m.loc[common]
    bench_m = bench_m.loc[common]

    def _ann(vals: pd.Series) -> float:
        k = len(vals)
        if k == 0:
            return float("nan")
        return float((1.0 + vals).prod() ** (12.0 / k) - 1.0)

    def _ratio(s_vals: pd.Series, b_vals: pd.Series) -> float:
        if len(b_vals) == 0:
            return float("nan")
        b_ann = _ann(b_vals)
        if abs(b_ann) < _EPS:
            return float("nan")
        return _ann(s_vals) / b_ann

    up_mask   = bench_m > 0.0
    down_mask = bench_m < 0.0
    return _ratio(strat_m[up_mask], bench_m[up_mask]), _ratio(strat_m[down_mask], bench_m[down_mask])


def annual_turnover(turnover_arr: pd.Series) -> float:
    """
    one-way 실제 체결 합 / 연 수 (연 회전율).
    입력: result["turnover"] — 매 거래일 실제 체결 크기(≥0).
    w_held.diff() 방식 폐기: band=0 ee에서 w_held=상수 → diff=0으로 실거래 숨김.
    """
    n = len(turnover_arr)
    if n == 0:
        return float("nan")
    n_years = n / _ANNUAL
    if n_years == 0.0:
        return float("nan")
    return float(turnover_arr.sum() / n_years)


def avg_exposure(weights: pd.Series) -> float:
    """mean(w_t) — 시장체류시간"""
    return float(weights.mean())


def summary(
    equity_net: pd.Series,
    returns_net: pd.Series,
    benchmark_returns: pd.Series,
    rf: pd.Series,
    weights: pd.Series,
    turnover_arr: pd.Series | None = None,
) -> dict:
    """
    전체 지표 dict 반환 (모두 net 기준).
    turnover_arr: result["turnover"] 필수. None이면 ValueError.
    """
    up_cap, down_cap = capture_ratios(returns_net, benchmark_returns)
    if turnover_arr is None:
        raise ValueError("turnover_arr(result['turnover']) 필수 — band=0 ee 표시 오류 방지")
    return {
        "cagr":            cagr(equity_net),
        "annual_vol":      annual_vol(returns_net),
        "sharpe":          sharpe(returns_net, rf),
        "sortino":         sortino(returns_net, rf),
        "mdd":             max_drawdown(equity_net),
        "calmar":          calmar(equity_net),
        "up_capture":      up_cap,
        "down_capture":    down_cap,
        "annual_turnover": annual_turnover(turnover_arr),
        "avg_exposure":    avg_exposure(weights),
    }
```

---

### `src/indicators/volatility.py`

```python
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

        sigma_t = sigma_t.where(sigma_t > 0)   # σ=0 방지
        w = (sigma_target / sigma_t).clip(0.0, w_max)
        w.name = "w_vol"
        return w


def _sigma_vix(data: dict[str, pd.Series]) -> pd.Series:
    """VIX_t / 100. VIX는 t일 종가 기준 → 룩어헤드 없음."""
    return data["vix"] / 100.0


def _sigma_realized(data: dict[str, pd.Series], cfg: dict) -> pd.Series:
    """
    N일 rolling std × √252.
    sp_ret[t] = t일 종가 기준 수익률. rolling은 [t-N+1, t] causal 구간.
    """
    lookback = int(cfg.get("realized_lookback", 21))
    sp_ret = data["sp500tr"].pct_change()
    sigma = sp_ret.rolling(lookback).std() * np.sqrt(252)
    _assert_realized_alignment(sp_ret, sigma)
    return sigma


def _sigma_blend(data: dict[str, pd.Series], cfg: dict) -> pd.Series:
    """VIX + realized 가중 평균."""
    bw = cfg.get("blend_weights", {"vix": 0.5, "realized": 0.5})
    w_vix  = float(bw.get("vix",      0.5))
    w_real = float(bw.get("realized", 0.5))
    return w_vix * _sigma_vix(data) + w_real * _sigma_realized(data, cfg)


def _assert_realized_alignment(sp_ret: pd.Series, sigma: pd.Series) -> None:
    """
    realized σ_t이 t일 수익률까지만 사용함을 단언.
    (1) 인덱스 동일성: 시프트·길이 불일치 모두 감지.
    (2) 유효값 끝날 일치: last valid σ == last valid sp_ret.
    위반 시 ValueError("룩어헤드").
    """
    if sp_ret.dropna().empty or sigma.dropna().empty:
        return
    if not sp_ret.index.equals(sigma.index):
        raise ValueError(
            f"realized σ 인덱스({len(sigma)})가 sp_ret 인덱스({len(sp_ret)})와 다름 — "
            "σ에 인위적 시프트 또는 길이 불일치: 룩어헤드 위험"
        )
    last_ret = sp_ret.dropna().index[-1]
    last_sig = sigma.dropna().index[-1]
    if last_ret != last_sig:
        raise ValueError(
            f"realized σ 마지막 유효일({last_sig.date()}) ≠ "
            f"수익률 마지막 유효일({last_ret.date()}): 룩어헤드 위험"
        )
```

---

### `src/indicators/base.py`

```python
"""
src/indicators/base.py — 공통 신호 인터페이스

─── 단독 평가(M3) eval_start 규약 [규약 A] ───────────────────────────────────
CLAUDE.md: "단독=각자 최대 기간" — 정보원 간 비대칭 허용,
한 정보원 내 추정기 변형끼리는 동일 입력 기준.

규약:
  1. 각 정보원에는 등록 시작일(SOURCE_STARTS)이 있다.
  2. 동일 정보원의 모든 추정기 변형은 source_start로 자른 데이터를 입력받아
     각자 워밍업 후 eval_start를 결정한다.
  3. period.start 슬라이싱은 비교·결합(M4~)에서만 사용.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


SOURCE_STARTS: dict[str, str] = {
    "volatility": "1990-01-01",   # VIX 신방식 1990~, VXO 금지
    "credit":     "1986-01-01",   # BAA10Y(Moody's Baa−10Y) 1986~
    "trend":      "1988-01-01",   # sp500tr 가용일 기준
}


class BaseIndicator(ABC):
    @abstractmethod
    def signal(self, data: dict[str, pd.Series], cfg: dict) -> pd.Series:
        """
        t일 종가 정보로 w_target ∈ [0, w_max] 산출 → 백테스트 엔진이 t+1 체결.
        워밍업 구간은 NaN 반환.
        """
        ...


def standalone_data(
    data_raw: dict[str, pd.Series],
    source: str,
) -> dict[str, pd.Series]:
    """
    단독 평가용 데이터 준비 — 규약 A.

    SOURCE_STARTS[source]로 data_raw를 슬라이싱.
    동일 정보원의 모든 추정기 변형이 이 함수를 통해 동일한 입력을 받는다.

    사용 예:
        data_vol = standalone_data(data_raw, "volatility")  # M3-A
        data_crd = standalone_data(data_raw, "credit")      # M3-B
        data_trd = standalone_data(data_raw, "trend")       # M3-C
    """
    if source not in SOURCE_STARTS:
        raise ValueError(f"알 수 없는 source: {source!r}. 등록값: {list(SOURCE_STARTS)}")
    start = pd.Timestamp(SOURCE_STARTS[source])
    return {k: v.loc[start:] for k, v in data_raw.items()}


class ConstantWeightIndicator(BaseIndicator):
    """더미 신호: 항상 고정 비중 반환. 테스트·M2 엔드투엔드 전용."""

    def __init__(self, weight: float = 1.0) -> None:
        self._w = weight

    def signal(self, data: dict[str, pd.Series], cfg: dict) -> pd.Series:
        ref = next(iter(data.values()))
        w = min(self._w, cfg.get("w_max", 1.0))
        return pd.Series(w, index=ref.index, name="w_dummy")
```

---

### `src/indicators/credit.py` (핵심 로직)

```python
def _percentile_rank(s: pd.Series, window: int) -> pd.Series:
    """trailing W일 causal 백분위. arr[-1]=현재값 기준."""
    def _rank_last(arr: np.ndarray) -> float:
        return float((arr <= arr[-1]).sum()) / len(arr)
    return s.rolling(window, min_periods=window).apply(_rank_last, raw=True)


def _monotone_map(p, theta_low, theta_high, w_max) -> pd.Series:
    """DEFINITIONS 1.4: p<θ_low→w_max, p>θ_high→0, 사이 선형보간."""
    span = theta_high - theta_low
    w_vals = np.where(p.isna(), np.nan,
        np.where(p < theta_low, w_max,
            np.where(p > theta_high, 0.0,
                w_max * (theta_high - p.values) / span)))
    return pd.Series(w_vals, index=p.index)


def _assert_percentile_alignment(baa10y, p) -> None:
    """p 인덱스 정합 단언 (_assert_realized_alignment 동급)."""
    if not baa10y.index.equals(p.index):
        raise ValueError("percentile p 인덱스 불일치 — 룩어헤드 위험")
    if baa10y.dropna().index[-1] != p.dropna().index[-1]:
        raise ValueError("percentile 마지막 유효일 불일치 — 룩어헤드 위험")
```

---

### `src/indicators/trend.py` (핵심 로직)

```python
class TrendIndicator(BaseIndicator):
    """DEFINITIONS 1.3 추세 신호 (ma200 | tsmom12)."""

    def signal(self, data: dict[str, pd.Series], cfg: dict) -> pd.Series:
        rule    = cfg.get("rule", "ma200")
        w_floor = float(cfg.get("w_floor", 0.0))
        w_max   = float(cfg.get("w_max",   1.0))
        prices  = data["sp500tr"].dropna()

        if rule == "ma200":
            w = _ma200_signal(prices, w_floor, w_max)
        elif rule == "tsmom12":
            w = _tsmom12_signal(prices, w_floor, w_max)
        else:
            raise ValueError(f"알 수 없는 trend rule: {rule!r}")
        w.name = "w_trend"
        return w


def _ma200_signal(prices, w_floor, w_max):
    """
    rolling(200, min_periods=200).mean() = MA200_t.
    t 위치에서 P[t-199:t+1] causal 구간 사용 (closed='right' 기본).
    w_t = w_max if P_t > MA200_t else w_floor.
    첫 199일 NaN (워밍업).
    """
    ma200 = prices.rolling(200, min_periods=200).mean()
    _assert_ma200_alignment(prices, ma200)
    above = prices > ma200
    w = above.map({True: w_max, False: w_floor}).astype(float)
    w[ma200.isna()] = np.nan
    return w


def _tsmom12_signal(prices, w_floor, w_max):
    """
    r12_t = P[i] / P[i-252] - 1  (iloc 직접 참조, off-by-one 방지).
    w_t = w_max if r12 > 0 else w_floor.
    첫 252일 NaN (워밍업).
    """
    n, lag = len(prices), 252
    w_vals = np.full(n, np.nan)
    for i in range(lag, n):
        r12 = prices.iloc[i] / prices.iloc[i - lag] - 1.0
        w_vals[i] = w_max if r12 > 0.0 else w_floor
    w = pd.Series(w_vals, index=prices.index)
    _assert_tsmom_alignment(prices, w)
    return w


def _assert_ma200_alignment(prices, ma200):
    """MA200 인덱스 정합 단언 (인덱스 동일성 + 마지막 유효일 일치)."""
    if prices.dropna().empty or ma200.dropna().empty:
        return
    if not prices.index.equals(ma200.index):
        raise ValueError("MA200 인덱스 불일치 — 시프트 또는 길이 불일치: 룩어헤드 위험")
    if prices.dropna().index[-1] != ma200.dropna().index[-1]:
        raise ValueError("MA200 마지막 유효일 불일치: 룩어헤드 위험")


def _assert_tsmom_alignment(prices, w):
    """TSMOM 워밍업 경계 단언 (첫 252일 NaN, iloc[252] non-NaN, 인덱스 동일성)."""
    if prices.dropna().empty or w.dropna().empty:
        return
    if not prices.index.equals(w.index):
        raise ValueError("TSMOM w 인덱스 불일치 — 룩어헤드 위험")
    if len(w) > 252:
        if not w.iloc[:252].isna().all():
            raise ValueError("TSMOM 워밍업 구간(첫 252일)에 non-NaN — 룩어헤드 위험")
        if np.isnan(w.iloc[252]):
            raise ValueError("TSMOM w.iloc[252] NaN — 252일 후 신호 미생성")
    if prices.dropna().index[-1] != w.dropna().index[-1]:
        raise ValueError("TSMOM 마지막 유효일 불일치: 룩어헤드 위험")
```

---

## 참고: 파이프라인 흐름

```
load_all(cfg)                              # data_loader.py
    └─ standalone_data(data_raw, source)   # base.py (단독 M3; source = volatility|credit|trend)
        └─ indicator.signal(data, cfg)     # VolatilityIndicator / CreditIndicator / TrendIndicator
            └─ w.dropna().index[0] = eval_start
                └─ backtest.run(w, sp_r, rf, cfg)
                    ├─ buy_and_hold(sp_r, rf, cfg)               # benchmarks.py
                    ├─ equal_exposure(mean_w, sp_r, rf, cfg)     # benchmarks.py (band=0)
                    └─ metrics.summary(..., result["turnover"])  # turnover_arr 필수

# M3 단독 검증 eval_start 요약
# volatility: vix=1990-01-02, realized/blend=1990-01-31
# credit:     baa10y=1988-12-29  (sp500tr inner join → 1988-01-04 기준, rolling 252)
# trend:      ma200=1988-10-14, tsmom12=1988-12-30  (sp500tr inner join → 1988-01-04 기준)

# M4 독립성 분석 (notebooks/03_independence.ipynb)
# 공통기간: 1990-01-31~ (realized·blend 워밍업 바닥)
# 능동수익: 전략 − ee (동일노출 정적, band=0)
# Gate 1 결과: 변동성 INCLUDE / 추세 INCLUDE / 신용 EXCLUDE
# 신용 판정 근거: (i)·(A) 병기 종합 — 기준1 실패 + 변동성 (A) 고상관 + 추세와 전략 수준 고상관
# M5 이월 결정: 변동성 대표 변형 선정 (사전등록 원칙 적용)
```
