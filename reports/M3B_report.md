# M3-B 실행 보고서 — 신용 신호 단독 검증

| 항목 | 내용 |
|---|---|
| 날짜 | 2026-06-03 |
| 커밋 | a674721 |

---

## 1. 목표

신용 정보원의 대표 신호(BAA10Y)를 동일 엔진·net 기준으로 단독 백테스트하고,
buy&hold와 동일노출 정적 배분(equal_exposure) 두 대조군과 비교한다.

---

## 2. 구현 파일

### `src/indicators/credit.py`

신호 흐름: `s_t = baa10y[t]` → trailing 252일 백분위 `p_t` → 단조 감소 매핑 → `w_t`

**`_percentile_rank(s, W)`**
- trailing rolling(W) 백분위: t 위치에서 `s[t-W+1:t+1]` 구간 안에서 `s[t]`의 순위
- `raw=True` + `arr[-1]` 기준 → `(arr <= arr[-1]).sum() / len(arr)` (causal ✓)
- 첫 W−1일 NaN (워밍업)

**`_monotone_map(p, theta_low, theta_high, w_max)`**
DEFINITIONS 1.4 임계 방식:

| 조건 | w |
|---|---|
| `p < 0.5` | `w_max` (정상여건, 풀 노출) |
| `p > 0.9` | `0` (극단 스트레스, 완전 방어) |
| `0.5 ≤ p ≤ 0.9` | `w_max × (0.9 − p) / 0.4` (선형 보간) |

theta 0.5/0.9는 `credit.yaml` 사전 등록값 그대로. 결과를 보고 바꾸지 않음.

**`_assert_percentile_alignment(baa10y, p)`**
`_assert_realized_alignment`(volatility)와 동급 룩어헤드 단언:
1. 인덱스 동일성: `p.index == baa10y.index` (시프트 감지)
2. 마지막 유효일 일치 확인

**eval_start 규약**: `standalone_data(data_raw, "credit")` → 1986-01-01 기준 입력,
rolling(252) 워밍업 후 eval_start = **1988-12-29**.

> NFCI·STLFSI4·HY OAS는 이 파일에서 사용 금지. M4 robustness 전용.

---

### 이번 마일스톤에서 수정된 공통 모듈

#### `src/benchmarks.py` — equal_exposure band=0 고정

**수정 이유**: band=0.05이면 강세장에서 drift가 방치되어 realized mean(w_t) > target
편향이 발생 (예: vix에서 0.684 → 0.699, Δ=+0.015).
"동일 평균노출" 비교 취지에 어긋나므로 **band=0 고정**으로 변경.

| band | ee 실현 mean (vix 기준) | 전략 대비 Δ | Turn/yr |
|---|---|---|---|
| 0.05 (이전) | 0.6994 | +0.0155 | 0.43 |
| **0.00 (현재)** | **0.6839** | **+0.000** | **0.43** |

Band=0이면 w_held = mean_w 상수, 실제 drift 교정 비용 = **0.86bp/yr** (CAGR 차이 0.94bp로 확인).

#### `src/metrics.py` — annual_turnover: turn_arr 기준으로 변경

**수정 이유**: `w_held.diff()` 기준이면 band=0 ee는 w_held=상수라 Turn/yr=0.00 집계.
실제 drift 교정 거래(0.43/yr)가 숨겨지는 표시 오류.

**변경 내용**: `annual_turnover`가 `result["turnover"]`(turn_arr) 를 직접 입력받음.
`summary()`에 `turnover_arr` 파라미터 추가 (필수).

| | 이전 w_held.diff | 수정 turn_arr |
|---|---|---|
| vix 전략 | 4.66 | **4.31** |
| ee(0.684) | **0.00** (오류) | **0.43** |
| credit 전략 | 7.56 | **7.54** |

---

## 3. 룩어헤드 단언 (`tests/test_no_lookahead.py`)

| 테스트 | 내용 | 결과 |
|---|---|---|
| `test_credit_future_perturbation_invariant` | baa10y[300:] 극단 교란 → p_t[:300] 불변 | PASS |
| `test_credit_signal_execution_lag` | r_gross[t] = weights[t-1]×r_eq[t]+… (400일 전수) | PASS |
| `test_credit_percentile_alignment_catches_shifted_index` | 인덱스 시프트 → ValueError 발생 | PASS |

---

## 4. 완료 기준 실행 결과

**pytest: 32/32 통과**

```
tests/test_metrics.py          18 passed
tests/test_no_lookahead.py     14 passed  (credit 3개 신규)
```

### credit_metrics_full.csv — 전체 지표표

| strategy | eval_start | CAGR | Vol | Sharpe | Sortino | MDD | Calmar | UpCap | DnCap | Turn/yr | AvgExp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vol_credit | 1988-12-29 | 8.38% | 10.61% | 0.541 | 0.763 | -30.81% | 0.272 | 57.36% | 59.03% | 7.54 | 0.661 |
| buy&hold | 1988-12-29 | 11.49% | 17.89% | 0.537 | 0.765 | -55.25% | 0.208 | 99.90% | 100.00% | 0.03 | 1.000 |
| ee(0.661) | 1988-12-29 | 8.89% | 11.84% | 0.537 | 0.766 | -39.46% | 0.225 | 64.86% | 67.75% | 0.44 | 0.661 |

**동일노출 고정비중 일치**: credit 전략 mean(w_t) = **0.661**, ee 실현 mean(w_t) = **0.661** (Δ = 0.000) ✓

---

## 5. 신호 해석 (실패 양상)

### 신호 드묾 — BAA10Y IG 하단 특성
- 전체 기간의 **52.7%가 w=1.0** (p < 0.5, 정상여건). 방어 개입 자체가 희귀.
- BAA10Y는 Baa(BBB) 등급 스프레드라 HY OAS(BB 이하)보다 위기 반응 폭이 작음.
  → 신용 방어 강도 과소평가 가능성 내재 (사전 등록 제약, 수용)

### 회전율 7.54 — 임계 근처 대형 점프 구조

| 구간 | credit | credit% | vix | vix% |
|---|---|---|---|---|
| Δw = 0 | 6,758 | 71.7% | 705 | 7.7% |
| 0 < Δw < 0.01 | 310 | 3.3% | 2,030 | 22.1% |
| 0.01 ≤ Δw < 0.05 | 707 | 7.5% | 4,856 | 53.0% |
| Δw ≥ 0.05 | 1,647 | 17.5% | 1,577 | 17.2% |
| 비0 Δw 평균 | 0.1123 | — | 0.0309 | — |

VIX는 매일 소폭 연속 변동, credit은 71.7% 불변 후 위기 시 대형 점프(평균 0.112).
p=0.5 교차 308일(3.3%), p=0.9 교차 286일(3.0%) — **임계 근처 휩소로 회전율 집중 발생**.

### 조용한 강세장 열위
- 저스프레드 지속 → 신호 없음 (w=1.0). 회전율(7.54)로 인한 비용이 net 열위 원인.
- CAGR 8.38% < ee(0.661) 8.89%: 타이밍 가치 없음, 순비용 손실.

### 임계 구조의 비선형 취약점
- 위기 초입에서 p < 0.5이면 방어 무발동 구간 존재.
- 스프레드가 급격히 상승 시작 전 BAA10Y는 아직 낮은 백분위 → 초기 방어 지연.

---

## 6. 이슈 및 처리

| 이슈 | 원인 | 처리 |
|---|---|---|
| ee 실현 mean > 전략 mean (+0.015) | band=0.05에서 강세장 drift 방치 | equal_exposure band=0 고정 |
| ee Turn/yr=0.00 (표시 오류) | w_held=상수라 w_held.diff=0, 실제 0.43/yr 거래 숨김 | annual_turnover를 turn_arr 기준으로 변경 |
| 비용 공정성 | ee drift 교정 비용 0.86bp/yr — 전략 8.6bp/yr의 10% | CAGR 차이 0.94bp로 반영 확인, 과소계상 없음 |
