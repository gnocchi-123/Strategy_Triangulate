# M3-A 실행 보고서 — 변동성 신호 단독 검증

| 항목 | 내용 |
|---|---|
| 날짜 | 2026-06-03 |
| 커밋 | 47e0fc6 (M3-A 최초) → 66c9fbd (캡처·규약 수정) |

---

## 1. 목표

변동성 정보원의 세 추정기 변형(VIX / Realized / Blend)을 동일 엔진·net 기준으로
단독 백테스트하고, **buy&hold**와 **동일노출 정적 배분(equal_exposure)** 두 대조군과
비교한다.

---

## 2. 구현 파일

### `src/indicators/volatility.py`

| 추정기 | 수식 | 룩어헤드 처리 |
|---|---|---|
| `vix` | `σ_t = VIX_t / 100` | t일 종가 확정 → t+1 체결 |
| `realized` | rolling(N=21) std × √252 | `_assert_realized_alignment()` 인덱스 동일성 단언 |
| `blend` | `0.5 × σ_vix + 0.5 × σ_realized` | 두 추정기 모두 위 규약 준수 |

공통 목표비중: `w = clip(σ_target / σ_t, 0, w_max)`

워밍업 규약: rolling NaN 구간 → `w_target = NaN` 반환. 엔진이 `fillna(0)` 으로
현금 유지 처리. 평가 시작일 = 신호 첫 유효일.

### `src/indicators/base.py` — 단독 평가 규약 추가

**`SOURCE_STARTS` 딕셔너리** — 정보원별 등록 시작일:

```python
SOURCE_STARTS = {
    "volatility": "1990-01-01",   # VIX 신방식 1990~, VXO 금지
    "credit":     "1986-01-01",   # BAA10Y(Moody's Baa−10Y) 1986~
    "trend":      "1988-01-01",   # sp500tr 가용일 기준
}
```

**`standalone_data(data_raw, source)`** — 규약 A 헬퍼:
동일 정보원의 모든 추정기 변형이 같은 source_start로 자른 데이터를 입력받는다.
vix·realized·blend 모두 1990-01-01 입력 기준 → 워밍업 후 eval_start 결정.

> M3-B(신용)에서 `standalone_data(data_raw, "credit")`,
> M3-C(추세)에서 `standalone_data(data_raw, "trend")`를 호출하면 자동 통일.

### `src/metrics.py` — `capture_ratios` 수정

| 구분 | 이전 (버그) | 현재 (표준) |
|---|---|---|
| 수식 | `(∏(1+r_m) − 1) / (∏(1+r_m) − 1)` | `(∏(1+r_m))^(12/k) − 1` (Morningstar 표준) |
| 289 상승달 vix UpCap | **3.27%** (분모 폭발) | **59.04%** |

전체 누적곱 방식은 k가 수백이면 분모가 수천 배로 폭발해 비율이 0에 수렴하는
가짜값을 만든다. 연율화하면 k에 독립적인 기하평균 연율 수익으로 환산된다.

### `DEFINITIONS_AND_CONVENTIONS.md` — 4절 캡처 명확화

`ann = (∏(1+r_m))^(12/k) − 1, k = 해당 집합의 달 수` 로 계산법 명시.
전체 누적곱 방식 사용 금지 명기. 합격 기준(하방<70%, 상방>85%) 유지.

### `tests/test_metrics.py` — 캡처 테스트 추가

| 테스트 | 검증 내용 |
|---|---|
| `test_capture_ratios_closed_form` | 12개월 균등 수익 닫힌형: UpCap = (1.01^12−1)/(1.02^12−1) |
| `test_capture_ratios_scale_invariance` | k=12/120/240 동일 결과 — 비연율화는 k=120→0.24, k=240→0.09로 붕괴 |
| `test_capture_ratios_long_series_sanity` | 20년·65% 노출 → UpCap ∈ [0.50, 0.80] |
| `test_capture_ratios_symmetry` | 전략 = 벤치 → 1.0 |

### `tests/test_no_lookahead.py` — M3-A 추가 단언

| 테스트 | 내용 |
|---|---|
| `test_vol_vix_future_perturbation_invariant` | VIX t+k 교란 → t<40 신호 불변 |
| `test_vol_realized_future_perturbation_invariant` | sp500tr t+k 교란 → t<50 신호 불변 |
| `test_vol_signal_execution_lag` | VIX 모드 체결 lag: `r_gross[t] = weights[t-1] × r_eq[t] + …` |
| `test_realized_alignment_catches_shifted_index` | σ 인덱스 시프트 시 ValueError 발생 |
| `test_equal_exposure_uses_realized_weights` | `mean_w = avg_exposure(result["weights"])` 패턴 단언 |

---

## 3. 단독 평가 규약

**규약 A** — 한 정보원 내 모든 추정기 변형은 그 정보원의 등록 시작일로 통일.

- "각자 최대 기간"은 정보원 간(변동성1990/신용1986/추세1988) 비대칭을 허용하는 것
- 한 정보원 내 변형끼리(vix·realized·blend)는 동일 입력 기준
- `period.start` 슬라이싱은 비교·결합(M4~)에서만 사용

| 추정기 | 입력 기준 | eval_start | 이유 |
|---|---|---|---|
| vix | 1990-01-01 | 1990-01-02 | VIX 데이터 첫 유효일 |
| realized | 1990-01-01 | 1990-01-31 | 1990-01-02 + rolling(21) 워밍업 |
| blend | 1990-01-01 | 1990-01-31 | realized 워밍업이 바인딩 |

---

## 4. 완료 기준 실행 결과

### pytest: 29/29 통과

```
tests/test_metrics.py          18 passed
tests/test_no_lookahead.py     11 passed
```

### vol_metrics_full.csv — 전체 지표표

| variant | strategy | eval_start | CAGR | Vol | Sharpe | Sortino | MDD | Calmar | UpCap | DnCap | Turn/yr | AvgExp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vix | vol_vix | 1990-01-02 | 7.59% | 9.27% | 0.542 | 0.761 | -28.33% | 0.268 | 59.04% | 64.43% | 4.66 | 0.684 |
| vix | buy&hold | 1990-01-02 | 10.95% | 18.01% | 0.516 | 0.734 | -55.25% | 0.198 | 100.00% | 100.00% | 0.00 | 1.000 |
| vix | ee(0.684) | 1990-01-02 | 8.79% | 12.40% | 0.521 | 0.743 | -40.80% | 0.215 | 68.28% | 71.34% | 0.43 | 0.699 |
| realized | vol_realized | 1990-01-31 | 9.43% | 11.48% | 0.607 | 0.854 | -37.42% | 0.252 | 73.17% | 75.95% | 2.45 | 0.812 |
| realized | buy&hold | 1990-01-31 | 11.24% | 18.01% | 0.531 | 0.757 | -55.25% | 0.204 | 99.76% | 100.00% | 0.00 | 1.000 |
| realized | ee(0.812) | 1990-01-31 | 9.96% | 14.67% | 0.536 | 0.763 | -46.70% | 0.213 | 81.16% | 83.87% | 0.29 | 0.828 |
| blend | vol_blend | 1990-01-31 | 8.46% | 10.28% | 0.578 | 0.811 | -32.06% | 0.264 | 65.34% | 70.09% | 2.77 | 0.751 |
| blend | buy&hold | 1990-01-31 | 11.24% | 18.01% | 0.531 | 0.757 | -55.25% | 0.204 | 99.76% | 100.00% | 0.00 | 1.000 |
| blend | ee(0.751) | 1990-01-31 | 9.50% | 13.63% | 0.535 | 0.762 | -44.65% | 0.213 | 74.85% | 77.92% | 0.37 | 0.766 |

---

## 5. 신호 해석 (실패 양상 포함)

### VIX (AvgExp 0.684, CAGR 7.59% vs buy&hold 10.95%)

- **장점**: 변동성 급등 구간(2008, 2020 등)에서 노출 축소로 MDD -28.3%(buy&hold -55.3% 대비 크게 방어). Calmar 0.268 vs 0.198.
- **열위**: CAGR이 동일노출 대조군 ee(0.684, 8.79%)에도 못 미침. 낮은 UpCap(59%)과 높은 회전율(4.66회/년)이 net 수익을 깎음.
- **실패 양상**: 조용한 강세장(저변동 지속 구간)에서 과도한 진입·청산 반복(휩소). VIX 낮을 때 w_max=1.0 상한에 걸려 추가 수익 불가.

### Realized (AvgExp 0.812, CAGR 9.43% vs buy&hold 11.24%)

- **장점**: 세 추정기 중 최고 Sharpe(0.607) · Sortino(0.854). 회전율 2.45로 VIX보다 낮음. UpCap 73.2%로 상승 추종력 가장 강함.
- **열위**: CAGR이 동일노출 대조군 ee(0.812, 9.96%)에 못 미침.
- **실패 양상**: 변동성 후행 특성 — 급락 후 시그마가 높아지면 이미 손실 이후에 방어, 급등 회복기에는 노출을 늦게 늘림.

### Blend (AvgExp 0.751, CAGR 8.46% vs buy&hold 11.24%)

- **장점**: VIX의 즉각성 + Realized의 안정성 결합. MDD -32.1%로 방어 Realized보다 우수. Sharpe 0.578으로 VIX보다 높음.
- **열위**: CAGR이 동일노출 대조군 ee(0.751, 9.50%)에 못 미침.
- **실패 양상**: 두 추정기 모두 w=1.0 상한에 걸리는 강세장에서 blend 가산 효과 없음.

### 공통 패턴

세 추정기 모두 **동일노출 정적 배분(equal_exposure)을 CAGR 기준으로 이기지 못함**.
위험조정(Sharpe·Sortino·MDD) 개선은 있으나 알파(타이밍 가치)는 아직 불명확.
→ Gate 1 기준(단독 승리 또는 저상관+결합 개선)으로 M4 독립성 검증에서 판정 예정.

---

## 6. 이슈 및 처리

| 이슈 | 원인 | 처리 |
|---|---|---|
| UpCap 3.27% 이상값 | 전체 누적곱 방식이 분모 폭발(289 상승달 → 12,000배) | Morningstar 표준 연율화 수식으로 교체, DEFINITIONS 4절 명확화 |
| realized CAGR 왕복(9.43↔9.83) | CSV 재생성 스크립트마다 period.start 슬라이싱 적용 여부 불일치 | 규약 A 확정: `standalone_data()` 헬퍼로 고정. realized = 1990 입력 기준 |
| `test_capture_ratios_closed_form` 기대값 변경 | k=12이면 연율화·비연율화 우연히 동치 → 두 수식 미구분 | `test_capture_ratios_scale_invariance` 추가로 변별 |
