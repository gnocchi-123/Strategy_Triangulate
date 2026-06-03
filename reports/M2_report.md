# M2 실행 보고서 — 공통 엔진 · 인터페이스 · 지표 · 대조군

| 항목 | 내용 |
|---|---|
| 날짜 | 2026-06-03 |
| 커밋 | d396602 (엔진·테스트) → f79112d (metrics 수식 정합) |

---

## 1. 목표

모든 신호(M3~)가 공유하는 백테스트 엔진, 신호 인터페이스, 성과 지표, 대조군을 구현한다.
이후 마일스톤에서 신호 로직만 `src/indicators/`에 추가하면 엔드투엔드 파이프라인이 동작하는 상태를 만든다.

---

## 2. 구현 내용

### `src/indicators/base.py` — 공통 신호 인터페이스

```python
class BaseIndicator(ABC):
    def signal(self, data: dict[str, pd.Series], cfg: dict) -> pd.Series: ...
```

- 입력: `data` (거래일 인덱스 정합 완료된 시계열 dict), `cfg` (신호별 파라미터)
- 출력: `w_target` ∈ [0, w_max], 거래일 인덱스. t일 종가 기준 산출 → 엔진이 t+1 체결.
- `ConstantWeightIndicator`: 항상 고정 비중 반환하는 더미 신호 (M2 엔드투엔드·테스트 전용).

---

### `src/backtest.py` — 백테스트 엔진

체결 순서 (매 거래일 t):

1. **gross return**: `w_eff × r_eq[t] + (1 − w_eff) × rf_d[t]`
   - 레버리지: `w_eff = w_prev × L`, 초과분(`w_eff > 1`) 차입비용 일간 차감
2. **자연 표류 비중**: `w_drifted = V_eq / (V_eq + V_cash)` (수익 반영 후)
3. **밴드 검사**: `|w_target − w_drifted| > band` → 거래 실행, `cost = Δw × cost_bps / 10000`
4. **net return**: `gross − cost`
5. **다음날 적용 비중** 확정 (`w_held[t]`)

반환값: `equity_gross`, `equity_net`, `returns_gross`, `returns_net`, `weights`, `turnover`

주요 파라미터 (`config/base.yaml`):

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `rebalance_band` | 0.05 | 리밸런싱 불감대 (5%) |
| `cost_bps` | 2.0 | 편도 거래비용 |
| `borrow_spread_bps` | 50 | 레버리지 차입비용 (Phase 3용) |
| `w_max` | 1.0 | 최대 주식 노출 상한 |

---

### `src/benchmarks.py` — 대조군

| 대조군 | 설명 |
|---|---|
| `buy_and_hold` | 100% 주식, w_target=1.0 전 기간. 시작일 비용 1회 발생. |
| `equal_exposure(mean_w)` | 전략의 실현 평균 노출(`avg_exposure(result["weights"])`)을 고정비중으로 상수 배분. 동일 엔진·동일 비용 모델 재사용. |

**설계 원칙**: `equal_exposure`는 전략과 동일한 `rebalance_band`·`cost_bps`로 드리프트 교정 회전율까지 처리 → 타이밍이 없는 동일 노출 대조군으로 공정 비교.

`mean_w`는 전략 `backtest.run()` 후 실현 비중 평균으로 주입한다(목표비중 평균 사용 금지).

---

### `src/metrics.py` — 성과 지표 (DEFINITIONS_AND_CONVENTIONS.md 4절)

equity 규약: `(1+r).cumprod()`, 암시적 초기=1.0, N=len(equity).

| 지표 | 수식 |
|---|---|
| CAGR | `equity[-1]^(252/N) − 1` |
| 연율 변동성 | `std(r_t) × √252` |
| Sharpe | `[mean(r_t − rf_t) × 252] / [std(r_t) × √252]` |
| Sortino | `[mean(r_t − rf_t) × 252] / [DD × √252]`, DD=하방편차(MAR=0) |
| MDD | `min_t(equity_t / running_max_t − 1)` (음수) |
| Calmar | `CAGR / |MDD|` |
| 상/하방 캡처 | 월간 복리수익 비율 (연율화 없음) |
| 연 회전율 | `Σ|Δw| / 연 수` (one-way) |
| 평균 노출 | `mean(w_t)` |

`metrics.summary()`: 위 전부를 dict로 반환.

---

### `src/combine.py` — 스텁 (M5에서 완성)

`combine_equal_weight()`, `combine_vote()` 시그니처만 등록. 호출 시 `NotImplementedError`.

---

### `tests/test_no_lookahead.py` — 룩어헤드 방지 단언

| 테스트 ID | 내용 |
|---|---|
| (a) 미래 교란 불변성 | `w_target[50:]` 변경 시 `weights[:49]` 불변 |
| (a) 미래 수익 교란 불변성 | `r_equity[70:]` 교란 시 `weights[:69]` 불변 |
| (b) 체결 lag | `r_gross[t] = weights[t−1] × r_eq[t] + (1 − weights[t−1]) × rf[t]` |
| (b) band·cost 하 체결 lag | 위와 동일, band=0.05·cost=2bp 조건 |
| 엔드투엔드 (더미 신호) | ConstantWeightIndicator(0.7) → backtest → metrics.summary 전 파이프라인 |
| 엔드투엔드 (대조군) | buy_and_hold·equal_exposure 정상 반환값 확인 |
| (a-vix) VIX 모드 교란 불변 | M3-A에서 추가: VolatilityIndicator(vix) 미래 교란 불변 |
| (a-real) realized 모드 교란 불변 | M3-A에서 추가: realized rolling std 미래 교란 불변 |
| (b-vol) VIX 모드 체결 lag | M3-A에서 추가: VolatilityIndicator 실제 신호로 lag 단언 |
| (c) _assert_realized_alignment | M3-A에서 추가: σ 인덱스 시프트 시 ValueError 발생 |
| (d) equal_exposure 실현 비중 패턴 | M3-A에서 추가: mean_w = avg_exposure(result["weights"]) 검증 |

---

### `tests/test_metrics.py` — 지표 기대값 단위 테스트

| 테스트 | 핵심 검증 |
|---|---|
| `test_cagr_varied_handcalc` | 손계산 기대값 일치 + 잘못된 수식(`equity[-1]/equity[0]`)과 10% 이상 차이 변별 |
| `test_cagr_zero_return` | 전 기간 0수익 → CAGR=0 |
| `test_cagr_exact_annual` | N=252, 일정 r → `(1+r)^252 − 1` 닫힌형 일치 |
| `test_annual_vol_*` | 상수 수익=0, 알려진 std×√252 |
| `test_sharpe_*` | 부동소수점 잔차에 의한 0나눗셈 → nan; 양의 초과수익 → 양수 |
| `test_mdd_planted` | 고점110→저점60: `60/110−1` 손계산 일치 |
| `test_calmar_consistent` | CAGR/|MDD| 내적 일관성 |
| `test_sortino_*` | 양수 확인; Sortino ≥ Sharpe (excess>0, MAR=0) |
| `test_annual_turnover_alternating` | 교대 비중 → one-way 연 회전율 손계산 |
| `test_capture_ratios_closed_form` | 상승달+10%/전략+20%, 하락달−5%/전략−2.5% → up=2.0, down=0.5 닫힌형 |
| `test_capture_ratios_symmetry` | 전략=벤치마크 → up=down=1.0 |

---

## 3. 완료 기준 실행 결과

```
pytest tests/ -v --tb=no -q
27 passed in 13.57s
```

| 기준 | 결과 |
|---|---|
| pytest 전부 통과 | ✅ 27/27 (test_metrics 16개, test_no_lookahead 11개) |
| 룩어헤드 단언 | ✅ 미래 교란 불변 · 체결 lag 2종 |
| 지표 기대값 | ✅ 손계산·닫힌형으로 수식 변별 확인 |
| gross ≥ net | ✅ 비용 차감 구조 정상 |
| 더미 신호 엔드투엔드 | ✅ ConstantWeightIndicator(0.7) → backtest → metrics.summary 전 경로 동작 |
| 대조군 엔드투엔드 | ✅ buy_and_hold(avg_exposure≈1.0), equal_exposure(avg_exposure≈0.6) |

---

## 4. 설계 결정 사항

### equity 규약: `equity[-1]^(252/N) − 1`
잘못된 수식 `(equity[-1]/equity[0])^(252/(N−1))`은 첫날 수익이 분모로 묻혀 N−1기간만 반영된다. M2에서 `test_cagr_varied_handcalc`로 두 수식이 10% 이상 차이나는 것을 확인하고 올바른 수식으로 고정했다.

### equal_exposure mean_w: 목표비중 평균 아닌 실현 비중 평균
`w_target.mean()`은 밴드·드리프트 이전 목표값이다. 대조군의 노출을 전략과 공정하게 맞추려면 엔진 실행 후 `avg_exposure(result["weights"])`(실현값)를 사용해야 한다. `test_equal_exposure_uses_realized_weights`로 패턴을 단언으로 고정했다.

### 상/하방 캡처: 연율화 없음
`DEFINITIONS_AND_CONVENTIONS.md` 4절에 연율화 공식이 명시되지 않았으므로 월간 복리수익 비율 그대로 사용. `test_capture_ratios_closed_form`의 닫힌형 기대값으로 수식을 고정했다.

---

## 5. 이슈 및 처리

- **CAGR 수식 불일치 (커밋 f79112d)**: 초기 구현에서 `equity[-1]/equity[0]` 형태로 작성. `DEFINITIONS_AND_CONVENTIONS.md` 4절 수식 재검토 후 `equity[-1]^(252/N)` 로 수정. 테스트도 손계산 기대값과 변별 구조로 강화.
- **Sharpe 0나눗셈**: 상수 수익률에서 `std()` 부동소수점 잔차(≈2e-19)가 0 나눗셈을 일으킬 수 있음. `_EPS=1e-10` 임계치로 처리해 `nan` 반환.
