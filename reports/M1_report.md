# M1 실행 보고서 — 데이터 레이어

| 항목 | 내용 |
|---|---|
| 날짜 | 2026-06-01 |
| 커밋 | b580925 → c69ea11 → 670072f → d1db95b → 7207a0e → 5d821bb → 4526d4e |

---

## 1. 목표

시세 + FRED 신용/거시 수집·거래일 정합·이상치 검증·parquet 캐시·주간 지표 발표 시차 시프트 구현

---

## 2. 구현 파일

### `src/data_loader.py`

| 함수 | 역할 |
|---|---|
| `load_prices()` | yfinance에서 sp500tr·vix·vix3m 수집. 시장 시세 ffill 금지. |
| `load_fred()` | FRED에서 rf·baa10y·hy_oas·nfci·stlfsi raw 수집. |
| `apply_weekly_lag()` | 주간 시리즈 발표 시차 시프트 → reindex → ffill. |
| `validate()` | 시리즈별 시작·종료·결측·중복·이상치 검증. |
| `load_all()` | 전체 파이프라인. `dict[str, pd.Series]` 반환. inner join 없음. |

### `notebooks/01_data_audit.ipynb`
완료 기준 1~7 전수 통과 확인.

### `config/base.yaml` 추가
```yaml
nfci_lag_bdays: 3
stlfsi_lag_bdays: 5
```

### `config/indicators/credit.yaml` (신규)
```yaml
series: BAA10Y
percentile_window: 252
map: {theta_low_pct: 0.5, theta_high_pct: 0.9}
```

---

## 3. 핵심 설계 원칙

### inner join 없음
`load_all()`은 시리즈별 dict 반환. 각 시리즈는 가용 최대 기간으로 보존.
공통 기간 교집합은 비교·결합 단계(M4/M5)에서만 수행.
→ vix3m(2006~)이어도 vix(1990~)·sp500tr(1988~) 전 기간 온전히 보존.

### 시장 시세 ffill 금지
sp500tr·vix·vix3m: 거래일 결측 = 에러 신호. 결측은 결측으로 유지.
공표 시리즈(rf·baa10y·hy_oas·nfci·stlfsi): ffill 허용 (정상 정보집합).

### 주간 지표 발표 시차 처리 (룩어헤드 차단 핵심)
- NFCI: 금요일 기준치 → 다음 주 수요일 공표 = **+3 영업일**
- STLFSI4: 불확실. 보수적으로 **+5 영업일** (over-shift 안전)
- BDay 한계(공휴일 미인식): lag_bdays를 실제 거리 이상으로 설정해 over-shift 보장
- reindex 후 ffill 단계에서 비거래일 공표일 → 다음 거래일 자동 push (추가 over-shift)
- 구현 주의: `s.dropna()` 후 시프트 (union-index 중복 방지)

### 캐시 원칙
`raw_prices.parquet`, `raw_fred.parquet`: 시차 처리 이전 raw만 저장.
`apply_weekly_lag`는 캐시 읽은 후 config lag로 적용. 시프트값 캐시 금지.

---

## 4. 완료 기준 실행 결과

### 시리즈별 감사 리포트

| 시리즈 | 시작 | 종료 | 유효관측 | 결측 | 이상치 |
|---|---|---|---|---|---|
| sp500tr | 1988-01-04 | 2026-05-29 | 9,674 | 0 | 없음 |
| vix | 1990-01-02 | 2026-05-29 | 9,169 | 505 | 없음 |
| vix3m | 2006-07-17 | 2026-05-29 | 4,999 | 4,675 | 없음 (짧은 표본 = 사실) |
| rf | 1988-01-04 | 2026-05-29 | 9,674 | 0 | 없음 |
| baa10y | 1986-01-02 | 2026-05-29 | 10,101 | 0 | 없음 (최솟값 1.16, 음수 0건) |
| hy_oas | 2023-05-30 | 2026-05-29 | 753 | 8,921 | ⚠ FRED rolling 3년 윈도우 |
| nfci | 1988-01-06 | 2026-05-29 | 9,672 | 2 | 없음 |
| stlfsi | 1994-01-07 | 2026-05-29 | 8,152 | 1,522 | 없음 |

### 나머지 기준

| 기준 | 결과 |
|---|---|
| VXO 미혼입·^SP500TR 총수익·^GSPC 미사용·TEDRATE 미사용 | ✓ |
| 시장 시세 ffill 미적용 (vix3m 시작 전 100% NaN) | ✓ |
| NFCI +3 영업일 시프트, 공표일 이전 NaN 보장 | ✓ |
| 비거래일 공표 21건 → 다음 거래일 자동 반영 | ✓ |
| 캐시 재로드 완전 재현, inner join 없음 | ✓ |

---

## 5. 주요 발견 및 결정 사항

### hy_oas 데이터 단축 — FRED rolling 3년 윈도우 정책

**증상:** BAMLH0A0HYM2가 2023-05-30~만 반환.

**원인 규명:** FRED 2026년 4월부터 BAML 계열 전체에 rolling 3년 윈도우 정책 적용.

| 경로 | 결과 |
|---|---|
| fredgraph.csv 직접 확인 | 2023-05-30~ (파라미터 무시됨) |
| BAML 계열 4개 시리즈 교차 확인 | 시작일·종료일·행 수 완전 동일 → FRED 서버 측 일괄 처리 |
| FRED API (키 보유) | 동일 결과 |
| ALFRED vintage `realtime_end=2023-01-01` | "does not exist in ALFRED" |

→ 1996~ 전체 이력 모든 경로에서 획득 완전 불가. **영구·진행성 제약.**

### 신용 대표 신호 결정 과정

| 단계 | 후보 | 기각 사유 |
|---|---|---|
| 초안 | HY OAS (BAMLH0A0HYM2) | 데이터 소멸 |
| 1차 | NFCI | 합성 금융여건지수 — 변동성·위험선호 성분 혼재, 독립성 분석 오염 위험 |
| **최종** | **BAA10Y** | 단일 신용 스프레드(순수성), FRED 무료·1986~·일간·재현 가능 |

**사전 등록 일탈 사유:** 데이터 가용성 기반 불가피 대체 (결과 기반 아님).

**BAA10Y 제약 (명시):** IG 하단이라 위기 반응 폭 HY OAS보다 작음 → 신용 방어 강도 과소평가 가능. 독립성·순수성 우선으로 수용.

### NFCI·STLFSI4 위치
사전 등록된 보조·대안 신호로 유지. 제거 금지.
독립성 분석(M4)에서 신용 신호와의 상관 측정에 활용 예정.

### BAA10Y 이상치 규칙
두 수익률의 차라 이론상 음수 가능 (hy_oas OAS와 다름).
실측: 1986~2026 전 기간 음수 0건, 최솟값 1.16 (1989-03-20, 수익률 곡선 역전기).
`validate()`: 음수 발생 건수 리포트만. 에러 중단 없음.

---

## 6. Cleanup 내용

| 삭제 항목 | 사유 |
|---|---|
| `_get_fred_api_key()` | `_fetch_hy_oas_full`만 사용, 함께 삭제 |
| `_fetch_hy_oas_full()` | FRED API도 2023~만 반환 — 동작하지 않는 사문 코드 |
| `load_fred()` api_key 분기 | 위 두 함수 삭제 후 불필요 |
| `import os` | 미사용 |
| `.env.example` | FRED_API_KEY 경로 삭제 후 사문 |
| `load_dotenv` 호출 | 동일 |
| `python-dotenv` 의존성 | 동일 |

---

## 7. 데이터 시리즈 요약

| 시리즈 | 소스 | 기간 | 역할 |
|---|---|---|---|
| sp500tr | yfinance | 1988~ | 벤치마크 |
| vix | yfinance | 1990~ | 변동성 대표 |
| vix3m | yfinance | 2006~ | 변동성 기간구조 (보조) |
| rf | FRED (DTB3) | 1988~ | 무위험수익률 (연율%) |
| baa10y | FRED (BAA10Y) | 1986~ | **신용 대표** (Moody's Baa − 10Y) |
| hy_oas | FRED (BAMLH0A0HYM2) | 2023-05-30~ | 신용 보조 (최근 robustness 전용) |
| nfci | FRED (NFCI) | 1971~ | 신용 등록 보조/대안 (주간, +3 BDay) |
| stlfsi | FRED (STLFSI4) | 1993~ | 신용 등록 보조 (주간, +5 BDay) |
