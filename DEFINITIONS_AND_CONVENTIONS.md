# 정의 · 규약 참고 (Definitions & Conventions)

> 모든 대화·코드가 동일한 수식·규약을 쓰도록 고정하는 참고 문서. 여기서 벗어난 정의를 임의로 쓰지 않는다.
> 세 정보원(변동성·신용·추세)은 **대칭**이며, 모두 동일한 신호 인터페이스를 따른다.

## 0. 공통 신호 인터페이스

모든 신호 모듈은 `signal(data, cfg) -> w_target` 형태로, **목표 주식노출 시계열** `w_target ∈ [0, w_max]`를 반환한다. 이 인터페이스 덕분에 어떤 정보원이든 동일한 백테스트 엔진에 꽂아 공정 비교한다.

- 입력 `data`: 정합된 가격·지표 시계열 (거래일 인덱스)
- 출력 `w_target`: t일 종가 정보로 산출, **t+1 체결** (룩어헤드 금지)
- 주간 지표 입력은 **발표 시차만큼 시프트**한 뒤 사용

## 1. 정보원별 신호 수식

### 1.1 변동성
**σ_t (연율화 소수)** — config로 선택:
- `vix`: `σ_t = VIX_t / 100`
- `realized`: 일간수익률 N일 표준편차 × √252 (또는 EWMA, λ는 config)
- `blend`: 위 둘의 가중 평균 (가중치 config)

**목표비중:** `w = clip( σ_target / σ_t , 0 , w_max )`

**기간구조(보조, 2007~):** `ts_t = VIX_t / VIX3M_t`. `ts_t > 1`(백워데이션)이면 방어. 매핑은 아래 1.4의 단조 규칙 사용.

### 1.2 신용
스트레스 지표 `s_t` (HY OAS 또는 NFCI 등, 발표 시차 시프트 적용). 높을수록 스트레스.
- 트레일링 윈도우 W에서 백분위 `p_t = percentile_rank(s_t)` 또는 z-score `z_t`.
- 목표비중은 1.4의 **단조 감소 매핑**으로 `p_t`(또는 `z_t`) → `w ∈ [0, w_max]`.

### 1.3 추세
- **200일선:** `w = w_max` if `P_t > MA200_t` else `w_floor` (기본 `w_floor=0`; 변형으로 0.5).
- **12M TSMOM:** `w = w_max` if `(P_t / P_{t-252} − 1) > 0` else `w_floor`.
- 변형: 추세 강도/거리로 [0, w_max] 스케일.

### 1.4 방어 신호 공통 매핑 (변동성 역타게팅 외)
지표값 `x_t`(높을수록 방어)를 단조 규칙으로 목표비중에 매핑. **config로 규칙·파라미터 지정, 그리드·민감도 대상.**
- 임계 방식: `x_t < θ_low → w_max`, `x_t > θ_high → 0`, 사이는 선형 보간.
- 또는 백분위 선형: `w = clip( w_max · (1 − (p_t − p0)/(1 − p0)) , 0 , w_max )`.
> 절대값 드리프트를 피하기 위해 **상대화(백분위/z-score)** 를 기본으로 한다.

## 2. 결합 규칙 (Phase 2)

Gate 1을 통과한 신호 `w^(i)` 들을 단순 결합. **신호 가중치 데이터 최적화 금지.**
- 기본: **등가중 평균** `w_comb = mean_i( w^(i) )`
- 변형: 중앙값, 또는 **투표**(방어 신호 과반 시 노출 축소)
- 결합 후 리밸런스 밴드·t+1 체결 동일 적용

## 3. 리밸런스·체결·레버리지

- **리밸런스 밴드:** `|w_target − w_current| > band` 일 때만 거래 (band는 config, 회전율 제어).
- **체결:** t일 종가 정보로 `w_target` 산출 → t+1일 수익률에 적용.
- **레버리지 되감기(Phase 3):** 전략 변동성 `σ_strat < σ_SP`이면 노출에 `L = clip(σ_SP / σ_strat, 1, L_max)` 적용. 차입분 `(L−1)`은 **차입금리(= rf + 스프레드)로 비용 차감.** 무레버리지(`w_max=1.0`, `L=1`)가 1차 기준.

## 4. 성과 지표 (모두 net 기준, 일간수익률 기반)

표기: `r_t`=전략 일간수익률, `b_t`=벤치마크(S&P500 총수익) 일간수익률, `rf_t`=무위험 일간수익률, `N`=거래일 수, 연율화 계수=252.

- **CAGR** = (최종자산/초기자산)^(252/N) − 1
- **연율 변동성** = std(r_t) × √252
- **Sharpe** = [mean(r_t − rf_t) × 252] / [std(r_t) × √252]
- **Sortino** = [mean(r_t − rf_t) × 252] / [DD × √252], `DD = sqrt(mean(min(r_t − MAR, 0)^2))`, 일간 목표수익 `MAR = 0`(기본). 변경 시 명시.
- **MDD** = min_t (자산_t / 직전최고_t − 1) (음수)
- **Calmar** = CAGR / |MDD|
- **상/하방 캡처 (월간 수익률):**
  - 상방 = (b_월>0 인 달들에서 전략 복리수익) / (그 달들 벤치마크 복리수익)
  - 하방 = (b_월<0 인 달들에서 전략 복리수익) / (그 달들 벤치마크 복리수익)
  - 목표: 하방 < 70%, 상방 > 85%
- **연 회전율** = 한 해 각 리밸런스의 one-way 회전율 합. 2자산(주식/현금)에서 one-way = |Δw_equity|.
- **시장체류시간(평균 노출)** = mean(w_t)

## 5. Config 스키마 (YAML)

`config/base.yaml` (공통)
```yaml
period:        {start: "1990-01-01", end: null}
benchmark:     "SP500TR"          # 가격지수 아님, 총수익
sigma_target:  0.12               # 연율 목표변동성(변동성·결합 공통 타깃)
w_max:         1.0                # 레버리지 상한(신호 단위)
rebalance_band: 0.05              # 비중 차이 임계
cost_bps:      2.0                # 거래 노션널당 bp
borrow_spread_bps: 50             # 레버리지 차입 스프레드(rf 위)
leverage_max:  2.0                # 되감기 L 상한
exec_lag:      1                  # t+1 체결
seed:          42
```

`config/indicators/*.yaml` (신호별 — 예)
```yaml
# volatility.yaml
vol_estimator: "blend"            # vix | realized | blend
realized_lookback: 21
ewma_lambda:   0.94
blend_weights: {vix: 0.5, realized: 0.5}

# credit.yaml
series:        "BAMLH0A0HYM2"      # HY OAS (대안 NFCI)
publish_lag_days: 1               # 주간 지표면 발표 시차
percentile_window: 252
map: {theta_low_pct: 0.5, theta_high_pct: 0.9}

# trend.yaml
rule:          "ma200"            # ma200 | tsmom12
w_floor:       0.0
```

## 6. 네이밍 규약

- 시리즈: `sp500tr`, `vix`, `vix3m`, `rf`, `hy_oas`, `nfci`, `ief`, `tlt`
- 컬럼: 수익률 `ret`, 목표비중 `w`, 자산곡선 `equity`, 낙폭 `drawdown`, 회전율 `turnover`
- 신호 출력 컬럼: `w_vol`, `w_credit`, `w_trend`, 결합 `w_comb`
- 결과 산출물: `results/{config_name}_{metric|curve|grid|corr|attr}.{png|csv}`

## 7. 벤치마크·대조군 (고정)

1. S&P500 buy&hold (총수익)
2. **동일 평균노출 정적 배분** — 전략의 실현 `mean(w_t)`와 같은 고정비중. ← 아노말리 판정의 기준
3. (비교) 최고 단일 신호 — 결합의 정당화 기준 (Phase 2)
4. (참고) 60/40

## 8. 데이터 함정 (재확인)

- 벤치마크는 **총수익**(`^SP500TR`/`SPY` adj), 가격지수(`^GSPC`) 아님
- `^VIX`(신방식, 1990~)만, VXO 혼용 금지
- `^VIX3M`은 2007~ → 기간구조 피처 표본 축소
- 주간 지표(NFCI·STLFSI) **발표 시차 시프트** 필수, TED(`TEDRATE`) 단종 → 금지
- 표본 길이 비대칭은 사실: **단독 평가=각자 최대 기간, 비교·결합=공통 기간**
- 거래일 inner join, 신호 누수 유발 forward-fill 금지
