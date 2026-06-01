"""
src/data_loader.py — 데이터 레이어

원칙 (PROJECT_PLAN.md 4.2, DEFINITIONS_AND_CONVENTIONS.md):
  - load_all() → dict[str, pd.Series]. inner join 없음.
    "단독 평가=각자 최대 기간, 비교·결합만 공통 기간."
  - 캐시는 raw(시차 처리 전)만 저장. apply_weekly_lag는 캐시 읽은 후 적용.
  - 시장 시세(sp500tr·vix·vix3m): ffill 금지 — 결측은 결측으로 유지·보고.
  - 공표 시리즈(rf·hy_oas·nfci·stlfsi): 공표된 마지막 값 유지(ffill) = 정상 정보집합.

[신용 정보원 대표 신호 = BAA10Y]
  BAA10Y(Moody's Baa − 10Y Treasury): FRED, 1986~, 일간, 무료, ICE rolling 제약 없음.
  NFCI보다 순수한 단일 신용 스프레드 → 독립성 분석 오염 없음.
  HY OAS 대비 위기 반응 폭 작음(IG 하단) → 신용 방어 강도 과소평가 가능. 수용.
  hy_oas: FRED rolling 3년 윈도우로 2023-05-30~만 가용 → 보조(최근 robustness 전용).
  NFCI·STLFSI4: 사전 등록된 보조·대안 신호로 유지. 제거 금지.
  상세 근거: config/indicators/credit.yaml 주석 참고.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent

# 레포 루트의 .env 로드 (있으면 적용, os.environ 이미 설정된 값은 유지)
load_dotenv(_ROOT / ".env", override=False)

CACHE_DIR = _ROOT / "data"
PRICES_CACHE = CACHE_DIR / "raw_prices.parquet"
FRED_CACHE   = CACHE_DIR / "raw_fred.parquet"

# 금지: ^GSPC(가격지수), VXO 계열
_YFINANCE_TICKERS: dict[str, str] = {
    "sp500tr": "^SP500TR",  # 총수익지수. ^GSPC 절대 금지. auto_adjust 사실상 무효.
    "vix":     "^VIX",      # 신방식 1990~. VXO 혼입 금지.
    "vix3m":   "^VIX3M",   # 2007~. 짧은 표본은 사실 — 채우지 않음.
}

# 금지: TEDRATE(TED 스프레드, LIBOR 폐지로 단종)
_FRED_IDS: dict[str, str] = {
    "rf":     "DTB3",          # 연율 % 금리. 일간 rf 변환(÷100÷252)은 backtest.py(M2) 담당.
    "baa10y": "BAA10Y",        # 【신용 대표】Moody's Baa − 10Y Treasury, 1986~, 일간.
    # hy_oas: 보조. FRED rolling 3년 윈도우(2026-04~)로 2023-05-30~만 가용.
    # 최근 구간 robustness·향후 재검증 전용. 메인 분석 미사용.
    "hy_oas": "BAMLH0A0HYM2",
    "nfci":   "NFCI",          # 【신용 등록 보조/대안】주간. 시카고 연준. 발표 시차 필수.
    "stlfsi": "STLFSI4",       # 【신용 등록 보조】주간. 세인트루이스 연준.
}

# hy_oas 기대 시작 연도 — 이보다 늦으면 경고
_HY_OAS_EXPECTED_START_YEAR = 1996


def _extract_close(raw: pd.DataFrame, ticker: str) -> pd.Series:
    """
    yfinance 1.x는 단일 티커도 (Price, Ticker) MultiIndex 컬럼을 반환한다.
    Close 시리즈를 안전하게 추출하고 실제 컬럼 구조를 출력해 검증.
    """
    print(f"  [{ticker}] 실제 컬럼 구조: {raw.columns.tolist()[:4]}...")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        if "Close" not in raw.columns:
            raise ValueError(f"{ticker}: 'Close' 없음. 컬럼: {raw.columns.tolist()}")
        close = raw["Close"]
    return close


def load_prices(cfg: dict, force_refresh: bool = False) -> pd.DataFrame:
    """
    yfinance에서 sp500tr·vix·vix3m 수집.
    · 시장 시세: ffill 금지.
    · 캐시: raw close만 저장 (시차 처리 전).
    """
    if not force_refresh and PRICES_CACHE.exists():
        print(f"[캐시 사용] {PRICES_CACHE}")
        return pd.read_parquet(PRICES_CACHE)

    print("[yfinance 다운로드]")
    frames: list[pd.Series] = []
    for key, ticker in _YFINANCE_TICKERS.items():
        raw = yf.download(ticker, start="1980-01-01", auto_adjust=True, progress=False)
        close = _extract_close(raw, ticker).rename(key)
        frames.append(close)

    df = pd.concat(frames, axis=1)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PRICES_CACHE)
    print(f"[캐시 저장] {PRICES_CACHE}")
    return df


def _get_fred_api_key() -> str | None:
    """
    FRED API 키를 os.environ에서 읽는다.
    .env 파일은 모듈 로드 시 load_dotenv()로 이미 os.environ에 반영됨.
    키가 있으면 반환, 없으면 None.
    """
    return os.environ.get("FRED_API_KEY") or None


def _fetch_hy_oas_full(api_key: str) -> pd.Series:
    """
    FRED API (ALFRED vintage 경로)로 BAMLH0A0HYM2 전체 이력 획득.
    api_key: FRED 등록 API 키 (.env의 FRED_API_KEY).
    OAS는 소급 개정이 드물어 latest vintage 사용.
    반환: DatetimeIndex(일간), Series명 'hy_oas'.
    """
    import requests as _req

    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id=BAMLH0A0HYM2"
        f"&observation_start=1990-01-01"
        f"&file_type=json"
        f"&api_key={api_key}"
    )
    r = _req.get(url, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(
            f"FRED API 오류 (HTTP {r.status_code}): {r.text[:200]}"
        )
    data = r.json()
    if "observations" not in data:
        raise RuntimeError(f"FRED API 응답에 'observations' 없음: {str(data)[:200]}")

    obs = data["observations"]
    dates  = pd.to_datetime([o["date"] for o in obs])
    values = pd.to_numeric([o["value"] for o in obs], errors="coerce")
    s = pd.Series(values, index=dates, name="hy_oas")
    s = s.dropna()
    print(
        f"  [FRED API:BAMLH0A0HYM2(hy_oas)] "
        f"{s.index.min().date()} ~ {s.index.max().date()}, n={len(s)}"
    )
    return s


def load_fred(cfg: dict, force_refresh: bool = False) -> pd.DataFrame:
    """
    FRED에서 rf·hy_oas·nfci·stlfsi raw 수집.
    · 캐시: 시차 처리 전 raw만 저장.
    · STLFSI4: ID 실재 확인 실패 시 즉시 중단·보고.
    · hy_oas: FRED_API_KEY 있으면 ALFRED vintage로 전체 이력(1996~) 획득.
              없으면 fredgraph.csv rolling 3년 윈도우(2023-05-30~)로 폴백 + 경고.
    """
    if not force_refresh and FRED_CACHE.exists():
        print(f"[캐시 사용] {FRED_CACHE}")
        return pd.read_parquet(FRED_CACHE)

    api_key = _get_fred_api_key()
    if api_key:
        print(f"[FRED API 키 감지] BAMLH0A0HYM2 전체 이력 획득 경로 사용")
    else:
        print(
            "[FRED_API_KEY 없음] hy_oas는 rolling 3년 윈도우(2023-05-30~)로 로드됩니다.\n"
            "  전체 이력(1996~)이 필요하면 .env 파일에 FRED_API_KEY를 설정하세요.\n"
            "  (.env.example 참고 — https://fred.stlouisfed.org/docs/api/api_key.html)"
        )

    print("[FRED 다운로드]")
    frames: list[pd.Series] = []
    for key, fred_id in _FRED_IDS.items():
        # hy_oas: API 키 있으면 전체 이력 경로
        if key == "hy_oas" and api_key:
            try:
                col = _fetch_hy_oas_full(api_key)
                frames.append(col)
                continue
            except Exception as e:
                warnings.warn(
                    f"FRED API hy_oas 전체 이력 실패: {e}\n"
                    "rolling 3년 윈도우로 폴백합니다.",
                    stacklevel=2,
                )

        try:
            s = web.DataReader(fred_id, "fred", start="1950-01-01")
        except Exception as e:
            msg = f"FRED {fred_id}({key}) 조회 실패: {e}"
            if key == "stlfsi":
                raise RuntimeError(
                    f"[중단] {msg}\n"
                    "STLFSI4 ID가 변경됐을 수 있음. FRED 확인 후 보고 요망."
                ) from e
            raise RuntimeError(msg) from e

        if s.empty:
            msg = f"FRED {fred_id}({key}) 빈 데이터 반환"
            if key == "stlfsi":
                raise RuntimeError(f"[중단] {msg}. FRED ID 확인 요망.")
            warnings.warn(msg)
            continue

        col = s.iloc[:, 0].rename(key)
        start_yr = col.dropna().index.min().year
        print(
            f"  [FRED:{fred_id}({key})] "
            f"{col.dropna().index.min().date()} ~ {col.dropna().index.max().date()}, "
            f"n={col.dropna().count()}"
        )

        if key == "hy_oas" and start_yr > _HY_OAS_EXPECTED_START_YEAR:
            warnings.warn(
                f"[hy_oas 보조] BAMLH0A0HYM2 실제 시작: "
                f"{col.dropna().index.min().date()} "
                f"(FRED rolling 3년 윈도우, 2026-04~). "
                "메인 신용 신호는 BAA10Y. hy_oas는 최근 구간 robustness 전용.",
                stacklevel=2,
            )

        frames.append(col)

    df = pd.concat(frames, axis=1)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FRED_CACHE)
    print(f"[캐시 저장] {FRED_CACHE}")
    return df


def apply_weekly_lag(
    s: pd.Series,
    lag_bdays: int,
    trading_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    주간 시리즈를 발표 시차만큼 시프트 → 거래일 달력에 reindex → ffill.

    ── BDay 한계: 미국 공휴일 미인식 ──────────────────────────────────────
    pd.tseries.offsets.BusinessDay는 월~금만 세고 미국 연방 공휴일을 모른다.
    공휴일이 기준일~공표일 사이에 끼면 실제 공표일보다 이른 날짜로 시프트될 수
    있다(under-shift → 룩어헤드 누수).

    설계 선택: lag_bdays를 실제 영업일 거리 이상(≥)으로 설정해 over-shift를
    보장한다. config 주석 참고.
    reindex 후 ffill 단계에서 시프트된 날짜가 비거래일이면 자동으로 다음
    거래일로 밀림 → 추가 over-shift 효과 ✓. under-shift는 절대 금지.
    ─────────────────────────────────────────────────────────────────────────

    구현 주의:
      s가 concat된 DataFrame의 컬럼이면 union 인덱스(다른 시리즈 날짜 포함)를
      들고 있어 BDay 시프트 후 중복 인덱스가 생긴다.
      반드시 .dropna()로 실제 값만 걸러낸 뒤 시프트한다.

    흐름:
      1. dropna()                 → union-index NaN 제거, 실제 공표 기준일만 남김
      2. index + BDay(lag_bdays)  → 근사 공표일
      3. reindex(trading_index)   → 비거래일 공표일은 NaN (ffill로 다음 거래일 push)
      4. ffill()                  → 공표일 ~ 다음 공표일까지 마지막 값 유지 (정상 정보집합)
         공표일 이전 구간 = NaN 유지 (룩어헤드 차단)
    """
    # Step 1: 실제 값만 추출 (union-index NaN 제거)
    s_clean = s.dropna()

    # Step 2: 근사 공표일로 인덱스 이동
    shifted = s_clean.copy()
    shifted.index = s_clean.index + pd.tseries.offsets.BusinessDay(lag_bdays)

    # Step 3: 거래일 달력으로 확장 — 비거래일 공표일 → NaN
    result = shifted.reindex(trading_index)

    # Step 4: ffill — 공표된 마지막 값 유지 (누수 아님)
    result = result.ffill()

    return result


def validate(series_dict: dict[str, pd.Series]) -> dict:
    """
    시리즈별 시작·종료일·결측·중복·이상치 검증 (DEFINITIONS_AND_CONVENTIONS.md 4.3).
    VIX3M 2007~ 짧은 표본: 오류 아님 — 사실로 기록.
    hy_oas 2023~: ICE 라이선싱 이슈로 단축됨 — 사실로 기록.
    """
    report: dict[str, dict] = {}
    for key, s in series_dict.items():
        valid = s.dropna()
        info: dict = {
            "start":             str(valid.index.min().date()) if len(valid) else "N/A",
            "end":               str(valid.index.max().date()) if len(valid) else "N/A",
            "n_obs":             int(len(valid)),
            "n_missing":         int(s.isna().sum()),
            "n_duplicate_dates": int(s.index.duplicated().sum()),
            "anomalies":         [],
        }

        # 이상치 검증 (DEFINITIONS 4.3)
        if key == "vix":
            bad = valid[valid <= 0]
            if len(bad):
                info["anomalies"].append(f"VIX≤0: {len(bad)}건")

        if key == "sp500tr":
            ret = s.pct_change().dropna()
            bad = ret[ret.abs() > 0.5]
            if len(bad):
                info["anomalies"].append(f"|일간수익|>50%: {len(bad)}건")

        if key == "baa10y":
            bad = valid[valid < 0]
            if len(bad):
                info["anomalies"].append(f"BAA10Y<0: {len(bad)}건")

        if key == "hy_oas":
            bad = valid[valid < 0]
            if len(bad):
                info["anomalies"].append(f"HY_OAS<0: {len(bad)}건")
            if len(valid) > 0 and valid.index.min().year > _HY_OAS_EXPECTED_START_YEAR:
                info["anomalies"].append(
                    f"⚠ 기대 시작({_HY_OAS_EXPECTED_START_YEAR}~)보다 늦음 — FRED rolling 3년 윈도우"
                )

        if not [a for a in info["anomalies"] if not a.startswith("⚠")]:
            # 이상치(오류) 없음; 경고만 있거나 전혀 없는 경우
            if not info["anomalies"]:
                info["anomalies"] = ["없음"]

        report[key] = info
    return report


def load_all(cfg: dict, force_refresh: bool = False) -> dict[str, pd.Series]:
    """
    전체 파이프라인. dict[str, pd.Series] 반환.

    ── 핵심 원칙 ─────────────────────────────────────────────────────────────
    inner join 없음: 각 시리즈는 가용 최대 기간으로 보존.
    가용 밖 날짜 = NaN (잘라내지 않음).
    공통 기간 교집합은 비교·결합 단계(M4/M5)에서만 수행.

    시차 처리: raw 캐시 읽은 후 config lag 적용.
    시프트된 값은 캐시에 저장하지 않는다.
    ──────────────────────────────────────────────────────────────────────────
    """
    prices = load_prices(cfg, force_refresh)
    fred   = load_fred(cfg, force_refresh)

    # 거래일 앵커: sp500tr의 거래일 인덱스 (미 증시 기준)
    trading_index: pd.DatetimeIndex = prices["sp500tr"].dropna().index

    out: dict[str, pd.Series] = {}

    # 시장 시세: ffill 없음 (결측은 결측으로 유지)
    for key in ("sp500tr", "vix", "vix3m"):
        out[key] = prices[key].reindex(trading_index)

    # 일간 FRED: ffill 허용 (공표된 마지막 값 유지 = 정상 정보집합)
    for key in ("rf", "baa10y", "hy_oas"):
        out[key] = fred[key].reindex(trading_index).ffill()

    # 주간 FRED: raw → config lag 시차 처리 → ffill
    # 캐시에는 raw만 저장됨 — 시프트값 캐시 금지
    # apply_weekly_lag 내부에서 .dropna()로 union-index NaN 제거 후 시프트
    out["nfci"]   = apply_weekly_lag(
        fred["nfci"],   cfg.get("nfci_lag_bdays", 3),   trading_index
    )
    out["stlfsi"] = apply_weekly_lag(
        fred["stlfsi"], cfg.get("stlfsi_lag_bdays", 5), trading_index
    )

    return out
