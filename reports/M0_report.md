# M0 실행 보고서 — 환경 구성

| 항목 | 내용 |
|---|---|
| 날짜 | 2026-06-01 |
| 커밋 | fbbea92 |

---

## 1. 목표

GitHub Codespace에서 재현 가능한 연구 환경을 구성한다.
devcontainer, requirements.txt, 디렉터리 골격, config/base.yaml, .gitignore

---

## 2. 구현 내용

### `.devcontainer/devcontainer.json`
- 이미지: `mcr.microsoft.com/devcontainers/python:3.11`
  - 실제 런타임 3.11.15와 일치. b530dd6의 3.12 변경은 런타임 사실과 불일치 → 원복
- Node: `ghcr.io/devcontainers/features/node:1` (베이스 스왑 아닌 feature 방식)
- `postCreateCommand`: pip install + npm install -g @anthropic-ai/claude-code
- vscode extensions: ms-python.python, ms-toolsai.jupyter

### `requirements.txt`
```
pandas>=2.0,<3.0  numpy>=1.24  yfinance>=0.2.40  pandas-datareader>=0.10
scipy>=1.11  statsmodels>=0.14  matplotlib>=3.7  pyarrow>=14
pyyaml>=6  jupyter  pytest
```

### `config/base.yaml` (DEFINITIONS_AND_CONVENTIONS.md 5절 기본값)
```yaml
period.start: "1990-01-01"  benchmark: "SP500TR"
sigma_target: 0.12  w_max: 1.0  rebalance_band: 0.05
cost_bps: 2.0  borrow_spread_bps: 50  leverage_max: 2.0
exec_lag: 1  seed: 42
```

### 디렉터리 골격 (PROJECT_PLAN.md 3절)
`src/` `src/indicators/` `data/` `notebooks/` `results/` `tests/` `config/` `config/indicators/`

### `.gitignore`
`data/` `(!data/.gitkeep)` `__pycache__/` `.venv/` `.env` `.DS_Store`

---

## 3. 완료 기준 실행 결과

| 기준 | 결과 |
|---|---|
| `pip install -r requirements.txt` | 성공 (exit 0) |
| `import yfinance, pandas_datareader` | 성공 |
| `^VIX` 수신 (yfinance) | 2026-05-29 Close: **15.32** |
| `^SP500TR` 수신 (yfinance, 총수익지수) | 2026-05-29 Close: **16935.35** |
| `DTB3` 수신 (FRED) | 2026-05-28: **3.6%** |

---

## 4. 결정 사항

- **Python 이미지 버전: 3.11 확정** — 런타임 3.11.15, PROJECT_PLAN.md 2절과 일치
- Node는 베이스 이미지 교체 아닌 devcontainer feature로 추가

---

## 5. 이슈 및 처리

이전 커밋(b530dd6)이 python:3.12로 올렸으나 실제 런타임 3.11.15와 불일치.
PROJECT_PLAN.md 스펙(3.11)과 런타임에 맞춰 3.11로 원복. doc·code 일치 원칙 준수.
