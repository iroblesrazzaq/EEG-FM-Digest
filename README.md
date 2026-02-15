# EEG FM Digest

Monthly arXiv-based digest for EEG Foundation Model papers:
1) Stage 1: keyword retrieval (title+abstract) within relevant arXiv categories
2) Stage 2: Gemini triage on abstract
3) Stage 3: PDF download + text extraction + Gemini deep summary
4) Publish: static site in `/docs` with one page per month

## Setup
```bash
pip install -e ".[dev]"
export GEMINI_API_KEY="..."
export GEMINI_MODEL_TRIAGE="gemini-2.0-flash"
export GEMINI_MODEL_SUMMARY="gemini-2.0-pro"
```

You can also use `GOOGLE_API_KEY` instead of `GEMINI_API_KEY`.

## Main run command
Run the pipeline for a specific month (`YYYY-MM`):
```bash
python -m eegfm_digest.run --month 2025-01
```

Useful options:
```bash
python -m eegfm_digest.run --month 2025-01 --max-candidates 300 --max-accepted 60
python -m eegfm_digest.run --month 2025-01 --include-borderline
python -m eegfm_digest.run --month 2025-01 --no-pdf
python -m eegfm_digest.run --month 2025-01 --force
```

## How to test
Run all tests:
```bash
pytest -q
```

Run tests by component:
```bash
pytest -q tests/test_arxiv.py
pytest -q tests/test_schema_paths.py
pytest -q tests/test_render_site.py
```

## Component-level sanity checks
### 1) arXiv retrieval only
```bash
python - <<'PY'
from eegfm_digest.arxiv import fetch_month_candidates
rows = fetch_month_candidates(max_candidates=50, month="2025-01", rate_limit_seconds=2)
print(f"candidates={len(rows)}")
print(rows[0]["arxiv_id_base"] if rows else "none")
PY
```

### 2) Triage path only (with a fake/stub model)
Use `tests/test_schema_paths.py` for a no-network triage repair-path test.

### 3) Summary path only (with a fake/stub model)
Use `tests/test_schema_paths.py` fallback summary test to validate JSON-repair + fallback behavior.

### 4) HTML rendering only
```bash
pytest -q tests/test_render_site.py
```

## Where outputs go
For a month like `2025-01`, pipeline artifacts are written to:
- `outputs/2025-01/arxiv_raw.json`
- `outputs/2025-01/triage.jsonl`
- `outputs/2025-01/papers.jsonl`
- `outputs/2025-01/digest.json`

Site artifacts are written to:
- `docs/index.html`
- `docs/digest/2025-01/index.html`
- `docs/digest/2025-01/papers.json`
- `docs/.nojekyll`
