# EEG FM Digest

Monthly arXiv-based digest for EEG Foundation Model papers:
1) Stage 1: keyword retrieval (title+abstract) within relevant arXiv categories
2) Stage 2: LLM triage on title+abstract (Google AI Studio or OpenRouter)
3) Stage 3: PDF download + text extraction + LLM deep summary
4) Publish: static site in `/docs` with one page per month

## Setup
```bash
pip install -e ".[dev]"
export LLM_PROVIDER="google"
export GEMINI_API_KEY="..."
export GEMINI_MODEL_TRIAGE="gemma-4-31b-it"
export GEMINI_MODEL_SUMMARY="gemma-4-31b-it"
export ARXIV_CONNECT_TIMEOUT_SECONDS="10"
export ARXIV_READ_TIMEOUT_SECONDS="90"
export ARXIV_RETRIES="3"
```

OpenRouter remains supported:
```bash
export LLM_PROVIDER="openrouter"
export OPENROUTER_API_KEY="..."
export OPENROUTER_MODEL_TRIAGE="stepfun/step-3.5-flash:free"
export OPENROUTER_MODEL_SUMMARY="stepfun/step-3.5-flash:free"
```

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
python -m eegfm_digest.run --month 2025-01 --no-site
python -m eegfm_digest.run --month 2025-01 --force
```

`--no-site` runs backend-only mode: outputs and SQLite are updated, `docs/` is untouched.

## Batch runs (all months or one month)
Use the batch runner to triage multiple months first, then summarize accepted papers:
```bash
python -m eegfm_digest.batch --config configs/batch_all_months.json
python -m eegfm_digest.batch --config configs/batch_single_month.json
```

The checked-in batch configs still default to OpenRouter, but the runner now also accepts `triage_provider` / `summary_provider` values compatible with Google AI Studio.

Wrapper scripts:
```bash
./scripts/run_batch_all.sh
./scripts/run_batch_month.sh 2025-02
```

How incremental weekly runs work:
- Keep `triage_force=false` and `summary_force=false`.
- Keep `sync_cache_from_outputs=true`.
- Re-run the same month config; previously seen `arxiv_id_base` rows are reused from SQLite / existing JSONL and only new papers trigger LLM calls.

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

Or use `notebooks/arxiv_sanity.ipynb` for interactive query-by-query inspection.

### 2) Triage path only (with a fake/stub model)
Use `tests/test_schema_paths.py` for a no-network triage repair-path test.

### 3) Summary path only (with a fake/stub model)
Use `tests/test_schema_paths.py` fallback summary test to validate JSON-repair + fallback behavior.

Summary input mode is:
- send full extracted `fulltext` when prompt token count is within `SUMMARY_MAX_INPUT_TOKENS`
- otherwise send deterministic `fulltext_slices` fallback (`abstract`, `introduction`, `methods`, `results`, `conclusion`, `excerpt`)

### 4) HTML rendering only
```bash
pytest -q tests/test_render_site.py
```

## Manual triage eval (fish shell)
Build or refresh the frozen gold snapshot from the curated title list:
```fish
eegfm-triage-eval build-gold \
  --db data/digest.sqlite \
  --titles-file tests/fixtures/triage_eval_seed_titles_v1.txt \
  --out tests/fixtures/triage_eval_gold_v1.jsonl
```

Score current DB triage decisions against the gold snapshot (`accept` vs `not_pass`, where `reject+borderline => not_pass`):
```fish
eegfm-triage-eval score \
  --db data/digest.sqlite \
  --gold tests/fixtures/triage_eval_gold_v1.jsonl
```

Equivalent module invocation:
```fish
python -m eegfm_digest.eval_triage score \
  --db data/digest.sqlite \
  --gold tests/fixtures/triage_eval_gold_v1.jsonl
```

## Where outputs go
For a month like `2025-01`, pipeline artifacts are written to:
- `outputs/2025-01/arxiv_raw.json`
- `outputs/2025-01/triage.jsonl`
- `outputs/2025-01/papers.jsonl`
- `outputs/2025-01/backend_rows.jsonl` (canonical backend artifact; one merged row per candidate)
- `outputs/2025-01/digest.json`

Site artifacts are written to:
- `docs/index.html`
- `docs/digest/2025-01/index.html`
- `docs/digest/2025-01/papers.json`
- `docs/.nojekyll`

`backend_rows.jsonl` row shape:
- paper metadata (`arxiv_id`, `arxiv_id_base`, `version`, `title`, `summary`, `authors`, `categories`, `published`, `updated`, `links`)
- `triage` (`decision`, `confidence`, `reasons`)
- `paper_summary` (full `PaperSummary` JSON or `null`)
- `pdf` (`downloaded`, `pdf_path`, `text_path`, `extract_meta`)
