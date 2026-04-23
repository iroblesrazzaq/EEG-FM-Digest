# SPEC.md

## 1) Overview
Implement an incremental “EEG Foundation Model Digest” pipeline.  Historically the pipeline was run once per month; the canonical production mode is now a daily incremental run that re-triages only papers submitted since the last successful run.  The site is still organized one page per month.

Current provider plan:
- GitHub-hosted scheduled and manual runs use Google AI Studio with `gemma-4-31b-it`
- OpenRouter remains a supported fallback/provider option for local or batch runs

### Pipeline stages
- **Stage 1 (Retrieve):** Query arXiv by relevant categories and a high-recall **title+abstract keyword strategy**.
- **Stage 2 (Triage):** A configured **Google AI Studio or OpenRouter model** reviews title+abstract and decides whether it is an **EEG-FM** paper (strict JSON output).
- **Stage 3 (Deep summary):** Download PDF for accepted papers, extract text, and use a stronger configured **Google AI Studio or OpenRouter model** to generate a **deep structured summary** (strict JSON output).
- **Stage 4 (Publish):** Generate a static GitHub Pages site in `docs/` with **one subpage per month** containing paper cards.

### Outputs (per month `YYYY-MM`)
Pipeline artifacts:
- `outputs/YYYY-MM/arxiv_raw.json` — raw candidates from Stage 1
- `outputs/YYYY-MM/triage.jsonl` — one triage decision per candidate
- `outputs/YYYY-MM/papers.jsonl` — one deep summary per accepted paper
- `outputs/YYYY-MM/backend_rows.jsonl` — canonical backend artifact (one merged row per candidate)
- `outputs/YYYY-MM/digest.json` — structured digest index (sections + top picks)
- `data/digest.sqlite` — persistent state for incremental runs

Public site artifacts (GitHub Pages):
- `docs/index.html` — home (latest + archive)
- `docs/digest/YYYY-MM/index.html` — monthly page with paper cards
- `docs/digest/YYYY-MM/papers.json` — monthly structured data (for debugging + future UI)
- `docs/assets/style.css`
- `docs/assets/site.js` (optional; search/filter enhancement)
- `docs/.nojekyll`

## 2) Definitions

### 2.1 What counts as an EEG Foundation Model paper (EEG-FM)
A paper is EEG-FM if:
- EEG is a **primary modality** (EEG / electroencephalography / brainwaves / BCI using EEG),
AND
- The work centers on a **general-purpose pretrained representation/model** intended to transfer/generalize across tasks, datasets, subjects, or settings, often via SSL/pretraining, large-scale training, or “foundation model” framing.

Include borderline papers if plausibly relevant but unclear from abstract (`decision="borderline"`).

Exclude:
- purely supervised single-task EEG models with no pretraining/transfer/generalization framing
- non-EEG modalities only (MEG-only, fMRI-only) unless EEG is central

## 3) Stage 1: arXiv candidate retrieval

### 3.1 Categories to search (recall net)
Filter to candidates that have **any** of these categories (primary or cross-list):
- `q-bio.NC`
- `cs.LG`
- `stat.ML`
- `eess.SP`
- `cs.AI`
- `cs.NE` (optional but recommended)

### 3.2 Keyword strategy (title+abstract)

EEG anchor terms (OR):
- `eeg`
- `electroencephalograph*`
- `brainwave*`

FM / pretraining anchor terms (OR):
- `"foundation model"` OR `"brain foundation model"`
- `pretrain*` / `pre-training` / `pretrained`
- `"self-supervised"` / `"self supervised"`
- `"representation learning"`
- `masked`
- `transfer` OR `generaliz*`

### 3.3 arXiv query implementation (literal strings)
Use 2 queries and union results:

- Query A (explicit FM/pretrain language):
  - `all:(eeg OR electroencephalograph* OR brainwave*) AND all:("foundation model" OR pretrain OR pretrained OR "self-supervised" OR "self supervised")`

- Query B (masked/representation/transfer language):
  - `all:(eeg OR electroencephalograph* OR brainwave*) AND all:("representation learning" OR masked OR transfer OR generaliz*)`

Implementation notes:
- Use arXiv API endpoint `https://export.arxiv.org/api/query`
- Parameters: `search_query`, `start`, `max_results`, `sortBy=submittedDate`, `sortOrder=descending`
- Paginate until fewer than `max_results` returned or `start` exceeds a configured cap.

### 3.4 Time window
For run month `YYYY-MM`, include papers with `published` in:
- `[start_of_month, start_of_next_month)`

Use `published` for inclusion, not `updated`.

### 3.5 Dedupe/versioning
- `arxiv_id_base`: strip version suffix `vN` if present.
- Keep latest version metadata (largest vN) but store under base key.

### 3.6 Rate limiting
- Sleep at least `ARXIV_RATE_LIMIT_SECONDS` (default 2) between paginated API calls.

## 4) Stage 2: Cheap LLM triage on abstract

### 4.1 Inputs to triage model
Provide:
- title
- abstract

### 4.2 Output schema: `TriageDecision`
Write JSON only, validate with `schemas/triage.json`.

Model output keys:
- `decision` in `["accept", "reject", "borderline"]`
- `confidence` in `[0,1]`
- `reasons` (2-4 short evidence-based strings)

Persistence rule:
- pipeline deterministically attaches `arxiv_id_base` outside model output.

Decision logic:
- trust model `decision` as returned (no local decision rewrite policy).

### 4.3 JSON repair
If parse/schema fails:
- retry once with `prompts/repair_json.md`
- if still failing: store error, mark reject with `confidence=0.0` and reason `triage_json_error`

## 5) Stage 3: Deep summary (PDF + extraction + strong LLM model)

### 5.1 Which papers proceed
Proceed for:
- accept
- borderline optionally (cap with `MAX_BORDERLINE_PDFS`)

### 5.2 PDF download
- Save to: `outputs/YYYY-MM/pdfs/{arxiv_id_base}.pdf`
- Rate limit: `PDF_RATE_LIMIT_SECONDS` default 5
- Cache: skip if exists

### 5.3 Text extraction
- Extract to: `outputs/YYYY-MM/text/{arxiv_id_base}.txt`
- Preferred: `pymupdf`
- Fallback chain: `pypdf`, then `pdfminer.six`
- Save extraction metadata: pages, chars, tool, errors

### 5.4 Deep summary input strategy
Summary input always includes:
- `arxiv_id_base`, `title`, `published_date`, `categories`, `abstract`, and triage decision fields.

For extracted paper text:
- Prefer full extracted `fulltext` if token count is within `SUMMARY_MAX_INPUT_TOKENS`.
- If fulltext is too large (or token count fails), send deterministic `fulltext_slices` with fixed keys:
  - `abstract`, `introduction`, `methods`, `results`, `conclusion`, `excerpt`.
- Missing slices are empty strings; `excerpt` is always populated as fallback.

### 5.5 Output schema: `PaperSummary`
Validate with `schemas/summary.json`.
Rules:
- no hallucinations; if absent, set null/unknown
- concise, digest-oriented

### 5.6 JSON repair
Same as triage.

## 6) Digest + Site rendering (deterministic)

### 6.1 Digest JSON
Write `outputs/YYYY-MM/digest.json` including:
- stats (candidates, accepted, summarized)
- `featured_paper` nullable arXiv id; defaults to `null` unless explicitly set via CLI
- `top_picks` ids
- sections mapping

### 6.2 Site pages
Generate:
- `docs/digest/YYYY-MM/index.html` with paper cards grouped by paper_type + top picks
- `docs/digest/YYYY-MM/papers.json` containing all PaperSummary objects for that month
- update `docs/index.html` archive (link list) and latest pointer

Site constraints:
- Static HTML must include all content (no JS required to see cards).
- JS (optional) may add search/filter UI but must not be required.

## 7) Persistence (SQLite)
Tables:
- `papers` (candidate metadata)
- `triage` (per arxiv_id_base)
- `summaries` (per arxiv_id_base)
- `runs` (per month)

Incremental:
- skip already-triaged and already-summarized unless `--force`

## 8) CLI
Monthly backfill / single-month run:
`python -m eegfm_digest.run --month YYYY-MM [--feature-paper ARXIV_ID] [--max-candidates N] [--max-accepted N] [--include-borderline] [--no-pdf] [--no-site] [--force]`

Daily incremental run:
`python -m eegfm_digest.run --daily [--since ISO8601] [--until ISO8601] [--overlap-hours H] [--dry-run] [--max-candidates N] [--max-accepted N] [--include-borderline] [--no-pdf] [--no-site] [--force]`

`--feature-paper`:
- optional explicit featured paper id for that month
- defaults to null when omitted
- this is the only supported way to set the featured paper; the pipeline and GitHub Actions must not auto-select one
- not supported in `--daily` mode

`--no-site` mode:
- still writes all `outputs/YYYY-MM/*` artifacts and updates SQLite.
- skips all `docs/` writes.

Defaults:
- month: previous calendar month
- include-borderline: false
- max-candidates: 500
- max-accepted: 80
- overlap-hours: 6 (daily mode)

## 8a) Daily mode semantics

State file: `data/last_successful_run.json` (git-tracked, version 1).
```json
{
  "last_success_utc": "2026-04-22T14:37:00Z",
  "last_query_end_utc": "2026-04-22T14:37:00Z",
  "papers_fetched": 12,
  "papers_accepted": 3,
  "affected_months": ["2026-04"],
  "run_id": "2026-04-22T14:37:00Z",
  "version": 1
}
```

Window resolution:
- `until = --until` or wall-clock UTC.
- `since = --since`, else `last_query_end_utc - overlap_hours` from the run log, else `until - 24h` (first-ever run, with a warning).
- arXiv query: `QUERY_A` / `QUERY_B` appended with `AND submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM]`.
- Window results are grouped by `published[:7]`; each distinct month triggers a normal `run_month` invocation.  SQLite cache ensures already-triaged papers do not re-invoke the LLM.

Success definition: the run is "successful" iff `run_window` returns without raising `ArxivFetchError` or `LLMRateLimitError`.  On success the run log is overwritten with the new window end.  On failure the run log is left untouched and the next run auto-covers the same window via the overlap.

Never catch `ArxivFetchError` or `LLMRateLimitError` inside `run_window`/`run_month`; daily callers depend on propagation to make the "no run-log advance" decision.

## 9) Configuration (LLM providers)
Env vars:
- `LLM_PROVIDER` (`google` or `openrouter`; default inferred from available API keys)
- `GEMINI_API_KEY`
- `GEMINI_MODEL_TRIAGE`
- `GEMINI_MODEL_SUMMARY`
- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL_TRIAGE`
- `OPENROUTER_MODEL_SUMMARY`
- `ARXIV_RATE_LIMIT_SECONDS`
- `PDF_RATE_LIMIT_SECONDS`
- `OUTPUT_DIR`
- `DATA_DIR`

Optional:
- `GOOGLE_API_KEY` (alias for `GEMINI_API_KEY`)
- `LLM_TEMPERATURE_TRIAGE` default 0.2
- `LLM_TEMPERATURE_SUMMARY` default 0.2
- `LLM_MAX_OUTPUT_TOKENS_TRIAGE` default 1024
- `LLM_MAX_OUTPUT_TOKENS_SUMMARY` default 2048
- `SUMMARY_MAX_INPUT_TOKENS` default 120000

## 10) Testing
- Unit tests: arXiv parsing, month boundaries, dedupe, schema validation, render snapshots
- Fixture-based integration test: cached `arxiv_raw.json`, stubbed LLM outputs

## 11) GitHub Actions
- `.github/workflows/daily-digest.yml`: production incremental workflow.  Cron line starts commented out; enable only after verifying a `workflow_dispatch` round-trip.  Runs `python -m eegfm_digest.run --daily`, then makes two commits per run — outputs + `data/digest.sqlite` first, `data/last_successful_run.json` second — so a run-log advance is never orphaned from the data it describes.
- Workflows must never auto-set `featured_paper`; leave it `null` unless the CLI flag is provided manually.
- Test workflow: runs pytest on push/PR.
- Manual / ad-hoc single-month runs use the CLI locally with `--month` (no scheduled workflow).
