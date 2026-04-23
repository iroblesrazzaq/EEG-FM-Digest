# AGENTS.md
# Instructions for Codex (development-time). Not runtime “agents”.

## Goal
Implement SPEC.md using:
- arXiv API (Atom feed)
- OpenRouter chat completions API
- SQLite for state
- deterministic HTML site output in docs/

## Key choices (do not deviate without updating SPEC.md)
- LLM provider: OpenRouter
- Publishing: GitHub Pages from /docs (static HTML, no-Jekyll)

## Environment variables
- OPENROUTER_API_KEY
- OPENROUTER_MODEL_TRIAGE
- OPENROUTER_MODEL_SUMMARY

## Implementation rules
- Keep pipeline staged: fetch -> triage -> (pdf+extract) -> summarize -> render -> publish.
- PDFs are downloaded ONLY after acceptance/borderline triage.
- All LLM outputs must validate against JSON Schemas in /schemas.
- If JSON invalid: retry once using prompts/repair_json.md; then log error and continue.
- Do not commit PDFs by default (add to .gitignore).

## Commit hygiene
- Prefer small, focused commits before starting a new logical change area when there is already completed work worth preserving.
- Do not mix unrelated staged or unstaged work into the current task's commit.
- When a task naturally splits into phases, commit the finished phase before starting the next one.
- Do not create checkpoint commits for trivial edits unless they materially reduce risk or protect important progress.
- Never amend or rewrite existing commits unless explicitly requested.

## PR workflow
- Prefer working in PR-sized slices: one logical change area per commit/PR whenever practical.
- If a task expands beyond a clean review scope, split it into sequential phases rather than one large all-in diff.
- Keep tests, docs, and config changes in the same PR when they are necessary to understand or validate the code change.
- Before starting the next major slice, leave the repo in a state that could plausibly be opened as a reviewable PR.
- When summarizing work, frame it in PR terms: user-visible outcome, key implementation points, tests run, and notable risks.

## Repo layout to create
- src/eegfm_digest/
  - run.py (CLI)
  - config.py
  - arxiv.py
  - keywords.py
  - db.py
  - llm.py
  - triage.py
  - pdf.py
  - summarize.py
  - render.py
  - site.py (HTML generation helpers)
- prompts/
- schemas/
- docs/
- tests/
- data/
- outputs/

## Commands
- install: pip install -e .
- run (monthly backfill): python -m eegfm_digest.run --month YYYY-MM
- run (incremental daily): python -m eegfm_digest.run --daily
- tests: pytest -q

## Daily mode
- `--daily` resolves `since = last_query_end_utc - overlap_hours` (default 6h) from `data/last_successful_run.json`, or falls back to a 24h lookback when that file is absent.
- `--until` defaults to wall-clock UTC.  `--since` / `--until` override everything (useful for backfills).
- The run log is written only when `run_window` returns without raising; `ArxivFetchError` and `LLMRateLimitError` intentionally propagate so the next run re-covers the window via the 6h overlap.
- Scheduled by `.github/workflows/daily-digest.yml` (cron initially disabled pending manual verification; `workflow_dispatch` exposes `since`/`until`/`dry_run` inputs).
- Commit pattern: outputs+SQLite in one commit, `data/last_successful_run.json` in a second commit on the same push.  Never amend the run log into a prior failed-pipeline commit.

## Publishing
- docs/.nojekyll must live inside docs/ (not repo root).
- docs/index.html should be updated on each run to link to the newest month.
