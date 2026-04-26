# TODO

## Open PRs to land
- [ ] **#3 triage-single** (`feat/triage-single`) — standalone single-paper triage CLI (`--triage-single`) plus arXiv helper updates.
- [ ] **#4 fork-template** (`feat/fork-template`) — make the project forkable / BYOD-friendly: `TEMPLATE.md`, `configs/topics/*`, notebook walkthrough, docs refresh.
- [ ] **#6 awesome-sync** (`feat/awesome-sync`) — sync digest entries to an awesome-list format (`scripts/sync_to_awesome.py`).

## Near-term follow-ups
- [ ] Enable the daily cron by uncommenting `- cron: '0 10 * * *'` in `.github/workflows/daily-digest.yml` after one more clean `workflow_dispatch` verification run.
- [ ] Count PDF download / extract failures in the pipeline failure signal (`summary_failures`) so partial PDF failures fail the run and do not advance `data/last_successful_run.json`.
- [ ] Decide whether CSV export still needs a CLI / bulk mode. Browser-side CSV export already exists in [site.js](/Users/ismaelrobles-razzaq/2_cs_projects/EEG-FM-Digest/docs/assets/site.js).

## Low-priority feature ideas
- [ ] Multi-judge triage mode: run 2-3 LLM model calls or model instances at higher temperature (`0.9`-`1.0`) and aggregate decisions to reduce variance in paper acceptance/borderline/rejection.

## Stale Branches To Preserve Before Deletion
- [ ] `suggest-paper-triage-api` — suggested-paper triage API + UI experiment. No PR is open for it; salvage only if still wanted.
- [ ] `digest/feb-mar-apr-2026` — content refresh branch to regenerate fallback summaries in older published digests. This is data refresh work, not a new product feature.
- [ ] `build-your-own-template` — older template / BYOD attempt that is likely superseded by `feat/fork-template`.
- [ ] `fix/json-repair-diagnostics` — diagnostics / tests for summary JSON repair failures if we still want deeper observability around repair paths.

## Already Merged
- [x] Search date filtering / published-date UI (`feat/search-date-filter`, PR #11).
- [x] Topic config (`feat/topic-config`, PR #8).

## PR Hygiene
- [ ] Fill in the empty PR bodies (Summary / Key Changes / Testing / Risks) for #3, #4, #6 before merging.
