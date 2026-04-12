#!/usr/bin/env fish
# overnight_build.fish
# 6 features via Codex headless. Run and sleep.
# Usage: fish overnight_build.fish

# ---- EDIT THIS ----
set REPO_DIR ~/2_cs_projects/EEG-FM-Digest
# --------------------

cd $REPO_DIR; or begin; echo "ERROR: $REPO_DIR not found"; exit 1; end

set log_dir .overnight-logs
mkdir -p $log_dir
set ts (date +%Y%m%d-%H%M%S)

function run_feat
    set feat_name $argv[1]
    set prompt $argv[2]
    set branch "feat/$feat_name"
    set logfile "$log_dir/$feat_name-$ts.log"

    echo ""
    echo "========================================"
    echo "  FEATURE: $feat_name"
    echo "  Branch:  $branch"
    echo "========================================"

    git checkout main
    git checkout -b $branch 2>/dev/null; or git checkout $branch

    echo "started: "(date) > $logfile
    codex exec --full-auto "$prompt" 2>&1 | tee -a $logfile

    git add -A
    if not git diff --staged --quiet
        git commit -m "$feat_name: codex implementation"
        echo "  COMMITTED on $branch"
    else
        echo "  no changes on $branch"
    end

    git checkout main
    echo "finished: "(date) >> $logfile
end


# ============================================================
# 1. Paper-of-month auth guard
# ============================================================
run_feat "paper-of-month-guard" "In this EEG-FM Digest repo:

1. In the digest.json schema, the featured_paper field should default to null.
2. In the site renderer (src/eegfm_digest/), when featured_paper is null, render a clean card that says 'No featured paper this month. Check back soon!' with a subtle dashed border and muted text. No broken layout.
3. Add a CLI flag: python -m eegfm_digest.run --month 2025-01 --feature-paper 2401.12345
   This is the ONLY way to set a featured paper. The GitHub Actions workflow should NEVER auto-set it.
4. Add a test in tests/ that verifies the site renders correctly with featured_paper=null.

Read SPEC.md and AGENTS.md first. Follow existing code patterns."


# ============================================================
# 2. CSV export
# ============================================================
run_feat "csv-export" "Add CSV export to this EEG-FM Digest repo:

1. Add a --csv flag to eegfm_digest.run that exports accepted papers to outputs/YYYY-MM/digest.csv
   Columns: arxiv_id, title, published, authors (semicolon-separated), decision, confidence, one_liner, tags (comma-separated), arxiv_url
2. Create scripts/export_all_csv.py that reads ALL months from outputs/ and writes outputs/all_papers.csv
3. Add a test that verifies CSV output matches expected schema.

Read SPEC.md for the backend_rows.jsonl format. Follow existing code patterns."


# ============================================================
# 3. triage_single.py for paper suggestions
# ============================================================
run_feat "triage-single" "Create scripts/triage_single.py for this EEG-FM Digest repo:

1. Takes an arxiv_id as CLI argument (e.g. '2401.12345')
2. Validates it is a real arXiv paper by fetching metadata via the arxiv API
3. Checks if it already exists in any outputs/*/backend_rows.jsonl
4. If already reviewed: print existing triage decision as GitHub-flavored markdown
5. If new: run triage (fetch abstract, call triage LLM via existing eegfm_digest.triage module) and print result as markdown with: title, authors, decision, confidence, reasons, arxiv link
6. Exit code 0 on success, 1 on invalid/missing paper

The triage LLM model is already hardcoded in the repo config. Do not change model settings.

Use existing code from eegfm_digest.arxiv and eegfm_digest.triage.
Read SPEC.md first. Match existing import patterns and error handling."


# ============================================================
# 4. Extract keyword config from hardcoded to JSON
# ============================================================
run_feat "topic-config" "Refactor eegfm_digest to support multiple topics/modalities:

1. Create configs/topics/eeg-fm.json with the current hardcoded keywords, arxiv categories, and prompt paths extracted from the codebase
2. Add a --topic flag to eegfm_digest.run (default: 'eeg-fm')
3. fetch_month_candidates() should read keyword config from the topic JSON file instead of hardcoded values
4. The triage prompt path should come from the topic config
5. Outputs go to outputs/{topic_slug}/{month}/ but for backward compat, 'eeg-fm' topic also writes to outputs/{month}/ (symlink or copy)
6. Do NOT change the site renderer yet, just the backend pipeline
7. Do NOT create any new topic configs besides eeg-fm.json -- just extract what already exists

Read SPEC.md and AGENTS.md. Preserve all existing tests. Add a test that loads eeg-fm.json and verifies it matches the current hardcoded keywords."


# ============================================================
# 5. awesome-eeg-fm sync script
# ============================================================
run_feat "awesome-sync" "Create scripts/sync_to_awesome.py for this EEG-FM Digest repo:

1. Reads accepted papers from outputs/{month}/backend_rows.jsonl
2. Formats each as a markdown list entry: '- [Title](arxiv_url) - one_liner (YYYY)'
3. Uses subprocess to call gh CLI to:
   a. Clone iroblesrazzaq/awesome-eeg-fm to /tmp/ if not already there
   b. Create branch 'digest-{month}'
   c. Append entries to README.md
   d. Commit and open PR
4. Takes --month as required argument
5. Deduplicates: skip papers whose arxiv_id already appears in the README
6. Requires gh auth to be configured
7. Add a --dry-run flag that prints what would be added without touching the repo

Add argparse and proper error handling."


# ============================================================
# 6. Fork-and-run template
# ============================================================
run_feat "fork-template" "Make this EEG-FM Digest repo easily forkable for other research areas:

1. Create TEMPLATE.md with step-by-step instructions:
   - How to fork and rename
   - How to create your own topic config in configs/topics/
   - How to set GitHub Secrets (GEMINI_API_KEY is the only secret needed -- the model is hardcoded in the repo)
   - How to enable GitHub Pages
   - How to customize site branding

2. Create configs/site_config.json with current branding:
   {
     \"title\": \"EEG Foundation Model Digest\",
     \"author\": \"Ismael Robles-Razzaq\",
     \"description\": \"Monthly arXiv digest for EEG-FM papers\",
     \"theme_color\": \"#1e40af\",
     \"links\": {
       \"github\": \"https://github.com/iroblesrazzaq\",
       \"linkedin\": \"https://www.linkedin.com/in/ismaelroblesrazzaq\",
       \"website\": \"https://iroblesrazzaq.github.io/\",
       \"email\": \"ismaelroblesrazzaq@gmail.com\"
     }
   }
   Update the site renderer to read branding from this config file.

3. Create notebooks/setup_your_digest.ipynb that walks through:
   - Choosing arXiv keywords (with sample queries and hit counts)
   - Testing triage LLM on 5 sample papers
   - Generating a topic config file
   Use existing eegfm_digest modules.

Read SPEC.md and AGENTS.md."


# ============================================================
# DONE
# ============================================================
echo ""
echo "========================================"
echo "  ALL 6 FEATURES COMPLETE"
echo ""
echo "  Branches:"
git branch --list 'feat/*'
echo ""
echo "  Review a branch:"
echo "    git diff main...feat/csv-export"
echo ""
echo "  Push all as PRs for Greptile review:"
echo "    for b in (git branch --list 'feat/*' | string trim)"
echo "        git push -u origin \$b"
echo "        gh pr create --base main --head \$b --fill"
echo "    end"
echo "========================================"
