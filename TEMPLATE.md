# Forking This Digest

This repo is set up so you can fork it for a different research area without rewriting the pipeline.

## 1. Fork and rename

1. Fork this repository on GitHub.
2. Rename the fork to match your topic.
   Example: `sleep-eeg-digest`, `ecg-foundation-model-digest`, or `robotics-arxiv-digest`.
3. Clone your fork locally and install dependencies:

```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
pip install -e ".[dev]"
```

4. Update the checked-in branding in `configs/site_config.json`.
5. If you want to keep multiple topic configs around, set `TOPIC_CONFIG_PATH` when you run the pipeline or in GitHub Actions.

```bash
export TOPIC_CONFIG_PATH="configs/topics/my_topic.json"
python -m eegfm_digest.run --month 2026-03
```

If you only need one topic, you can also overwrite `configs/topics/eeg_fm.json`.

## 2. Create your topic config

Topic configs live in `configs/topics/`.

1. Copy the starter file:

```bash
cp configs/topics/eeg_fm.json configs/topics/my_topic.json
```

2. Edit these fields:
   `slug`: short machine-readable name.
   `title`: topic title for the digest.
   `description`: short one-line description.
   `categories`: arXiv categories you want to allow.
   `queries`: one or more arXiv API query strings.

3. Start with broad recall-oriented queries, then tighten them after testing hit quality.

Example:

```json
{
  "slug": "sleep_eeg_fm",
  "title": "Sleep EEG Foundation Model Digest",
  "description": "Monthly arXiv digest for sleep EEG foundation model papers",
  "categories": ["cs.LG", "q-bio.NC", "stat.ML"],
  "queries": [
    "all:(sleep EEG OR polysomnography) AND all:(foundation model OR pretrain OR pretrained)",
    "all:(sleep EEG OR polysomnography) AND all:(representation learning OR self-supervised OR transfer)"
  ]
}
```

4. Use `notebooks/setup_your_digest.ipynb` to test query quality, run 5 triage samples, and generate the final JSON file.

## 3. Set GitHub Secrets

Only one secret is required:

- `GEMINI_API_KEY`

Set it in:
`Settings` -> `Secrets and variables` -> `Actions` -> `New repository secret`

Important:
- Despite the secret name, the repo uses the OpenRouter-compatible API path in code.
- The model is hardcoded in the repo, so you do not need to add model secrets or vars for a basic fork.

## 4. Enable GitHub Pages

1. Go to `Settings` -> `Pages`.
2. Under `Build and deployment`, choose `Deploy from a branch`.
3. Select your default branch.
4. Select the `/docs` folder.
5. Save.

The pipeline writes static output to `docs/`, and `docs/.nojekyll` is already handled by the renderer.

## 5. Customize site branding

Edit `configs/site_config.json`.

Current fields:

```json
{
  "title": "Your Digest Title",
  "author": "Your Name",
  "description": "Short description shown in page metadata and About text",
  "theme_color": "#1e40af",
  "links": {
    "github": "https://github.com/your-user",
    "linkedin": "https://www.linkedin.com/in/your-profile",
    "website": "https://your-site.example",
    "email": "you@example.com"
  }
}
```

What this controls:
- Site title in the nav and page titles.
- Author byline.
- About-page description.
- Theme accent color.
- Header contact links.

After editing branding or topic config, rerun:

```bash
python -m eegfm_digest.run --month 2026-03
```

## 6. Recommended first run

1. Finalize `configs/site_config.json`.
2. Finalize `configs/topics/my_topic.json`.
3. Export:

```bash
export GEMINI_API_KEY="..."
export TOPIC_CONFIG_PATH="configs/topics/my_topic.json"
```

4. Run one month first:

```bash
python -m eegfm_digest.run --month 2026-03 --max-candidates 100 --max-accepted 20
```

5. Review:
   `outputs/2026-03/`
   `docs/`

6. Then enable the scheduled workflow on your fork.
