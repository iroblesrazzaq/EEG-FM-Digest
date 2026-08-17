from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _load_sync_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "sync_to_awesome.py"
    spec = importlib.util.spec_from_file_location("sync_to_awesome", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _paper(
    *,
    arxiv_id: str,
    title: str,
    paper_type: str,
    one_liner: str = "A useful summary.",
    published_date: str = "2026-07-02",
    decision: str = "accept",
    authors: list[str] | None = None,
    code_url: str | None = None,
) -> dict:
    return {
        "arxiv_id_base": arxiv_id,
        "title": title,
        "authors": authors or ["Ada Lovelace", "Alan Turing", "Grace Hopper"],
        "published_date": published_date,
        "triage": {"decision": decision},
        "summary": {
            "paper_type": paper_type,
            "one_liner": one_liner,
            "published_date": published_date,
            "open_source": {"code_url": code_url, "weights_url": None, "license": None},
        },
    }


def test_load_digest_papers_keeps_new_models_only(tmp_path, monkeypatch):
    module = _load_sync_module()
    month = "2026-07"
    month_dir = tmp_path / "docs" / "digest" / month
    month_dir.mkdir(parents=True)
    payload = {
        "papers": [
            _paper(arxiv_id="2607.00001", title="New FM", paper_type="new_model"),
            _paper(arxiv_id="2607.00001", title="Duplicate FM", paper_type="new_model"),
            _paper(arxiv_id="2607.00002", title="LoRA method", paper_type="method"),
            _paper(arxiv_id="2607.00003", title="Rejected FM", paper_type="new_model", decision="reject"),
            _paper(
                arxiv_id="2607.00004",
                title="Broken summary FM",
                paper_type="new_model",
                one_liner="Summary unavailable due to JSON validation failure.",
            ),
            _paper(
                arxiv_id="2607.00005",
                title="Open FM",
                paper_type="new_model",
                code_url="https://github.com/example/open-fm",
            ),
        ]
    }
    (month_dir / "papers.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    papers = module.load_digest_papers(month, docs_dir=Path("docs"))

    assert [paper.arxiv_id for paper in papers] == ["2607.00001", "2607.00005"]
    assert papers[0].year == "2026"
    assert papers[0].authors[0] == "Ada Lovelace"
    assert papers[1].code_url == "https://github.com/example/open-fm"
    assert "method" not in {paper.title for paper in papers}


def test_format_markdown_entry_matches_awesome_list_style():
    module = _load_sync_module()
    entry = module.PaperEntry(
        arxiv_id="2607.03925",
        title="NeuroOnline: Bridging Pretraining and Online Adaptation",
        one_liner="A one liner.",
        year="2026",
        authors=("Weibin Li", "Wendu Li", "Yushan You"),
        code_url="https://github.com/example/neuroonline",
    )

    rendered = module.format_markdown_entry(entry)

    assert rendered == (
        "- **NeuroOnline: Bridging Pretraining and Online Adaptation**  \n"
        "  [paper](https://arxiv.org/abs/2607.03925) · *Weibin Li et al.* "
        "(arXiv 2026; [arXiv:2607.03925](https://arxiv.org/abs/2607.03925)) "
        "· [code](https://github.com/example/neuroonline)\n"
    )


def test_insert_fm_entries_prepends_to_existing_year_and_creates_missing_year():
    module = _load_sync_module()
    readme = """# Awesome EEG Foundation Models

## EEG Foundation Models

### 2026 
- **Uni-NTFM: A Unified Foundation Model**  
  [paper](https://openreview.net/forum?id=oUMiuYHW21) · *Chen et al.* (ICLR 2026)

### 2025
- **ALFEE: Adaptive Large Foundation Model for EEG Representation**  
  [paper](https://arxiv.org/abs/2505.06291) · *Wang et al.* (arXiv; [arXiv:2505.06291](https://arxiv.org/abs/2505.06291))

---

## Multimodal brainwave Foundation Models (EEG + other modalities)

### 2025
- **BrainOmni**  
  [paper](https://arxiv.org/abs/2505.18185) · *Xiao et al.*
"""
    new_2026 = module.PaperEntry(
        arxiv_id="2607.03925",
        title="NeuroOnline",
        one_liner="line",
        year="2026",
        authors=("Weibin Li",),
        code_url=None,
    )
    new_2022 = module.PaperEntry(
        arxiv_id="2204.03272",
        title="MAEEG",
        one_liner="line",
        year="2022",
        authors=("Author A", "Author B"),
        code_url=None,
    )

    updated = module.insert_fm_entries(readme, [new_2026, new_2022])

    fm_start = updated.index("## EEG Foundation Models")
    multimodal_start = updated.index("## Multimodal")
    fm_section = updated[fm_start:multimodal_start]
    assert fm_section.index("### 2026") < fm_section.index("### 2025")
    assert fm_section.index("### 2025") < fm_section.index("### 2022")
    assert fm_section.index("NeuroOnline") < fm_section.index("Uni-NTFM")
    assert "\n\n- **Uni-NTFM" in fm_section or "\n- **Uni-NTFM" in fm_section
    neuro_block = fm_section[fm_section.index("NeuroOnline"):fm_section.index("Uni-NTFM")]
    assert "2607.03925" in neuro_block
    assert "2204.03272" in fm_section
    assert updated[multimodal_start:].count("2505.18185") == 1
    assert "BrainOmni" in updated[multimodal_start:]


def test_extract_arxiv_ids_matches_base_and_versioned_urls():
    module = _load_sync_module()
    readme = """
    - [Paper A](https://arxiv.org/abs/2501.00001v2) - Summary. (2025)
    - [Paper B](https://arxiv.org/pdf/2501.00002.pdf) - Summary. (2025)
    Mentioned as arXiv:2501.00003 in text.
    """

    found = module.extract_arxiv_ids(readme)

    assert found == {"2501.00001", "2501.00002", "2501.00003"}


def test_default_months_includes_january_wrap():
    module = _load_sync_module()
    assert module.default_months(datetime(2026, 8, 16, tzinfo=timezone.utc)) == ["2026-07", "2026-08"]
    assert module.default_months(datetime(2026, 1, 2, tzinfo=timezone.utc)) == ["2025-12", "2026-01"]


def test_resolve_months_all_months_includes_older_digest_folders(tmp_path):
    module = _load_sync_module()
    docs_dir = tmp_path / "docs"
    for month in ("2025-03", "2026-07", "2026-08"):
        month_dir = docs_dir / "digest" / month
        month_dir.mkdir(parents=True)
        (month_dir / "papers.json").write_text("{}", encoding="utf-8")
    (docs_dir / "digest" / "notes").mkdir()

    args = module.parse_args(["--all-months", "--docs-dir", str(docs_dir)])
    months, missing_ok = module.resolve_months(args, docs_dir)

    assert months == ["2025-03", "2026-07", "2026-08"]
    assert missing_ok is False


def test_automatic_workflow_scans_all_published_months():
    workflow = Path(".github/workflows/awesome-sync.yml").read_text(encoding="utf-8")
    assert 'if [ "${GITHUB_EVENT_NAME}" = "workflow_run" ] || [ "${INPUT_ALL_MONTHS:-}" = "true" ]; then' in workflow
    assert "args+=(--all-months)" in workflow
    assert "gh auth setup-git" in workflow
    assert "default_months()" not in workflow


def test_configure_github_https_auth_sets_bearer_header(tmp_path, monkeypatch):
    module = _load_sync_module()
    calls: list[list[str]] = []

    def fake_run(args, *, cwd=None, check=True):  # noqa: ANN001
        calls.append(list(args))
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setenv("GH_TOKEN", "test-token")
    monkeypatch.setattr(module, "run_command", fake_run)
    module.configure_github_https_auth(tmp_path)

    assert calls[0][0:3] == ["git", "config", "http.https://github.com/.extraheader"]
    assert calls[0][3].startswith("AUTHORIZATION: basic ")


def test_configure_git_identity_sets_bot_author(tmp_path, monkeypatch):
    module = _load_sync_module()
    calls: list[list[str]] = []

    def fake_run(args, *, cwd=None, check=True):  # noqa: ANN001
        calls.append(list(args))
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(module, "run_command", fake_run)
    module.configure_git_identity(tmp_path)

    assert ["git", "config", "user.name", module.GIT_AUTHOR_NAME] in calls
    assert ["git", "config", "user.email", module.GIT_AUTHOR_EMAIL] in calls


def test_prepare_month_branch_reuses_remote_monthly_branch(tmp_path, monkeypatch):
    module = _load_sync_module()
    calls: list[list[str]] = []

    def fake_run(args, *, cwd=None, check=True):  # noqa: ANN001
        calls.append(list(args))
        joined = " ".join(args)
        if "rev-parse" in joined and "origin/digest-2025-01" in joined:
            return type("R", (), {"returncode": 0, "stdout": "abc", "stderr": ""})()
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(module, "run_command", fake_run)
    module.prepare_month_branch(
        tmp_path,
        default_branch="main",
        branch_name="digest-2025-01",
    )

    assert ["git", "checkout", "-B", "digest-2025-01", "origin/digest-2025-01"] in calls
    assert ["git", "checkout", "-b", "digest-2025-01"] not in calls


def test_prepare_month_branch_creates_fresh_branch_when_remote_missing(tmp_path, monkeypatch):
    module = _load_sync_module()
    calls: list[list[str]] = []

    def fake_run(args, *, cwd=None, check=True):  # noqa: ANN001
        calls.append(list(args))
        # Remote and local branch absent.
        if "rev-parse" in args:
            return type("R", (), {"returncode": 1, "stdout": "", "stderr": ""})()
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(module, "run_command", fake_run)
    module.prepare_month_branch(
        tmp_path,
        default_branch="main",
        branch_name="digest-2025-01",
    )

    assert ["git", "checkout", "-b", "digest-2025-01"] in calls
    assert ["git", "checkout", "-B", "digest-2025-01", "origin/digest-2025-01"] not in calls


def test_commit_and_open_pr_configures_identity_before_commit(tmp_path, monkeypatch):
    module = _load_sync_module()
    calls: list[list[str]] = []

    def fake_run(args, *, cwd=None, check=True):  # noqa: ANN001
        calls.append(list(args))
        if args[:3] == ["gh", "pr", "list"]:
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        if args[:3] == ["gh", "pr", "create"]:
            return type(
                "R",
                (),
                {"returncode": 0, "stdout": "https://example.com/pr/1\n", "stderr": ""},
            )()
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(module, "run_command", fake_run)
    url = module.commit_and_open_pr(
        tmp_path,
        label="2025-01",
        branch_name="digest-2025-01",
        base_branch="main",
    )

    name_idx = calls.index(["git", "config", "user.name", module.GIT_AUTHOR_NAME])
    commit_idx = next(i for i, args in enumerate(calls) if args[:2] == ["git", "commit"])
    assert name_idx < commit_idx
    assert url == "https://example.com/pr/1"
    create_call = next(args for args in calls if args[:3] == ["gh", "pr", "create"])
    assert "paper_type=new_model" in " ".join(create_call)
    assert "backend_rows.jsonl" not in " ".join(create_call)
