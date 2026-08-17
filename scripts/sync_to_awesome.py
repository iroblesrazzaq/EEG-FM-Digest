#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


AWESOME_REPO = "iroblesrazzaq/awesome-eeg-fm"
CLONE_DIR = Path("/tmp/awesome-eeg-fm")
ARXIV_ID_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|arXiv:)?(?P<id>\d{4}\.\d{4,5})(?:v\d+)?")
GIT_AUTHOR_NAME = "eeg-fm-daily-digest[bot]"
GIT_AUTHOR_EMAIL = "eeg-fm-daily-digest[bot]@users.noreply.github.com"
NEW_MODEL_TYPE = "new_model"
FM_SECTION_HEADER = "## EEG Foundation Models"
YEAR_SPLIT_RE = re.compile(r"(?=^### )", re.MULTILINE)
MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


class SyncError(RuntimeError):
    """Raised when the sync flow cannot continue safely."""


@dataclass(frozen=True)
class PaperEntry:
    arxiv_id: str
    title: str
    one_liner: str
    year: str
    authors: tuple[str, ...]
    code_url: str | None

    @property
    def arxiv_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add new EEG foundation models from the digest to awesome-eeg-fm and open a PR."
    )
    parser.add_argument(
        "--month",
        action="append",
        dest="months",
        help="Digest month YYYY-MM. Repeatable. Default: current and previous UTC months.",
    )
    parser.add_argument(
        "--all-months",
        action="store_true",
        help="Sync every month under docs/digest that has papers.json.",
    )
    parser.add_argument(
        "--docs-dir",
        default="docs",
        help="Path to the published docs/ directory (default: docs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the entries that would be added without modifying the awesome repo checkout.",
    )
    args = parser.parse_args(argv)
    if args.all_months and args.months:
        parser.error("--all-months cannot be combined with --month")
    return args


def run_command(args: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            check=check,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SyncError(f"Required command not found: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        command = " ".join(args)
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = stderr or stdout or f"exit code {exc.returncode}"
        raise SyncError(f"Command failed: {command}\n{details}") from exc


def require_gh_auth() -> None:
    run_command(["gh", "auth", "status"])


def default_months(now: datetime | None = None) -> list[str]:
    current = now or datetime.now(timezone.utc)
    this_month = f"{current.year:04d}-{current.month:02d}"
    if current.month == 1:
        previous = f"{current.year - 1:04d}-12"
    else:
        previous = f"{current.year:04d}-{current.month - 1:02d}"
    return [previous, this_month]


def list_digest_months(docs_dir: Path) -> list[str]:
    digest_root = docs_dir / "digest"
    if not digest_root.is_dir():
        return []
    months: list[str] = []
    for path in digest_root.iterdir():
        if path.is_dir() and MONTH_RE.fullmatch(path.name) and (path / "papers.json").exists():
            months.append(path.name)
    return sorted(months)


def resolve_months(args: argparse.Namespace, docs_dir: Path) -> tuple[list[str], bool]:
    if args.all_months:
        months = list_digest_months(docs_dir)
        if not months:
            raise SyncError(f"No digest months found under {docs_dir / 'digest'}")
        return months, False
    if args.months:
        for month in args.months:
            if not MONTH_RE.fullmatch(month):
                raise SyncError(f"Invalid month {month!r}; expected YYYY-MM")
        return args.months, False
    return default_months(), True


def load_digest_papers(month: str, *, docs_dir: Path) -> list[PaperEntry]:
    papers_path = docs_dir / "digest" / month / "papers.json"
    if not papers_path.exists():
        raise SyncError(f"Digest papers file not found: {papers_path}")

    try:
        payload = json.loads(papers_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SyncError(f"Invalid JSON in {papers_path}: {exc}") from exc

    if isinstance(payload, dict):
        rows = payload.get("papers", [])
    elif isinstance(payload, list):
        rows = payload
    else:
        raise SyncError(f"Unexpected papers payload in {papers_path}")
    if not isinstance(rows, list):
        raise SyncError(f"Unexpected papers list in {papers_path}")

    accepted: list[PaperEntry] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        if not _is_accepted_new_model(row):
            continue
        entry = build_entry(row, source=f"{papers_path}[{index}]")
        if entry is None or entry.arxiv_id in seen_ids:
            continue
        seen_ids.add(entry.arxiv_id)
        accepted.append(entry)
    return accepted


def collect_new_model_papers(
    months: list[str],
    *,
    docs_dir: Path,
    missing_ok: bool,
) -> list[PaperEntry]:
    collected: list[PaperEntry] = []
    seen_ids: set[str] = set()
    for month in months:
        papers_path = docs_dir / "digest" / month / "papers.json"
        if not papers_path.exists():
            if missing_ok:
                print(f"No digest payload for {month}; skipping.")
                continue
            raise SyncError(f"Digest papers file not found: {papers_path}")
        for entry in load_digest_papers(month, docs_dir=docs_dir):
            if entry.arxiv_id in seen_ids:
                continue
            seen_ids.add(entry.arxiv_id)
            collected.append(entry)
    return collected


def _is_accepted_new_model(row: dict) -> bool:
    triage = row.get("triage")
    if isinstance(triage, dict):
        decision = str(triage.get("decision", "")).strip()
        if decision and decision != "accept":
            return False
    summary = row.get("summary")
    if not isinstance(summary, dict):
        return False
    paper_type = str(summary.get("paper_type") or row.get("paper_type") or "").strip()
    return paper_type == NEW_MODEL_TYPE


def build_entry(row: dict, *, source: str) -> PaperEntry | None:
    summary = row.get("summary")
    if not isinstance(summary, dict):
        return None
    one_liner = normalize_whitespace(str(summary.get("one_liner") or ""))
    if not one_liner or one_liner.startswith("Summary unavailable"):
        return None
    try:
        arxiv_id = require_string(row.get("arxiv_id_base"), f"{source} missing arxiv_id_base")
        title = normalize_whitespace(require_string(row.get("title"), f"{source} missing title"))
    except SyncError:
        return None
    published_value = summary.get("published_date") or row.get("published_date") or row.get("published")
    try:
        year = extract_year(require_string(published_value, f"{source} missing published date"))
    except SyncError:
        return None
    authors = tuple(item.strip() for item in row.get("authors", []) if isinstance(item, str) and item.strip())
    open_source = summary.get("open_source")
    code_url = None
    if isinstance(open_source, dict):
        raw_code = open_source.get("code_url")
        if isinstance(raw_code, str) and raw_code.strip():
            code_url = raw_code.strip()
    return PaperEntry(
        arxiv_id=arxiv_id,
        title=title,
        one_liner=one_liner,
        year=year,
        authors=authors,
        code_url=code_url,
    )


def require_string(value: object, message: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SyncError(message)
    return value.strip()


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def extract_year(value: str) -> str:
    match = re.match(r"^(?P<year>\d{4})", value)
    if not match:
        raise SyncError(f"Could not extract year from date value: {value!r}")
    return match.group("year")


def format_authors(authors: tuple[str, ...]) -> str:
    names = [name.strip() for name in authors if name.strip()]
    if not names:
        return "et al."
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{names[0]} et al."


def format_markdown_entry(entry: PaperEntry) -> str:
    title = entry.title.replace("\n", " ").strip()
    authors = format_authors(entry.authors)
    meta = (
        f"[paper]({entry.arxiv_url}) · *{authors}* "
        f"(arXiv {entry.year}; [arXiv:{entry.arxiv_id}]({entry.arxiv_url}))"
    )
    if entry.code_url:
        meta += f" · [code]({entry.code_url})"
    return f"- **{title}**  \n  {meta}\n"


def extract_arxiv_ids(readme_text: str) -> set[str]:
    return {match.group("id") for match in ARXIV_ID_RE.finditer(readme_text)}


def insert_fm_entries(readme: str, entries: list[PaperEntry]) -> str:
    if not entries:
        return readme
    start = readme.find(FM_SECTION_HEADER)
    if start < 0:
        raise SyncError(f"README.md is missing {FM_SECTION_HEADER!r}")
    after_header = start + len(FM_SECTION_HEADER)
    next_h2 = re.search(r"\n## ", readme[after_header:])
    end = after_header + next_h2.start() if next_h2 else len(readme)
    section = readme[start:end]
    return readme[:start] + insert_into_fm_section(section, entries) + readme[end:]


def insert_into_fm_section(section: str, entries: list[PaperEntry]) -> str:
    by_year: dict[str, list[PaperEntry]] = defaultdict(list)
    for entry in entries:
        by_year[entry.year].append(entry)

    chunks = YEAR_SPLIT_RE.split(section)
    preamble = chunks[0]
    year_blocks: list[tuple[int, str]] = []
    for chunk in chunks[1:]:
        first_line = chunk.split("\n", 1)[0]
        match = re.match(r"###\s+(\d{4})", first_line)
        if match is None:
            preamble += chunk
            continue
        year_blocks.append((int(match.group(1)), chunk))

    for year, year_entries in sorted(by_year.items(), key=lambda item: item[0], reverse=True):
        year_entries = sorted(year_entries, key=lambda item: item.title.casefold())
        block = "\n\n".join(format_markdown_entry(item).rstrip("\n") for item in year_entries) + "\n\n"
        year_int = int(year)
        existing_idx = next((index for index, (value, _) in enumerate(year_blocks) if value == year_int), None)
        if existing_idx is None:
            new_chunk = f"### {year}\n{block}"
            updated: list[tuple[int, str]] = []
            inserted = False
            for value, chunk in year_blocks:
                if not inserted and year_int > value:
                    updated.append((year_int, new_chunk))
                    inserted = True
                updated.append((value, chunk))
            if not inserted:
                updated.append((year_int, new_chunk))
            year_blocks = updated
            continue

        heading, _, body = year_blocks[existing_idx][1].partition("\n")
        year_blocks[existing_idx] = (year_int, heading + "\n" + block + body.lstrip("\n"))

    return preamble + "".join(chunk for _, chunk in year_blocks)


def fetch_remote_readme() -> str:
    result = run_command(
        [
            "gh",
            "api",
            "-H",
            "Accept: application/vnd.github.raw",
            f"repos/{AWESOME_REPO}/contents/README.md",
        ]
    )
    return result.stdout


def configure_github_https_auth(repo_dir: Path) -> None:
    """Let git push/fetch use GH_TOKEN on Actions runners that have no interactive prompt."""
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        return
    basic = base64.b64encode(f"x-access-token:{token}".encode("ascii")).decode("ascii")
    run_command(
        [
            "git",
            "config",
            "http.https://github.com/.extraheader",
            f"AUTHORIZATION: basic {basic}",
        ],
        cwd=repo_dir,
    )


def ensure_repo_checkout() -> Path:
    if CLONE_DIR.exists():
        if not (CLONE_DIR / ".git").exists():
            raise SyncError(f"Clone path exists but is not a git repo: {CLONE_DIR}")
        configure_github_https_auth(CLONE_DIR)
        run_command(["git", "fetch", "origin"], cwd=CLONE_DIR)
        return CLONE_DIR

    run_command(["gh", "repo", "clone", AWESOME_REPO, str(CLONE_DIR)])
    configure_github_https_auth(CLONE_DIR)
    return CLONE_DIR


def ensure_clean_worktree(repo_dir: Path) -> None:
    result = run_command(["git", "status", "--short"], cwd=repo_dir)
    if result.stdout.strip():
        raise SyncError(f"Repository has uncommitted changes: {repo_dir}")


def get_default_branch(repo_dir: Path) -> str:
    result = run_command(
        ["git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
        cwd=repo_dir,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        ref = result.stdout.strip()
        if "/" in ref:
            return ref.rsplit("/", 1)[-1]

    fallback = run_command(["git", "branch", "--show-current"], cwd=repo_dir, check=False)
    if fallback.returncode == 0 and fallback.stdout.strip():
        return fallback.stdout.strip()
    return "main"


def configure_git_identity(repo_dir: Path) -> None:
    """Set local author identity so commits work on fresh Actions runners."""
    run_command(["git", "config", "user.name", GIT_AUTHOR_NAME], cwd=repo_dir)
    run_command(["git", "config", "user.email", GIT_AUTHOR_EMAIL], cwd=repo_dir)


def remote_branch_exists(repo_dir: Path, branch_name: str) -> bool:
    result = run_command(
        ["git", "rev-parse", "--verify", f"refs/remotes/origin/{branch_name}"],
        cwd=repo_dir,
        check=False,
    )
    return result.returncode == 0


def prepare_month_branch(repo_dir: Path, *, default_branch: str, branch_name: str) -> None:
    """Check out ``branch_name``, reusing ``origin/branch_name`` when it exists.

    Fresh month: branch from updated default. Rerun: continue from the remote
    monthly branch so pushes remain fast-forward and open PRs stay updateable.
    """
    run_command(["git", "fetch", "origin"], cwd=repo_dir)
    run_command(["git", "checkout", default_branch], cwd=repo_dir)
    run_command(
        ["git", "pull", "--ff-only", "origin", default_branch],
        cwd=repo_dir,
        check=False,
    )

    if remote_branch_exists(repo_dir, branch_name):
        run_command(
            ["git", "checkout", "-B", branch_name, f"origin/{branch_name}"],
            cwd=repo_dir,
        )
        return

    local = run_command(
        ["git", "rev-parse", "--verify", f"refs/heads/{branch_name}"],
        cwd=repo_dir,
        check=False,
    )
    if local.returncode == 0:
        run_command(["git", "branch", "-D", branch_name], cwd=repo_dir)
    run_command(["git", "checkout", "-b", branch_name], cwd=repo_dir)


def find_existing_pr_url(branch_name: str) -> str | None:
    owner = AWESOME_REPO.split("/", 1)[0]
    result = run_command(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            AWESOME_REPO,
            "--head",
            f"{owner}:{branch_name}",
            "--state",
            "open",
            "--json",
            "url",
            "--jq",
            ".[0].url // empty",
        ],
        check=False,
    )
    if result.returncode != 0:
        return None
    url = result.stdout.strip()
    return url or None


def commit_and_open_pr(repo_dir: Path, *, label: str, branch_name: str, base_branch: str) -> str:
    configure_git_identity(repo_dir)
    commit_message = f"Add EEG foundation models from digest ({label})"
    run_command(["git", "add", "README.md"], cwd=repo_dir)
    run_command(["git", "commit", "-m", commit_message], cwd=repo_dir)
    run_command(["git", "push", "--set-upstream", "origin", branch_name], cwd=repo_dir)

    existing_url = find_existing_pr_url(branch_name)
    if existing_url:
        return existing_url

    pr_title = commit_message
    pr_body = (
        f"Adds new EEG foundation models (`paper_type=new_model`) from EEG-FM Digest ({label}).\n\n"
        "Source: `docs/digest/*/papers.json`.\n"
        "Site: https://iroblesrazzaq.github.io/EEG-FM-Digest/"
    )
    result = run_command(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            AWESOME_REPO,
            "--base",
            base_branch,
            "--head",
            branch_name,
            "--title",
            pr_title,
            "--body",
            pr_body,
        ],
        cwd=repo_dir,
    )
    return result.stdout.strip()


def _new_papers_against_readme(papers: list[PaperEntry], readme_text: str) -> list[PaperEntry]:
    existing_ids = extract_arxiv_ids(readme_text)
    return [paper for paper in papers if paper.arxiv_id not in existing_ids]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        require_gh_auth()
        docs_dir = Path(args.docs_dir)
        months, missing_ok = resolve_months(args, docs_dir)
        papers = collect_new_model_papers(months, docs_dir=docs_dir, missing_ok=missing_ok)
        if not papers:
            print("No new EEG foundation models found in the selected digest months.")
            return 0

        if args.all_months:
            label = "all months"
            branch_name = "digest-sync"
        elif args.months and len(months) == 1:
            label = months[0]
            branch_name = f"digest-{months[0]}"
        else:
            label = ", ".join(months)
            branch_name = "digest-sync"

        if args.dry_run:
            readme_text = fetch_remote_readme()
            new_papers = _new_papers_against_readme(papers, readme_text)
            if not new_papers:
                print(
                    f"No new papers to add for {label}; "
                    "all new_model arXiv IDs already appear in README.md."
                )
                return 0
            print(f"Would add {len(new_papers)} new_model entries to {AWESOME_REPO}/README.md:")
            for paper in new_papers:
                print(format_markdown_entry(paper).rstrip())
            return 0

        repo_dir = ensure_repo_checkout()
        ensure_clean_worktree(repo_dir)
        base_branch = get_default_branch(repo_dir)
        prepare_month_branch(
            repo_dir,
            default_branch=base_branch,
            branch_name=branch_name,
        )

        readme_path = repo_dir / "README.md"
        readme_text = readme_path.read_text(encoding="utf-8")
        new_papers = _new_papers_against_readme(papers, readme_text)
        if not new_papers:
            print(
                f"No new papers to add for {label}; "
                "all new_model arXiv IDs already appear in README.md."
            )
            return 0

        readme_path.write_text(insert_fm_entries(readme_text, new_papers), encoding="utf-8")
        pr_url = commit_and_open_pr(
            repo_dir,
            label=label,
            branch_name=branch_name,
            base_branch=base_branch,
        )
        print(f"Opened PR: {pr_url}")
        return 0
    except SyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
