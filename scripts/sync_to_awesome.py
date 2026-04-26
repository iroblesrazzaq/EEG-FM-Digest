#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


AWESOME_REPO = "iroblesrazzaq/awesome-eeg-fm"
CLONE_DIR = Path("/tmp/awesome-eeg-fm")
ARXIV_ID_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|arXiv:)?(?P<id>\d{4}\.\d{4,5})(?:v\d+)?")


class SyncError(RuntimeError):
    """Raised when the sync flow cannot continue safely."""


@dataclass(frozen=True)
class PaperEntry:
    arxiv_id: str
    title: str
    one_liner: str
    year: str

    @property
    def arxiv_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append accepted EEG-FM Digest papers to awesome-eeg-fm and open a PR."
    )
    parser.add_argument("--month", required=True, help="Digest month in YYYY-MM format.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the entries that would be added without modifying the awesome repo checkout.",
    )
    return parser.parse_args(argv)


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


def load_backend_rows(month: str) -> list[PaperEntry]:
    backend_path = Path("outputs") / month / "backend_rows.jsonl"
    if not backend_path.exists():
        raise SyncError(f"Backend rows file not found: {backend_path}")

    accepted: list[PaperEntry] = []
    seen_ids: set[str] = set()

    with backend_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SyncError(f"Invalid JSON in {backend_path} line {line_number}: {exc}") from exc

            triage = row.get("triage") or {}
            if triage.get("decision") != "accept":
                continue

            entry = build_entry(row, source=f"{backend_path}:{line_number}")
            if entry.arxiv_id in seen_ids:
                continue
            seen_ids.add(entry.arxiv_id)
            accepted.append(entry)

    return accepted


def build_entry(row: dict, *, source: str) -> PaperEntry:
    arxiv_id = require_string(row.get("arxiv_id_base"), f"{source} missing arxiv_id_base")
    title = normalize_whitespace(require_string(row.get("title"), f"{source} missing title"))
    summary = row.get("paper_summary")
    if not isinstance(summary, dict):
        raise SyncError(f"{source} accepted paper missing paper_summary")
    one_liner = normalize_whitespace(
        require_string(summary.get("one_liner"), f"{source} missing paper_summary.one_liner")
    )
    published_value = summary.get("published_date") or row.get("published")
    year = extract_year(require_string(published_value, f"{source} missing published date"))
    return PaperEntry(arxiv_id=arxiv_id, title=title, one_liner=one_liner, year=year)


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


def escape_markdown_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("[", r"\[").replace("]", r"\]")


def format_markdown_entry(entry: PaperEntry) -> str:
    title = escape_markdown_text(entry.title)
    return f"- [{title}]({entry.arxiv_url}) - {entry.one_liner} ({entry.year})"


def extract_arxiv_ids(readme_text: str) -> set[str]:
    return {match.group("id") for match in ARXIV_ID_RE.finditer(readme_text)}


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


def ensure_repo_checkout() -> Path:
    if CLONE_DIR.exists():
        if not (CLONE_DIR / ".git").exists():
            raise SyncError(f"Clone path exists but is not a git repo: {CLONE_DIR}")
        return CLONE_DIR

    run_command(["gh", "repo", "clone", AWESOME_REPO, str(CLONE_DIR)])
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


def ensure_branch_absent(repo_dir: Path, branch_name: str) -> None:
    result = run_command(["git", "rev-parse", "--verify", branch_name], cwd=repo_dir, check=False)
    if result.returncode == 0:
        raise SyncError(f"Branch already exists in {repo_dir}: {branch_name}")


def checkout_branch(repo_dir: Path, *, default_branch: str, branch_name: str) -> None:
    run_command(["git", "checkout", default_branch], cwd=repo_dir)
    ensure_branch_absent(repo_dir, branch_name)
    run_command(["git", "checkout", "-b", branch_name], cwd=repo_dir)


def append_entries(readme_path: Path, entries: list[str]) -> None:
    original = readme_path.read_text(encoding="utf-8")
    suffix = "\n".join(entries)
    if original.endswith("\n\n"):
        updated = f"{original}{suffix}\n"
    elif original.endswith("\n"):
        updated = f"{original}\n{suffix}\n"
    else:
        updated = f"{original}\n\n{suffix}\n"
    readme_path.write_text(updated, encoding="utf-8")


def commit_and_open_pr(repo_dir: Path, *, month: str, branch_name: str, base_branch: str) -> str:
    commit_message = f"Add EEG-FM digest papers for {month}"
    run_command(["git", "add", "README.md"], cwd=repo_dir)
    run_command(["git", "commit", "-m", commit_message], cwd=repo_dir)
    run_command(["git", "push", "--set-upstream", "origin", branch_name], cwd=repo_dir)

    pr_title = commit_message
    pr_body = f"Adds accepted EEG-FM Digest papers for {month} from `outputs/{month}/backend_rows.jsonl`."
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        require_gh_auth()
        papers = load_backend_rows(args.month)
        if not papers:
            print(f"No accepted papers found in outputs/{args.month}/backend_rows.jsonl.")
            return 0

        repo_dir: Path | None = None
        if args.dry_run:
            readme_text = fetch_remote_readme()
        else:
            repo_dir = ensure_repo_checkout()
            readme_text = (repo_dir / "README.md").read_text(encoding="utf-8")

        existing_ids = extract_arxiv_ids(readme_text)
        new_entries = [format_markdown_entry(paper) for paper in papers if paper.arxiv_id not in existing_ids]

        if not new_entries:
            print(f"No new papers to add for {args.month}; all accepted arXiv IDs already appear in README.md.")
            return 0

        if args.dry_run:
            print(f"Would append {len(new_entries)} entries to {AWESOME_REPO}/README.md:")
            for entry in new_entries:
                print(entry)
            return 0

        assert repo_dir is not None
        ensure_clean_worktree(repo_dir)
        branch_name = f"digest-{args.month}"
        base_branch = get_default_branch(repo_dir)
        checkout_branch(repo_dir, default_branch=base_branch, branch_name=branch_name)

        append_entries(repo_dir / "README.md", new_entries)
        pr_url = commit_and_open_pr(
            repo_dir,
            month=args.month,
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
