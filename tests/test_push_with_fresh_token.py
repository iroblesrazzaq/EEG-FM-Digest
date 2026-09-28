"""Daily digest push must drop the expired checkout token and retry."""

from __future__ import annotations

import os
import stat
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "push_with_fresh_token.sh"


def _run(args: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(args, cwd=cwd, env=merged, check=False, capture_output=True, text=True)


def _git(repo: Path, *args: str) -> None:
    result = _run(["git", *args], cwd=repo)
    assert result.returncode == 0, result.stderr


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")


def test_strip_removes_checkout_credential_include_and_file(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    runner_temp = tmp_path / "runner"
    runner_temp.mkdir()
    creds = runner_temp / "git-credentials-abc.config"
    creds.write_text(
        '[http "https://github.com/"]\n\textraheader = AUTHORIZATION: basic c3RhbGU=\n',
        encoding="utf-8",
    )
    git_dir = str(repo / ".git").replace("\\", "/")
    _git(repo, "config", "--local", f"includeIf.gitdir:{git_dir}.path", str(creds))
    _git(
        repo,
        "config",
        "--local",
        "http.https://github.com/.extraheader",
        "AUTHORIZATION: basic c3RhbGU=",
    )

    result = _run(
        ["bash", "-c", f'source "{SCRIPT}"; strip_persisted_checkout_auth'],
        cwd=repo,
        env={"RUNNER_TEMP": str(runner_temp), "GIT_TOKEN": "fresh-token-value"},
    )
    assert result.returncode == 0, result.stderr
    config = (repo / ".git" / "config").read_text(encoding="utf-8")
    assert "git-credentials" not in config
    assert "extraheader" not in config
    assert "fresh-token-value" not in config
    assert not creds.exists()


def test_strip_leaves_unrelated_include_and_files_outside_runner_temp(tmp_path: Path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    runner_temp = tmp_path / "runner"
    runner_temp.mkdir()
    outside = tmp_path / "git-credentials-outside.config"
    outside.write_text("keep\n", encoding="utf-8")
    other = tmp_path / "other.config"
    other.write_text("other\n", encoding="utf-8")
    git_dir = str(repo / ".git").replace("\\", "/")
    _git(repo, "config", "--local", f"includeIf.gitdir:{git_dir}.path", str(outside))
    _git(repo, "config", "--local", f"includeIf.gitdir:{git_dir}/other.path", str(other))

    result = _run(
        ["bash", "-c", f'source "{SCRIPT}"; strip_persisted_checkout_auth'],
        cwd=repo,
        env={"RUNNER_TEMP": str(runner_temp)},
    )
    assert result.returncode == 0, result.stderr
    config = (repo / ".git" / "config").read_text(encoding="utf-8")
    assert "git-credentials-outside.config" not in config
    assert other.name in config
    assert outside.exists()
    assert outside.read_text(encoding="utf-8") == "keep\n"


def _install_stub(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_push_configures_gh_and_retries_pull(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    trace = tmp_path / "trace"
    state = tmp_path / "pulls"
    _install_stub(
        bin_dir / "git",
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            printf '%s\\n' "$*" >> "{trace}"
            cmd="$1"
            if [[ "$cmd" == "config" ]]; then
              exit 0
            fi
            if [[ "$cmd" == "pull" ]]; then
              count=0
              if [[ -f "{state}" ]]; then
                count="$(cat "{state}")"
              fi
              count=$((count + 1))
              printf '%s' "$count" > "{state}"
              if [[ "$count" -eq 1 ]]; then
                exit 1
              fi
              exit 0
            fi
            if [[ "$cmd" == "rebase" || "$cmd" == "push" ]]; then
              exit 0
            fi
            exit 1
            """
        ),
    )
    _install_stub(
        bin_dir / "gh",
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            printf 'gh %s\\n' "$*" >> "{trace}"
            if [[ "$*" == "auth setup-git --hostname github.com --force" ]]; then
              if [[ "${{GH_TOKEN:-}}" != "fresh-token-value" || "${{GITHUB_TOKEN:-}}" != "fresh-token-value" ]]; then
                echo "token env mismatch" >&2
                exit 3
              fi
              exit 0
            fi
            echo "unexpected gh args: $*" >&2
            exit 2
            """
        ),
    )

    result = _run(
        ["bash", str(SCRIPT), "main"],
        cwd=tmp_path,
        env={
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "GIT_TOKEN": "fresh-token-value",
            "PUSH_RETRY_SLEEP_SECONDS": "0",
            "RUNNER_TEMP": str(tmp_path / "runner"),
        },
    )
    assert result.returncode == 0, result.stderr
    trace_text = trace.read_text(encoding="utf-8")
    assert "fresh-token-value" not in trace_text
    assert "fresh-token-value" not in result.stdout
    assert "fresh-token-value" not in result.stderr
    assert trace_text.index("gh auth setup-git --hostname github.com --force") < trace_text.index("pull --rebase origin main")
    assert trace_text.count("pull --rebase origin main") == 2
    assert "push origin HEAD:main" in trace_text
    assert "push rejected (attempt 1)" in result.stderr


def test_push_fails_when_token_missing(tmp_path: Path):
    result = _run(
        ["bash", str(SCRIPT), "main"],
        cwd=tmp_path,
        env={"GIT_TOKEN": "", "PUSH_RETRY_SLEEP_SECONDS": "0"},
    )
    assert result.returncode == 1
    assert "GIT_TOKEN is required" in result.stderr


def test_push_reports_failure_after_two_attempts(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install_stub(
        bin_dir / "git",
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            if [[ "$1" == "config" || "$1" == "rebase" ]]; then
              exit 0
            fi
            exit 1
            """
        ),
    )
    _install_stub(
        bin_dir / "gh",
        "#!/usr/bin/env bash\nexit 0\n",
    )
    result = _run(
        ["bash", str(SCRIPT), "main"],
        cwd=tmp_path,
        env={
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "GIT_TOKEN": "fresh-token-value",
            "PUSH_RETRY_SLEEP_SECONDS": "0",
        },
    )
    assert result.returncode == 1
    assert "failed to push daily digest after rebase retries" in result.stderr
    assert "fresh-token-value" not in result.stderr
