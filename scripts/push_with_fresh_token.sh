#!/usr/bin/env bash
# Push the current checkout using a GitHub App token minted after the long
# daily pipeline. actions/checkout persists the job's first installation
# token in an includeIf credentials file. That token expires after one hour,
# and the daily pipeline regularly runs longer, so git pull then fails with
# "could not read Username" and the commits never land.
set -euo pipefail

strip_persisted_checkout_auth() {
  local key value keys values base
  keys="$(git config --local --name-only --get-regexp '^include[Ii]f\.gitdir:' || true)"
  while IFS= read -r key; do
    [ -n "$key" ] || continue
    values="$(git config --local --get-all "$key" || true)"
    while IFS= read -r value; do
      [ -n "$value" ] || continue
      base="$(basename "$value")"
      case "$base" in
        git-credentials-*.config)
          # --unset treats the value as a regex. Credential paths contain
          # dots, and a '+' would leave the expired include in place.
          git config --local --unset --fixed-value "$key" "$value" || true
          remove_checkout_credentials_file "$value"
          ;;
      esac
    done <<< "$values"
  done <<< "$keys"
  git config --local --unset-all 'http.https://github.com/.extraheader' || true
  if git config --local --get-regexp '^include[Ii]f\.gitdir:' 2>/dev/null | grep -q 'git-credentials-.*\.config'; then
    echo "::error::expired checkout credentials are still configured" >&2
    return 1
  fi
}

remove_checkout_credentials_file() {
  local path="$1"
  local base runner_temp
  base="$(basename "$path")"
  case "$base" in
    git-credentials-*.config) ;;
    *) return 0 ;;
  esac
  if [ -L "$path" ] || [ ! -f "$path" ]; then
    return 0
  fi
  runner_temp="${RUNNER_TEMP:-}"
  if [ -n "$runner_temp" ] && [[ "$path" == "$runner_temp"/* ]]; then
    rm -f "$path"
  fi
}

configure_fresh_github_auth() {
  if [ -z "${GIT_TOKEN:-}" ]; then
    echo "::error::GIT_TOKEN is required to push the daily digest" >&2
    return 1
  fi
  # The default Actions GITHUB_TOKEN cannot bypass main branch protection.
  # gh reads GH_TOKEN before GITHUB_TOKEN; set both so the fresh app token wins.
  export GH_TOKEN="$GIT_TOKEN"
  export GITHUB_TOKEN="$GIT_TOKEN"
  gh auth setup-git --hostname github.com --force
}

push_with_fresh_token() {
  local branch="${1:-}"
  local attempt sleep_seconds
  if [ -z "$branch" ]; then
    echo "::error::push branch is required" >&2
    return 1
  fi
  strip_persisted_checkout_auth
  configure_fresh_github_auth
  # Fail immediately if the fresh token is not offered, instead of prompting.
  export GIT_TERMINAL_PROMPT=0
  sleep_seconds="${PUSH_RETRY_SLEEP_SECONDS:-2}"
  for attempt in 1 2; do
    if git pull --rebase origin "$branch" && git push origin "HEAD:${branch}"; then
      return 0
    fi
    git rebase --abort >/dev/null 2>&1 || true
    echo "push rejected (attempt ${attempt}); retrying after rebase" >&2
    if [ "$attempt" -lt 2 ]; then
      sleep "$sleep_seconds"
    fi
  done
  echo "::error::failed to push daily digest after rebase retries" >&2
  return 1
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  push_with_fresh_token "${1:-}"
fi
