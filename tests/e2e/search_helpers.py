"""Synthetic docs, browser fixtures, and helpers for search E2E tests."""

import hashlib
import json
import re
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

playwright_sync_api = pytest.importorskip("playwright.sync_api")
sync_playwright = playwright_sync_api.sync_playwright


MONTH_REQ_RE = re.compile(r"/digest/\d{4}-\d{2}/papers\.json(?:\?|$)")
MONTH_CACHE_PREFIX = "eegfm:monthPayload"
MONTH_CACHE_SCHEMA_VERSION = "v1"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _paper(
    arxiv_id_base: str,
    title: str,
    month: str,
    tags: dict[str, list[str]],
    one_liner: str,
    published: str,
) -> dict:
    return {
        "arxiv_id": f"{arxiv_id_base}v1",
        "arxiv_id_base": arxiv_id_base,
        "authors": ["Author A"],
        "categories": ["cs.LG"],
        "links": {"abs": f"https://arxiv.org/abs/{arxiv_id_base}", "pdf": ""},
        "published_date": published,
        "summary_failed_reason": "",
        "title": title,
        "triage": {"decision": "accept", "confidence": 0.9, "reasons": ["fit"]},
        "summary": {
            "arxiv_id_base": arxiv_id_base,
            "categories": ["cs.LG"],
            "data_scale": {"datasets": [], "subjects": None, "eeg_hours": None, "channels": None},
            "detailed_summary": f"{one_liner} Detailed summary for {title}.",
            "evaluation": {"tasks": [], "benchmarks": [], "headline_results": []},
            "key_points": [
                f"{title} point one.",
                f"{title} point two.",
                f"{title} point three.",
            ],
            "limitations": ["limitation one", "limitation two"],
            "method": {"architecture": "Transformer", "objective": "Masked", "pretraining": "SSL", "finetuning": None},
            "notes": f"synthetic:{month}",
            "one_liner": one_liner,
            "open_source": {"code_url": "", "weights_url": "", "license": ""},
            "paper_type": "new_model",
            "published_date": published,
            "tags": tags,
            "title": title,
            "unique_contribution": f"Unique contribution of {title}.",
            "used_fulltext": True,
        },
    }


def _build_synthetic_docs(
    root: Path,
    *,
    include_missing_month: bool = False,
    null_featured_month: str | None = None,
) -> None:
    asset_src = Path("docs/assets/site.js")
    asset_dst = root / "assets" / "site.js"
    asset_dst.parent.mkdir(parents=True, exist_ok=True)
    asset_dst.write_text(asset_src.read_text(encoding="utf-8"), encoding="utf-8")

    month_a = "2025-01"
    month_b = "2025-02"

    payload_a = {
        "month": month_a,
        "featured_paper_id": None if null_featured_month == month_a else "2501.00001",
        "stats": {"candidates": 3, "accepted": 2, "summarized": 2},
        "top_picks": ["2501.00001"],
        "papers": [
            _paper(
                "2501.00001",
                "Alpha EEG Foundation Model",
                month_a,
                {
                    "paper_type": ["new-model"],
                    "backbone": ["transformer"],
                    "objective": ["masked-reconstruction"],
                    "tokenization": ["time-patch"],
                    "topology": ["fixed-montage"],
                },
                "Alpha one-liner with transformer pretraining.",
                "2025-01-10",
            ),
            _paper(
                "2501.00002",
                "Beta Survey of EEG Foundation Models",
                month_a,
                {
                    "paper_type": ["survey"],
                    "backbone": ["transformer"],
                    "objective": ["contrastive"],
                    "tokenization": ["time-patch"],
                    "topology": ["channel-flexible"],
                },
                "Beta survey one-liner.",
                "2025-01-12",
            ),
        ],
    }
    payload_b = {
        "month": month_b,
        "featured_paper_id": None if null_featured_month == month_b else "2502.00001",
        "stats": {"candidates": 2, "accepted": 1, "summarized": 1},
        "top_picks": ["2502.00001"],
        "papers": [
            _paper(
                "2502.00001",
                "Gamma Benchmark for EEG-FM Transfer",
                month_b,
                {
                    "paper_type": ["benchmark"],
                    "backbone": ["mamba-ssm"],
                    "objective": ["autoregressive"],
                    "tokenization": ["discrete-tokens"],
                    "topology": ["channel-flexible"],
                },
                "Gamma benchmark one-liner.",
                "2025-02-03",
            ),
        ],
    }

    _write_json(root / "digest" / month_a / "papers.json", payload_a)
    _write_json(root / "digest" / month_b / "papers.json", payload_b)

    rev_a = hashlib.sha256((root / "digest" / month_a / "papers.json").read_bytes()).hexdigest()[:16]
    rev_b = hashlib.sha256((root / "digest" / month_b / "papers.json").read_bytes()).hexdigest()[:16]
    month_rows = [
        {
            "month": month_b,
            "month_label": "February 2025",
            "href": f"digest/{month_b}/index.html",
            "json_path": f"digest/{month_b}/papers.json",
            "month_rev": rev_b,
            "stats": payload_b["stats"],
            "empty_state": "has_papers",
            "featured": None
            if null_featured_month == month_b
            else {
                "arxiv_id_base": "2502.00001",
                "title": "Gamma Benchmark for EEG-FM Transfer",
                "one_liner": "Gamma benchmark one-liner.",
                "abs_url": "https://arxiv.org/abs/2502.00001",
            },
        },
        {
            "month": month_a,
            "month_label": "January 2025",
            "href": f"digest/{month_a}/index.html",
            "json_path": f"digest/{month_a}/papers.json",
            "month_rev": rev_a,
            "stats": payload_a["stats"],
            "empty_state": "has_papers",
            "featured": None
            if null_featured_month == month_a
            else {
                "arxiv_id_base": "2501.00001",
                "title": "Alpha EEG Foundation Model",
                "one_liner": "Alpha one-liner with transformer pretraining.",
                "abs_url": "https://arxiv.org/abs/2501.00001",
            },
        },
    ]
    fallback_months = [month_b, month_a]
    latest_month = month_b
    if include_missing_month:
        month_c = "2025-03"
        month_rows.insert(
            0,
            {
                "month": month_c,
                "month_label": "March 2025",
                "href": f"digest/{month_c}/index.html",
                "json_path": f"digest/{month_c}/papers.json",
                "month_rev": "missing-month-rev",
                "stats": {"candidates": 1, "accepted": 1, "summarized": 1},
                "empty_state": "has_papers",
                "featured": {
                    "arxiv_id_base": "2503.00001",
                    "title": "Missing Month Placeholder",
                    "one_liner": "This month payload intentionally returns 404.",
                    "abs_url": "https://arxiv.org/abs/2503.00001",
                },
            },
        )
        fallback_months.insert(0, month_c)
        latest_month = month_c
    months_manifest = {
        "latest": latest_month,
        "months": month_rows,
    }
    _write_json(root / "data" / "months.json", months_manifest)

    fallback_months_json = json.dumps(fallback_months, ensure_ascii=False)
    (root / "index.html").write_text(
        (
            "<!doctype html><html><body>"
            f"<main id='digest-app' class='container' data-view='home' data-month='' data-manifest-json='data/months.json' "
            f"data-fallback-months='{fallback_months_json}'>"
            "<section id='home-controls' class='controls'></section><section id='home-results'></section></main>"
            "<script src='assets/site.js'></script></body></html>"
        ),
        encoding="utf-8",
    )
    (root / "explore").mkdir(parents=True, exist_ok=True)
    (root / "explore" / "index.html").write_text(
        (
            "<!doctype html><html><body>"
            f"<main id='digest-app' class='container' data-view='explore' data-month='' data-manifest-json='../data/months.json' "
            f"data-fallback-months='{fallback_months_json}'>"
            "<h1>Search</h1><section id='controls' class='controls'></section>"
            "<p id='results-meta' class='small'></p><section id='results'></section></main>"
            "<script src='../assets/site.js'></script></body></html>"
        ),
        encoding="utf-8",
    )
    (root / "digest" / month_a).mkdir(parents=True, exist_ok=True)
    (root / "digest" / month_b).mkdir(parents=True, exist_ok=True)
    (root / "digest" / month_a / "index.html").write_text(
        (
            "<!doctype html><html><body>"
            f"<main id='digest-app' class='container' data-view='month' data-month='{month_a}' "
            "data-manifest-json='../../data/months.json' data-month-json='../../digest/2025-01/papers.json'>"
            "<section id='featured-paper'></section>"
            "<section id='controls' class='controls'></section>"
            "<p id='results-meta' class='small'></p><section id='results'></section></main>"
            "<script src='../../assets/site.js'></script></body></html>"
        ),
        encoding="utf-8",
    )
    (root / "digest" / month_b / "index.html").write_text(
        (
            "<!doctype html><html><body>"
            f"<main id='digest-app' class='container' data-view='month' data-month='{month_b}' "
            "data-manifest-json='../../data/months.json' data-month-json='../../digest/2025-02/papers.json'>"
            "<section id='featured-paper'></section>"
            "<section id='controls' class='controls'></section>"
            "<p id='results-meta' class='small'></p><section id='results'></section></main>"
            "<script src='../../assets/site.js'></script></body></html>"
        ),
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def synthetic_site(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_docs")
    _build_synthetic_docs(root)
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "root": root,
            "base_url": f"http://127.0.0.1:{server.server_port}",
        }
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


@pytest.fixture(scope="module")
def synthetic_site_with_missing_month(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_docs_missing_month")
    _build_synthetic_docs(root, include_missing_month=True)
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "root": root,
            "base_url": f"http://127.0.0.1:{server.server_port}",
        }
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


@pytest.fixture(scope="module")
def synthetic_site_with_null_featured_month(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_docs_null_featured")
    _build_synthetic_docs(root, null_featured_month="2025-01")
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "root": root,
            "base_url": f"http://127.0.0.1:{server.server_port}",
        }
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"Playwright browser unavailable: {exc}")
        try:
            yield browser
        finally:
            browser.close()


def _tracked_page(context):
    urls: list[str] = []
    page = context.new_page()
    page.on("request", lambda req: urls.append(req.url))
    return page, urls


def _month_payload_request_count(urls: list[str]) -> int:
    return sum(1 for url in urls if MONTH_REQ_RE.search(url))


def _wait_for_stats(page, predicate_js: str, timeout_ms: int = 4000):
    page.wait_for_function(predicate_js, timeout=timeout_ms)
    return page.evaluate("() => window.__digestTestHooks.getCacheStats()")


def _cumulative(stats: dict) -> dict:
    return stats.get("cumulative", stats)


def _last_run(stats: dict) -> dict:
    return stats.get("last_run") or {}


def _goto_explore_and_run_search_to_three_cards(page, synthetic_site: dict) -> None:
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page.get_by_test_id("search-run-btn").click()
    page.wait_for_function("() => document.querySelectorAll('.paper-card').length === 3")


def _synthetic_published_dates() -> tuple[str, ...]:
    """Top-level published_date on each synthetic fixture paper (see _paper)."""
    return ("2025-01-10", "2025-01-12", "2025-02-03")


def _month_cache_key(month: str, month_rev: str) -> str:
    return f"{MONTH_CACHE_PREFIX}:{MONTH_CACHE_SCHEMA_VERSION}:{month}:{month_rev}"


def _storage_set_item(page, store: str, key: str, value: str) -> None:
    page.evaluate(
        """([store, key, value]) => {
          const target = store === "local" ? window.localStorage : window.sessionStorage;
          target.setItem(key, value);
        }""",
        [store, key, value],
    )


def _storage_get_item(page, store: str, key: str):
    return page.evaluate(
        """([store, key]) => {
          const target = store === "local" ? window.localStorage : window.sessionStorage;
          return target.getItem(key);
        }""",
        [store, key],
    )


def _seed_storage_with_month_payload(page, store: str, key: str, payload_url: str) -> None:
    page.evaluate(
        """async ([store, key, payloadUrl]) => {
          const target = store === "local" ? window.localStorage : window.sessionStorage;
          const payload = await fetch(payloadUrl, { cache: "no-store" }).then((r) => r.json());
          target.setItem(key, JSON.stringify(payload));
        }""",
        [store, key, payload_url],
    )

