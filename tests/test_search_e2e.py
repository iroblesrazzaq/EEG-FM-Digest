from __future__ import annotations

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


def _build_synthetic_docs(root: Path, *, include_missing_month: bool = False) -> None:
    asset_src = Path("docs/assets/site.js")
    asset_dst = root / "assets" / "site.js"
    asset_dst.parent.mkdir(parents=True, exist_ok=True)
    asset_dst.write_text(asset_src.read_text(encoding="utf-8"), encoding="utf-8")

    month_a = "2025-01"
    month_b = "2025-02"

    payload_a = {
        "month": month_a,
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
            "featured": {
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
            "featured": {
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


def test_explore_has_search_button_and_no_auto_load(browser, synthetic_site):
    context = browser.new_context()
    page, urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page.wait_for_timeout(150)

    assert page.get_by_test_id("search-run-btn").count() == 1
    assert page.locator("input[data-tag-category]").count() > 0
    assert _month_payload_request_count(urls) == 0
    assert page.locator(".paper-card").count() == 0
    assert "Showing" not in page.get_by_test_id("results-meta").inner_text()

    context.close()


def test_explore_empty_search_click_loads_all_months(browser, synthetic_site):
    context = browser.new_context()
    page, urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page.get_by_test_id("search-run-btn").click()
    stats = _wait_for_stats(page, "() => window.__digestTestHooks.getCacheStats().cumulative.network_hits >= 2")
    cumulative = _cumulative(stats)
    last_run = _last_run(stats)

    assert _month_payload_request_count(urls) == 2
    assert cumulative["network_hits"] == 2
    assert last_run["months_total"] == 2
    assert last_run["months_loaded"] == 2
    assert page.locator(".paper-card").count() == 3
    assert page.get_by_test_id("results-meta").inner_text() == "3 results"

    context.close()


def test_explore_keyword_search_filters_results_after_click(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page.get_by_test_id("search-input").fill("Alpha")
    page.get_by_test_id("search-run-btn").click()
    page.wait_for_function("() => document.querySelectorAll('.paper-card').length === 1")

    title = page.locator(".paper-card h3").first.inner_text()
    assert "Alpha EEG Foundation Model" in title

    context.close()


def test_explore_tag_search_filters_results_after_click(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page.locator("input[data-tag-category='paper_type'][data-tag-value='survey']").check()
    page.get_by_test_id("search-run-btn").click()
    page.wait_for_function("() => document.querySelectorAll('.paper-card').length === 1")

    title = page.locator(".paper-card h3").first.inner_text()
    assert "Beta Survey of EEG Foundation Models" in title

    context.close()


def test_explore_tag_term_requires_checkbox_filter(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")

    page.get_by_test_id("search-input").fill("time patch")
    page.get_by_test_id("search-run-btn").click()
    _wait_for_stats(page, "() => window.__digestTestHooks.getCacheStats().cumulative.network_hits >= 2")
    assert page.locator(".paper-card").count() == 0

    page.get_by_test_id("search-input").fill("")
    page.locator("input[data-tag-category='tokenization'][data-tag-value='time-patch']").check()
    page.get_by_test_id("search-run-btn").click()
    page.wait_for_function("() => document.querySelectorAll('.paper-card').length === 2")
    assert page.locator(".paper-card").count() == 2

    context.close()


def test_explore_meta_does_not_increment_before_search(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")

    meta = page.get_by_test_id("results-meta").inner_text()
    assert "Loading " not in meta
    assert "Showing " not in meta

    context.close()


def test_month_page_network_path_on_full_cache_miss(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/digest/2025-01/index.html", wait_until="networkidle")
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.clearPersistentCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )
    page.evaluate(
        """() => window.__digestTestHooks.loadMonthPayloadForTest({
          month: "2025-01",
          jsonPath: "/digest/2025-01/papers.json",
          view: "month",
          monthRev: "rev-network-test"
        })"""
    )
    stats = page.evaluate("() => window.__digestTestHooks.getCacheStats()")
    cumulative = _cumulative(stats)
    assert cumulative["network_hits"] == 1
    assert cumulative["cache_writes"] == 1
    assert cumulative["local_hits"] == 0
    assert cumulative["map_hits"] == 0

    context.close()


def test_month_page_local_path_when_mem_miss_local_hit(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/digest/2025-01/index.html", wait_until="networkidle")
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.clearPersistentCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )
    page.evaluate(
        """() => window.__digestTestHooks.loadMonthPayloadForTest({
          month: "2025-01",
          jsonPath: "/digest/2025-01/papers.json",
          view: "month",
          monthRev: "rev-local-test"
        })"""
    )
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )
    page.evaluate(
        """() => window.__digestTestHooks.loadMonthPayloadForTest({
          month: "2025-01",
          jsonPath: "/digest/2025-01/papers.json",
          view: "month",
          monthRev: "rev-local-test"
        })"""
    )
    stats = page.evaluate("() => window.__digestTestHooks.getCacheStats()")
    cumulative = _cumulative(stats)
    assert cumulative["local_hits"] == 1
    assert cumulative["network_hits"] == 0
    assert cumulative["map_hits"] == 0

    context.close()


def test_month_page_map_path_when_mem_hit(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/digest/2025-01/index.html", wait_until="networkidle")
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.clearPersistentCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )
    page.evaluate(
        """() => window.__digestTestHooks.loadMonthPayloadForTest({
          month: "2025-01",
          jsonPath: "/digest/2025-01/papers.json",
          view: "month",
          monthRev: "rev-map-test"
        })"""
    )
    page.evaluate("() => window.__digestTestHooks.resetCacheStatsForTest()")
    page.evaluate(
        """() => window.__digestTestHooks.loadMonthPayloadForTest({
          month: "2025-01",
          jsonPath: "/digest/2025-01/papers.json",
          view: "month",
          monthRev: "rev-map-test"
        })"""
    )
    stats = page.evaluate("() => window.__digestTestHooks.getCacheStats()")
    cumulative = _cumulative(stats)
    assert cumulative["map_hits"] == 1
    assert cumulative["network_hits"] == 0

    context.close()


def test_search_network_fallback_on_miss(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.clearPersistentCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )
    page.get_by_test_id("search-run-btn").click()
    stats = _wait_for_stats(page, "() => window.__digestTestHooks.getCacheStats().cumulative.network_hits >= 2")
    cumulative = _cumulative(stats)
    assert cumulative["network_hits"] == 2
    assert cumulative["cache_writes"] >= 2

    context.close()


def test_search_local_hit_after_same_tab_navigation(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.clearPersistentCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )
    page.get_by_test_id("search-run-btn").click()
    _wait_for_stats(page, "() => window.__digestTestHooks.getCacheStats().cumulative.network_hits >= 2")

    # Navigate within the same tab so persistent cache survives while JS memory is rebuilt.
    page.goto(f"{synthetic_site['base_url']}/index.html", wait_until="networkidle")
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )
    page.get_by_test_id("search-run-btn").click()
    stats = _wait_for_stats(page, "() => window.__digestTestHooks.getCacheStats().cumulative.local_hits >= 2")
    cumulative = _cumulative(stats)
    assert cumulative["local_hits"] >= 2
    assert cumulative["network_hits"] == 0

    context.close()


def test_search_map_hit_in_same_runtime(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.clearPersistentCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )
    page.get_by_test_id("search-run-btn").click()
    _wait_for_stats(page, "() => window.__digestTestHooks.getCacheStats().cumulative.network_hits >= 2")
    page.evaluate("() => window.__digestTestHooks.resetCacheStatsForTest()")
    page.get_by_test_id("search-run-btn").click()
    stats = _wait_for_stats(page, "() => window.__digestTestHooks.getCacheStats().cumulative.map_hits >= 2")
    cumulative = _cumulative(stats)
    assert cumulative["map_hits"] >= 2
    assert cumulative["network_hits"] == 0

    context.close()


def test_search_local_hit_in_new_tab_same_origin(browser, synthetic_site):
    context = browser.new_context()
    page1, _urls1 = _tracked_page(context)
    page1.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page1.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.clearPersistentCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )
    page1.get_by_test_id("search-run-btn").click()
    _wait_for_stats(page1, "() => window.__digestTestHooks.getCacheStats().cumulative.network_hits >= 2")

    page2, _urls2 = _tracked_page(context)
    page2.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page2.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )
    page2.get_by_test_id("search-run-btn").click()
    stats = _wait_for_stats(page2, "() => window.__digestTestHooks.getCacheStats().cumulative.local_hits >= 2")
    cumulative = _cumulative(stats)
    assert cumulative["local_hits"] >= 2
    assert cumulative["network_hits"] == 0

    context.close()


def test_search_last_run_resets_each_click_and_cumulative_accumulates(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.clearPersistentCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )

    page.get_by_test_id("search-run-btn").click()
    first = _wait_for_stats(
        page,
        """() => {
          const stats = window.__digestTestHooks.getCacheStats();
          return Boolean(stats.last_run) && stats.last_run.months_loaded === 2 && stats.last_run.network_hits >= 2;
        }""",
    )
    first_cumulative = _cumulative(first)
    first_last_run = _last_run(first)
    assert first_last_run["months_total"] == 2
    assert first_last_run["months_loaded"] == 2
    assert first_last_run["network_hits"] == 2
    assert first_last_run["map_hits"] == 0
    assert first_last_run["local_hits"] == 0

    page.get_by_test_id("search-run-btn").click()
    second = _wait_for_stats(
        page,
        """() => {
          const stats = window.__digestTestHooks.getCacheStats();
          return Boolean(stats.last_run)
            && stats.last_run.months_loaded === 2
            && stats.last_run.network_hits === 0
            && stats.last_run.map_hits >= 2;
        }""",
    )
    second_cumulative = _cumulative(second)
    second_last_run = _last_run(second)
    assert second_last_run["months_total"] == 2
    assert second_last_run["months_loaded"] == 2
    assert second_last_run["network_hits"] == 0
    assert second_last_run["local_hits"] == 0
    assert second_last_run["map_hits"] >= 2
    assert second_cumulative["network_hits"] == first_cumulative["network_hits"]
    assert second_cumulative["map_hits"] >= first_cumulative["map_hits"] + 2

    context.close()


def test_month_page_month_rev_change_forces_network_refetch(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/digest/2025-01/index.html", wait_until="networkidle")
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.clearPersistentCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )
    page.evaluate(
        """() => window.__digestTestHooks.loadMonthPayloadForTest({
          month: "2025-01",
          jsonPath: "/digest/2025-01/papers.json",
          view: "month",
          monthRev: "rev-A"
        })"""
    )
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )
    page.evaluate(
        """() => window.__digestTestHooks.loadMonthPayloadForTest({
          month: "2025-01",
          jsonPath: "/digest/2025-01/papers.json",
          view: "month",
          monthRev: "rev-B"
        })"""
    )
    stats = page.evaluate("() => window.__digestTestHooks.getCacheStats()")
    cumulative = _cumulative(stats)
    assert cumulative["network_hits"] == 1
    assert cumulative["local_hits"] == 0
    assert cumulative["map_hits"] == 0

    context.close()


def test_month_page_legacy_session_entry_migrates_to_local(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    month = "2025-01"
    month_rev = "rev-migration-test"
    key = _month_cache_key(month, month_rev)
    page.goto(f"{synthetic_site['base_url']}/digest/2025-01/index.html", wait_until="networkidle")
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.clearPersistentCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )
    _seed_storage_with_month_payload(page, "session", key, "/digest/2025-01/papers.json")
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )
    page.evaluate(
        f"""() => window.__digestTestHooks.loadMonthPayloadForTest({{
          month: "{month}",
          jsonPath: "/digest/2025-01/papers.json",
          view: "month",
          monthRev: "{month_rev}"
        }})"""
    )
    stats = page.evaluate("() => window.__digestTestHooks.getCacheStats()")
    cumulative = _cumulative(stats)
    assert cumulative["network_hits"] == 0
    assert cumulative["local_hits"] == 1
    assert _storage_get_item(page, "local", key) is not None
    assert _storage_get_item(page, "session", key) is None

    context.close()


def test_month_page_corrupt_local_entry_removed_and_refetched(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    month = "2025-01"
    month_rev = "rev-corrupt-local-test"
    key = _month_cache_key(month, month_rev)
    page.goto(f"{synthetic_site['base_url']}/digest/2025-01/index.html", wait_until="networkidle")
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.clearPersistentCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )
    _storage_set_item(page, "local", key, "{this-is:not-json")
    page.evaluate(
        f"""() => window.__digestTestHooks.loadMonthPayloadForTest({{
          month: "{month}",
          jsonPath: "/digest/2025-01/papers.json",
          view: "month",
          monthRev: "{month_rev}"
        }})"""
    )
    stats = page.evaluate("() => window.__digestTestHooks.getCacheStats()")
    cumulative = _cumulative(stats)
    assert cumulative["network_hits"] == 1
    assert cumulative["local_hits"] == 0
    stored = _storage_get_item(page, "local", key)
    assert stored is not None
    parsed = json.loads(stored)
    assert parsed.get("month") == "2025-01"

    context.close()


def test_search_localstorage_failures_fallback_to_mem_and_network(browser, synthetic_site):
    context = browser.new_context()
    context.add_init_script(
        """() => {
          const proto = Storage.prototype;
          const originalGetItem = proto.getItem;
          const originalSetItem = proto.setItem;
          const originalRemoveItem = proto.removeItem;
          proto.getItem = function(key) {
            if (this === window.localStorage) {
              throw new Error("local_get_blocked");
            }
            return originalGetItem.call(this, key);
          };
          proto.setItem = function(key, value) {
            if (this === window.localStorage) {
              throw new Error("local_set_blocked");
            }
            return originalSetItem.call(this, key, value);
          };
          proto.removeItem = function(key) {
            if (this === window.localStorage) {
              throw new Error("local_remove_blocked");
            }
            return originalRemoveItem.call(this, key);
          };
        }"""
    )
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.clearPersistentCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )
    page.get_by_test_id("search-run-btn").click()
    first = _wait_for_stats(page, "() => window.__digestTestHooks.getCacheStats().cumulative.network_hits >= 2")
    first_cumulative = _cumulative(first)
    assert first_cumulative["network_hits"] == 2
    assert first_cumulative["local_hits"] == 0
    assert page.locator(".paper-card").count() == 3
    assert page.get_by_test_id("results-meta").inner_text() == "3 results"

    page.goto(f"{synthetic_site['base_url']}/index.html", wait_until="networkidle")
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )
    page.get_by_test_id("search-run-btn").click()
    second = _wait_for_stats(page, "() => window.__digestTestHooks.getCacheStats().cumulative.network_hits >= 2")
    second_cumulative = _cumulative(second)
    assert second_cumulative["network_hits"] == 2
    assert second_cumulative["local_hits"] == 0
    assert second_cumulative["map_hits"] == 0

    context.close()


def test_explore_no_partial_cards_rendered_while_loading(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.clearPersistentCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )

    def _delayed_month_payloads(route, request):
        if MONTH_REQ_RE.search(request.url):
            time.sleep(0.2)
        route.continue_()

    page.route("**/digest/*/papers.json", _delayed_month_payloads)
    page.get_by_test_id("search-run-btn").click()
    page.wait_for_function(
        """() => {
          const meta = document.querySelector("[data-testid='results-meta']");
          return Boolean(meta) && meta.textContent === "Searching...";
        }"""
    )
    assert page.locator(".paper-card").count() == 0
    page.wait_for_timeout(80)
    assert page.get_by_test_id("results-meta").inner_text() == "Searching..."
    assert page.locator(".paper-card").count() == 0

    _wait_for_stats(page, "() => window.__digestTestHooks.getCacheStats().cumulative.network_hits >= 2")
    assert page.get_by_test_id("results-meta").inner_text() == "3 results"
    assert page.locator(".paper-card").count() == 3

    context.close()


def test_explore_partial_month_fetch_failure_still_returns_final_count(browser, synthetic_site_with_missing_month):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site_with_missing_month['base_url']}/explore/index.html", wait_until="networkidle")
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.clearPersistentCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )

    page.get_by_test_id("search-run-btn").click()
    stats = _wait_for_stats(
        page,
        """() => {
          const cacheStats = window.__digestTestHooks.getCacheStats();
          return Boolean(cacheStats.last_run) && cacheStats.last_run.months_total === 3 && cacheStats.last_run.months_loaded === 3;
        }""",
    )
    cumulative = _cumulative(stats)
    last_run = _last_run(stats)
    meta = page.get_by_test_id("results-meta").inner_text()
    assert cumulative["network_hits"] == 2
    assert last_run["months_total"] == 3
    assert last_run["months_loaded"] == 3
    assert page.locator(".paper-card").count() == 3
    assert meta == "3 results"
    assert "Showing" not in meta

    context.close()


def test_clear_session_alias_clears_local_persistent_cache(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    month = "2025-01"
    month_rev = "rev-alias-clear-test"
    key = _month_cache_key(month, month_rev)
    page.goto(f"{synthetic_site['base_url']}/digest/2025-01/index.html", wait_until="networkidle")
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.clearPersistentCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )
    _storage_set_item(page, "local", key, "{}")
    assert _storage_get_item(page, "local", key) is not None
    page.evaluate("() => window.__digestTestHooks.clearSessionCacheForTest()")
    assert _storage_get_item(page, "local", key) is None

    context.close()
