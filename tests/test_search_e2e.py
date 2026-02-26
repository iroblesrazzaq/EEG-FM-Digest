from __future__ import annotations

import hashlib
import json
import re
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

playwright_sync_api = pytest.importorskip("playwright.sync_api")
sync_playwright = playwright_sync_api.sync_playwright


MONTH_REQ_RE = re.compile(r"/digest/\d{4}-\d{2}/papers\.json(?:\?|$)")


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


def _build_synthetic_docs(root: Path) -> None:
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
    months_manifest = {
        "latest": month_b,
        "months": [
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
        ],
    }
    _write_json(root / "data" / "months.json", months_manifest)

    fallback_months = json.dumps([month_b, month_a], ensure_ascii=False)
    (root / "explore").mkdir(parents=True, exist_ok=True)
    (root / "explore" / "index.html").write_text(
        (
            "<!doctype html><html><body>"
            f"<main id='digest-app' class='container' data-view='explore' data-month='' data-manifest-json='../data/months.json' "
            f"data-fallback-months='{fallback_months}'>"
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


def test_explore_has_search_button_and_no_auto_load(browser, synthetic_site):
    context = browser.new_context()
    page, urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page.wait_for_timeout(150)

    assert page.get_by_test_id("search-run-btn").count() == 1
    assert _month_payload_request_count(urls) == 0
    assert page.locator(".paper-card").count() == 0
    assert "Showing" not in page.get_by_test_id("results-meta").inner_text()

    context.close()


def test_explore_empty_search_click_loads_all_months(browser, synthetic_site):
    context = browser.new_context()
    page, urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page.get_by_test_id("search-run-btn").click()
    stats = _wait_for_stats(page, "() => window.__digestTestHooks.getCacheStats().network_hits >= 2")

    assert _month_payload_request_count(urls) == 2
    assert stats["network_hits"] == 2
    assert page.locator(".paper-card").count() == 3
    assert "Showing 3 of 3 accepted papers" in page.get_by_test_id("results-meta").inner_text()

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
    page.get_by_test_id("search-run-btn").click()
    _wait_for_stats(page, "() => window.__digestTestHooks.getCacheStats().network_hits >= 2")

    page.locator("input[data-tag-category='paper_type'][data-tag-value='survey']").check()
    page.get_by_test_id("search-run-btn").click()
    page.wait_for_function("() => document.querySelectorAll('.paper-card').length === 1")

    title = page.locator(".paper-card h3").first.inner_text()
    assert "Beta Survey of EEG Foundation Models" in title

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
          window.__digestTestHooks.clearSessionCacheForTest();
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
    assert stats["network_hits"] == 1
    assert stats["cache_writes"] == 1
    assert stats["session_hits"] == 0
    assert stats["map_hits"] == 0

    context.close()


def test_month_page_session_path_when_mem_miss_session_hit(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/digest/2025-01/index.html", wait_until="networkidle")
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.clearSessionCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )
    page.evaluate(
        """() => window.__digestTestHooks.loadMonthPayloadForTest({
          month: "2025-01",
          jsonPath: "/digest/2025-01/papers.json",
          view: "month",
          monthRev: "rev-session-test"
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
          monthRev: "rev-session-test"
        })"""
    )
    stats = page.evaluate("() => window.__digestTestHooks.getCacheStats()")
    assert stats["session_hits"] == 1
    assert stats["network_hits"] == 0
    assert stats["map_hits"] == 0

    context.close()


def test_month_page_map_path_when_mem_hit(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/digest/2025-01/index.html", wait_until="networkidle")
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.clearSessionCacheForTest();
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
    assert stats["map_hits"] == 1
    assert stats["network_hits"] == 0

    context.close()


def test_search_network_fallback_on_miss(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.clearSessionCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )
    page.get_by_test_id("search-run-btn").click()
    stats = _wait_for_stats(page, "() => window.__digestTestHooks.getCacheStats().network_hits >= 2")
    assert stats["network_hits"] == 2
    assert stats["cache_writes"] >= 2

    context.close()


def test_search_session_hit_after_same_tab_navigation(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.clearSessionCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )
    page.get_by_test_id("search-run-btn").click()
    _wait_for_stats(page, "() => window.__digestTestHooks.getCacheStats().network_hits >= 2")

    # Navigate within the same tab so sessionStorage survives while JS memory is rebuilt.
    page.goto(f"{synthetic_site['base_url']}/digest/2025-01/index.html", wait_until="networkidle")
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )
    page.get_by_test_id("search-run-btn").click()
    stats = _wait_for_stats(page, "() => window.__digestTestHooks.getCacheStats().session_hits >= 2")
    assert stats["session_hits"] >= 2
    assert stats["network_hits"] == 0

    context.close()


def test_search_map_hit_in_same_runtime(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page.evaluate(
        """() => {
          window.__digestTestHooks.clearMemCacheForTest();
          window.__digestTestHooks.clearSessionCacheForTest();
          window.__digestTestHooks.resetCacheStatsForTest();
        }"""
    )
    page.get_by_test_id("search-run-btn").click()
    _wait_for_stats(page, "() => window.__digestTestHooks.getCacheStats().network_hits >= 2")
    page.evaluate("() => window.__digestTestHooks.resetCacheStatsForTest()")
    page.get_by_test_id("search-run-btn").click()
    stats = _wait_for_stats(page, "() => window.__digestTestHooks.getCacheStats().map_hits >= 2")
    assert stats["map_hits"] >= 2
    assert stats["network_hits"] == 0

    context.close()
