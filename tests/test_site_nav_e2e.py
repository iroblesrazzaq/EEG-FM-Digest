from __future__ import annotations

import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

playwright_sync_api = pytest.importorskip("playwright.sync_api")
sync_playwright = playwright_sync_api.sync_playwright


REPO_DOCS_ROOT = Path(__file__).resolve().parent.parent / "docs"


@pytest.fixture(scope="module")
def real_docs_site():
    """Serve the committed docs/ directory as a static site.

    Used for smoke tests that assert on the real, production-built HTML
    (titles, nav, About page). Interactive behaviour tests should continue
    to use the synthetic_site fixture in test_search_e2e.py so they
    remain deterministic regardless of content churn.
    """
    if not REPO_DOCS_ROOT.is_dir():
        pytest.skip(f"docs/ not found at {REPO_DOCS_ROOT}")
    handler = partial(SimpleHTTPRequestHandler, directory=str(REPO_DOCS_ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "root": REPO_DOCS_ROOT,
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


def test_home_page_renders(browser, real_docs_site):
    context = browser.new_context()
    try:
        page = context.new_page()
        page.goto(f"{real_docs_site['base_url']}/index.html")

        assert "EEG-FM Digest" in page.title()

        nav_links = page.locator("nav.site-nav a.site-nav-link")
        nav_texts = nav_links.all_text_contents()
        assert "Monthly Digest" in nav_texts
        assert "Search" in nav_texts
        assert "Models" in nav_texts
        assert "About" in nav_texts
        assert "GitHub Repo" in nav_texts

        app = page.locator("main#digest-app")
        assert app.get_attribute("data-view") == "home"

        page.wait_for_selector("details.year-block", timeout=5000)
        open_years = page.locator("details.year-block[open]")
        assert open_years.count() >= 1, "expected at least one year-block to be open on load"

        newest_open_year = open_years.first
        month_cards = newest_open_year.locator(".month-card")
        assert month_cards.count() >= 1, "expected month cards inside the open year"
    finally:
        context.close()


def test_models_tab_renders_reve_diagram(browser, real_docs_site):
    graph = real_docs_site["root"] / "data" / "arch" / "2510.21585.json"
    if not graph.is_file():
        pytest.skip("REVE architecture graph is not committed yet")
    context = browser.new_context()
    try:
        page = context.new_page()
        page.goto(f"{real_docs_site['base_url']}/models/index.html")
        assert "Models" in page.title()
        nav_texts = page.locator("nav.site-nav a.site-nav-link").all_text_contents()
        assert "Models" in nav_texts
        page.wait_for_selector(".arch-graph-svg", timeout=8000)
        assert page.locator(".model-arch-card").count() >= 1
        svg = page.locator(".arch-graph-svg").first
        assert "Transformer encoder" in svg.inner_text()
        toggle = page.locator("[data-arch-toggle]").first
        toggle.click()
        page.wait_for_function(
            "() => document.body.innerText.includes('RMSNorm')",
            timeout=5000,
        )
        assert "GeGLU" in svg.inner_text()
        assert page.locator("a[href*='hfviewer']").count() == 0
    finally:
        context.close()


def test_october_reve_card_mounts_local_graph(browser, real_docs_site):
    graph = real_docs_site["root"] / "data" / "arch" / "2510.21585.json"
    if not graph.is_file():
        pytest.skip("REVE architecture graph is not committed yet")
    context = browser.new_context()
    try:
        page = context.new_page()
        page.goto(f"{real_docs_site['base_url']}/digest/2025-10/index.html")
        page.wait_for_selector("#2510.21585", timeout=8000)
        card = page.locator("#2510.21585")
        assert card.locator(".arch-graph-host").count() == 1
        page.wait_for_selector("#2510.21585 .arch-graph-svg", timeout=8000)
        assert card.locator("a[href*='hfviewer']").count() == 0
        assert "View architecture" not in card.inner_text()
        assert "Transformer encoder" in card.inner_text()
    finally:
        context.close()

