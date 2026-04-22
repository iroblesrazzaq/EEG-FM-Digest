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
