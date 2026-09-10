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
        assert "Model Gallery" in nav_texts
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
    context = browser.new_context(viewport={"width": 1400, "height": 900})
    try:
        page = context.new_page()
        page.goto(f"{real_docs_site['base_url']}/models/index.html")
        assert "Model Gallery" in page.title()
        nav_texts = page.locator("nav.site-nav a.site-nav-link").all_text_contents()
        assert "Model Gallery" in nav_texts
        page.wait_for_selector('[id="arch-2510.21585"] .arch-graph-svg', timeout=8000)
        assert page.locator(".model-arch-card").count() >= 7
        card = page.locator('[id="arch-2510.21585"]')
        svg = card.locator(".arch-graph-svg")
        text = svg.text_content() or ""
        assert "Multi-head attention" in text
        assert "Feed forward" in text
        assert "RMSNorm" in text
        assert "22 ×" in text
        assert "GeGLU" in text
        brace = card.locator("[data-arch-brace='repeat']")
        block = card.locator(".arch-repeat-block")
        brace_box = brace.bounding_box()
        block_box = block.bounding_box()
        assert brace_box and block_box
        assert brace_box["height"] >= block_box["height"] * 0.8
        assert card.locator("[data-arch-ffn='gated']").count() == 1
        assert "GELU activation" in text
        layout = page.evaluate(
            """() => {
              const ffn = document.querySelector('[id="arch-2510.21585"] [data-arch-ffn="gated"]');
              const mul = ffn && ffn.querySelector("circle");
              const labels = {};
              if (ffn) {
                ffn.querySelectorAll("text").forEach((node) => {
                  labels[node.textContent || ""] = Number(node.getAttribute("y"));
                });
              }
              return {
                mulCy: mul ? Number(mul.getAttribute("cy")) : null,
                geluY: labels["GELU activation"] ?? null,
                topY: labels["Linear layer"] ?? null,
              };
            }"""
        )
        assert layout["mulCy"] is not None and layout["geluY"] is not None
        assert layout["geluY"] > layout["mulCy"], "GELU must sit on the spine below the multiply"
        toggle = card.locator("[data-arch-toggle]").first
        toggle.click()
        assert "GELU" in (svg.text_content() or "")
        assert page.locator("a[href*='hfviewer']").count() == 0
        for arxiv_id in ("2405.18765", "2410.19779", "2412.07236", "2502.06438", "2505.18185", "2510.22257", "2607.27308"):
            assert page.locator(f'[id="arch-{arxiv_id}"] .arch-graph-svg').count() == 1
        labram = page.locator('[id="arch-2405.18765"] .arch-graph-svg').text_content() or ""
        assert "VQ-VAE codebook" in labram
        assert "Frozen codebook" in labram
        assert "QK-Norm" in labram
        cbramod = page.locator('[id="arch-2412.07236"] .arch-graph-svg').text_content() or ""
        assert "Criss-cross attention" in cbramod
        assert "Spatial attention" not in cbramod
        assert "Temporal attention" not in cbramod
        assert "O(N²T)" in cbramod
        assert "O(NT²)" in cbramod
        assert "2-layer MLP" in cbramod
        luna = page.locator('[id="arch-2510.22257"] .arch-graph-svg').text_content() or ""
        assert "Channel unifier" in luna
        assert "Learned queries" in luna
        brainomni = page.locator('[id="arch-2505.18185"] .arch-graph-svg').text_content() or ""
        assert "Sensor encoder" in brainomni
        assert "VQ-VAE codebook" in brainomni
        femba = page.locator('[id="arch-2502.06438"] .arch-graph-svg').text_content() or ""
        assert "Bidirectional Mamba" in femba
        eegpt = page.locator('[id="arch-2410.19779"] .arch-graph-svg').text_content() or ""
        assert "BrainGPT" in eegpt
        assert "Autoregressive transformer" in eegpt
        assert "Causal attention" in eegpt
        assert "Next-token head" in eegpt
        assert "Electrode embedding" in eegpt
        assert "Causal mask" in eegpt
        families = page.locator(".arch-family").all_text_contents()
        assert len(families) >= 7
        assert len(set(families)) >= 6
        assert "Autoregressive transformer" in families
        assert "Bidirectional Mamba" in families
        assert "Criss-cross transformer" in families
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
        page.wait_for_selector('[id="2510.21585"]', timeout=8000)
        card = page.locator('[id="2510.21585"]')
        assert card.locator(".arch-graph-host").count() == 1
        page.wait_for_selector('[id="2510.21585"] .arch-graph-svg', timeout=8000)
        assert card.locator("a[href*='hfviewer']").count() == 0
        text = card.text_content() or ""
        assert "View architecture" not in text
        assert "Multi-head attention" in text
        assert "Feed forward" in text
        assert "22 ×" in text
    finally:
        context.close()

