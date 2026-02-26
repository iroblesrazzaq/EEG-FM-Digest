from pathlib import Path


def test_site_js_defines_month_cache_primitives():
    site_js = Path("docs/assets/site.js").read_text(encoding="utf-8")
    assert 'const MONTH_CACHE_SCHEMA_VERSION = "v1";' in site_js
    assert 'const MONTH_CACHE_PREFIX = "eegfm:monthPayload";' in site_js
    assert "const monthPayloadMem = new Map();" in site_js
    assert "function buildMonthCacheKey(month, monthRev)" in site_js
    assert "function getMonthPayloadFromCache(month, monthRev)" in site_js
    assert "function setMonthPayloadCache(month, monthRev, payload)" in site_js
    assert "async function loadMonthPayloadCached({ month, jsonPath, view, monthRev })" in site_js
    assert "function currentMonthCacheStats()" in site_js


def test_site_js_uses_month_cache_loader_in_month_and_search_paths():
    site_js = Path("docs/assets/site.js").read_text(encoding="utf-8")
    assert "payload = await loadMonthPayloadCached({" in site_js
    assert "monthPayload = await loadMonthPayloadCached({" in site_js
    assert "const monthRev = normalizeMonthRev(item.month_rev);" in site_js
    assert "const monthRev = normalizeMonthRev(initialMonthRow?.month_rev);" in site_js


def test_site_js_exposes_test_hooks_and_submit_driven_search():
    site_js = Path("docs/assets/site.js").read_text(encoding="utf-8")
    assert "window.__digestTestHooks = {" in site_js
    assert "loadMonthPayloadForTest: (args) => loadMonthPayloadCached(args)" in site_js
    assert "getCacheStats: () => currentMonthCacheStats()" in site_js
    assert "clearMemCacheForTest: () => clearMonthMemCache()" in site_js
    assert "clearSessionCacheForTest: () => clearMonthSessionCache()" in site_js
    assert 'id="search-run-btn"' in site_js
    assert "void runExploreSearch(app, state);" in site_js
    assert "await loadExploreMonthsLazy(app, state, state.monthRows, \"explore\");" in site_js
    assert "void loadExploreMonthsLazy(app, state, manifest.months, view);" not in site_js
