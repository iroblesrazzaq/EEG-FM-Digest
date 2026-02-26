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


def test_site_js_uses_month_cache_loader_in_month_and_search_paths():
    site_js = Path("docs/assets/site.js").read_text(encoding="utf-8")
    assert "payload = await loadMonthPayloadCached({" in site_js
    assert "monthPayload = await loadMonthPayloadCached({" in site_js
    assert "const monthRev = normalizeMonthRev(item.month_rev);" in site_js
    assert "const monthRev = normalizeMonthRev(initialMonthRow?.month_rev);" in site_js

