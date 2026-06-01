"""Search and cache E2E tests (Playwright)."""

from __future__ import annotations

from e2e.search_helpers import (
    MONTH_CACHE_PREFIX,
    MONTH_CACHE_SCHEMA_VERSION,
    MONTH_REQ_RE,
    _cumulative,
    _goto_explore_and_run_search_to_three_cards,
    _last_run,
    _month_cache_key,
    _month_payload_request_count,
    _seed_storage_with_month_payload,
    _storage_get_item,
    _storage_set_item,
    _synthetic_published_dates,
    _tracked_page,
    _wait_for_stats,
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
    meta = page.get_by_test_id("results-meta").inner_text()
    assert "Showing" not in meta
    assert "Loading" not in meta

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


def test_explore_clear_search_resets_query_tags_and_results(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page.get_by_test_id("search-input").fill("Alpha")
    page.locator("input[data-tag-category='paper_type'][data-tag-value='new-model']").check()
    page.get_by_test_id("search-run-btn").click()
    page.wait_for_function("() => document.querySelectorAll('.paper-card').length === 1")

    page.get_by_role("button", name="Clear search").click()
    page.wait_for_function("() => document.querySelectorAll('.paper-card').length === 3")

    assert page.get_by_test_id("search-input").input_value() == ""
    assert page.locator("input[data-tag-category]:checked").count() == 0
    assert page.get_by_test_id("results-meta").inner_text() == "3 results"

    context.close()


def test_explore_export_csv_downloads_current_filtered_results(browser, synthetic_site):
    context = browser.new_context(accept_downloads=True)
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page.get_by_test_id("search-input").fill("Alpha")
    page.get_by_test_id("search-run-btn").click()
    page.wait_for_function("() => document.querySelectorAll('.paper-card').length === 1")

    with page.expect_download() as download_info:
        page.get_by_test_id("export-results-btn").click()
    download = download_info.value
    csv_text = Path(download.path()).read_text(encoding="utf-8")

    assert download.suggested_filename == "eegfm-digest-search-all-months.csv"
    assert "Alpha EEG Foundation Model" in csv_text
    assert "Beta Survey of EEG Foundation Models" not in csv_text
    assert "Gamma Benchmark for EEG-FM Transfer" not in csv_text

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


def test_month_page_renders_null_featured_card(browser, synthetic_site_with_null_featured_month):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site_with_null_featured_month['base_url']}/digest/2025-01/index.html", wait_until="networkidle")

    page.wait_for_function("() => document.querySelectorAll('.paper-card').length === 2")
    card = page.get_by_test_id("featured-empty-card")
    assert card.count() == 1
    assert card.inner_text().strip() == "No featured paper this month. Check back soon!"

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

    paused_routes = []

    def _pause_month_payloads(route, request):
        if MONTH_REQ_RE.search(request.url):
            paused_routes.append(route)
            return
        route.continue_()

    page.route("**/digest/*/papers.json", _pause_month_payloads)
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
    page.wait_for_function("() => window.__digestTestHooks.getCacheStats().last_run?.months_total === 2")
    for _ in range(40):
        if len(paused_routes) >= 2:
            break
        page.wait_for_timeout(25)
    assert len(paused_routes) == 2
    for route in paused_routes:
        route.continue_()

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


def test_date_from_filter_narrows_results(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    _goto_explore_and_run_search_to_three_cards(page, synthetic_site)
    page.get_by_test_id("date-from").fill("2025-01-11")
    page.get_by_test_id("date-from").press("Tab")
    page.wait_for_function("() => document.querySelectorAll('.paper-card').length === 2")
    assert page.get_by_test_id("results-meta").inner_text() == "2 results"
    titles = page.locator(".paper-card h3").all_text_contents()
    assert any("Beta Survey" in t for t in titles)
    assert any("Gamma Benchmark" in t for t in titles)
    context.close()


def test_date_to_filter_narrows_results(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    _goto_explore_and_run_search_to_three_cards(page, synthetic_site)
    # Beta is 2025-01-12; cap must be >= that day to include both January papers.
    page.get_by_test_id("date-to").fill("2025-01-12")
    page.get_by_test_id("date-to").press("Tab")
    page.wait_for_function("() => document.querySelectorAll('.paper-card').length === 2")
    assert page.get_by_test_id("results-meta").inner_text() == "2 results"
    titles = page.locator(".paper-card h3").all_text_contents()
    assert any("Alpha EEG" in t for t in titles)
    assert any("Beta Survey" in t for t in titles)
    context.close()


def test_date_range_filter(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    _goto_explore_and_run_search_to_three_cards(page, synthetic_site)
    page.get_by_test_id("date-from").fill("2025-01-11")
    page.get_by_test_id("date-from").press("Tab")
    page.get_by_test_id("date-to").fill("2025-01-31")
    page.get_by_test_id("date-to").press("Tab")
    page.wait_for_function("() => document.querySelectorAll('.paper-card').length === 1")
    assert page.get_by_test_id("results-meta").inner_text() == "1 results"
    assert "Beta Survey of EEG Foundation Models" in page.locator(".paper-card h3").first.inner_text()
    context.close()


def test_date_filter_ands_with_tags(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    page.locator("input[data-tag-category='tokenization'][data-tag-value='time-patch']").check()
    page.get_by_test_id("search-run-btn").click()
    page.wait_for_function("() => document.querySelectorAll('.paper-card').length === 2")
    page.get_by_test_id("date-from").fill("2025-01-12")
    page.get_by_test_id("date-from").press("Tab")
    page.wait_for_function("() => document.querySelectorAll('.paper-card').length === 1")
    assert "Beta Survey of EEG Foundation Models" in page.locator(".paper-card h3").first.inner_text()
    context.close()


def test_date_filter_ands_with_text_query(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    _goto_explore_and_run_search_to_three_cards(page, synthetic_site)
    page.get_by_test_id("search-input").fill("Benchmark")
    page.get_by_test_id("search-run-btn").click()
    page.wait_for_function("() => document.querySelectorAll('.paper-card').length === 1")
    page.get_by_test_id("date-from").fill("2025-02-01")
    page.get_by_test_id("date-from").press("Tab")
    page.get_by_test_id("date-to").fill("2025-02-28")
    page.get_by_test_id("date-to").press("Tab")
    page.wait_for_function("() => document.querySelectorAll('.paper-card').length === 1")
    assert "Gamma Benchmark" in page.locator(".paper-card h3").first.inner_text()
    context.close()


def test_date_preset_last_3_months_sets_inputs_and_filters(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    _goto_explore_and_run_search_to_three_cards(page, synthetic_site)
    page.get_by_test_id("date-preset-3m").click()
    d_from = page.get_by_test_id("date-from").input_value()
    d_to = page.get_by_test_id("date-to").input_value()
    assert len(d_from) == 10 and len(d_to) == 10
    assert d_from <= d_to
    pubs = _synthetic_published_dates()
    expected = sum(1 for p in pubs if d_from <= p <= d_to)
    page.wait_for_function(
        f"() => document.querySelectorAll('.paper-card').length === {expected}",
    )
    assert page.get_by_test_id("results-meta").inner_text() == f"{expected} results"
    context.close()


def test_clear_search_resets_dates(browser, synthetic_site):
    context = browser.new_context()
    page, _urls = _tracked_page(context)
    _goto_explore_and_run_search_to_three_cards(page, synthetic_site)
    page.get_by_test_id("date-from").fill("2025-02-01")
    page.get_by_test_id("date-from").press("Tab")
    page.wait_for_function("() => document.querySelectorAll('.paper-card').length === 1")
    page.get_by_role("button", name="Clear search").click()
    page.wait_for_function("() => document.querySelectorAll('.paper-card').length === 3")
    assert page.get_by_test_id("date-from").input_value() == ""
    assert page.get_by_test_id("date-to").input_value() == ""
    assert page.get_by_test_id("results-meta").inner_text() == "3 results"
    context.close()


def test_tag_checkbox_auto_triggers_load(browser, synthetic_site):
    """Checking a tag before clicking Search should auto-load papers and filter live."""
    context = browser.new_context()
    page, urls = _tracked_page(context)
    page.goto(f"{synthetic_site['base_url']}/explore/index.html", wait_until="networkidle")
    # No search yet — zero cards, zero network requests for month payloads.
    assert page.locator(".paper-card").count() == 0
    assert _month_payload_request_count(urls) == 0

    # Check the "survey" tag — should auto-trigger load and show only Beta Survey.
    page.locator("input[data-tag-category='paper_type'][data-tag-value='survey']").check()
    page.wait_for_function("() => document.querySelectorAll('.paper-card').length === 1")

    assert _month_payload_request_count(urls) == 2  # both months loaded
    title = page.locator(".paper-card h3").first.inner_text()
    assert "Beta Survey of EEG Foundation Models" in title
    assert page.get_by_test_id("results-meta").inner_text() == "1 results"

    context.close()


def test_csv_export_respects_date_filter(browser, synthetic_site):
    context = browser.new_context(accept_downloads=True)
    page, _urls = _tracked_page(context)
    _goto_explore_and_run_search_to_three_cards(page, synthetic_site)
    page.get_by_test_id("date-from").fill("2025-01-11")
    page.get_by_test_id("date-from").press("Tab")
    page.get_by_test_id("date-to").fill("2025-01-31")
    page.get_by_test_id("date-to").press("Tab")
    page.wait_for_function("() => document.querySelectorAll('.paper-card').length === 1")

    with page.expect_download() as download_info:
        page.get_by_test_id("export-results-btn").click()
    download = download_info.value
    csv_text = Path(download.path()).read_text(encoding="utf-8")
    lines = [ln for ln in csv_text.strip().splitlines() if ln.strip()]
    assert len(lines) >= 2
    assert "Beta Survey of EEG Foundation Models" in csv_text
    assert "Alpha EEG Foundation Model" not in csv_text
    assert "Gamma Benchmark for EEG-FM Transfer" not in csv_text
    context.close()
