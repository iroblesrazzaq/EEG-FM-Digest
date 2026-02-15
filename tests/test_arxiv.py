from eegfm_digest.arxiv import dedupe_latest, in_month, parse_arxiv_id


def test_parse_arxiv_id_version():
    base, ver = parse_arxiv_id("http://arxiv.org/abs/2501.01234v3")
    assert base == "2501.01234"
    assert ver == 3


def test_month_boundaries():
    assert in_month("2025-01-01T00:00:00Z", "2025-01")
    assert in_month("2025-01-31T23:59:59Z", "2025-01")
    assert not in_month("2024-12-31T23:59:59Z", "2025-01")
    assert not in_month("2025-02-01T00:00:00Z", "2025-01")


def test_dedupe_keeps_latest_version():
    rows = [
        {"arxiv_id_base": "2501.00001", "version": 1, "published": "2025-01-02T00:00:00Z"},
        {"arxiv_id_base": "2501.00001", "version": 3, "published": "2025-01-02T00:00:00Z"},
        {"arxiv_id_base": "2501.00002", "version": 1, "published": "2025-01-03T00:00:00Z"},
    ]
    out = dedupe_latest(rows)
    assert len(out) == 2
    keep = [r for r in out if r["arxiv_id_base"] == "2501.00001"][0]
    assert keep["version"] == 3
