from eegfm_digest.volume_chart import accepted_counts_from_manifest


def test_accepted_counts_from_manifest_sorted_and_highlights_latest():
    manifest = {
        "latest": "2026-07",
        "months": [
            {"month": "2026-07", "stats": {"accepted": 6}},
            {"month": "2026-01", "stats": {"accepted": 2}},
            {"month": "2025-12", "stats": {"accepted": "3"}},
        ],
    }
    rows = accepted_counts_from_manifest(manifest)
    assert [row["month"] for row in rows] == ["2025-12", "2026-01", "2026-07"]
    assert [row["accepted"] for row in rows] == [3, 2, 6]
    assert [row["is_current"] for row in rows] == [False, False, True]


def test_accepted_counts_ignores_bad_rows():
    rows = accepted_counts_from_manifest(
        {
            "latest": "2026-01",
            "months": [
                {"month": "", "stats": {"accepted": 1}},
                "bad",
                {"month": "2026-01", "stats": {"accepted": None}},
            ],
        }
    )
    assert rows == [{"month": "2026-01", "accepted": 0, "is_current": True}]
