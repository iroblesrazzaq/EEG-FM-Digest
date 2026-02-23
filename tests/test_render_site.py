from pathlib import Path

from eegfm_digest.site import render_month_page


def test_month_page_snapshot():
    digest = {"top_picks": ["2501.00001"], "sections": [{"title": "new_model", "paper_ids": ["2501.00001"]}]}
    summaries = [
        {
            "arxiv_id_base": "2501.00001",
            "title": "EEG FM One",
            "published_date": "2025-01-04",
            "paper_type": "new_model",
            "one_liner": "A concise line",
            "detailed_summary": (
                "This paper proposes a reusable EEG foundation model backbone for cross-task transfer. "
                "Its key novelty is a compact architecture that preserves performance while reducing compute."
            ),
            "unique_contribution": "A unique thing",
            "tags": {
                "paper_type": ["eeg-fm"],
                "backbone": ["transformer"],
                "objective": ["masked-reconstruction"],
                "tokenization": ["time-patch"],
                "topology": ["fixed-montage"],
            },
            "open_source": {"code_url": "https://code", "weights_url": None, "license": None},
        }
    ]
    metadata = {
        "2501.00001": {
            "authors": ["Alice", "Bob"],
            "links": {"abs": "https://arxiv.org/abs/2501.00001"},
        }
    }
    html = render_month_page("2025-01", summaries, metadata, digest)
    snapshot = Path("tests/fixtures/month_page_snapshot.html").read_text(encoding="utf-8")
    assert html.strip() == snapshot.strip()
