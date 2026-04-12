import json

from eegfm_digest.keywords import load_topic_config
from eegfm_digest.llm import load_api_key
from eegfm_digest.site import SiteBranding, load_site_config, render_month_page


def test_load_api_key_accepts_gemini_alias(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    assert load_api_key() == "test-key"


def test_load_topic_config_reads_custom_file(tmp_path):
    config_path = tmp_path / "my_topic.json"
    config_path.write_text(
        json.dumps(
            {
                "slug": "sleep_eeg",
                "title": "Sleep EEG Digest",
                "description": "Monthly digest for sleep EEG papers",
                "categories": ["cs.LG", "q-bio.NC"],
                "queries": ["all:(sleep EEG) AND all:(pretrain OR transfer)"],
            }
        ),
        encoding="utf-8",
    )

    topic = load_topic_config(config_path)

    assert topic.slug == "sleep_eeg"
    assert topic.title == "Sleep EEG Digest"
    assert topic.categories == ("cs.LG", "q-bio.NC")
    assert topic.queries == ("all:(sleep EEG) AND all:(pretrain OR transfer)",)


def test_render_month_page_uses_site_branding(monkeypatch):
    monkeypatch.setattr(
        "eegfm_digest.site.load_site_config",
        lambda: SiteBranding(
            title="Custom Digest",
            author="Researcher",
            description="Custom monthly digest",
            theme_color="#123456",
            links={"github": "https://github.com/example"},
        ),
    )

    html = render_month_page("2026-03", [], {}, {})

    assert "Custom Digest" in html
    assert "Researcher" in html
    assert "#123456" in html
    assert "Custom monthly digest for March 2026." in html
