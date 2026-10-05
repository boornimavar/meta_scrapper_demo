from types import SimpleNamespace

import meta_design_scraper_session as scraper


def fake_ad(i):
    return {
        "id": str(i),
        "page": {"name": f"Brand {i}", "id": f"p{i}"},
        "creatives": [{"image_url": f"https://example.com/{i}.jpg", "body": "copy"}],
    }


class FakeCollector:
    """Stands in for MetaAdsCollector; never touches the network.

    With `fail_with`, the search raises that error after yielding `ads`,
    like the real collector does when Meta rate-limits partway through.
    """

    def __init__(self, ads, fail_with=None):
        self.ads = ads
        self.fail_with = fail_with

    def search(self, **kwargs):
        yield from self.ads
        if self.fail_with:
            raise self.fail_with


def use_fake_collector(monkeypatch, ads, fail_with=None):
    monkeypatch.setattr(scraper, "MetaAdsCollector", lambda: FakeCollector(ads, fail_with))


def test_rate_limit_partway_keeps_collected_ads(monkeypatch):
    error = RuntimeError("Max retries exceeded due to rate limiting")
    use_fake_collector(monkeypatch, [fake_ad(i) for i in range(3)], fail_with=error)
    interrupted = []

    records = scraper.search_ads_broad(query="beauty", static_only=True, on_interrupted=interrupted.append)

    assert len(records) == 3
    assert interrupted == [error]


def test_error_before_any_ad_is_raised(monkeypatch):
    use_fake_collector(monkeypatch, [], fail_with=RuntimeError("Max retries exceeded due to rate limiting"))

    try:
        scraper.search_ads_broad(query="beauty", static_only=True)
    except RuntimeError as exc:
        assert "rate limiting" in str(exc)
    else:
        raise AssertionError("expected the collector error to be raised")


def test_on_collect_reports_each_ad(monkeypatch):
    use_fake_collector(monkeypatch, [fake_ad(i) for i in range(3)])
    seen = []

    records = scraper.search_ads_broad(query="beauty", static_only=True, on_collect=seen.append)

    assert seen == [1, 2, 3]
    assert [r["ad_id"] for r in records] == ["0", "1", "2"]


def test_stop_check_ends_collection_early_and_keeps_partial_results(monkeypatch):
    use_fake_collector(monkeypatch, [fake_ad(i) for i in range(10)])
    seen = []

    records = scraper.search_ads_broad(
        query="beauty", static_only=True, on_collect=seen.append, stop_check=lambda: len(seen) >= 4,
    )

    assert seen == [1, 2, 3, 4]
    assert len(records) == 4


def test_pipeline_skips_analysis_when_stopped_during_collection(monkeypatch, tmp_path):
    use_fake_collector(monkeypatch, [fake_ad(i) for i in range(5)])
    monkeypatch.setattr(scraper, "load_models", lambda path: (_ for _ in ()).throw(AssertionError("analysis should not start")))
    stages = []
    args = SimpleNamespace(
        search_query="beauty", search_country="IN", search_status="ACTIVE", search_type="KEYWORD_UNORDERED",
        search_max_results=500, search_page_size=10, search_media_type="IMAGE", platforms=None,
        search_start_date=None, search_end_date=None, model="unused", search_sort_by=None,
    )

    scraper.run_broad_search_pipeline(
        args,
        progress_callback=lambda done, total, stage: stages.append((stage, done)),
        stop_check=lambda: len(stages) >= 3,
        db_path=tmp_path / "session.db",
        save_run_artifacts=False,
    )

    assert stages[0] == ("collecting", 0)
    assert stages[-1][0] == "stopped"
