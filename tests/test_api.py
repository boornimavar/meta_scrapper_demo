import sqlite3

from conftest import insert_rows, make_row, wait_for_stage


QUERY = "category=skincare&country=IN"


def test_index_serves_dashboard(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Creative Lens" in res.data


def test_creatives_without_category_reads_session_db_only(client, app_env, fake_pipeline):
    insert_rows(app_env.SESSION_DB_PATH, [
        make_row(image_hash="aaaa000000000001", rank_score=0.6),
        make_row(image_hash="aaaa000000000002", rank_score=0.95),
        make_row(image_hash="aaaa000000000003", rank_score=0.2),
    ])

    data = client.get("/api/creatives").get_json()

    assert fake_pipeline["calls"] == []
    assert data["on_demand_scraped"] is False
    # design-led only by default (rank_score >= 0.5), highest score first
    assert [r["image_hash"] for r in data["rows"]] == ["aaaa000000000002", "aaaa000000000001"]


def test_design_led_only_false_includes_low_scores(client, app_env):
    insert_rows(app_env.SESSION_DB_PATH, [make_row(rank_score=0.1)])

    data = client.get("/api/creatives?design_led_only=false").get_json()

    assert len(data["rows"]) == 1


def test_new_category_starts_one_search_and_streams_results(client, fake_pipeline):
    fake_pipeline["rows"] = [make_row(image_hash="bbbb000000000001")]

    first = client.get(f"/api/creatives?{QUERY}").get_json()
    assert first["on_demand_scraped"] is True
    assert first["scrape_target"] == {"category": "skincare", "country": "IN"}

    status = wait_for_stage(client, QUERY, {"done"})
    assert status["matches"] == 1

    second = client.get(f"/api/creatives?{QUERY}").get_json()
    assert second["on_demand_scraped"] is False
    assert [r["image_hash"] for r in second["rows"]] == ["bbbb000000000001"]
    assert fake_pipeline["calls"] == ["skincare"]


def test_category_case_and_spacing_count_as_same_search(client, fake_pipeline):
    client.get(f"/api/creatives?{QUERY}")
    wait_for_stage(client, QUERY, {"done"})

    again = client.get("/api/creatives?category=%20SkinCare%20&country=in").get_json()

    assert again["on_demand_scraped"] is False
    assert len(fake_pipeline["calls"]) == 1


def test_scrape_status_idle_when_nothing_ran(client):
    assert client.get(f"/api/scrape-status?{QUERY}").get_json() == {"stage": "idle"}


def test_stop_with_nothing_running(client):
    data = client.post(f"/api/stop-scrape?{QUERY}").get_json()
    assert data["stopping"] is False


def test_stop_running_search(client, fake_pipeline):
    fake_pipeline["behaviour"] = "wait"
    client.get(f"/api/creatives?{QUERY}")

    assert client.post(f"/api/stop-scrape?{QUERY}").get_json()["stopping"] is True
    assert wait_for_stage(client, QUERY, {"stopped"})["stage"] == "stopped"


def test_failed_search_can_be_retried(client, fake_pipeline):
    fake_pipeline["behaviour"] = "fail"
    client.get(f"/api/creatives?{QUERY}")
    assert wait_for_stage(client, QUERY, {"failed"})["stage"] == "failed"

    fake_pipeline["behaviour"] = "insert"
    retry = client.get(f"/api/creatives?{QUERY}").get_json()

    assert retry["on_demand_scraped"] is True
    assert wait_for_stage(client, QUERY, {"done"})["stage"] == "done"
    assert len(fake_pipeline["calls"]) == 2


def test_categories_lists_distinct_representatives(client, app_env):
    insert_rows(app_env.SESSION_DB_PATH, [
        make_row(image_hash="cccc000000000001", category="skincare"),
        make_row(image_hash="cccc000000000002", category="fashion"),
        make_row(image_hash="cccc000000000003", category="fashion"),
        make_row(image_hash="cccc000000000004", category="hidden", is_creative_representative=0),
    ])

    assert client.get("/api/categories").get_json() == ["fashion", "skincare"]


def test_save_snapshot_survives_session_reset(client, app_env):
    insert_rows(app_env.SESSION_DB_PATH, [make_row(image_hash="dddd000000000001", brand="Kept Brand")])

    res = client.post("/api/save/dddd000000000001").get_json()
    assert res == {"ok": True, "saved": True, "snapshot_saved": True}

    # Simulate a server restart: the session DB is wiped, saved DB is not.
    app_env.SESSION_DB_PATH.unlink()

    saved = client.get("/api/saved").get_json()["rows"]
    assert len(saved) == 1
    assert saved[0]["brand"] == "Kept Brand"
    assert saved[0]["is_saved"] is True


def test_saved_flag_shown_in_creatives(client, app_env):
    insert_rows(app_env.SESSION_DB_PATH, [
        make_row(image_hash="eeee000000000001"),
        make_row(image_hash="eeee000000000002"),
    ])
    client.post("/api/save/eeee000000000001")

    rows = {r["image_hash"]: r["is_saved"] for r in client.get("/api/creatives").get_json()["rows"]}

    assert rows == {"eeee000000000001": True, "eeee000000000002": False}


def test_unsave_removes_bookmark(client, app_env):
    insert_rows(app_env.SESSION_DB_PATH, [make_row(image_hash="ffff000000000001")])
    client.post("/api/save/ffff000000000001")

    res = client.delete("/api/save/ffff000000000001").get_json()

    assert res["saved"] is False
    assert client.get("/api/saved").get_json()["rows"] == []


def test_old_saved_table_is_migrated_without_losing_bookmarks(client, app_env):
    insert_rows(app_env.SESSION_DB_PATH, [make_row(image_hash="abcd000000000001")])
    conn = sqlite3.connect(app_env.SAVED_DB_PATH)
    conn.execute("CREATE TABLE saved (image_hash TEXT PRIMARY KEY, saved_at TEXT)")
    conn.execute("INSERT INTO saved VALUES ('abcd000000000001', '2026-01-01 00:00:00')")
    conn.commit()
    conn.close()

    saved = client.get("/api/saved").get_json()["rows"]

    assert [r["image_hash"] for r in saved] == ["abcd000000000001"]
    conn = sqlite3.connect(app_env.SAVED_DB_PATH)
    snapshot = conn.execute("SELECT creative_json FROM saved").fetchone()[0]
    conn.close()
    assert snapshot  # the missing snapshot was backfilled
