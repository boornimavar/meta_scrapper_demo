"""Shared test setup.

CREATIVE_LENS_DATA_DIR must point at a temporary folder *before* the app
modules are imported, because BASE_DIR (and every database path derived
from it) is computed at import time. This keeps the real
saved_creatives.db / session_creatives.db untouched by tests.
"""
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest

_DATA_DIR = Path(tempfile.mkdtemp(prefix="creative_lens_tests_"))
os.environ["CREATIVE_LENS_DATA_DIR"] = str(_DATA_DIR)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import meta_design_scraper_session as scraper  # noqa: E402
import dashboard_server_temporary as server  # noqa: E402


def make_row(**overrides):
    """A complete ad record accepted by insert_ad()."""
    row = {
        "category": "skincare",
        "brand": "Brand A",
        "page_name": "Brand A",
        "page_id": "100",
        "ad_id": "1",
        "creative_index": 0,
        "image_file": "img.jpg",
        "image_url": "https://example.com/img.jpg",
        "body": "Body copy",
        "title": "Title",
        "cta": "Shop now",
        "image_hash": "ffff0000ffff0000",
        "duplicate": 0,
        "rank_score": 0.9,
        "country": "IN",
        "platform": "INSTAGRAM",
        "media_type": "IMAGE",
        "search_query": "skincare",
        "is_creative_representative": 1,
    }
    row.update(overrides)
    return row


def insert_rows(db_path, rows):
    conn = scraper.create_database(Path(db_path))
    for row in rows:
        scraper.insert_ad(conn, row)
    conn.commit()
    conn.close()


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "ads.db"


@pytest.fixture
def app_env():
    """Fresh session + saved databases and cleared in-memory search state."""
    for path in (server.SESSION_DB_PATH, server.SAVED_DB_PATH):
        if path.exists():
            path.unlink()
    scraper.create_database(server.SESSION_DB_PATH).close()
    with server._scrapes_lock:
        server._scrapes_in_progress.clear()
        server._session_searched_targets.clear()
    with server._progress_lock:
        server._scrape_progress.clear()
    with server._stop_events_lock:
        server._stop_events.clear()
    with server._progress_lock:
        server._failures.clear()
        server._last_activity.clear()
    yield server
    # Let any background search a test started finish, so it isn't still
    # holding the session DB open when the next test deletes it (Windows
    # refuses to delete open files).
    deadline = time.time() + 5
    while time.time() < deadline:
        with server._scrapes_lock:
            if not server._scrapes_in_progress:
                break
        time.sleep(0.01)


@pytest.fixture
def client(app_env):
    return app_env.app.test_client()


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Replace the live Meta search with a controllable fake.

    Set `behaviour` to choose what the fake does: "insert" writes `rows`
    to the session DB, "wait" loops until the Stop button is pressed,
    "fail" / "rate_limited" / "timeout" raise like the collector does.
    """
    state = {"behaviour": "insert", "rows": [], "calls": []}
    errors = {
        "fail": "Meta unavailable",
        "rate_limited": "Max retries exceeded due to rate limiting",
        "timeout": "Failed to perform, curl: (28) Operation timed out after 450847 milliseconds with 0 bytes received.",
    }

    def fake(args, progress_callback=None, stop_check=None, db_path=None, save_run_artifacts=True,
             on_collection_interrupted=None):
        state["calls"].append(args.search_query)
        state["page_size"] = args.search_page_size
        if state["behaviour"] == "partial":
            # Meta rate-limits partway: the rows collected so far are still analyzed.
            for n in range(1, len(state["rows"]) + 1):
                progress_callback(n, 0, "collecting")
            on_collection_interrupted(RuntimeError(errors["rate_limited"]))
            progress_callback(0, len(state["rows"]), "analyzing")
            insert_rows(db_path, state["rows"])
            progress_callback(len(state["rows"]), len(state["rows"]), "done")
            return
        if state["behaviour"] in errors:
            raise RuntimeError(errors[state["behaviour"]])
        if state["behaviour"] == "wait":
            deadline = time.time() + 5
            while not stop_check() and time.time() < deadline:
                time.sleep(0.01)
            return
        for n in range(1, len(state["rows"]) + 1):
            progress_callback(n, 0, "collecting")
        progress_callback(0, len(state["rows"]), "analyzing")
        insert_rows(db_path, state["rows"])
        progress_callback(len(state["rows"]), len(state["rows"]), "analyzing")

    monkeypatch.setattr(server, "run_broad_search_pipeline", fake)
    return state


def wait_for_stage(client, query, stages, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/scrape-status?{query}").get_json()
        if status["stage"] in stages:
            return status
        time.sleep(0.02)
    raise AssertionError(f"scrape never reached {stages}; last status: {status}")


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_DATA_DIR, ignore_errors=True)
