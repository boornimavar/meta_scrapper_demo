"""
Creative Intelligence Dashboard - local web server.

No more count dropdown. Searching something new (or stale) kicks off a
live scrape that writes results to the database batch by batch as they're
found - the dashboard polls and shows them growing in real time, and the
person can stop whenever they've seen enough, or let it run until it
genuinely runs out of new content (internally capped at a generous 500
per search, which costs nothing extra to request - Meta just returns
whatever's really available either way).

Run with:  python dashboard_server_temporary.py
Then open: http://127.0.0.1:5000
"""
import sqlite3
import json
import math
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from flask import Flask, jsonify, request, send_from_directory

from meta_design_scraper_session import (
    MASTER_DB_PATH, browse_master_db, BASE_DIR, create_database,
    run_broad_search_pipeline, ML_MODEL_PATH,
)

app = Flask(__name__)
SAVED_DB_PATH = BASE_DIR / "saved_creatives.db"
SESSION_DB_PATH = BASE_DIR / "session_creatives.db"

# Gunicorn imports this module rather than executing the __main__ block.
# Ensure the session schema exists before the first API request.
if not SESSION_DB_PATH.exists():
    _initial_conn = create_database(SESSION_DB_PATH)
    _initial_conn.close()

STALE_AFTER_HOURS = 48
SCRAPE_REQUEST_SIZE = 500  # always ask Meta for a generous amount - costs nothing extra,
                           # since Meta only ever returns what's genuinely available anyway

_scrapes_in_progress = set()
_scrapes_lock = threading.Lock()

_scrape_progress = {}
_progress_lock = threading.Lock()

# A target is searched at most once per running dashboard session, even if it
# returned zero qualifying creatives. This prevents category switching from
# silently launching duplicate searches.
_session_searched_targets = set()

_stop_events = {}  # target -> threading.Event(), set when the user clicks Stop
_stop_events_lock = threading.Lock()

# Meta throttles by network. A failed search is never restarted automatically
# (that only extends the throttling); the person retries explicitly, and only
# after a cooldown that depends on why it failed.
COLLECT_STALL_SECONDS = 180  # give up if Meta sends no new ad for this long
WATCHDOG_INTERVAL_SECONDS = 5
RETRY_COOLDOWN_SECONDS = {"rate_limited": 15 * 60, "no_response": 5 * 60, "error": 0}
FAILURE_MESSAGES = {
    "rate_limited": "Meta is rate-limiting searches from this network. Wait a while - ideally a few hours - before searching again.",
    "no_response": "Meta stopped responding on this network. Try again later or from a different connection.",
    "error": "The search could not finish. Check the server terminal for details.",
}

_failures = {}  # target -> {"reason", "failed_at"}; guarded by _progress_lock
_last_activity = {}  # target -> time of the last progress update; guarded by _progress_lock


def classify_failure(exc):
    """Map a collector exception to a reason the dashboard can explain."""
    text = str(exc).lower()
    if "rate limit" in text:
        return "rate_limited"
    if any(marker in text for marker in ("timed out", "timeout", "connection was reset", "connection reset", "recv failure", "could not connect")):
        return "no_response"
    return "error"


def _mark_failed_locked(target, reason):
    """Record a failed search. Caller must hold _progress_lock."""
    prior = _scrape_progress.get(target, {})
    _scrape_progress[target] = {
        "stage": "failed",
        "reason": reason,
        "message": FAILURE_MESSAGES[reason],
        "done": 0,
        "total": 0,
        "collected": int(prior.get("collected", 0)),
    }
    _failures[target] = {"reason": reason, "failed_at": time.time()}


def retry_wait_seconds(target):
    """Seconds until a failed target may be retried, or None if it has not failed.
    Caller must hold _progress_lock."""
    failure = _failures.get(target)
    if not failure:
        return None
    ready_at = failure["failed_at"] + RETRY_COOLDOWN_SECONDS[failure["reason"]]
    return max(0, math.ceil(ready_at - time.time()))


def watch_for_stall(target, stop_event):
    """Fail a search whose collection has gone quiet, instead of letting the
    dashboard sit on "collecting" until the collector's own retries give up."""
    while True:
        time.sleep(WATCHDOG_INTERVAL_SECONDS)
        with _scrapes_lock:
            if target not in _scrapes_in_progress:
                return
        with _progress_lock:
            stage = _scrape_progress.get(target, {}).get("stage")
            quiet_for = time.time() - _last_activity.get(target, time.time())
            stalled = stage in ("starting", "collecting") and quiet_for > COLLECT_STALL_SECONDS
            if stalled:
                _mark_failed_locked(target, "no_response")
        if stalled:
            print(f"[ON-DEMAND] No response from Meta for {COLLECT_STALL_SECONDS}s - marking '{target[0]}' as failed.")
            stop_event.set()
            return


def get_saved_conn():
    conn = sqlite3.connect(SAVED_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS saved (
            image_hash TEXT PRIMARY KEY,
            saved_at TEXT,
            creative_json TEXT
        )
    """)
    # Migrate databases created by earlier versions without deleting bookmarks.
    columns = {row[1] for row in conn.execute("PRAGMA table_info(saved)").fetchall()}
    if "creative_json" not in columns:
        conn.execute("ALTER TABLE saved ADD COLUMN creative_json TEXT")
        conn.commit()
    return conn


def find_creative_by_hash(image_hash):
    """Find a full creative record to snapshot into persistent Saved ads."""
    for db_path in (SESSION_DB_PATH, MASTER_DB_PATH):
        if not db_path.exists():
            continue
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT * FROM ads WHERE image_hash = ? LIMIT 1",
                (image_hash,),
            ).fetchone()
        except sqlite3.OperationalError:
            row = None
        finally:
            conn.close()
        if row:
            record = dict(row)
            record.pop("id", None)
            record["is_saved"] = True
            return record
    return None


def normalize_target(category, country):
    """Normalize a category/market pair for per-session cache keys."""
    return ((category or "").strip().casefold(), (country or "").strip().upper())


def is_stale(category, country, min_rank_score=None):
    """True if we have nothing for this category+country, or what we have
    hasn't been checked recently enough to trust (48h). There's no more
    count-driven staleness - since a search now always requests a generous
    amount and can be watched/stopped live, "ask for more" isn't a
    separate action anymore the way it used to be.
    """
    if not SESSION_DB_PATH.exists():
        return True
    conn = sqlite3.connect(SESSION_DB_PATH)
    query = "SELECT MAX(last_seen_at), COUNT(*) FROM ads WHERE is_creative_representative=1"
    params = []
    if category:
        query += " AND category LIKE ?"; params.append(f"%{category}%")
    if country:
        query += " AND country = ?"; params.append(country.upper())
    if min_rank_score is not None:
        query += " AND rank_score >= ?"; params.append(min_rank_score)
    row = conn.execute(query, params).fetchone()
    conn.close()
    last_seen, count = row
    if not count or not last_seen:
        return True
    age = datetime.now() - datetime.strptime(last_seen, "%Y-%m-%d %H:%M:%S")
    return age > timedelta(hours=STALE_AFTER_HOURS)


def run_on_demand_scrape(category, country, stop_event):
    target = normalize_target(category, country)
    print(f"[ON-DEMAND] '{category}' / {country or 'any country'} - scraping live, results will stream in...")

    def on_progress(done, total, stage):
        # During collection `done` is the number of ads Meta has returned so
        # far. Once analysis starts, `total` is the number of creatives to analyze.
        with _progress_lock:
            _last_activity[target] = time.time()
            prior = _scrape_progress.get(target, {})
            if prior.get("stage") == "failed":
                return  # the stall watchdog already gave up on this search
            if stage == "collecting":
                collected = int(done or 0)
                if collected and collected % 10 == 0:
                    print(f"[ON-DEMAND] '{category}': {collected} ads collected from Meta so far")
            elif stage == "analyzing":
                collected = int(total or 0)
            else:
                collected = int(prior.get("collected", 0))
            _scrape_progress[target] = {
                "stage": stage,
                "done": 0 if stage == "collecting" else int(done or 0),
                "total": 0 if stage == "collecting" else int(total or 0),
                "collected": collected,
            }

    args = SimpleNamespace(
        search_query=category,
        search_country=country or "IN",
        search_status="ACTIVE",
        search_type="KEYWORD_UNORDERED",
        search_max_results=SCRAPE_REQUEST_SIZE,
        search_page_size=10,
        search_media_type="IMAGE",
        platforms=None,
        search_start_date=None,
        search_end_date=None,
        model=str(ML_MODEL_PATH),
        skip_browser_open=True,
        search_sort_by=None,
    )
    failure_reason = None
    try:
        run_broad_search_pipeline(args, progress_callback=on_progress, stop_check=stop_event.is_set, db_path=SESSION_DB_PATH, save_run_artifacts=False)
    except Exception as exc:
        failure_reason = classify_failure(exc)
        print(f"[ON-DEMAND] Scrape failed ({failure_reason}): {exc}")
    finally:
        # Publish a terminal state so the dashboard can stop polling and
        # transition cleanly from the live analysis panel to the gallery.
        with _progress_lock:
            current = _scrape_progress.get(target, {})
            if current.get("stage") != "failed":
                if failure_reason:
                    _mark_failed_locked(target, failure_reason)
                else:
                    current["stage"] = "stopped" if stop_event.is_set() else "done"
                    current.setdefault("done", 0)
                    current.setdefault("total", 0)
                    current.setdefault("collected", current.get("total", 0))
                    _scrape_progress[target] = current
        # The target stays "searched" even when it failed, so refreshing the
        # gallery never silently starts another search. Retrying is explicit.
        with _scrapes_lock:
            _scrapes_in_progress.discard(target)
            _session_searched_targets.add(target)
        with _stop_events_lock:
            _stop_events.pop(target, None)


@app.route("/")
def index():
    return send_from_directory(Path(__file__).resolve().parent, "dashboard.html")


@app.route("/api/creatives")
def api_creatives():
    category = request.args.get("category") or None
    country = request.args.get("country") or None
    platform = request.args.get("platform") or None
    media_type = request.args.get("media_type") or None
    status = request.args.get("status") or None
    brand = request.args.get("brand") or None
    days = request.args.get("days") or None
    design_led_only = request.args.get("design_led_only", "true").lower() == "true"

    min_rank_score = 0.5 if design_led_only else None

    retry_requested = request.args.get("retry") == "1"

    on_demand_triggered = False
    retry_after = None
    target = normalize_target(category, country)
    should_scrape = False
    if category:
        with _scrapes_lock:
            if target not in _scrapes_in_progress:
                if target not in _session_searched_targets:
                    should_scrape = True
                elif retry_requested:
                    with _progress_lock:
                        wait = retry_wait_seconds(target)
                        if wait == 0:
                            _failures.pop(target, None)
                            should_scrape = True
                        else:
                            retry_after = wait
                if should_scrape:
                    _scrapes_in_progress.add(target)
                    # Mark it immediately to close the race between rapid repeated requests.
                    _session_searched_targets.add(target)
    if should_scrape:
        stop_event = threading.Event()
        with _stop_events_lock:
            _stop_events[target] = stop_event
        with _progress_lock:
            _scrape_progress[target] = {"stage": "starting", "done": 0, "total": 0, "collected": 0}
            _last_activity[target] = time.time()
        threading.Thread(target=run_on_demand_scrape, args=(category, country, stop_event), daemon=True).start()
        threading.Thread(target=watch_for_stall, args=(target, stop_event), daemon=True).start()
        on_demand_triggered = True

    rows = browse_master_db(
        category=category, country=country, platform=platform,
        media_type=media_type, status=status, brand=brand,
        min_rank_score=min_rank_score, days=days, db_path=SESSION_DB_PATH,
    )

    conn = get_saved_conn()
    saved_hashes = {r[0] for r in conn.execute("SELECT image_hash FROM saved").fetchall()}
    conn.close()
    for r in rows:
        r["is_saved"] = r.get("image_hash") in saved_hashes

    rows.sort(key=lambda r: float(r.get("rank_score") or 0), reverse=True)
    return jsonify({
        "rows": rows,
        "on_demand_scraped": on_demand_triggered,
        "scrape_target": {"category": category, "country": (country or "").upper()} if on_demand_triggered else None,
        "retry_after": retry_after,
    })


@app.route("/api/scrape-status")
def api_scrape_status():
    category = request.args.get("category") or None
    country = request.args.get("country") or None
    target = normalize_target(category, country)
    with _progress_lock:
        progress = dict(_scrape_progress.get(target, {}))
        if progress.get("stage") == "failed":
            progress["retry_after"] = retry_wait_seconds(target) or 0
    if not progress:
        return jsonify({"stage": "idle"})

    # Use the exact same browse function as /api/creatives so this count
    # respects representative/deduplication rules and the gallery's filters.
    # That prevents the live counter from counting rows the gallery excludes.
    design_led_only = request.args.get("design_led_only", "true").lower() == "true"
    matches = browse_master_db(
        category=category,
        country=country,
        platform=request.args.get("platform") or None,
        media_type=request.args.get("media_type") or None,
        status=request.args.get("status") or None,
        brand=request.args.get("brand") or None,
        min_rank_score=0.5 if design_led_only else None,
        days=request.args.get("days") or None,
        db_path=SESSION_DB_PATH,
    )
    progress["matches"] = len(matches)
    progress.setdefault("collected", progress.get("total", 0) if progress.get("stage") in ("analyzing", "done", "stopped") else 0)
    return jsonify(progress)


@app.route("/api/stop-scrape", methods=["POST"])
def api_stop_scrape():
    category = request.args.get("category") or None
    country = request.args.get("country") or None
    target = normalize_target(category, country)
    with _stop_events_lock:
        event = _stop_events.get(target)
    if event is not None:
        event.set()
        return jsonify({"ok": True, "stopping": True})
    return jsonify({"ok": True, "stopping": False, "note": "nothing in progress for this target"})


@app.route("/api/categories")
def api_categories():
    if not SESSION_DB_PATH.exists():
        return jsonify([])
    conn = sqlite3.connect(SESSION_DB_PATH)
    cats = [r[0] for r in conn.execute(
        "SELECT DISTINCT category FROM ads WHERE is_creative_representative=1 AND category IS NOT NULL ORDER BY category"
    ).fetchall()]
    conn.close()
    return jsonify(cats)


@app.route("/api/saved")
def api_saved():
    """Return saved creatives from persistent snapshots, even after session reset."""
    conn = get_saved_conn()
    saved_rows = conn.execute(
        "SELECT image_hash, creative_json FROM saved ORDER BY saved_at DESC"
    ).fetchall()
    results = []
    missing_snapshots = []
    for image_hash, creative_json in saved_rows:
        record = None
        if creative_json:
            try:
                record = json.loads(creative_json)
            except (TypeError, json.JSONDecodeError):
                record = None
        if record is None:
            record = find_creative_by_hash(image_hash)
            if record:
                missing_snapshots.append((json.dumps(record, ensure_ascii=False), image_hash))
        if record:
            record["image_hash"] = image_hash
            record["is_saved"] = True
            results.append(record)
    if missing_snapshots:
        conn.executemany("UPDATE saved SET creative_json=? WHERE image_hash=?", missing_snapshots)
        conn.commit()
    conn.close()
    return jsonify({"rows": results})


@app.route("/api/save/<image_hash>", methods=["POST"])
def api_save(image_hash):
    record = find_creative_by_hash(image_hash)
    conn = get_saved_conn()
    conn.execute(
        "INSERT OR REPLACE INTO saved (image_hash, saved_at, creative_json) VALUES (?, ?, ?)",
        (image_hash, time.strftime("%Y-%m-%d %H:%M:%S"), json.dumps(record, ensure_ascii=False) if record else None),
    )
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "saved": True, "snapshot_saved": bool(record)})


@app.route("/api/save/<image_hash>", methods=["DELETE"])
def api_unsave(image_hash):
    conn = get_saved_conn()
    conn.execute("DELETE FROM saved WHERE image_hash=?", (image_hash,))
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "saved": False})


if __name__ == "__main__":
    print("=" * 70)
    print("CREATIVE INTELLIGENCE DASHBOARD")
    print("=" * 70)
    # Search results are temporary: start each server session with an empty DB.
    # Bookmarks live in saved_creatives.db and are deliberately not cleared.
    if SESSION_DB_PATH.exists():
        SESSION_DB_PATH.unlink()
    conn = create_database(SESSION_DB_PATH)
    conn.close()
    print(f"Temporary search results: {SESSION_DB_PATH.resolve()}")
    print(f"Persistent bookmarks: {SAVED_DB_PATH.resolve()}")
    print("Searches stream results in live as they're found. No count to set - stop anytime, or let it run out naturally.")
    print("Open in your browser: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(host="0.0.0.0", debug=False, use_reloader=False, port=int(os.environ.get("PORT", "5000")))

