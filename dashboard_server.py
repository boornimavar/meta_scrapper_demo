"""
Creative Intelligence Dashboard - local web server.

No more count dropdown. Searching something new (or stale) kicks off a
live scrape that writes results to the database batch by batch as they're
found - the dashboard polls and shows them growing in real time, and the
person can stop whenever they've seen enough, or let it run until it
genuinely runs out of new content (internally capped at a generous 500
per search, which costs nothing extra to request - Meta just returns
whatever's really available either way).

Run with:  python dashboard_server.py
Then open: http://127.0.0.1:5000
"""
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from flask import Flask, jsonify, request, send_from_directory

from meta_design_scraper_app30 import (
    MASTER_DB_PATH, browse_master_db, BASE_DIR, create_database,
    run_broad_search_pipeline, ML_MODEL_PATH,
)

app = Flask(__name__)
SAVED_DB_PATH = BASE_DIR / "saved_creatives.db"

STALE_AFTER_HOURS = 48
SCRAPE_REQUEST_SIZE = 500  # always ask Meta for a generous amount - costs nothing extra,
                           # since Meta only ever returns what's genuinely available anyway

_scrapes_in_progress = set()
_scrapes_lock = threading.Lock()

_scrape_progress = {}
_progress_lock = threading.Lock()

_stop_events = {}  # target -> threading.Event(), set when the user clicks Stop
_stop_events_lock = threading.Lock()


def get_saved_conn():
    conn = sqlite3.connect(SAVED_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS saved (
            image_hash TEXT PRIMARY KEY,
            saved_at TEXT
        )
    """)
    return conn


def is_stale(category, country, min_rank_score=None):
    """True if we have nothing for this category+country, or what we have
    hasn't been checked recently enough to trust (48h). There's no more
    count-driven staleness - since a search now always requests a generous
    amount and can be watched/stopped live, "ask for more" isn't a
    separate action anymore the way it used to be.
    """
    if not MASTER_DB_PATH.exists():
        return True
    conn = sqlite3.connect(MASTER_DB_PATH)
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
    target = (category, (country or "").upper())
    print(f"[ON-DEMAND] '{category}' / {country or 'any country'} - scraping live, results will stream in...")

    def on_progress(done, total, stage):
        with _progress_lock:
            _scrape_progress[target] = {"stage": stage, "done": done, "total": total}

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
    try:
        run_broad_search_pipeline(args, progress_callback=on_progress, stop_check=stop_event.is_set)
    except Exception as exc:
        print(f"[ON-DEMAND] Scrape failed: {exc}")
        with _progress_lock:
            _scrape_progress[target] = {"stage": "failed", "done": 0, "total": 0}
    finally:
        with _scrapes_lock:
            _scrapes_in_progress.discard(target)
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

    on_demand_triggered = False
    target = (category, (country or "").upper())
    should_scrape = False
    if category:
        with _scrapes_lock:
            if target not in _scrapes_in_progress and is_stale(category, country, min_rank_score=min_rank_score):
                _scrapes_in_progress.add(target)
                should_scrape = True
    if should_scrape:
        stop_event = threading.Event()
        with _stop_events_lock:
            _stop_events[target] = stop_event
        with _progress_lock:
            _scrape_progress[target] = {"stage": "starting", "done": 0, "total": 0}
        thread = threading.Thread(target=run_on_demand_scrape, args=(category, country, stop_event), daemon=True)
        thread.start()
        on_demand_triggered = True

    rows = browse_master_db(
        category=category, country=country, platform=platform,
        media_type=media_type, status=status, brand=brand,
        min_rank_score=min_rank_score, days=days,
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
    })


@app.route("/api/scrape-status")
def api_scrape_status():
    category = request.args.get("category") or None
    country = request.args.get("country") or None
    target = (category, (country or "").upper())
    with _progress_lock:
        progress = _scrape_progress.get(target)
    if not progress:
        return jsonify({"stage": "idle"})
    return jsonify(progress)


@app.route("/api/stop-scrape", methods=["POST"])
def api_stop_scrape():
    category = request.args.get("category") or None
    country = request.args.get("country") or None
    target = (category, (country or "").upper())
    with _stop_events_lock:
        event = _stop_events.get(target)
    if event is not None:
        event.set()
        return jsonify({"ok": True, "stopping": True})
    return jsonify({"ok": True, "stopping": False, "note": "nothing in progress for this target"})


@app.route("/api/categories")
def api_categories():
    if not MASTER_DB_PATH.exists():
        return jsonify([])
    conn = sqlite3.connect(MASTER_DB_PATH)
    cats = [r[0] for r in conn.execute(
        "SELECT DISTINCT category FROM ads WHERE is_creative_representative=1 AND category IS NOT NULL ORDER BY category"
    ).fetchall()]
    conn.close()
    return jsonify(cats)


@app.route("/api/save/<image_hash>", methods=["POST"])
def api_save(image_hash):
    conn = get_saved_conn()
    conn.execute(
        "INSERT OR REPLACE INTO saved (image_hash, saved_at) VALUES (?, ?)",
        (image_hash, time.strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "saved": True})


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
    if MASTER_DB_PATH.exists():
        conn = create_database(MASTER_DB_PATH)
        conn.close()
        print("[MIGRATE] Checked/backfilled database schema.")
    print(f"Reading from: {MASTER_DB_PATH.resolve()}")
    print("Searches stream results in live as they're found. No count to set - stop anytime, or let it run out naturally.")
    print("Open in your browser: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(debug=True, port=5000)

