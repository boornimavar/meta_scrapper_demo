import sqlite3
from datetime import datetime, timedelta

import meta_design_scraper_session as scraper
from conftest import insert_rows, make_row


def test_insert_same_image_hash_twice_keeps_one_row(db_path):
    insert_rows(db_path, [make_row(), make_row(brand="Other brand")])

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT brand FROM ads").fetchall()
    conn.close()

    assert rows == [("Brand A",)]


def test_browse_missing_database_returns_empty(tmp_path):
    assert scraper.browse_master_db(db_path=tmp_path / "missing.db") == []


def test_browse_hides_non_representative_variants(db_path):
    insert_rows(db_path, [
        make_row(image_hash="1000000000000001"),
        make_row(image_hash="1000000000000002", is_creative_representative=0),
    ])

    rows = scraper.browse_master_db(db_path=db_path)

    assert [r["image_hash"] for r in rows] == ["1000000000000001"]


def test_browse_filters(db_path):
    insert_rows(db_path, [
        make_row(image_hash="2000000000000001", category="skincare", country="IN", platform="INSTAGRAM", brand="Plum", rank_score=0.9),
        make_row(image_hash="2000000000000002", category="fashion", country="GB", platform="FACEBOOK", brand="Myntra", rank_score=0.3),
        make_row(image_hash="2000000000000003", category="skincare serum", country="IN", platform="ALL", brand="Minimalist", rank_score=0.7),
    ])

    def hashes(**filters):
        return sorted(r["image_hash"] for r in scraper.browse_master_db(db_path=db_path, **filters))

    assert hashes(category="skincare") == ["2000000000000001", "2000000000000003"]
    assert hashes(country="gb") == ["2000000000000002"]
    # rows stored with platform ALL match any platform filter
    assert hashes(platform="FACEBOOK") == ["2000000000000002", "2000000000000003"]
    assert hashes(platform="ALL") == ["2000000000000001", "2000000000000002", "2000000000000003"]
    assert hashes(brand="plum") == ["2000000000000001"]
    assert hashes(min_rank_score=0.5) == ["2000000000000001", "2000000000000003"]


def test_browse_days_filter_uses_last_seen(db_path):
    insert_rows(db_path, [
        make_row(image_hash="3000000000000001"),
        make_row(image_hash="3000000000000002"),
    ])
    old = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE ads SET last_seen_at=? WHERE image_hash='3000000000000002'", (old,))
    conn.commit()
    conn.close()

    rows = scraper.browse_master_db(db_path=db_path, days="30")

    assert [r["image_hash"] for r in rows] == ["3000000000000001"]


def test_create_database_backfills_missing_market_fields(db_path):
    insert_rows(db_path, [make_row(country="", platform="", media_type="")])

    scraper.create_database(db_path).close()

    row = scraper.browse_master_db(db_path=db_path)[0]
    assert (row["country"], row["platform"], row["media_type"]) == ("IN", "ALL", "IMAGE")
