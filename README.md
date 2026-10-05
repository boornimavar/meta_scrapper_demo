# Creative Lens

A web dashboard for finding design-led static ads in the Meta Ad Library.
Search an industry or theme, and Creative Lens collects live image ads,
scores how "designed" each creative is, removes duplicates, and streams
the results into a gallery as they are found. Creatives you like can be
saved and stay saved across restarts.

## How it works

```
Search (industry / theme + market)
        ↓
Live Meta Ad Library collection (meta_ads_collector)
        ↓
Image analysis — CLIP embeddings, OCR text density, visual complexity
        ↓
Design classifier + rank model → rank_score
        ↓
Deduplication — perceptual hashes, crop hashes, SIFT, OCR text overlap
        ↓
SQLite (session results) → dashboard gallery, updated live
        ↓
Save / bookmark → saved_creatives.db (persistent)
```

## Project layout

| Path | Purpose |
|------|---------|
| `dashboard_server_temporary.py` | Flask server — serves the dashboard and the JSON API |
| `dashboard.html` | Single-page frontend |
| `meta_design_scraper_session.py` | Collection, image analysis, scoring, dedup and database helpers |
| `design_classifier.pkl` | Design vs. non-design classifier |
| `rank_model_v1.pkl` | Rank model that produces `rank_score` |
| `meta_ads_collector_source.zip` | Source of the `meta_ads_collector` package (unpacked in the Docker build) |
| `Dockerfile`, `render.yaml`, `DEPLOY.md` | Render deployment |

## Dashboard features

- Search by industry, brand or campaign theme
- Filter by market (India, UK, US, all), platform, ad status, date range and brand
- "Design-led only" view (`rank_score ≥ 0.5`)
- Live progress while a search runs, with a Stop button
- Save / unsave creatives; saved ads keep a full snapshot of the ad record

## Requirements

- Python 3.11+ (local development uses Python 3.13 in `.venv313`)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and on `PATH`
- PyTorch and OpenAI CLIP
- The `meta_ads_collector` package

## Setup

```bash
python -m venv .venv313
.venv313\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/openai/CLIP.git
```

Install `meta_ads_collector` from `meta_ads_collector_source.zip` if it is
not already available in the environment.

## Running

```bash
.venv313\Scripts\python.exe dashboard_server_temporary.py
```

Then open http://127.0.0.1:5000. Set the `PORT` environment variable to use
a different port.

Notes:

- Search results are temporary. `session_creatives.db` is cleared every
  time the server starts.
- Saved creatives live in `saved_creatives.db` and are never cleared.
- The first search is slower while CLIP loads its model.
- Set `CREATIVE_LENS_DATA_DIR` to store databases and model files somewhere
  other than the project folder (the Docker image uses `/var/data`).

## API

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Dashboard |
| GET | `/api/creatives` | Creatives matching the filters; starts a live search for a new category |
| GET | `/api/scrape-status` | Progress of the current search |
| POST | `/api/stop-scrape` | Stop the current search |
| GET | `/api/categories` | Categories in the session database |
| GET | `/api/saved` | Saved creatives |
| POST / DELETE | `/api/save/<image_hash>` | Save / unsave a creative |

## Deployment

See [DEPLOY.md](DEPLOY.md) for deploying to Render with Docker.

## Limitations

- Depends on Meta's Ad Library and the unofficial `meta_ads_collector`
  package. Changes on Meta's side, rate limits or blocking can break
  collection. Check that collection is permitted for your use.
- Search progress and stop state are held in memory, so the server must
  run as a single process (one Gunicorn worker).
- The design classifier is trained on a small labelled set, so some
  borderline creatives will be misclassified.
- Deduplication is visual and textual, not semantic: different designs
  from the same campaign stay separate.
