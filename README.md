# Meta Ad Scraper

A local bulk scraper for collecting image-based Meta Ad Library creatives
from selected advertisers, removing duplicate creatives, storing metadata,
and providing a browser-based interface for reviewing and selecting
design-led advertisements.

## Workflow

Meta Ad Library
        ↓
Advertiser / Page Discovery
        ↓
Active Image Ad Collection
        ↓
Creative Download
        ↓
pHash Deduplication
        ↓
SQLite + CSV
        ↓
HTML Visual Review
        ↓
Design-led Ad Selection
        ↓
Selected CSV

## Features

- Search and collect ads by advertiser/page
- Configurable categories and brands
- Collect active image creatives
- Filter out video creatives
- Download creatives locally
- Perceptual-hash (pHash) based duplicate detection
- Store ad metadata in SQLite
- Export metadata to CSV
- Generate a browser-based visual review interface
- Filter review results by category and brand
- Search ad copy and metadata
- Select individual or visible creatives
- Export selected creatives to CSV
- Combined review interface when running all categories
- Per-brand timeout handling so a stalled Meta request does not stop
  the entire scraping process

## Installation

Create a virtual environment:

```bash
python -m venv .venv

Install dependencies:

pip install -r requirements.txt

On Windows, the virtual environment Python executable can also be used
directly:

.venv\Scripts\python.exe
Usage
Scrape one category
.venv\Scripts\python.exe .\bulk_scraper.py --category "Beauty & Skincare"

This scrapes the configured advertisers for the selected category and
generates a browser-based review page.

Scrape all categories
.venv\Scripts\python.exe .\bulk_scraper.py --all

This processes all configured categories and creates a combined review
interface.

Change the number of ads collected per brand

The default is 20 qualifying ads per brand.

For example:

.venv\Scripts\python.exe .\bulk_scraper.py --all --max-ads 50
Generate the selected-only review

After selecting ads and downloading the selection CSV:

.venv\Scripts\python.exe .\bulk_scraper.py --selected-review
Output

For each category:

bulk_output/
└── Category_Name/
    ├── ads.db
    ├── ads.csv
    ├── images/
    └── duplicates/

The combined review is generated under:

bulk_output/
└── ALL_REVIEW/
    └── review.html
Design-led Ad Selection

The main purpose of this project is to collect design-led advertising
creatives, rather than simply collecting any image advertisement.

Examples of design-led creatives include:

Typography-heavy advertisements
Promotional / offer graphics
Product + graphic compositions
Testimonial graphics
Comparison graphics
Illustrated or highly designed creatives
Campaign-oriented visual layouts

The scraper intentionally separates collection from design judgment.

The collection stage handles bulk retrieval, image filtering and
deduplication. The review interface then allows the user to visually
select the creatives that match the desired design criteria.

Current Limitations
1. Design classification is not automated yet

The current pipeline can identify image creatives and remove duplicates,
but it does not reliably distinguish between:

A designed advertising creative
A plain product photograph
A lifestyle photograph
UGC/social-media screenshots
Catalog/product listing images
Other low-design visual content

A purely rule-based image classifier was explored using visual properties
such as image structure, edges and perceptual characteristics. This was
not sufficiently reliable for the required precision.

Therefore, the current implementation uses a human-in-the-loop review
step for the final design selection.

2. Meta collection can be affected by external changes

The project depends on the current behavior of Meta's Ad Library and the
meta-ads-collector package.

Changes to Meta's internal interfaces, rate limits, availability or
anti-automation behavior may affect scraping reliability.

The scraper therefore includes per-brand timeout handling so that a
single stalled request does not stop the entire run.

3. Duplicate detection is not semantic

pHash is useful for identifying identical or visually similar images, but
it does not understand whether two creatives belong to the same campaign
or communicate the same concept.

Different designs can therefore still remain as separate creatives, while
very similar creatives may be grouped as duplicates.

4. Collection is currently advertiser-driven

The current approach uses configured advertisers/pages instead of attempting
to discover every possible advertiser automatically.

This improves the quality and relevance of the collected data, but means
the quality of the dataset depends partly on the brands selected in the
configuration.

Planned Improvements
AI-assisted design classification

A future version can use a pretrained computer-vision model to generate
image embeddings and classify creatives as:

Designed
Not designed
Uncertain

Rather than training a vision model from scratch, the initial approach
would use pretrained visual embeddings with a lightweight classifier.

Human-in-the-loop learning

The existing review interface can become the source of training data.

Each human selection can be stored as a label:

image → designed / not designed

The collected labels can then be used to improve the classifier over time.

A possible workflow:

Bulk Scraping
      ↓
Deduplication
      ↓
AI Design Classifier
      ↓
High-confidence → Auto-select
      ↓
Low-confidence → Human Review
      ↓
Human Corrections
      ↓
Training Dataset
      ↓
Improved Classifier

This would gradually reduce the amount of manual review while keeping a
human verification step for uncertain creatives.

Additional metadata

Future versions can store fields such as:

design_label
design_score
human_verified
model_version

This would make it possible to evaluate classifier performance and track
how the model improves over time.

Global deduplication

The current deduplication is performed within the scraping workflow.
A future version could maintain a global creative hash index so that the
same creative appearing across multiple advertisers or categories can be
identified consistently.

Project Status

The current version is a working proof of concept focused on:

Bulk Meta Ad Library collection
Image creative extraction
Duplicate removal
Metadata storage
Browser-based visual review
Human selection of design-led creatives

The next major improvement is AI-assisted design classification with
human-in-the-loop feedback.