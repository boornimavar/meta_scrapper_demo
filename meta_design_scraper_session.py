import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import io
import json
import math
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta
import webbrowser
from pathlib import Path

import requests
import cv2
import numpy as np
from PIL import Image, ImageStat
import imagehash
import joblib
import torch
import clip
import pytesseract
from meta_ads_collector import MetaAdsCollector, FilterConfig
# import meta_ads_collector.client as _mac_client
# _mac_client.DOC_ID_SEARCH = "24922295957467452"  # Meta changed this on/before Oct 2026
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows only; Linux uses PATH

# ============================================================
# CONFIGURATION
# ============================================================

COUNTRY = "IN"
BASE_DIR = Path(os.environ.get("CREATIVE_LENS_DATA_DIR", str(Path(__file__).resolve().parent))).resolve()
BASE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = BASE_DIR / "bulk_output"
MASTER_DB_PATH = BASE_DIR / "master.db"  # one accumulating database, across every run/category, ever

# Optional legacy classifier is diagnostics only. It NEVER affects KEEP.
ML_MODEL_PATH = BASE_DIR / "design_classifier.pkl"
RANK_MODEL_PATH = BASE_DIR / "rank_model_v1.pkl"

_rank_model_bundle = None
RANK_FEATURES = ['clip_design_score','clip_product_score','broad_graphic_prob','broad_photo_prob',
                 'broad_video_prob','broad_product_prob','ocr_word_count','ocr_text_density','visual_complexity']


def retrain_rank_model(data_dir):
    """Reads every *.csv file in data_dir (the labeled_training_data*.csv
    files downloaded from review.html's "Download Labeled CSV" button),
    combines them, removes real duplicate creatives (same image_hash seen
    across multiple runs/categories), trains a logistic regression on the
    same features rank_score() already uses, and overwrites
    rank_model_v1.pkl. This is the whole "retrain the model" step - it
    needs no external help, run it any time you have new labeled CSVs.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"[RETRAIN] No CSV files found in {data_dir.resolve()}")
        print("          Put your downloaded labeled_training_data*.csv files there first.")
        return

    seen_hashes = set()
    rows = []
    for path in csv_files:
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                label = (row.get("human_label") or "").strip().upper()
                if label not in ("SAVE", "REJECT"):
                    continue
                img_hash = row.get("image_hash", "")
                if img_hash and img_hash in seen_hashes:
                    continue  # real duplicate creative, seen in an earlier file - skip
                if img_hash:
                    seen_hashes.add(img_hash)
                rows.append(row)

    if len(rows) < 20:
        print(f"[RETRAIN] Only {len(rows)} labeled rows found across {len(csv_files)} file(s) - "
              f"too few to train on meaningfully. Keep labeling and try again later.")
        return

    X = np.array([[float(r.get(f) or 0) for f in RANK_FEATURES] for r in rows])
    y = np.array([1 if (r.get("human_label") or "").strip().upper() == "SAVE" else 0 for r in rows])
    n_save, n_reject = int(y.sum()), int(len(y) - y.sum())

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced").fit(Xs, y)

    # Cross-validated AUC on this same data, purely as an honest sanity
    # check printed to the terminal - not used to pick the deployed model,
    # just so you can see whether this training run looks reasonable
    # before trusting it, and watch whether more data keeps helping or
    # has plateaued.
    cv_auc = None
    if n_save >= 5 and n_reject >= 5:
        cv = StratifiedKFold(n_splits=min(5, n_save, n_reject), shuffle=True, random_state=42)
        probs = cross_val_predict(model, Xs, y, cv=cv, method="predict_proba")[:, 1]
        cv_auc = roc_auc_score(y, probs)

    bundle = {
        "model": model, "scaler": scaler, "features": RANK_FEATURES,
        "trained_on_n": len(rows), "trained_on_files": [p.name for p in csv_files],
    }
    joblib.dump(bundle, RANK_MODEL_PATH)

    print("=" * 70)
    print("RETRAIN COMPLETE")
    print("=" * 70)
    print(f"Source files:        {len(csv_files)}  ({', '.join(p.name for p in csv_files)})")
    print(f"Labeled rows used:   {len(rows)}  (Save: {n_save}, Reject: {n_reject}, duplicates removed automatically)")
    if cv_auc is not None:
        print(f"Cross-validated AUC: {cv_auc:.3f}  <- compare this to your last retrain's number")
        print("                     If this number has stopped improving as you add more data,")
        print("                     that's your signal you can stop labeling for now.")
    print(f"Saved:               {RANK_MODEL_PATH.resolve()}")
    print("Nothing else needs to change - rank_score() already loads this file automatically.")


def _load_rank_model():
    """Loads the trained ranking model once, if the file exists. Returns
    None (silently) if it doesn't - callers fall back to the hand-set
    rank_score formula, so the pipeline still works without this file."""
    global _rank_model_bundle
    if _rank_model_bundle is None and RANK_MODEL_PATH.exists():
        try:
            _rank_model_bundle = joblib.load(RANK_MODEL_PATH)
            print(f"[MODEL] Loaded trained ranking model: {RANK_MODEL_PATH.name} "
                  f"(trained on {_rank_model_bundle.get('trained_on_n', '?')} labeled examples "
                  f"across {len(_rank_model_bundle.get('trained_on_categories', []))} categories)")
        except Exception as exc:
            print(f"[MODEL] Could not load {RANK_MODEL_PATH.name}, falling back to hand-set rank_score: {exc}")
            _rank_model_bundle = False
    return _rank_model_bundle or None

KEEP_THRESHOLD = 0.62
REVIEW_THRESHOLD = 0.44
STRONG_PHOTO_CLIP_MAX = 0.43
KEEP_MIN_TEXT_DENSITY = 0.05

# Product/catalog-shot guard. This is a second CLIP signal specifically for
# catching packshots, ecommerce/catalog photos, and packaging-only creatives.
# It is intentionally a targeted veto: product-looking imagery is rejected when
# graphic-ad evidence is not strong enough. Designed product ads can still pass.
PRODUCT_PHOTO_VETO_THRESHOLD = 0.54
PRODUCT_VETO_MARGIN = 0.025
PRODUCT_VETO_MAX_CREATIVE_SCORE = 0.72

# Broad-search is intended to surface genuinely design-led STATIC creatives.
# Keep this stricter than the legacy brand workflow so lifestyle/video-preview
# imagery with text overlays does not flood the app review page.
BROAD_MIN_DESIGN_CLIP_KEEP = 0.55
BROAD_MIN_SCORE_WITH_BORDERLINE_CLIP = 0.65
BROAD_MAX_PRODUCT_CLIP_FOR_BORDERLINE_KEEP = 0.44
BROAD_MIN_TEXT_DENSITY_FOR_BORDERLINE_KEEP = 0.10
BROAD_GRAPHIC_PROB_KEEP = 0.34
BROAD_VIDEO_PROB_MAX = 0.30
BROAD_PHOTO_PROB_MAX = 0.58
# Visually matching media-frame variants are grouped across different ad IDs.
# A center-cropped square pHash is used so the same creative in landscape, square,
# and portrait frames can collapse to one representative.
FRAME_VARIANT_PHASH_DISTANCE = 12
# Multi-crop matching is intentionally conservative. It is used only to find
# the same underlying creative delivered in different aspect-ratio frames.
CREATIVE_CROP_PHASH_DISTANCE = 12
CREATIVE_OCR_JACCARD = 0.5
CREATIVE_OCR_STRONG_JACCARD = 0.88

# Performance settings. These do NOT change prompts, thresholds, OCR passes,
# frame grouping rules, or final scoring; they only parallelize/batch work.
ANALYSIS_FETCH_WORKERS = 4
OCR_WORKERS = 4
CLIP_BATCH_SIZE = 8

_ml_model = None
_clip_model = None
_clip_preprocess = None
_clip_device = None
_clip_text_features = None
_product_text_features = None
_broad_class_features = None


def load_models(model_path: Path):
    global _ml_model, _clip_model, _clip_preprocess
    global _clip_device, _clip_text_features, _product_text_features, _broad_class_features

    if _clip_model is not None:
        # Already loaded - this matters a lot once load_models() gets called
        # from a long-running process (like the dashboard server triggering
        # an on-demand scrape) instead of once per CLI run. Reloading CLIP
        # from disk on every request would be slow and pointless.
        return

    _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MODEL] Loading CLIP on {_clip_device}...")
    _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=_clip_device)
    _clip_model.eval()

    if model_path.exists():
        print(f"[MODEL] Loading legacy classifier for diagnostics only: {model_path}")
        _ml_model = joblib.load(model_path)
    else:
        print("[MODEL] Legacy classifier not found; using CLIP + OCR only.")

    design_prompts = [
        "a professionally designed static advertising creative with typography, graphic layout, and visual hierarchy",
        "a polished social media advertisement with designed text, graphics, shapes, and product presentation",
        "a marketing ad creative, not just a photograph, with intentional graphic design and composition",
    ]
    photo_prompts = [
        "a plain product photograph or simple product shot used as an advertisement",
        "a simple photograph of a person or model with little graphic design",
        "an ordinary lifestyle or product photograph without a designed ad layout",
    ]
    product_prompts = [
        "an ecommerce catalog product photo or isolated product packshot",
        "a retail product packaging photograph with text printed on the package",
        "a simple product listing photo showing the product or package as the main subject",
        "a product packshot photographed for an online store, not a designed graphic ad",
    ]

    # Broad-search classifier: explicitly distinguish a designed static ad from
    # an ordinary photo with a social caption, a catalog/packshot, or a frame
    # grabbed from a video/reel. This is separate from the legacy POC score.
    broad_graphic_prompts = [
        "a static social media advertising graphic with deliberate typography, layout, shapes, color blocks, and visual hierarchy",
        "a professionally designed beauty or skincare ad poster with product, typography, offer text, and graphic composition",
        "a polished ecommerce marketing creative designed as a poster, not an ordinary photograph",
        "a static promotional ad artwork with intentional graphic design and arranged text",
    ]
    broad_photo_prompts = [
        "an ordinary photograph of a person or product with a caption or text overlay",
        "a lifestyle photograph used as an ad with social-media style caption text",
        "a real-world photo with text overlaid on top, but without a designed poster layout",
    ]
    broad_video_prompts = [
        "a screenshot or still frame taken from a social media video or reel",
        "a video frame from an Instagram Reel, TikTok, or Facebook video with caption text",
        "a paused frame from a short-form social video, not a designed static poster",
    ]
    broad_product_prompts = [
        "a simple ecommerce product packshot or catalog listing image",
        "a product photograph dominated by the package or bottle, with little graphic layout",
    ]

    with torch.no_grad():
        design_tokens = clip.tokenize(design_prompts).to(_clip_device)
        photo_tokens = clip.tokenize(photo_prompts).to(_clip_device)
        product_tokens = clip.tokenize(product_prompts).to(_clip_device)
        design_features = _clip_model.encode_text(design_tokens)
        photo_features = _clip_model.encode_text(photo_tokens)
        product_features = _clip_model.encode_text(product_tokens)
        design_features /= design_features.norm(dim=-1, keepdim=True)
        photo_features /= photo_features.norm(dim=-1, keepdim=True)
        product_features /= product_features.norm(dim=-1, keepdim=True)
        _clip_text_features = (design_features, photo_features)
        _product_text_features = product_features

        broad_groups = []
        for prompts in (broad_graphic_prompts, broad_photo_prompts, broad_video_prompts, broad_product_prompts):
            tokens = clip.tokenize(prompts).to(_clip_device)
            features = _clip_model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
            # Mean normalized prompt embeddings gives one prototype per class.
            prototype = features.mean(dim=0, keepdim=True)
            prototype /= prototype.norm(dim=-1, keepdim=True)
            broad_groups.append(prototype)
        _broad_class_features = torch.cat(broad_groups, dim=0)


def predict_ml(image: Image.Image):
    if _ml_model is None:
        return None
    try:
        image_tensor = _clip_preprocess(image).unsqueeze(0).to(_clip_device)
        with torch.no_grad():
            embedding = _clip_model.encode_image(image_tensor).cpu().numpy()
        return float(_ml_model.predict_proba(embedding)[0][1])
    except Exception as exc:
        print(f"[ML] Failed: {exc}")
        return None


def predict_clip_design(image: Image.Image):
    try:
        design_features, photo_features = _clip_text_features
        image_tensor = _clip_preprocess(image).unsqueeze(0).to(_clip_device)
        with torch.no_grad():
            image_features = _clip_model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            design_sim = (image_features @ design_features.T).mean().item()
            photo_sim = (image_features @ photo_features.T).mean().item()
        return 1.0 / (1.0 + math.exp(-(design_sim - photo_sim) * 8.0))
    except Exception as exc:
        print(f"[CLIP] Failed: {exc}")
        return None


def normalize_ocr_tokens(text):
    """Normalize OCR words so the same ad copy survives crop/frame changes."""
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9%+₹$€£]+", " ", text)
    tokens = [t for t in text.split() if len(t) >= 2 or t.isdigit()]
    return sorted(set(tokens))


def calculate_ocr(image: Image.Image):
    """Multi-pass OCR; keep strongest word-count/density result and its text."""
    try:
        from PIL import ImageEnhance, ImageFilter, ImageOps
        base = image.convert("RGB")
        scale = 2
        up = base.resize((base.width * scale, base.height * scale))
        gray = ImageOps.grayscale(up)
        gray = ImageEnhance.Contrast(gray).enhance(2.0)
        sharp = gray.filter(ImageFilter.SHARPEN)
        variants = [
            up, gray, sharp, ImageOps.autocontrast(gray),
            gray.point(lambda p: 255 if p > 170 else 0),
            gray.point(lambda p: 255 if p > 210 else 0),
        ]
        best = (0, 0.0, "")
        for variant in variants:
            data = pytesseract.image_to_data(
                variant, config="--psm 11", output_type=pytesseract.Output.DICT
            )
            detected = []
            for i, raw in enumerate(data["text"]):
                value = raw.strip()
                if not value:
                    continue
                try:
                    confidence = float(data["conf"][i])
                except Exception:
                    confidence = -1
                if confidence >= 30:
                    detected.append((
                        value,
                        int(data["left"][i]), int(data["top"][i]),
                        int(data["width"][i]), int(data["height"][i]),
                    ))
            word_count = len(detected)
            # Union of bounding boxes, not sum of their areas - naive summing
            # double-counts any overlapping/duplicate boxes (common with
            # multi-pass OCR on busy images) and can exceed the actual image
            # area entirely. A boolean coverage mask at native resolution
            # gives the real covered-pixel count.
            mask = np.zeros((base.height, base.width), dtype=bool)
            for _, x, y, w, h in detected:
                x0, y0 = max(0, x // scale), max(0, y // scale)
                x1, y1 = min(base.width, (x + w) // scale), min(base.height, (y + h) // scale)
                if x1 > x0 and y1 > y0:
                    mask[y0:y1, x0:x1] = True
            image_area = base.width * base.height
            density = float(mask.sum()) / image_area if image_area else 0.0
            raw_text = " ".join(v for v, *_ in detected)
            # Preserve the old selection rule exactly: word count first, density second.
            if (word_count, density) > (best[0], best[1]):
                best = (word_count, density, raw_text)
        word_count, density, raw_text = best
        word_signal = min(word_count / 14.0, 1.0)
        density_signal = min(density / 0.04, 1.0)
        ocr_score = 0.65 * word_signal + 0.35 * density_signal
        return word_count, density, ocr_score, " ".join(normalize_ocr_tokens(raw_text))
    except Exception as exc:
        print(f"[OCR] Failed: {exc}")
        return 0, 0.0, 0.0, ""


def visual_complexity(image: Image.Image):
    try:
        small = image.convert("RGB").resize((96, 96))
        stat = ImageStat.Stat(small)
        mean_variance = sum(stat.var) / 3.0
        return min(math.sqrt(mean_variance) / 80.0, 1.0)
    except Exception:
        return 0.0


def background_uniformity(image: Image.Image):
    """How plain/flat the image's border region is - a direct signal for
    "product photographed against a plain or gradient background," which
    reweighting CLIP/OCR features alone (hand-set or trained) can't fully
    capture, since those features don't explicitly represent background
    structure. High value = very uniform border = classic packshot cue.
    Returns 0-1, where 1.0 means the border is nearly a single flat color.
    """
    try:
        small = np.asarray(image.convert("RGB").resize((96, 96)))
        border = 8
        top = small[:border, :, :]
        bottom = small[-border:, :, :]
        left = small[:, :border, :]
        right = small[:, -border:, :]
        border_pixels = np.concatenate([
            top.reshape(-1, 3), bottom.reshape(-1, 3),
            left.reshape(-1, 3), right.reshape(-1, 3),
        ], axis=0).astype(np.float32)
        std = border_pixels.std(axis=0).mean()
        # std of ~0 (perfectly flat) -> 1.0; std of ~60+ (busy/textured
        # border, typical of a graphic with edge-to-edge design elements)
        # -> near 0.0.
        return float(max(0.0, min(1.0, 1.0 - (std / 60.0))))
    except Exception:
        return 0.0


def combined_score(clip_score, ocr_score, complexity):
    values = []
    if clip_score is not None:
        values.append((0.65, clip_score))
    values.append((0.25, ocr_score))
    values.append((0.10, complexity))
    total_weight = sum(w for w, _ in values)
    return sum(w*v for w,v in values) / total_weight if total_weight else 0.0


def classify(record):
    score = float(record["creative_score"])
    clip_score = record.get("clip_design_score")
    clip_score = float(clip_score) if clip_score not in (None, "") else None
    product_score = record.get("clip_product_score")
    product_score = float(product_score) if product_score not in (None, "") else None
    ocr_score = float(record.get("ocr_score", 0) or 0)
    density = float(record.get("ocr_text_density", 0) or 0)
    word_count = int(float(record.get("ocr_word_count", 0) or 0))

    if word_count == 0:
        return "REJECT", "no usable text detected on image"

    # Product/catalog guard:
    # OCR cannot tell whether text is an ad overlay or text printed on a package.
    # The product-photo signal is therefore used RELATIVE to the design signal.
    # A product-looking image is rejected only when product-photo confidence
    # clearly beats graphic-ad confidence. This avoids killing designed food/product
    # ads such as price-offer cards that happen to contain a prominent product.
    product_wins = (
        product_score is not None
        and clip_score is not None
        and product_score >= PRODUCT_PHOTO_VETO_THRESHOLD
        and product_score >= clip_score + PRODUCT_VETO_MARGIN
        and score < PRODUCT_VETO_MAX_CREATIVE_SCORE
    )

    # Borderline packaging shots: when both CLIP signals are around the same level,
    # only reject if product is still clearly the stronger signal and the design
    # score is not above the borderline range. This catches simple packshots while
    # preserving designed promotional layouts.
    borderline_product_photo = (
        product_score is not None
        and clip_score is not None
        and product_score >= 0.52
        and product_score > clip_score
        and clip_score <= 0.52
        and score < 0.70
    )

    if product_wins or borderline_product_photo:
        return "REJECT", "product/catalog photo with packaging text, weak graphic-ad evidence"

    if clip_score is not None:
        if clip_score < STRONG_PHOTO_CLIP_MAX and word_count <= 3 and density < 0.015:
            return "REJECT", "photo-like image with minimal text"
        if clip_score < 0.47 and ocr_score < 0.35:
            return "REJECT", "weak design evidence"
    if score >= KEEP_THRESHOLD and (clip_score is None or clip_score >= 0.48):
        if density < KEEP_MIN_TEXT_DENSITY:
            return "REJECT", "strong score but insufficient external ad-text coverage"
        return "KEEP", "strong visual design + text signals"

    # Phase 1 decision: there is no REVIEW bucket in the HTML.
    # Anything that does not meet the KEEP bar is rejected so the manual review
    # page stays focused on genuine design-led candidates.
    return "REJECT", "below strict design-led KEEP threshold"


def _rank_score_heuristic(record):
    """Hand-set fallback ranking formula. Used only if rank_model_v1.pkl
    (the trained model - see rank_score() below) is missing or fails to
    load. Weights here were hand-recalibrated once, guided by a diagnostic
    fit on an early 87-row labeled set, but this is not the trained model
    itself - see rank_score() for the real, current scoring path.
    """
    graphic = float(record.get("broad_graphic_prob", 0) or 0)
    photo = float(record.get("broad_photo_prob", 0) or 0)
    video = float(record.get("broad_video_prob", 0) or 0)
    product = float(record.get("broad_product_prob", 0) or 0)
    clip_design = float(record.get("clip_design_score", 0) or 0)
    clip_product = float(record.get("clip_product_score", 0) or 0)
    density = float(record.get("ocr_text_density", 0) or 0)
    complexity = float(record.get("visual_complexity", 0) or 0)
    bg_uniformity = float(record.get("background_uniformity", 0) or 0)

    text_signal = min(density / 0.05, 1.0)
    graphic_margin = graphic - max(photo, product)

    score = (
        0.35 * clip_design
        + 0.25 * graphic_margin
        + 0.10 * text_signal
        + 0.10 * complexity
        - 0.22 * video
        - 0.15 * clip_product
        - 0.15 * bg_uniformity
    )
    return round(score, 4)


def rank_score(record):
    """Continuous ranking score for broad-search candidates.

    This intentionally does NOT return KEEP/REJECT. Every past attempt to
    force these signals into a hard binary decision produced a threshold
    sitting inside a genuine overlap zone between design-led and
    non-design examples. A ranking only needs good examples to trend
    higher than bad ones on average, which these signals CAN support.

    ONE exception: zero real text (ocr_word_count == 0). Checked directly
    against every real, human-approved SAVE example ever labeled (88 of
    them) - not one had zero text. A genuine design-led ad essentially
    always has some real overlay text; a bare building photo or stock
    wildlife shot does not. This is a hard, evidence-backed veto, not a
    guessed threshold - it forces the score down regardless of what the
    model otherwise thinks.

    Supports two model formats, detected via bundle["uses_rich_embedding"]:
      - Old format: the original 9 engineered features only.
      - Rich format: CLIP's full embedding (PCA-reduced) combined with the
        same 9 features - built by retrain_rich.py, giving the model real
        visual detail instead of only a narrow numeric summary. Falls back
        to the 9-feature score if the embedding is missing for this record
        (e.g. it was scored before embedding capture was added), so nothing
        breaks on older or partially-migrated data.

    If no model file exists at all, falls back to the hand-set heuristic
    formula in _rank_score_heuristic() so the pipeline still works.
    """
    word_count = record.get("ocr_word_count")
    if word_count is not None and float(word_count or 0) == 0:
        return 0.05

    bundle = _load_rank_model()
    if bundle is None:
        return _rank_score_heuristic(record)

    features = bundle["features"]

    if bundle.get("uses_rich_embedding"):
        embedding_str = record.get("clip_embedding") or ""
        if embedding_str:
            try:
                embedding = np.array([[float(v) for v in embedding_str.split(",")]])
                reduced = bundle["pca"].transform(embedding)
                old_vals = np.array([[float(record.get(f, 0) or 0) for f in features]])
                combined = np.hstack([reduced, old_vals])
                scaled = bundle["scaler"].transform(combined)
                prob = bundle["model"].predict_proba(scaled)[0][1]
                return round(float(prob), 4)
            except Exception as exc:
                print(f"[RANK] Rich scoring failed, falling back to heuristic: {exc}")
                return _rank_score_heuristic(record)
        else:
            # No embedding available for this record - can't use the rich
            # model's expected input shape, so fall back rather than crash.
            return _rank_score_heuristic(record)

    row = [[float(record.get(f, 0) or 0) for f in features]]
    scaled = bundle["scaler"].transform(row)
    prob = bundle["model"].predict_proba(scaled)[0][1]
    return round(float(prob), 4)


def classify_broad_static(record):
    """Classify broad-search images using a four-way CLIP visual type gate.

    Broad search is judged differently from the legacy brand POC: a creative
    should look like an intentionally designed static ad, not merely contain
    enough OCR text to push its aggregate score over a threshold.
    """
    graphic = float(record.get("broad_graphic_prob", 0) or 0)
    photo = float(record.get("broad_photo_prob", 0) or 0)
    video = float(record.get("broad_video_prob", 0) or 0)
    product = float(record.get("broad_product_prob", 0) or 0)
    score = float(record.get("creative_score", 0) or 0)
    density = float(record.get("ocr_text_density", 0) or 0)
    words = int(float(record.get("ocr_word_count", 0) or 0))

    if words == 0:
        return "REJECT", "broad static gate: no usable text detected"

    # A visual video-frame signal overrides the fact that Meta supplied an
    # image URL. This catches paused TikTok/Reels frames that look like images.
    if video > BROAD_VIDEO_PROB_MAX and video >= graphic:
        return "REJECT", f"broad static gate: likely video/reel frame (CLIP video {video:.2f})"

    # The central rule: the image must actually look more like a designed
    # graphic than an ordinary photo. This prevents OCR-heavy lifestyle shots
    # from becoming KEEP simply because captions were detected.
    if graphic >= BROAD_GRAPHIC_PROB_KEEP and graphic > photo and graphic > product:
        return "KEEP", f"broad static gate: designed graphic (CLIP graphic {graphic:.2f})"

    # Allow a few borderline designed posters when the aggregate evidence is
    # strong, but only when the photo/video signals remain controlled.
    if (
        graphic >= 0.28
        and score >= 0.64
        and density >= 0.08
        and video <= 0.25
        and photo <= BROAD_PHOTO_PROB_MAX
        and graphic >= photo * 0.90
    ):
        return "KEEP", f"broad static gate: borderline designed graphic (CLIP graphic {graphic:.2f})"

    return "REJECT", (
        f"broad static gate: photo/video-like rather than design-led "
        f"(graphic {graphic:.2f}, photo {photo:.2f}, video {video:.2f}, product {product:.2f})"
    )


def fetch_image_in_memory(url):
    """Fetch remote creative bytes for analysis only; never writes them to disk."""
    if not url:
        return None
    try:
        response = requests.get(
            url, timeout=(15, 45),
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()
        if not response.content:
            return None
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image.load()
        return image
    except Exception as exc:
        print(f"[IMAGE FETCH] Failed: {exc}")
        return None


def frame_variant_hash(image: Image.Image):
    """Return several square crop hashes so aspect-ratio changes are tolerated."""
    from PIL import ImageOps
    base = image.convert("RGB")
    positions = [
        (0.5, 0.5),
        (0.25, 0.5), (0.75, 0.5),
        (0.5, 0.25), (0.5, 0.75),
    ]
    hashes = []
    for center in positions:
        square = ImageOps.fit(
            base, (256, 256), method=Image.Resampling.LANCZOS, centering=center
        )
        hashes.append(str(imagehash.phash(square)))
    return hashes


def crop_hash_distance(hashes_a, hashes_b):
    """Minimum pHash distance across multiple crop hypotheses."""
    try:
        a = hashes_a if isinstance(hashes_a, list) else json.loads(hashes_a)
        b = hashes_b if isinstance(hashes_b, list) else json.loads(hashes_b)
        best = 999
        for ha in a:
            for hb in b:
                try:
                    best = min(best, imagehash.hex_to_hash(str(ha)) - imagehash.hex_to_hash(str(hb)))
                except Exception:
                    pass
        return best
    except Exception:
        return 999


def ocr_token_jaccard(a, b):
    try:
        aa = set(a if isinstance(a, list) else str(a).split())
        bb = set(b if isinstance(b, list) else str(b).split())
        if not aa or not bb:
            return 0.0
        return len(aa & bb) / len(aa | bb)
    except Exception:
        return 0.0


def compute_sift_descriptors(image: Image.Image):
    """Local-feature signature for recognizing the same artwork after crop/resize."""
    try:
        arr = np.asarray(image.convert("RGB"))
        h, w = arr.shape[:2]
        scale = min(1.0, 900.0 / max(h, w))
        if scale < 1.0:
            arr = cv2.resize(arr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create(nfeatures=300, contrastThreshold=0.015)
        keypoints, desc = sift.detectAndCompute(gray, None)
        if desc is None or len(keypoints) < 6:
            found = 0 if keypoints is None else len(keypoints)
            print(f"    [SIFT] Too few keypoints ({found}) - no signature for this image")
            return None
        points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        proc_h, proc_w = gray.shape[:2]
        return {"points": points, "desc": desc.astype(np.float32), "size": (proc_w, proc_h)}
    except Exception as exc:
        print(f"    [SIFT] Failed to compute descriptors: {type(exc).__name__}: {exc}")
        return None


def sift_same_creative(sig_a, sig_b):
    """Confirm same artwork using local matches + a geometric homography check.

    Ad templates from the same brand often share identical chrome (logo,
    price badge, border, button shapes) around a different product photo.
    Matching keypoints found only in that shared chrome can look like a
    strong match (many inliers) even when the actual photos are completely
    different. To guard against that, a real match must also have its
    inlier keypoints spread across a meaningful portion of the image, not
    clustered in one small corner where a logo/badge typically sits.
    """
    if not sig_a or not sig_b:
        return False, 0
    try:
        da, db = sig_a["desc"], sig_b["desc"]
        pa, pb = sig_a["points"], sig_b["points"]
        if len(da) < 6 or len(db) < 6:
            return False, 0
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        pairs = matcher.knnMatch(da, db, k=2)
        good = [m for m, n in pairs if m.distance < 0.72 * n.distance]
        if len(good) < 8:
            return False, len(good)
        src = np.float32([pa[m.queryIdx] for m in good]).reshape(-1, 1, 2)
        dst = np.float32([pb[m.trainIdx] for m in good]).reshape(-1, 1, 2)
        if len(good) >= 4:
            _H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            inliers = int(mask.sum()) if mask is not None else 0
            if inliers >= 7 and inliers / max(1, len(good)) >= 0.35:
                # Spatial-spread check: require the inlier points to cover a
                # real portion of BOTH images, not just a shared corner badge.
                mask_flat = mask.ravel().astype(bool)
                inlier_src = src.reshape(-1, 2)[mask_flat]
                inlier_dst = dst.reshape(-1, 2)[mask_flat]
                size_a = sig_a.get("size")
                size_b = sig_b.get("size")
                if size_a and size_b and len(inlier_src) >= 4:
                    wa, ha = size_a
                    wb, hb = size_b
                    spread_ax = (inlier_src[:, 0].max() - inlier_src[:, 0].min()) / max(1, wa)
                    spread_ay = (inlier_src[:, 1].max() - inlier_src[:, 1].min()) / max(1, ha)
                    spread_bx = (inlier_dst[:, 0].max() - inlier_dst[:, 0].min()) / max(1, wb)
                    spread_by = (inlier_dst[:, 1].max() - inlier_dst[:, 1].min()) / max(1, hb)
                    # Both images need matches spanning a real chunk of the
                    # frame in at least one direction; a shared logo/badge in
                    # a fixed corner produces a small, tight cluster instead.
                    if max(spread_ax, spread_ay) < 0.30 or max(spread_bx, spread_by) < 0.30:
                        return False, inliers
                return True, inliers
        return False, len(good)
    except Exception:
        return False, 0



def extract_clip_embeddings_batch(images):
    """Returns the FULL normalized CLIP embedding for each image (512 numbers
    for ViT-B/32), instead of collapsing it down to a handful of prompt-
    similarity scores. This is the raw visual fingerprint CLIP already
    computes internally for every other function above - it was just never
    being kept. Storing this lets a future classifier learn from real,
    rich visual detail instead of only 9 summary numbers.
    """
    if not images:
        return []
    try:
        tensors = torch.stack([_clip_preprocess(image) for image in images]).to(_clip_device)
        with torch.no_grad():
            image_features = _clip_model.encode_image(tensors)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return [row.detach().cpu().tolist() for row in image_features]
    except Exception as exc:
        print(f"[CLIP EMBEDDING] Failed: {exc}")
        return [None for _ in images]


def predict_clip_design_batch(images):
    """Batch CLIP inference using the exact same model/prompts/math as the single-image path."""
    if not images:
        return []
    try:
        design_features, photo_features = _clip_text_features
        tensors = torch.stack([_clip_preprocess(image) for image in images]).to(_clip_device)
        with torch.no_grad():
            image_features = _clip_model.encode_image(tensors)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            design_sim = (image_features @ design_features.T).mean(dim=1)
            photo_sim = (image_features @ photo_features.T).mean(dim=1)
            values = 1.0 / (1.0 + torch.exp(-(design_sim - photo_sim) * 8.0))
        return [float(x) for x in values.detach().cpu().tolist()]
    except Exception as exc:
        print(f"[CLIP BATCH] Failed; falling back to single-image inference: {exc}")
        return [predict_clip_design(image) for image in images]


def predict_product_photo_batch(images):
    """Return a 0..1 CLIP score for catalog/packshot/product-photography imagery.

    This is deliberately separate from the existing design-vs-photo score.
    The existing photo prompts are broad and can miss the specific failure mode
    where OCR comes mostly from text printed on product packaging.
    """
    if not images:
        return []
    try:
        product_features = _product_text_features
        design_features, _photo_features = _clip_text_features
        tensors = torch.stack([_clip_preprocess(image) for image in images]).to(_clip_device)
        with torch.no_grad():
            image_features = _clip_model.encode_image(tensors)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            product_sim = (image_features @ product_features.T).mean(dim=1)
            design_sim = (image_features @ design_features.T).mean(dim=1)
            values = 1.0 / (1.0 + torch.exp(-(product_sim - design_sim) * 10.0))
        return [float(x) for x in values.detach().cpu().tolist()]
    except Exception as exc:
        print(f"[PRODUCT CLIP BATCH] Failed; falling back to single-image inference: {exc}")
        return [predict_product_photo(image) for image in images]


def predict_product_photo(image: Image.Image):
    try:
        product_features = _product_text_features
        design_features, _photo_features = _clip_text_features
        image_tensor = _clip_preprocess(image).unsqueeze(0).to(_clip_device)
        with torch.no_grad():
            image_features = _clip_model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            product_sim = (image_features @ product_features.T).mean().item()
            design_sim = (image_features @ design_features.T).mean().item()
        return 1.0 / (1.0 + math.exp(-(product_sim - design_sim) * 10.0))
    except Exception as exc:
        print(f"[PRODUCT CLIP] Failed: {exc}")
        return None


def predict_broad_static_batch(images):
    """Return CLIP probabilities for graphic/photo/video-frame/product classes."""
    if not images:
        return []
    try:
        tensors = torch.stack([_clip_preprocess(image) for image in images]).to(_clip_device)
        with torch.no_grad():
            image_features = _clip_model.encode_image(tensors)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ _broad_class_features.T
            probs = torch.softmax(logits * 25.0, dim=1)
        return [row.detach().cpu().tolist() for row in probs]
    except Exception as exc:
        print(f"[BROAD CLIP] Failed: {exc}")
        return [[0.0, 0.0, 0.0, 0.0] for _ in images]


def predict_ml_batch(images):
    """Batch legacy-model diagnostics; this model never affects KEEP decisions."""
    if _ml_model is None or not images:
        return [None] * len(images)
    try:
        tensors = torch.stack([_clip_preprocess(image) for image in images]).to(_clip_device)
        with torch.no_grad():
            embeddings = _clip_model.encode_image(tensors).cpu().numpy()
        return [float(x) for x in _ml_model.predict_proba(embeddings)[:, 1]]
    except Exception as exc:
        print(f"[ML BATCH] Failed; falling back to single-image inference: {exc}")
        return [predict_ml(image) for image in images]


def analyze_remote_creatives_batch(urls):
    """Analyze a batch while preserving the exact existing scoring logic.

    Network fetches and OCR are parallelized. CLIP/legacy ML image inference is
    batched. OCR still runs the same six variants with the same thresholds.
    """
    if not urls:
        return []

    images = [None] * len(urls)
    # Fetch independently so one slow URL does not block every other image.
    with ThreadPoolExecutor(max_workers=ANALYSIS_FETCH_WORKERS) as pool:
        future_map = {pool.submit(fetch_image_in_memory, url): i for i, url in enumerate(urls)}
        for future in as_completed(future_map):
            i = future_map[future]
            try:
                images[i] = future.result()
            except Exception as exc:
                print(f"[IMAGE FETCH] Failed: {exc}")

    valid = [(i, image) for i, image in enumerate(images) if image is not None]
    results = [None] * len(urls)
    if not valid:
        return results

    valid_indices = [i for i, _ in valid]
    valid_images = [image for _, image in valid]

    # One batched CLIP pass and one batched legacy diagnostic pass.
    clip_scores = predict_clip_design_batch(valid_images)
    product_scores = predict_product_photo_batch(valid_images)
    broad_probs = predict_broad_static_batch(valid_images)
    ml_scores = predict_ml_batch(valid_images)
    rich_embeddings = extract_clip_embeddings_batch(valid_images)

    # OCR remains exactly the same function, simply run concurrently.
    with ThreadPoolExecutor(max_workers=OCR_WORKERS) as pool:
        ocr_future_map = {pool.submit(calculate_ocr, image): j for j, image in enumerate(valid_images)}
        ocr_results = [None] * len(valid_images)
        for future in as_completed(ocr_future_map):
            j = ocr_future_map[future]
            try:
                ocr_results[j] = future.result()
            except Exception as exc:
                print(f"[OCR] Failed: {exc}")
                ocr_results[j] = (0, 0.0, 0.0, "")

    for j, original_index in enumerate(valid_indices):
        image = valid_images[j]
        ml_score = ml_scores[j]
        clip_score = clip_scores[j]
        product_score = product_scores[j]
        word_count, density, ocr_score, ocr_text = ocr_results[j]
        complexity = visual_complexity(image)
        bg_uniformity = background_uniformity(image)
        score = combined_score(clip_score, ocr_score, complexity)
        embedding = rich_embeddings[j]
        clip_embedding_str = ",".join(f"{v:.5f}" for v in embedding) if embedding is not None else ""
        result = {
            "design_score": round(ml_score, 4) if ml_score is not None else "",
            "clip_design_score": round(clip_score, 4) if clip_score is not None else "",
            "clip_product_score": round(product_score, 4) if product_score is not None else "",
            "clip_embedding": clip_embedding_str,
            "broad_graphic_prob": round(broad_probs[j][0], 4),
            "broad_photo_prob": round(broad_probs[j][1], 4),
            "broad_video_prob": round(broad_probs[j][2], 4),
            "broad_product_prob": round(broad_probs[j][3], 4),
            "ocr_word_count": word_count,
            "ocr_text_density": round(density, 6),
            "ocr_score": round(ocr_score, 4),
            "visual_complexity": round(complexity, 4),
            "background_uniformity": round(bg_uniformity, 4),
            "creative_score": round(score, 4),
            "image_width": image.width,
            "image_height": image.height,
            "aspect_ratio": round(image.width / image.height, 4) if image.height else 0,
            "ocr_fingerprint": ocr_text,
        }
        status, reason = classify(result)
        result["filter_status"] = status
        result["filter_reason"] = reason
        # Computed here, while the image is still open, so duplicate detection
        # later in the pipeline has real descriptors instead of None.
        sift_signature = compute_sift_descriptors(image)
        results[original_index] = (
            result, str(imagehash.phash(image)), json.dumps(frame_variant_hash(image)), sift_signature,
        )

    for image in images:
        if image is not None:
            try:
                image.close()
            except Exception:
                pass
    return results


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_name(value):
    value = str(value or "").strip()

    output = []

    for char in value:
        if char.isalnum() or char in (" ", "-", "_"):
            output.append(char)

    value = "".join(output)
    value = "_".join(value.split())

    return value or "Unknown"


def clean_text(value):
    if value is None:
        return ""

    if isinstance(value, (list, tuple)):
        return " ".join(str(x) for x in value)

    if isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False
        )

    return str(value)


def get_attr(obj, *names, default=""):
    """
    Safely get an attribute from collector model objects.
    Also supports dictionaries.
    """

    for name in names:

        try:
            value = getattr(
                obj,
                name,
                None
            )

            if value is not None:
                return value

        except Exception:
            pass

        if isinstance(obj, dict):

            value = obj.get(name)

            if value is not None:
                return value

    return default


# ============================================================
# DATABASE
# ============================================================

def browse_master_db(category=None, country=None, platform=None, media_type=None,
                      status=None, brand=None, min_rank_score=None, min_clip_design=None, days=None,
                      db_path=None):
    """Query the ACCUMULATED master database - built up across every
    --broad-search run you've ever done - rather than any single run's
    results. This is what "search shouldn't be capped by max-results"
    actually means in practice: max-results controls how much a single
    ingestion run pulls; browsing queries however much has piled up in
    total across all runs so far. This is also the function the Flask
    backend (dashboard_server_temporary.py) calls directly for every API request -
    it's the single source of truth for querying creatives, whether from
    the CLI or the website.

    min_clip_design exists because rank_score alone was letting mediocre
    bold-text-banner ads (clip_design_score 0.52-0.60) through at
    near-maximum confidence (0.98+) - a real overconfidence problem in the
    trained model. Requiring a real minimum on the raw CLIP signal too is
    a direct, evidence-based guard against exactly that failure mode.
    """
    database_path = Path(db_path) if db_path is not None else MASTER_DB_PATH
    if not database_path.exists():
        return []
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    query = "SELECT " + ", ".join(CSV_COLUMNS) + " FROM ads WHERE is_creative_representative=1"
    params = []
    if category:
        query += " AND category LIKE ?"; params.append(f"%{category}%")
    if country:
        query += " AND country = ?"; params.append(country.upper())
    if platform and platform.upper() != "ALL":
        query += " AND (platform = ? OR platform = 'ALL')"; params.append(platform.upper())
    if media_type:
        query += " AND media_type = ?"; params.append(media_type.upper())
    # NOTE: there is no real "status" column - Meta's raw ACTIVE/INACTIVE
    # state was never actually captured anywhere in this schema (every
    # scrape only ever queries status=ACTIVE at acquisition time, so
    # everything stored was active as of when it was last seen - there's
    # no INACTIVE data to filter against). The dashboard's Status dropdown
    # used to crash the whole request with a 500 error whenever touched,
    # because it referenced a column that was never created. Silently
    # ignoring it here is honest given what data actually exists.
    if brand:
        query += " AND brand LIKE ?"; params.append(f"%{brand}%")
    if min_rank_score is not None:
        query += " AND rank_score >= ?"; params.append(min_rank_score)
    if min_clip_design is not None:
        query += " AND clip_design_score >= ?"; params.append(min_clip_design)
    if days:
        cutoff = (datetime.now() - timedelta(days=int(days))).strftime("%Y-%m-%d %H:%M:%S")
        query += " AND last_seen_at >= ?"; params.append(cutoff)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_database(
    db_path
):
    db_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    conn = sqlite3.connect(
        db_path
    )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            brand TEXT,
            page_name TEXT,
            page_id TEXT,
            ad_id TEXT,
            creative_index INTEGER,
            image_file TEXT,
            image_url TEXT,
            body TEXT,
            title TEXT,
            cta TEXT,
            image_hash TEXT,
            duplicate INTEGER DEFAULT 0,
            design_score REAL,
            clip_design_score REAL,
            clip_product_score REAL,
            broad_graphic_prob REAL,
            broad_photo_prob REAL,
            broad_video_prob REAL,
            broad_product_prob REAL,
            ocr_word_count INTEGER,
            ocr_text_density REAL,
            ocr_score REAL,
            visual_complexity REAL,
            background_uniformity REAL,
            creative_score REAL,
            filter_status TEXT,
            filter_reason TEXT,
            image_width INTEGER,
            image_height INTEGER,
            aspect_ratio REAL,
            frame_variant_hash TEXT,
            ocr_fingerprint TEXT,
            creative_group_id INTEGER,
            is_creative_representative INTEGER DEFAULT 1,
            creative_variant_of TEXT,
            creative_variant_count INTEGER DEFAULT 0,
            representative_reason TEXT,
            rank_score REAL,
            human_label TEXT,
            search_query TEXT,
            country TEXT,
            platform TEXT,
            media_type TEXT,
            first_seen_at TEXT,
            last_seen_at TEXT
        )
    """)

    # Migrate older databases created by the previous scraper/schema.
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(ads)").fetchall()}
    migrations = {
        "design_score": "REAL",
        "clip_design_score": "REAL",
        "clip_product_score": "REAL",
        "clip_embedding": "TEXT",
        "broad_graphic_prob": "REAL",
        "broad_photo_prob": "REAL",
        "broad_video_prob": "REAL",
        "broad_product_prob": "REAL",
        "ocr_word_count": "INTEGER",
        "ocr_text_density": "REAL",
        "ocr_score": "REAL",
        "visual_complexity": "REAL",
        "background_uniformity": "REAL",
        "creative_score": "REAL",
        "filter_status": "TEXT",
        "filter_reason": "TEXT",
        "image_width": "INTEGER",
        "image_height": "INTEGER",
        "aspect_ratio": "REAL",
        "frame_variant_hash": "TEXT",
        "ocr_fingerprint": "TEXT",
        "creative_group_id": "INTEGER",
        "is_creative_representative": "INTEGER DEFAULT 1",
        "creative_variant_of": "TEXT",
        "creative_variant_count": "INTEGER DEFAULT 0",
        "representative_reason": "TEXT",
        "rank_score": "REAL",
        "human_label": "TEXT",
        "search_query": "TEXT",
        "country": "TEXT",
        "platform": "TEXT",
        "media_type": "TEXT",
        "first_seen_at": "TEXT",
        "last_seen_at": "TEXT",
    }
    for column, sql_type in migrations.items():
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE ads ADD COLUMN {column} {sql_type}")

    # Older databases (created before this unique index existed) may already
    # contain duplicate image_hash rows, which would make CREATE UNIQUE
    # INDEX fail. Only attempt the index after cleaning those up.
    dupe_hashes = conn.execute("""
        SELECT image_hash FROM ads
        WHERE image_hash IS NOT NULL AND image_hash != ''
        GROUP BY image_hash HAVING COUNT(*) > 1
    """).fetchall()
    for (h,) in dupe_hashes:
        ids = [r[0] for r in conn.execute("SELECT id FROM ads WHERE image_hash=? ORDER BY id", (h,)).fetchall()]
        for old_id in ids[1:]:
            conn.execute("DELETE FROM ads WHERE id=?", (old_id,))
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_ads_image_hash
        ON ads(image_hash) WHERE image_hash IS NOT NULL AND image_hash != ''
    """)

    # Rows scraped before country/platform/media_type existed as columns
    # have NULL in them - and "country = 'IN'" never matches NULL in SQL,
    # so those older rows would silently vanish from every country/platform
    # filter forever otherwise. Backfill with the one safe default: every
    # run before this fix was India, so that's what NULL really means here.
    conn.execute("UPDATE ads SET country='IN' WHERE country IS NULL OR country=''")
    conn.execute("UPDATE ads SET platform='ALL' WHERE platform IS NULL OR platform=''")
    conn.execute("UPDATE ads SET media_type='IMAGE' WHERE media_type IS NULL OR media_type=''")
    conn.commit()

    conn.commit()

    return conn


def insert_ad(conn, data):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    img_hash = data.get("image_hash") or ""

    # If this exact creative (by image_hash) is already in the database from
    # an earlier run, don't insert a duplicate row - just note we saw it
    # again. This is what makes it safe to re-run the same search/category
    # repeatedly: the database accumulates NEW creatives over time instead
    # of growing a duplicate copy of everything already collected.
    if img_hash:
        existing = conn.execute("SELECT id FROM ads WHERE image_hash=?", (img_hash,)).fetchone()
        if existing:
            conn.execute("UPDATE ads SET last_seen_at=? WHERE image_hash=?", (now, img_hash))
            return

    conn.execute("""
        INSERT INTO ads (
            category, brand, page_name, page_id, ad_id, creative_index,
            image_file, image_url, body, title, cta, image_hash, duplicate,
            design_score, clip_design_score, clip_product_score, clip_embedding, broad_graphic_prob, broad_photo_prob, broad_video_prob, broad_product_prob, ocr_word_count, ocr_text_density,
            ocr_score, visual_complexity, creative_score, filter_status,
            filter_reason, image_width, image_height, aspect_ratio, frame_variant_hash, ocr_fingerprint,
            creative_group_id, is_creative_representative, creative_variant_of,
            creative_variant_count, representative_reason, rank_score, human_label, background_uniformity,
            search_query, country, platform, media_type, first_seen_at, last_seen_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["category"], data["brand"], data["page_name"], data["page_id"],
        data["ad_id"], data["creative_index"], data["image_file"],
        data["image_url"], data["body"], data["title"], data["cta"],
        data["image_hash"], data["duplicate"], data.get("design_score"),
        data.get("clip_design_score"), data.get("clip_product_score"), data.get("clip_embedding", ""), data.get("broad_graphic_prob"), data.get("broad_photo_prob"), data.get("broad_video_prob"), data.get("broad_product_prob"), data.get("ocr_word_count"),
        data.get("ocr_text_density"), data.get("ocr_score"),
        data.get("visual_complexity"), data.get("creative_score"),
        data.get("filter_status"), data.get("filter_reason"),
        data.get("image_width"), data.get("image_height"),
        data.get("aspect_ratio"), data.get("frame_variant_hash"), data.get("ocr_fingerprint"),
        data.get("creative_group_id"), data.get("is_creative_representative", 1),
        data.get("creative_variant_of", ""), data.get("creative_variant_count", 0),
        data.get("representative_reason", ""), data.get("rank_score"), data.get("human_label", ""), data.get("background_uniformity"),
        data.get("search_query", ""), data.get("country", ""), data.get("platform", ""), data.get("media_type", "IMAGE"), now, now,
    ))


# ============================================================
# BROAD SEARCH ACQUISITION — APP MODE
# ============================================================

def search_ads_broad(
    query="",
    country="IN",
    status="ACTIVE",
    search_type="KEYWORD_UNORDERED",
    max_results=100,
    page_size=10,
    media_type=None,
    publisher_platforms=None,
    start_date=None,
    end_date=None,
    static_only=False,
    sort_by=None,
    on_collect=None,
    stop_check=None,
    on_interrupted=None,
):
    """
    Broad Meta Ad Library acquisition for the future internal app.

    It searches by keyword without requiring a predefined brand/page ID and
    returns one normalized record per creative.

    No images are written to disk here. The existing intelligence pipeline can
    consume the returned image_url values later.

    on_collect(ad_count) is called after every ad Meta returns, so callers can
    show live collection progress and notice when Meta stops responding.
    stop_check() is checked before each ad; when it returns True collection
    ends early and the creatives gathered so far are returned.

    If Meta cuts collection off partway (rate limiting, timeouts), the
    creatives gathered so far are returned and on_interrupted(exc) is called,
    rather than discarding them. With nothing collected the error is raised.
    """
    collector = MetaAdsCollector()

    # Broad design search is a STATIC image library. Do not let the collector
    # return video ads whose preview frame happens to expose an image URL.
    # The user-facing media selector may still be ALL for acquisition tests,
    # but the design pipeline itself must request IMAGE + no video.
    if static_only:
        effective_media_type = "IMAGE"
        effective_has_image = True
        effective_has_video = False
    else:
        effective_media_type = media_type
        effective_has_image = True if media_type == "IMAGE" else None
        effective_has_video = None

    filter_config = FilterConfig(
        start_date=start_date,
        end_date=end_date,
        media_type=effective_media_type,
        publisher_platforms=publisher_platforms,
        has_image=effective_has_image,
        has_video=effective_has_video,
    )

    print("\n" + "=" * 70)
    print("BROAD META AD SEARCH — APP MODE")
    print("=" * 70)
    print(f"Query:       {query or '[ALL ADS]'}")
    print(f"Country:     {country}")
    print(f"Status:      {status}")
    print(f"Media:       {'IMAGE (static only)' if static_only else (media_type or 'ALL')}")
    print(f"Platforms:   {publisher_platforms or 'ALL'}")
    print(f"Max results: {max_results}")
    print(f"Sort by:     {sort_by or 'RELEVANCY (not impressions - avoids always surfacing the same big campaigns)'}")
    print("=" * 70)

    ads = collector.search(
        query=query,
        country=country,
        status=status,
        search_type=search_type,
        max_results=max_results,
        page_size=page_size,
        filter_config=filter_config,
        sort_by=sort_by,
    )

    records = []
    ad_count = 0
    interruption = []

    def until_interrupted(stream):
        try:
            yield from stream
        except Exception as exc:
            interruption.append(exc)

    for ad in until_interrupted(ads):
        if stop_check is not None and stop_check():
            print(f"\n[STOPPED] Collection ended early after {ad_count} ad(s).")
            break
        ad_count += 1
        if on_collect is not None:
            on_collect(ad_count)
        page = get_attr(ad, "page", default=None)
        page_name = clean_text(get_attr(page, "name", "page_name", default=""))
        page_id = clean_text(get_attr(page, "id", "page_id", default=""))
        creatives = get_attr(ad, "creatives", "creative", default=[]) or []
        if not isinstance(creatives, (list, tuple)):
            creatives = [creatives]

        for creative_index, creative in enumerate(creatives):
            image_url = clean_text(get_attr(creative, "image_url", "image", "image_uri", default=""))
            thumbnail_url = clean_text(get_attr(creative, "thumbnail_url", "thumbnail", default=""))
            video_url = clean_text(get_attr(creative, "video_url", "video", default=""))
            preview_url = clean_text(get_attr(creative, "video_preview_image_url", default=""))

            # STATIC means static. Never turn a video thumbnail/preview into a
            # design-library image. Some Meta collector responses expose an image
            # URL even when the underlying creative is video, so checking only
            # image_url is not sufficient.
            if static_only:
                creative_type = clean_text(get_attr(
                    creative, "type", "creative_type", "format", "media_type", "display_format", default=""
                )).upper()
                ad_type = clean_text(get_attr(ad, "ad_type", default="")).upper()

                raw = get_attr(ad, "raw_data", default={}) or {}
                raw_display = clean_text(get_attr(raw, "display_format", "format", default="")).upper()
                raw_videos = get_attr(raw, "videos", default=[]) or []

                video_markers = ("VIDEO", "REELS", "VIDEO_SLIDESHOW", "VIDEO_COLLECTION")
                looks_like_video = (
                    bool(video_url)
                    or bool(preview_url)
                    or any(marker in creative_type for marker in video_markers)
                    or any(marker in ad_type for marker in video_markers)
                    or any(marker in raw_display for marker in video_markers)
                )

                # If the collector gives us a raw videos payload but no explicit
                # static creative type, treat the creative as video unless it has
                # a real image asset and no video-specific fields. This prevents
                # video previews from slipping through while preserving genuine
                # image creatives in mixed responses.
                if raw_videos and not creative_type and not raw_display:
                    looks_like_video = True

                if looks_like_video:
                    continue

            # For the static pipeline we require the actual image asset.
            # Thumbnail/video-preview fallback is deliberately disabled.
            if not image_url:
                if not static_only:
                    image_url = thumbnail_url or (preview_url if video_url else "")
            if not image_url:
                continue

            records.append({
                "ad_id": clean_text(get_attr(ad, "id", "ad_id", default="")),
                "page_id": page_id,
                "page_name": page_name,
                "creative_index": creative_index,
                "image_url": image_url,
                "video_url": video_url,
                "body": clean_text(get_attr(creative, "body", default="")),
                "title": clean_text(get_attr(creative, "title", default="")),
                "cta": clean_text(get_attr(creative, "cta_text", "cta", default="")),
                "link_url": clean_text(get_attr(creative, "link_url", default="")),
                "is_active": get_attr(ad, "is_active", default=None),
                "delivery_start_time": clean_text(get_attr(ad, "delivery_start_time", default="")),
                "delivery_stop_time": clean_text(get_attr(ad, "delivery_stop_time", default="")),
                "publisher_platforms": clean_text(get_attr(ad, "publisher_platforms", default=[])),
                "impressions": clean_text(get_attr(ad, "impressions", default="")),
                "spend": clean_text(get_attr(ad, "spend", default="")),
                "ad_type": clean_text(get_attr(ad, "ad_type", default="")),
                "categories": clean_text(get_attr(ad, "categories", default=[])),
            })

    if interruption:
        if not records:
            raise interruption[0]
        print(f"\n[INTERRUPTED] Meta stopped the search after {ad_count} ad(s): {interruption[0]}")
        print("              Keeping the creatives collected so far.")
        if on_interrupted is not None:
            on_interrupted(interruption[0])

    print(f"\nAds received:      {ad_count}")
    if static_only:
        print("Static filter:    IMAGE + has_video=False; video previews/thumbnails excluded")
    print(f"Image creatives:   {len(records)}")

    return records


def deduplicate_analyzed_rows(analyzed_rows):
    """Group duplicate creatives across brands/ad IDs using the existing POC rules."""
    representatives = []
    duplicate_count = 0
    groups = {}
    hash_index = {}
    token_index = {}
    crophash_index = {}
    text_poor_indices = []
    group_counter = 0

    def hash_distance(a, b):
        try:
            return imagehash.hex_to_hash(str(a)) - imagehash.hex_to_hash(str(b))
        except Exception:
            return 999

    def row_tokens(row):
        return set(str(row.get("ocr_fingerprint") or "").split())

    def numeric_tokens(tokens):
        result = set()
        for t in tokens:
            core = t.strip("₹$€£%+")
            if core.isdigit() and 2 <= len(core) <= 4:
                result.add(core)
        return result

    def numbers_compatible(set_a, set_b):
        for a in set_a:
            for b in set_b:
                if a == b or a in b or b in a:
                    return True
                if len(a) >= 3 and len(b) >= 3 and a[-3:] == b[-3:]:
                    return True
        return False

    def same_underlying_creative(a, b):
        text_a = str(a.get("ocr_fingerprint") or "").split()
        text_b = str(b.get("ocr_fingerprint") or "").split()
        num_a = numeric_tokens(text_a)
        num_b = numeric_tokens(text_b)
        if num_a and num_b and not numbers_compatible(num_a, num_b):
            return False, f"numeric mismatch ({sorted(num_a)} vs {sorted(num_b)})"

        if a.get("image_hash") and a.get("image_hash") == b.get("image_hash"):
            return True, "exact pHash"

        text_similarity = ocr_token_jaccard(text_a, text_b)
        visual_distance = crop_hash_distance(
            a.get("frame_variant_hash", "[]"),
            b.get("frame_variant_hash", "[]")
        )

        if len(text_a) >= 3 and len(text_b) >= 3 and text_similarity >= CREATIVE_OCR_JACCARD:
            matched, match_count = sift_same_creative(
                a.get("_sift_descriptors"), b.get("_sift_descriptors")
            )
            if matched:
                return True, f"OCR {text_similarity:.2f} + SIFT inliers={match_count}"
            if visual_distance <= 8 and text_similarity >= CREATIVE_OCR_STRONG_JACCARD:
                return True, f"crop pHash {visual_distance} + OCR {text_similarity:.2f}"
            return False, f"OCR {text_similarity:.2f} + SIFT inliers={match_count} + crop pHash {visual_distance} (not enough)"

        if visual_distance <= 6:
            return True, f"crop pHash {visual_distance} (visual match, low/no OCR overlap)"

        matched, match_count = sift_same_creative(
            a.get("_sift_descriptors"), b.get("_sift_descriptors")
        )
        if matched:
            return True, f"SIFT inliers={match_count} (visual match, low/no OCR overlap)"
        return False, f"crop pHash {visual_distance} + SIFT inliers={match_count} (not enough, low/no OCR overlap)"

    def is_better_frame(candidate, current):
        # The real fix: which crop looks like the better DESIGN, according
        # to the actual trained model - not just which one is closest to a
        # square. A small margin (0.03) avoids meaningless swaps between
        # two genuinely near-identical scores; only a real difference in
        # quality changes the winner.
        cs = float(candidate.get("rank_score") or 0)
        rs = float(current.get("rank_score") or 0)
        if abs(cs - rs) > 0.03:
            return cs > rs

        cr = float(candidate.get("aspect_ratio") or 0)
        rr = float(current.get("aspect_ratio") or 0)
        cd = abs(cr - 1.0) if cr else 999
        rd = abs(rr - 1.0) if rr else 999
        if cd != rd:
            return cd < rd
        cp = int(candidate.get("image_width") or 0) * int(candidate.get("image_height") or 0)
        rp = int(current.get("image_width") or 0) * int(current.get("image_height") or 0)
        if cp != rp:
            return cp > rp
        return float(candidate.get("creative_score") or 0) > float(current.get("creative_score") or 0)

    def crop_prefixes(row):
        try:
            hashes = json.loads(row.get("frame_variant_hash") or "[]")
        except Exception:
            hashes = []
        return {str(h)[:4] for h in hashes if h}

    def register(idx, row):
        if row.get("image_hash"):
            hash_index.setdefault(row["image_hash"], idx)
        toks = row_tokens(row)
        if len(toks) >= 3:
            for t in toks:
                token_index.setdefault(t, set()).add(idx)
        else:
            text_poor_indices.append(idx)
        for prefix in crop_prefixes(row):
            crophash_index.setdefault(prefix, set()).add(idx)

    for row in analyzed_rows:
        candidate_idxs = set()
        if row.get("image_hash") and row["image_hash"] in hash_index:
            candidate_idxs.add(hash_index[row["image_hash"]])
        for prefix in crop_prefixes(row):
            candidate_idxs.update(crophash_index.get(prefix, ()))
        toks = row_tokens(row)
        if len(toks) >= 3:
            for token in toks:
                candidate_idxs.update(token_index.get(token, ()))
        else:
            candidate_idxs.update(text_poor_indices)

        matched_index = None
        matched_reason = ""
        tried = []
        for idx in candidate_idxs:
            same, reason = same_underlying_creative(row, representatives[idx])
            tried.append((idx, reason))
            if same:
                matched_index = idx
                matched_reason = reason
                break

        if matched_index is None:
            idx = len(representatives)
            representatives.append(row)
            group_counter += 1
            row["creative_group_id"] = group_counter
            groups[group_counter] = {"rep_idx": idx, "variants": []}
            register(idx, row)
            continue

        current = representatives[matched_index]
        group_id = current["creative_group_id"]
        duplicate_count += 1
        row["creative_group_id"] = group_id
        row["duplicate"] = 1

        if is_better_frame(row, current):
            current["duplicate"] = 1
            row["duplicate"] = 0  # this row just WON the swap and is now the representative -
                                   # the premature duplicate=1 set above must not stick
            representatives[matched_index] = row
            groups[group_id]["variants"].append((current, matched_reason))
            register(matched_index, row)
        else:
            groups[group_id]["variants"].append((row, matched_reason))

    for group_id, group in groups.items():
        rep = representatives[group["rep_idx"]]
        variants = group["variants"]
        rep["is_creative_representative"] = 1
        rep["creative_variant_of"] = ""
        rep["creative_variant_count"] = len(variants)
        rep["representative_reason"] = (
            f"square frame preferred (aspect ratio {float(rep.get('aspect_ratio') or 0):.2f})"
            if variants else "unique creative (no variants found)"
        )
        for variant, reason in variants:
            variant["is_creative_representative"] = 0
            variant["creative_variant_of"] = rep["image_file"]
            variant["creative_variant_count"] = 0
            variant["representative_reason"] = reason

    return representatives, duplicate_count, groups


def run_broad_search_pipeline(args, progress_callback=None, stop_check=None, db_path=None, save_run_artifacts=True,
                              on_collection_interrupted=None):
    """Acquire broad ads, run the existing image intelligence, deduplicate, and build review output."""
    from datetime import datetime

    def parse_date(value):
        if not value:
            return None
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            raise SystemExit(f"Invalid date '{value}'. Use YYYY-MM-DD.")

    platforms = None
    if args.platforms:
        platforms = [p.strip().upper() for p in args.platforms.split(",") if p.strip()]

    if progress_callback:
        progress_callback(0, 0, stage="collecting")

    records = search_ads_broad(
        query=args.search_query,
        country=args.search_country,
        status=args.search_status,
        search_type=args.search_type,
        max_results=args.search_max_results,
        page_size=args.search_page_size,
        media_type=args.search_media_type,
        publisher_platforms=platforms,
        start_date=parse_date(args.search_start_date),
        end_date=parse_date(args.search_end_date),
        static_only=True,
        sort_by=getattr(args, "search_sort_by", None),
        on_collect=(lambda n: progress_callback(n, 0, stage="collecting")) if progress_callback else None,
        stop_check=stop_check,
        on_interrupted=on_collection_interrupted,
    )

    if stop_check is not None and stop_check():
        print("\n[STOPPED] Stop requested during collection - skipping analysis.")
        if progress_callback:
            progress_callback(0, len(records), stage="stopped")
        return

    if not records:
        print("\nNo image creatives were returned. Nothing to analyze.")
        return

    print("\n" + "=" * 70)
    print("DESIGN INTELLIGENCE — BROAD SEARCH")
    print("=" * 70)
    print(f"Static image creatives to analyze: {len(records)}")
    print("Video creatives/previews are excluded from this design-led static pipeline.")
    print("Images are fetched in memory only; no creative images are saved during analysis.")
    print("=" * 70)

    load_models(Path(args.model).resolve())

    if progress_callback:
        progress_callback(0, len(records), stage="analyzing")

    # Opened early so every batch's results can be written to the live
    # database as soon as they're ready - this is what makes results
    # appear gradually while it's still running, instead of everything
    # only showing up once the whole run finishes.
    database_path = Path(db_path) if db_path is not None else MASTER_DB_PATH
    live_conn = create_database(database_path)

    analyzed_rows = []
    analyzed_count = 0
    excluded_count = 0
    candidate_count = 0
    sift_ok_count = 0
    stopped_early = False

    for batch_start in range(0, len(records), CLIP_BATCH_SIZE):
        batch = records[batch_start:batch_start + CLIP_BATCH_SIZE]
        urls = [clean_text(r.get("image_url", "")) for r in batch]
        print(f"  [ANALYZE {batch_start + 1}-{batch_start + len(batch)}/{len(records)}] broad search (batched, in memory only)")
        analyses = analyze_remote_creatives_batch(urls)

        batch_rows = []

        for local_index, (record, analysis) in enumerate(zip(batch, analyses)):
            if analysis is None:
                print(f"    [ANALYSIS FAILED] {record.get('page_name', '')} / Ad {record.get('ad_id', '')} / Creative {record.get('creative_index', 0)}")
                continue

            scores, image_hash, frame_hash, sift_signature = analysis
            analyzed_count += 1

            # Hard exclusion: a FACT (Meta says this is a video ad and gave
            # us no real static image), not a stylistic judgment. Everything
            # else, however "photo-like" it scores, becomes a ranked
            # candidate instead of a silent rejection.
            if record.get("is_probable_video_frame"):
                scores["filter_status"] = "EXCLUDED"
                scores["filter_reason"] = "video ad with no real static image (thumbnail fallback)"
                scores["rank_score"] = ""
                excluded_count += 1
            else:
                scores["filter_status"] = "CANDIDATE"
                scores["filter_reason"] = "(legacy heuristic note, NOT what set the rank score) " + classify_broad_static(scores)[1]
                scores["rank_score"] = rank_score(scores)
                candidate_count += 1

            if sift_signature is not None:
                sift_ok_count += 1

            image_file = (
                f"{safe_name(record.get('page_name', 'Unknown'))}_"
                f"{safe_name(record.get('ad_id', 'ad'))}_"
                f"{record.get('creative_index', local_index)}_{batch_start + local_index}.jpg"
            )
            analyzed_rows.append({
                "category": args.search_query or "Uncategorized",
                "brand": record.get("page_name", ""),
                "page_name": record.get("page_name", ""),
                "page_id": record.get("page_id", ""),
                "ad_id": record.get("ad_id", ""),
                "creative_index": record.get("creative_index", 0),
                "image_file": image_file,
                "image_url": record.get("image_url", ""),
                "body": record.get("body", ""),
                "title": record.get("title", ""),
                "cta": record.get("cta", ""),
                "image_hash": image_hash,
                "duplicate": 0,
                "frame_variant_hash": frame_hash,
                "ocr_fingerprint": scores.pop("ocr_fingerprint", ""),
                "_sift_descriptors": sift_signature,
                "creative_group_id": None,
                "is_creative_representative": 1,
                "creative_variant_of": "",
                "creative_variant_count": 0,
                "representative_reason": "",
                "search_query": args.search_query or "",
                "country": (args.search_country or "").upper(),
                "platform": (args.platforms or "ALL").upper(),
                "media_type": "IMAGE",
                **scores,
            })
            batch_rows.append(analyzed_rows[-1])

        # Write this batch's results to the live database right now, so the
        # dashboard can show them immediately instead of waiting for the
        # entire run to finish. Cross-batch duplicate cleanup (SIFT-based
        # frame-variant grouping) still happens once at the end, below.
        for row in batch_rows:
            insert_ad(live_conn, row)
        live_conn.commit()

        if progress_callback:
            progress_callback(min(batch_start + len(batch), len(records)), len(records), stage="analyzing")

        if stop_check is not None and stop_check():
            print(f"\n[STOPPED] User requested stop after {analyzed_count}/{len(records)} analyzed.")
            stopped_early = True
            break

    print("\n" + "=" * 70)
    print("INTELLIGENCE RESULTS")
    print("=" * 70)
    print(f"Analyzed:            {analyzed_count}")
    print(f"Excluded (factual):  {excluded_count}")
    print(f"Ranked candidates:   {candidate_count}")
    print(f"(sorted by rank_score in review.html — nothing is silently rejected)")
    print(f"SIFT signatures:     {sift_ok_count}/{analyzed_count}")

    representatives, duplicate_count, groups = deduplicate_analyzed_rows(analyzed_rows)
    unique_count = len(analyzed_rows) - duplicate_count

    print(f"Unique creatives:    {unique_count}")
    print(f"Duplicate variants: {duplicate_count}")
    print(f"Creative groups:    {len(representatives)}")

    # Save structured output for this broad-search run. Only metadata/URLs are
    # persisted; image bytes remain remote until the user explicitly downloads.
    stamp = time.strftime("%Y%m%d_%H%M%S")
    query_slug = safe_name(args.search_query or "all_ads")[:60]
    broad_dir = OUTPUT_DIR / "BROAD_SEARCH" / f"{query_slug}_{stamp}"
    if save_run_artifacts:
        broad_dir.mkdir(parents=True, exist_ok=True)

    # Every run writes into the SAME accumulating database (MASTER_DB_PATH),
    # not a fresh one per run. insert_ad() already skips true duplicates (by
    # image_hash) and just updates last_seen_at instead, so re-running the
    # same search/category repeatedly grows the pool with genuinely new
    # creatives instead of re-adding what's already there.
    # Rows were already written to master.db live, batch by batch, above.
    # This corrects their dedup flags now that the FULL cross-batch grouping
    # is known.
    for row in analyzed_rows:
        live_conn.execute(
            "UPDATE ads SET is_creative_representative=?, duplicate=?, creative_group_id=?, "
            "creative_variant_of=?, creative_variant_count=?, representative_reason=? WHERE image_hash=?",
            (row.get("is_creative_representative", 1), row.get("duplicate", 0), row.get("creative_group_id"),
             row.get("creative_variant_of", ""), row.get("creative_variant_count", 0),
             row.get("representative_reason", ""), row.get("image_hash", "")),
        )
    live_conn.commit()
    live_conn.close()
    print(f"\n[CREATIVES DB] Updated: {database_path.resolve()}")

    if save_run_artifacts:
        with (broad_dir / "ads.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
            for row in analyzed_rows:
                writer.writerow([row.get(c, "") for c in CSV_COLUMNS])
        run_stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        review_path = broad_dir / "review.html"
        make_review_html(
        [r for r in analyzed_rows if r.get("duplicate", 0) == 0],
        review_path,
        f"Broad Search — {args.search_query or 'All Ads'}",
        run_stamp=run_stamp,
    )

    if save_run_artifacts:
        results_path = broad_dir / "results.html"
        results_path, shown_n, hidden_n = make_results_html(
            [r for r in analyzed_rows if r.get("duplicate", 0) == 0],
            results_path,
            f"Design-led Creatives — {args.search_query or 'All Ads'}",
        )
        print("\n" + "=" * 70)
        print("BROAD SEARCH PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Database: {database_path.resolve()}")
        print(f"CSV:      {(broad_dir / 'ads.csv').resolve()}")
        print(f"Results:  {results_path.resolve()}  ({shown_n} design-led shown, {hidden_n} lower-confidence hidden)")
        print(f"Labeling tool (optional, for retraining later): {review_path.resolve()}")
        print("Images:   NOT saved locally (Meta URLs retained)")
        print("=" * 70)
        if not getattr(args, "skip_browser_open", False):
            webbrowser.open(results_path.resolve().as_uri())
    else:
        print(f"[SESSION DB] Temporary search results stored in {database_path.resolve()}; no CSV/HTML run artifacts saved.")

    if progress_callback:
        progress_callback(analyzed_count, len(records), stage="stopped" if stopped_early else "done")


# Backward-compatible acquisition-only test.
def run_broad_search_test(args):
    records = search_ads_broad(
        query=args.search_query,
        country=args.search_country,
        status=args.search_status,
        search_type=args.search_type,
        max_results=args.search_max_results,
        page_size=args.search_page_size,
        media_type=args.search_media_type,
        publisher_platforms=[p.strip().upper() for p in args.platforms.split(",") if p.strip()] if args.platforms else None,
    )
    print("\nFIRST 10 NORMALIZED CREATIVES")
    print("=" * 70)
    for i, record in enumerate(records[:10], 1):
        print(f"\n[{i}] {record['page_name']} | Ad {record['ad_id']} | Creative {record['creative_index']}")
        print(f"    Image: {record['image_url'][:180]}")
        print(f"    Title: {record['title']}")
        print(f"    Start: {record['delivery_start_time']}")
        print(f"    Platforms: {record['publisher_platforms']}")
    print("\nACQUISITION TEST COMPLETE")


# ============================================================
# CSV / RECORDS
# ============================================================

CSV_COLUMNS = [
    "category", "brand", "page_name", "page_id", "ad_id", "creative_index",
    "image_file", "image_url", "body", "title", "cta", "image_hash", "duplicate",
    "design_score", "clip_design_score", "clip_product_score", "clip_embedding", "broad_graphic_prob", "broad_photo_prob", "broad_video_prob", "broad_product_prob", "ocr_word_count", "ocr_text_density",
    "ocr_score", "visual_complexity", "background_uniformity", "creative_score", "filter_status", "filter_reason",
    "image_width", "image_height", "aspect_ratio", "frame_variant_hash", "ocr_fingerprint",
    "creative_group_id", "is_creative_representative", "creative_variant_of",
    "creative_variant_count", "representative_reason", "rank_score", "human_label",
    "search_query", "country", "platform", "media_type", "first_seen_at", "last_seen_at",
]


RESULTS_SCORE_THRESHOLD = 0.5  # Real precision/recall check on 222 labeled examples:
                                # 0.5 -> 78% precision, 89% recall. Raise toward 0.6-0.7
                                # for a stricter/cleaner (but smaller) results page.


def make_results_html(records, output_path, title="Design-led Creatives", threshold=RESULTS_SCORE_THRESHOLD):
    """The actual end-product page: only design-led creatives, sorted by
    confidence, no Save/Reject, no labeling UI at all. This is what a
    marketer using the finished tool should see - review.html (with the
    Save/Reject buttons) is a separate, optional tool only YOU use if you
    ever want to label more data and retrain later.
    """
    def score_of(r):
        try:
            return float(r.get("rank_score", ""))
        except (TypeError, ValueError):
            return -1.0

    shown = sorted([r for r in records if score_of(r) >= threshold], key=score_of, reverse=True)
    hidden_count = len(records) - len(shown)

    cards = []
    for r in shown:
        cards.append(f'''<div class="card">
          <img src="{html.escape(str(r.get('image_url','')))}" loading="lazy">
          <div class="meta"><b>{html.escape(str(r.get('brand','')))}</b><br>
          {html.escape(str(r.get('category','')))}</div></div>''')

    text = f'''<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:Arial,sans-serif;margin:20px;background:#f4f4f4}}h1{{margin-bottom:4px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px;margin-top:18px}}
.card{{background:white;border-radius:8px;padding:10px;border:1px solid #ddd}}
.card img{{width:100%;height:320px;object-fit:contain;background:#eee}}
.meta{{font-size:13px;line-height:1.4;margin-top:8px}}
.sub{{color:#777;font-size:13px;margin-top:4px}}</style></head>
<body>
<h1>{html.escape(title)}</h1>
<p class="sub">{len(shown)} design-led creative(s) shown{f' &middot; {hidden_count} lower-confidence result(s) not shown' if hidden_count else ''}.</p>
<div class="grid">{''.join(cards)}</div>
</body></html>'''
    output_path.write_text(text, encoding="utf-8")
    return output_path, len(shown), hidden_count


def make_review_html(records, output_path, title="Design-led Static Ad Review", run_stamp=None):
    run_stamp = run_stamp or time.strftime("%Y-%m-%d %H:%M:%S")

    def sort_key(r):
        # Excluded (factual) rows have no rank_score - push them to the very
        # bottom rather than mixing them into the ranked candidate list.
        rs = r.get("rank_score", "")
        try:
            return (1, float(rs))
        except (TypeError, ValueError):
            return (0, 0.0)
    records = sorted(records, key=sort_key, reverse=True)

    payload = [{k:v for k,v in r.items() if not k.startswith("_")} for r in records]
    json_data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    cards=[]
    for i,r in enumerate(records):
        status = r.get("filter_status", "")
        rank = r.get("rank_score", "")
        rank_display = f"{float(rank):.3f}" if rank not in (None, "") else "—"
        clip_score = float(r.get("clip_design_score", 0) or 0)
        words = int(float(r.get("ocr_word_count", 0) or 0))
        density = float(r.get("ocr_text_density", 0) or 0)
        variant_count = int(float(r.get("creative_variant_count", 0) or 0))
        variant_note = (
            f"<br><b>{variant_count} frame variant(s) hidden</b> (square/near-square kept)"
            if variant_count > 0 else ""
        )
        cards.append(f'''<div class="card {status.lower()}" data-index="{i}">
          <div class="labelrow">
            <button class="lbl-btn save" data-index="{i}" onclick="setLabel({i},'SAVE')">Save</button>
            <button class="lbl-btn reject" data-index="{i}" onclick="setLabel({i},'REJECT')">Reject</button>
            <span class="lbl-state" id="lbl-{i}">unreviewed</span>
          </div>
          <div class="rankbadge">rank {rank_display}{' &middot; ' + html.escape(status) if status == 'EXCLUDED' else ''}</div>
          <img src="{html.escape(str(r.get('image_url','')))}" loading="lazy">
          <div class="meta"><b>{html.escape(str(r.get('brand','')))}</b><br>
          Category: {html.escape(str(r.get('category','')))}<br>
          CLIP design: {clip_score:.2f}<br>
          CLIP product-photo: {float(r.get('clip_product_score', 0) or 0):.2f}<br>
          Broad visual: graphic {float(r.get('broad_graphic_prob', 0) or 0):.2f} / photo {float(r.get('broad_photo_prob', 0) or 0):.2f} / video {float(r.get('broad_video_prob', 0) or 0):.2f} / product {float(r.get('broad_product_prob', 0) or 0):.2f}<br>
          OCR: {words} words / {density:.1%} area<br>
          Frame: {float(r.get('aspect_ratio', 0) or 0):.2f}:1<br>
          Background uniformity: {float(r.get('background_uniformity', 0) or 0):.2f} (high = plain/packshot-like background)<br>
          Note: {html.escape(str(r.get('filter_reason','')))}{variant_note}</div></div>''')
    text=f'''<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:Arial,sans-serif;margin:20px;background:#f4f4f4}}h1{{margin-bottom:6px}}.stamp{{background:#222;color:#0f0;font-family:monospace;font-size:15px;padding:10px 14px;border-radius:6px;display:inline-block;margin-bottom:10px}}.controls{{position:sticky;top:0;background:white;padding:12px;z-index:10;border-bottom:1px solid #ccc}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px;margin-top:18px}}.card{{background:white;border-radius:8px;padding:10px;border:3px solid #ccc}}.card.excluded{{opacity:.45}}.card.labeled-save{{border-color:#3a8}}.card.labeled-reject{{border-color:#c55;opacity:.6}}.card img{{width:100%;height:320px;object-fit:contain;background:#eee;margin-top:8px}}.meta{{font-size:13px;line-height:1.45;margin-top:8px}}.rankbadge{{font-family:monospace;font-size:12px;color:#555;margin-top:6px}}.labelrow{{display:flex;align-items:center;gap:8px}}.lbl-btn{{padding:6px 12px;cursor:pointer;border:1px solid #ccc;border-radius:5px;background:#fff}}.lbl-btn.save.active{{background:#3a8;color:#fff}}.lbl-btn.reject.active{{background:#c55;color:#fff}}.overlay{{position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:100;display:none;align-items:center;justify-content:center;flex-direction:column}}.overlay.active{{display:flex}}.overlay img{{max-width:70vw;max-height:60vh;object-fit:contain;background:#111}}.overlay .info{{color:#eee;font-size:14px;max-width:70vw;margin:14px 0;text-align:center;line-height:1.5}}.overlay .keys{{display:flex;gap:16px;margin-top:6px}}.overlay .keys button{{font-size:16px;padding:12px 22px}}.overlay .progress{{color:#999;font-family:monospace;margin-bottom:10px}}</style></head>
<body><h1>{html.escape(title)}</h1>
<div class="stamp">GENERATED AT: {run_stamp} &nbsp; | &nbsp; Check this EXACT line matches the "[RUN TIMESTAMP]" printed in your terminal for this run. If it does not match, you are looking at an old page &mdash; close this tab and reopen the file.</div>
<p><b>Ranked candidates, not a hard filter.</b> Cards are sorted by rank_score (highest first) &mdash; nothing is silently rejected except creatives Meta itself marked as video ads with no real static image (shown faded at the bottom, labeled EXCLUDED). Click Save or Reject on any card as you review, or use <b>Focus Review</b> below to go fast with keyboard shortcuts (S = save, R = reject, &rarr; = skip). Your decisions export as real labeled data, which is how the ranking gets replaced with a properly trained model later.</p>
<div class="controls"><button onclick="startFocus()">&#9654; Focus Review (keyboard: S / R / &rarr;)</button><button onclick="downloadLabeled()">Download Labeled CSV (Save/Reject decisions)</button><button onclick="resetLabels()">Clear my decisions</button><span id="count">0 reviewed</span></div><div class="grid">{''.join(cards)}</div>
<div class="overlay" id="focusOverlay">
  <div class="progress" id="focusProgress"></div>
  <img id="focusImg" src="">
  <div class="info" id="focusInfo"></div>
  <div class="keys">
    <button style="background:#3a8;color:#fff" onclick="focusDecide('SAVE')">Save (S)</button>
    <button style="background:#c55;color:#fff" onclick="focusDecide('REJECT')">Reject (R)</button>
    <button onclick="focusSkip()">Skip (&rarr;)</button>
    <button onclick="endFocus()">Done for now (Esc)</button>
  </div>
</div>
<script>
const records={json_data};
const labels={{}};
function updateCount(){{document.getElementById('count').textContent=Object.keys(labels).length+' reviewed';}}
function applyLabelToCard(i,val){{
  const card=document.querySelector(`.card[data-index="${{i}}"]`);
  if(!card) return;
  card.classList.remove('labeled-save','labeled-reject');
  card.classList.add(val==='SAVE'?'labeled-save':'labeled-reject');
  document.querySelectorAll(`.lbl-btn[data-index="${{i}}"]`).forEach(b=>b.classList.remove('active'));
  const btn=card.querySelector(`.lbl-btn.${{val==='SAVE'?'save':'reject'}}`);
  if(btn) btn.classList.add('active');
  const state=document.getElementById('lbl-'+i);
  if(state) state.textContent=val;
}}
function setLabel(i,val){{
  labels[i]=val;
  applyLabelToCard(i,val);
  updateCount();
}}
function resetLabels(){{
  Object.keys(labels).forEach(k=>delete labels[k]);
  document.querySelectorAll('.card').forEach(c=>c.classList.remove('labeled-save','labeled-reject'));
  document.querySelectorAll('.lbl-btn').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.lbl-state').forEach(s=>s.textContent='unreviewed');
  updateCount();
}}
function downloadLabeled(){{
  const indices=Object.keys(labels);
  if(!indices.length){{alert('Click Save or Reject on at least one card first.');return;}}
  const selected=indices.map(i=>({{...records[Number(i)], human_label: labels[i]}}));
  const cols=Object.keys(selected[0]);
  const esc=v=>'"'+String(v??'').replace(/"/g,'""')+'"';
  const lines=[cols.map(esc).join(','),...selected.map(r=>cols.map(c=>esc(r[c])).join(','))];
  const blob=new Blob([lines.join('\\n')],{{type:'text/csv;charset=utf-8;'}});
  const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='labeled_training_data.csv';a.click();URL.revokeObjectURL(url);
}}

// --- Focus Review: one card at a time, keyboard-driven, for fast labeling ---
let focusQueue=[];
let focusPos=0;
function startFocus(){{
  // Only queue up candidates that aren't excluded and haven't been labeled yet.
  focusQueue=records.map((r,i)=>i).filter(i=>records[i].filter_status!=='EXCLUDED' && !(i in labels));
  focusPos=0;
  if(!focusQueue.length){{alert('Nothing left to review - everything is either excluded or already labeled.');return;}}
  document.getElementById('focusOverlay').classList.add('active');
  showFocusCard();
}}
function endFocus(){{
  document.getElementById('focusOverlay').classList.remove('active');
}}
function showFocusCard(){{
  if(focusPos>=focusQueue.length){{alert('Reviewed everything in the queue.');endFocus();return;}}
  const i=focusQueue[focusPos];
  const r=records[i];
  document.getElementById('focusImg').src=r.image_url||'';
  document.getElementById('focusProgress').textContent=`${{focusPos+1}} / ${{focusQueue.length}}  (rank_score ${{r.rank_score}})`;
  document.getElementById('focusInfo').innerHTML=`<b>${{r.brand||''}}</b> &middot; ${{r.category||''}}<br>CLIP design ${{Number(r.clip_design_score||0).toFixed(2)}} &middot; OCR ${{r.ocr_word_count||0}} words / ${{(Number(r.ocr_text_density||0)*100).toFixed(1)}}% area`;
}}
function focusDecide(val){{
  if(focusPos>=focusQueue.length) return;
  const i=focusQueue[focusPos];
  setLabel(i,val);
  focusPos+=1;
  showFocusCard();
}}
function focusSkip(){{
  focusPos+=1;
  showFocusCard();
}}
document.addEventListener('keydown',(e)=>{{
  if(!document.getElementById('focusOverlay').classList.contains('active')) return;
  if(e.key==='s'||e.key==='S') focusDecide('SAVE');
  else if(e.key==='r'||e.key==='R') focusDecide('REJECT');
  else if(e.key==='ArrowRight') focusSkip();
  else if(e.key==='Escape') endFocus();
}});

updateCount();
</script></body></html>'''
    output_path.write_text(text, encoding="utf-8")
    return output_path


def main():
    parser=argparse.ArgumentParser(description="Meta ad scraper + strict design-led static review workflow")
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--broad-search", action="store_true", help="Broad Meta acquisition + existing design intelligence + dedup + review")
    mode.add_argument("--broad-acquisition-test", action="store_true", help="Acquisition-only broad Meta test")
    mode.add_argument("--retrain", action="store_true", help="Retrain rank_model_v1.pkl from labeled CSVs - no external help needed")
    mode.add_argument("--browse", action="store_true", help="Show accumulated results from the master database (all past runs), not just one run")
    parser.add_argument("--browse-category", default=None, help="Filter browse results to a category (matches the search query used when it was collected)")
    parser.add_argument("--training-data-dir", default=str(BASE_DIR / "training_data"),
                         help="Folder containing your downloaded labeled_training_data*.csv files")
    parser.add_argument("--search-query", default="", help="Broad search keyword/phrase")
    parser.add_argument("--search-country", default=COUNTRY, help="Broad search country code")
    parser.add_argument("--search-status", default="ACTIVE", choices=["ACTIVE", "INACTIVE", "ALL"])
    parser.add_argument("--search-type", default="KEYWORD_UNORDERED")
    parser.add_argument("--search-max-results", type=int, default=500)
    parser.add_argument("--search-sort-by", default=None, choices=[None, "SORT_BY_TOTAL_IMPRESSIONS"],
                         help="Default (None) = relevancy, which surfaces a wider variety of advertisers. "
                              "SORT_BY_TOTAL_IMPRESSIONS always favors the same big-spend campaigns.")
    parser.add_argument("--search-page-size", type=int, default=10)
    parser.add_argument("--search-media-type", default=None, choices=["IMAGE", "VIDEO", "ALL"])
    parser.add_argument("--platforms", default=None, help="Comma-separated platforms, e.g. INSTAGRAM,FACEBOOK")
    parser.add_argument("--search-start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--search-end-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--model", default=str(ML_MODEL_PATH))
    args=parser.parse_args()

    # Meta's API wants real ISO country codes, not everyday abbreviations -
    # "UK" fails with a cryptic GraphQL error because the real code is "GB".
    # Translate the common mismatches so this doesn't bite again.
    COUNTRY_ALIASES = {
        "UK": "GB", "U.K.": "GB", "BRITAIN": "GB", "GREAT BRITAIN": "GB",
        "USA": "US", "U.S.": "US", "U.S.A.": "US", "AMERICA": "US",
        "UAE": "AE", "U.A.E.": "AE",
        "INDIA": "IN",
    }
    if args.search_country:
        normalized = COUNTRY_ALIASES.get(args.search_country.strip().upper())
        if normalized:
            print(f"[COUNTRY] '{args.search_country}' -> '{normalized}' (Meta's API needs the real ISO code)")
            args.search_country = normalized

    if args.retrain:
        retrain_rank_model(args.training_data_dir); return
    if args.browse:
        records = browse_master_db(args.browse_category)
        if not records:
            print(f"[BROWSE] No accumulated results found" + (f" for category matching '{args.browse_category}'" if args.browse_category else "") + ".")
            print(f"          Run --broad-search first to start building up the accumulated database at {MASTER_DB_PATH.resolve()}")
            return
        out_dir = OUTPUT_DIR / "BROWSE"
        out_dir.mkdir(parents=True, exist_ok=True)
        label = args.browse_category or "All Categories"
        path, shown_n, hidden_n = make_results_html(records, out_dir / "results.html", f"Design-led Creatives — {label}")
        print(f"[BROWSE] {len(records)} accumulated creative(s) found for '{label}'")
        print(f"         {shown_n} shown as design-led, {hidden_n} below the confidence threshold")
        print(f"         {path.resolve()}")
        webbrowser.open(path.resolve().as_uri())
        return
    if args.broad_search:
        run_broad_search_pipeline(args); return
    if args.broad_acquisition_test:
        run_broad_search_test(args); return
    parser.error("Use --broad-search, --broad-acquisition-test, --browse or --retrain")

if __name__ == "__main__": main()
