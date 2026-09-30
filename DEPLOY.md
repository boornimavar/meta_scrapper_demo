# Creative Lens — Render deployment

## What this bundle includes
- Flask dashboard and API
- Linux Tesseract OCR setup
- CPU-only PyTorch + OpenAI CLIP installation
- `meta_ads_collector` package source
- Render Docker configuration and a persistent disk for SQLite/model cache

## Deploy
1. Create a **private GitHub repository** and upload the files in this folder to its root. Do not upload credentials or personal data.
2. Push the repository to GitHub.
3. In Render, choose **New + → Blueprint**, connect the repository, and approve the `render.yaml` configuration.
4. Wait for the Docker build. The first build is large because it installs PyTorch and CLIP.
5. Open the generated Render URL and test the dashboard, a single category search, saving a creative, and reopening Saved after a restart.

## Important
- This app performs live Meta Ad Library collection. Verify that collection is permitted for your intended use and that the deployed host can reach Meta.
- The first search may take longer while CLIP downloads its model weights. The weights are cached on the persistent disk.
- The persistent disk is required to retain `saved_creatives.db`. Do not remove it or change its mount path without migrating the database.
- This initial setup uses one Gunicorn worker because the app tracks active searches in process memory. Multiple workers would make progress/stop state inconsistent.
- The deployment has not been tested against live Meta collection from Render yet.
