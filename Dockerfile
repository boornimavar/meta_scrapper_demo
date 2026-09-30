FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CREATIVE_LENS_DATA_DIR=/var/data \
    TORCH_HOME=/var/data/torch-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    unzip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install git+https://github.com/openai/CLIP.git

COPY meta_ads_collector_source.zip /tmp/meta_ads_collector_source.zip
RUN mkdir -p /app/meta_ads_collector \
    && unzip -q /tmp/meta_ads_collector_source.zip -d /app/meta_ads_collector \
    && rm /tmp/meta_ads_collector_source.zip

COPY dashboard.html dashboard_server_temporary.py meta_design_scraper_session.py ./
RUN mkdir -p /var/data && python -m py_compile dashboard_server_temporary.py meta_design_scraper_session.py

EXPOSE 10000
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 --timeout 600 dashboard_server_temporary:app"]
