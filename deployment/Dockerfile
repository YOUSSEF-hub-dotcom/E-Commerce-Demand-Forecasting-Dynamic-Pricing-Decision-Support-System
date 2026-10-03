# Application image for both services in docker-compose (api + dashboard).
# Build context = repository root (see docker-compose.yml: context: ..).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/app/.matplotlib

# lightgbm needs the OpenMP runtime (libgomp1) on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Runtime sources.
COPY app ./app
COPY dashboard ./dashboard
COPY src ./src
COPY deployment/healthcheck.py ./deployment/healthcheck.py

# Served data + model registry (read-only). The protected training/test
# splits and raw analytics are excluded via .dockerignore.
COPY data/processed/demand_matrix_weekly.csv.gz ./data/processed/demand_matrix_weekly.csv.gz
COPY artifacts/mlflow_tracking.db ./artifacts/mlflow_tracking.db
COPY mlruns ./mlruns

# Persisted Part-4 pricing state consumed read-only by the dashboard
# (dashboard/product_catalog.py). Only this one artifact is shipped; every
# other report stays excluded via .dockerignore.
COPY reports/forecasting/part4_pricing_engine.json ./reports/forecasting/part4_pricing_engine.json

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/artifacts /app/.matplotlib \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8501

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]