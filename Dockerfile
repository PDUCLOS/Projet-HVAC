# =============================================================================
# HVAC Market Analysis — Multi-stage Dockerfile
# =============================================================================
# Build : docker build -t hvac-market .
# Run   : docker run -p 8501:8501 -p 8000:8000 hvac-market
# Dev   : docker compose up
# =============================================================================

# --- Stage 1 : Base Python ---
FROM python:3.11-slim AS base

# Metadata
LABEL maintainer="Patrice DUCLOS"
LABEL description="HVAC Market Analysis - Pipeline ML + API + Dashboard"
LABEL version="2.0"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Stage 2 : Python Dependencies ---
FROM base AS dependencies

# Copy requirements first (Docker cache)
COPY requirements.txt requirements-api.txt ./

# Install main + API dependencies
RUN pip install -r requirements.txt -r requirements-api.txt

# --- Stage 3 : Application ---
FROM dependencies AS app

# Copy source code
COPY config/ ./config/
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY setup_project.py .
COPY .env.example .

# Create non-root user for security
RUN useradd -m -u 1000 -s /sbin/nologin hvac && \
    chown -R hvac:hvac /app

# Create data directories
RUN mkdir -p data/raw/weather data/raw/insee data/raw/eurostat data/raw/sitadel data/raw/dpe \
    data/processed/weather data/processed/dpe data/processed/insee data/processed/eurostat \
    data/features data/models/figures data/analysis/figures \
    && chown -R hvac:hvac data/

# Copy .env.example as default .env
RUN cp .env.example .env

# Expose ports : Streamlit (8501) + FastAPI (8000)
EXPOSE 8501 8000

# Startup script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Run as non-root user
USER hvac

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["all"]
