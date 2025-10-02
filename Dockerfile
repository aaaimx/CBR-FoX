# Dockerfile for cbr_fox - SLIM VERSION
# Optimized for minimal size using multi-stage build and alpine

# Stage 1: Build stage (if needed)
FROM python:3.11-slim as builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt .

# Install dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install with no cache and use binary wheels only
RUN pip install --no-cache-dir \
    --only-binary :all: \
    -r requirements.txt || \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage (minimal)
FROM python:3.11-slim

# Set metadata
LABEL maintainer="your-email@example.com"
LABEL description="cbr_fox: Case-Based Reasoning for Time Series Forecasting"
LABEL version="1.0.1"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy only necessary files (exclude large unnecessary files)
COPY cbr_fox/ ./cbr_fox/
COPY requirements.txt pyproject.toml README.md ./

# Install package in development mode
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /app/data /app/figures /app/outputs

# Clean up unnecessary files
RUN find /opt/venv -name "*.pyc" -delete && \
    find /opt/venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf /root/.cache

# Set the default command with fallback
CMD ["sh", "-c", "if [ -d 'tests' ]; then python -m pytest tests/ -v; else echo 'CBR-FoX ready. Use: docker exec -it <container> bash'; fi"]

# Usage examples:
# docker build -t cbr_fox:slim .
# docker run cbr_fox:slim python scripts/examples/reproduce_all_figures.py
# docker run -p 8888:8888 cbr_fox:slim jupyter notebook --ip=0.0.0.0 --no-browser --allow-root