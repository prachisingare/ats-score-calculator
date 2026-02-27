# ATS Score Calculator API - Optimized Dockerfile (< 500MB without embeddings)

# Build argument to control embeddings (default: false for smaller size)
ARG USE_EMBEDDINGS=false

# ====================
# Builder Stage
# ====================
FROM python:3.11-slim as builder

ARG USE_EMBEDDINGS

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY requirements.txt .

# Install base dependencies (without torch/sentence-transformers)
RUN pip install --upgrade pip && \
    pip install \
    fastapi==0.109.2 \
    uvicorn[standard]==0.27.1 \
    pydantic==2.6.1 \
    pydantic-settings==2.1.0 \
    python-multipart==0.0.9 \
    nltk==3.8.1 \
    scikit-learn==1.4.0 \
    pytest==8.0.0 \
    httpx==0.26.0 \
    PyPDF2==3.0.1 \
    python-docx==1.1.0 \
    pdfplumber==0.10.4

# Install embeddings only if requested (adds ~2GB)
RUN if [ "$USE_EMBEDDINGS" = "true" ]; then \
    pip install --no-cache-dir \
    torch==2.2.0 \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    sentence-transformers==2.5.1 \
    transformers==4.38.2 \
    tokenizers==0.15.2 \
    huggingface_hub==0.20.3 \
    safetensors==0.4.2; \
    fi

# Clean up pip cache
RUN pip cache purge

# ====================
# Production Stage
# ====================
FROM python:3.11-slim

ARG USE_EMBEDDINGS

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    ATS_USE_EMBEDDINGS=${USE_EMBEDDINGS}

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy only necessary application files
COPY --chown=app:app app/ ./app/
COPY --chown=app:app requirements.txt .
COPY --chown=app:app .env.example .

# Switch to non-root user
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with single worker (adjust based on your needs)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
