# Python 3.12-slim: the requirements.txt lock was resolved and tested on 3.12.
# (PROJECT.md suggested 3.11-slim; 3.12 satisfies the ">=3.11" requirement and
# matches the pinned wheels, so we use it for reproducibility.)
FROM python:3.12-slim

# Don't write .pyc, don't buffer stdout/stderr (logs appear immediately).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY src ./src
COPY scripts ./scripts
COPY tests/eval_dataset.json ./tests/eval_dataset.json

# Chroma data is mounted as a volume at runtime (see docker-compose.yml).
RUN mkdir -p /app/data/chroma /app/data/raw

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "src.main"]
