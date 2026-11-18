FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/data/hf_cache \
    API_HOST=0.0.0.0 \
    API_PORT=8000

WORKDIR /app

# CPU-only PyTorch keeps the image small (no CUDA libraries).
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY eval ./eval
COPY scripts ./scripts
RUN pip install --no-deps .

# Run as a non-root user.
RUN useradd --create-home appuser && mkdir -p /app/data /app/chroma_db && chown -R appuser /app
USER appuser

# Inside the container the server must listen on 0.0.0.0 to be reachable through the
# published port; docker-compose publishes it on the host's 127.0.0.1 only.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8)"

# Default: the web app. CLI example: docker compose run --rm cli python -m sec_analyst.cli count
CMD ["python", "-m", "sec_analyst.api"]
