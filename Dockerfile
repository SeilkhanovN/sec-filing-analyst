FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/data/hf_cache

WORKDIR /app

# CPU-only PyTorch keeps the image small (no CUDA libraries).
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

# Run as a non-root user.
RUN useradd --create-home appuser && mkdir -p /app/data /app/chroma_db && chown -R appuser /app
USER appuser

CMD ["python", "-m", "sec_analyst.cli", "--help"]
