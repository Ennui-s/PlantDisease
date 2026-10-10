# Inference service for the plant disease classifier.
# The image carries its own copy of the champion model (model_export/, created with
# `python -m registry export`), so it needs no registry database, no volume and no secrets.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/model_export

WORKDIR /app

# CPU-only PyTorch keeps the image small; the other dependencies come from PyPI.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Only what serving needs: the model code the packaged model imports, and the service itself.
COPY config.py ./
COPY models ./models
COPY registry ./registry
COPY serving ./serving
COPY model_export ./model_export

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready', timeout=3)"

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
