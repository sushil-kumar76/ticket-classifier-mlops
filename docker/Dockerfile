FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    MODEL_DIR=/app/models/final

COPY docker/requirements-serve.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 10 -r requirements-serve.txt

COPY src/ ./src/
COPY models/final/ ./models/final/

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
