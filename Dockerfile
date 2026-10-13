# The documents folder. By default it is ./data; to use another folder, pass it as a
# parameter: DOCS_DIR=<folder> with docker compose, or
#   docker build --build-context docs=<folder> .
FROM scratch AS docs
COPY data/ /

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ANONYMIZED_TELEMETRY=False \
    DATA_DIR=/data \
    DB_DIR=/chroma

WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

# Run as a normal user. /chroma is created here so that the volume mounted on it is writable.
RUN useradd --create-home app && mkdir /data /chroma && chown app:app /chroma
COPY app ./app

# The documents are part of the image. Rebuild it when they change.
COPY --from=docs / /data/

USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
