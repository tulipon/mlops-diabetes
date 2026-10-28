# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Etapa de construcción: instala las dependencias en un virtualenv aislado.
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Etapa final: solo runtime, usuario no-root y código de la aplicación.
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH" \
    HOME="/home/mlops" \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet

# libgomp1 es necesario para el soporte OpenMP de scikit-learn.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 mlops \
    && useradd --uid 10001 --gid mlops --create-home --home-dir /home/mlops \
        --shell /usr/sbin/nologin mlops

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=mlops:mlops api ./api
COPY --chown=mlops:mlops mlops ./mlops

# Los volúmenes con nombre heredan el propietario del directorio de la imagen.
RUN mkdir -p /app/data/raw /app/data/processed /app/reports \
        /mlflow/data /mlflow/artifacts \
    && chown -R mlops:mlops /app /mlflow

USER mlops

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
