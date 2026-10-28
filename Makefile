PYTHON ?= .venv/bin/python
UV ?= uv
DOCKER ?= docker
COMPOSE ?= docker compose
IMAGE ?= mlops-diabetes:local

.PHONY: setup pipeline test api mlflow-server clean \
	docker-build docker-up docker-pipeline docker-down docker-reset \
	docker-logs docker-ps

setup:
	@if [ ! -d .venv ]; then $(UV) venv --python 3.13 --seed .venv; fi
	$(UV) pip install --python $(PYTHON) -r requirements.txt -r requirements-dev.txt

pipeline:
	$(PYTHON) -m mlops.pipeline

test:
	$(PYTHON) -m pytest

api:
	$(PYTHON) -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

mlflow-server:
	$(PYTHON) -m mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns \
		--host 127.0.0.1 \
		--port 5000

docker-build:
	$(DOCKER) build -t $(IMAGE) .

docker-up:
	$(COMPOSE) up -d --build

docker-pipeline:
	$(COMPOSE) run --rm pipeline

docker-down:
	$(COMPOSE) down

docker-reset:
	$(COMPOSE) down -v

docker-logs:
	$(COMPOSE) logs -f api mlflow

docker-ps:
	$(COMPOSE) ps

clean:
	rm -rf mlruns mlflow.db mlflow.db-shm mlflow.db-wal .pytest_cache
	rm -f reports/*.json data/processed/*.csv
