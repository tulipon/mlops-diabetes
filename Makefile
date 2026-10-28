PYTHON ?= .venv/bin/python
UV ?= uv

.PHONY: setup pipeline test api mlflow-server clean

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

clean:
	rm -rf mlruns mlflow.db mlflow.db-shm mlflow.db-wal .pytest_cache
	rm -f reports/*.json data/processed/*.csv
