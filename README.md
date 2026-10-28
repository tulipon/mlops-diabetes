# Predicción de progresión de diabetes — Proyecto MLOps

Sistema MLOps de extremo a extremo para predecir la progresión de la diabetes
un año después de la medición basal. El proyecto parte del notebook exploratorio
`notebooks/Proyecto_MLOps.ipynb` y lo convierte en un repositorio con etapas
separadas de datos, entrenamiento, registro de modelos, serving y CI/CD.

El objetivo no es construir una plataforma de producción completa, sino
demostrar de forma práctica los conceptos clave de MLOps: extracción,
validación y limpieza de datos, tracking de experimentos, Model Registry,
selección de un modelo champion, API de inferencia, manejo de errores,
despliegue con Docker Compose y automatización con GitHub Actions.

---

## 1. Dataset

Se utiliza el **Diabetes Dataset** incluido en `scikit-learn`
(`sklearn.datasets.load_diabetes`).

- **Filas:** 442.
- **Features:** 10 variables numéricas (`age`, `sex`, `bmi`, `bp`, `s1`–`s6`).
- **Target:** `target`, una medida continua de progresión de la enfermedad.
- **Tipo de problema:** regresión.
- **Particularidad importante:** `load_diabetes` devuelve las features ya
  estandarizadas (centradas y escaladas), no en unidades clínicas originales.
  Por eso los ejemplos de la API usan valores como `0.038076`, no una edad en
  años.

El pipeline guarda una copia cruda en `data/raw/diabetes.csv` y una copia limpia
en `data/processed/diabetes_clean.csv`.

---

## 2. Estructura del repositorio

```text
pro-fin/
├── api/
│   ├── __init__.py
│   ├── main.py              # App FastAPI, endpoints y handlers de error
│   ├── model_loader.py      # Carga del champion desde MLflow
│   └── schemas.py           # Esquemas Pydantic de entrada/salida
├── mlops/
│   ├── __init__.py
│   ├── config.py            # Rutas, features, semilla, umbrales y nombres
│   ├── features.py          # Transformaciones compartidas train/serving
│   ├── pipeline.py          # Orquestador end-to-end
│   ├── data/
│   │   ├── extract.py       # scikit-learn -> CSV crudo
│   │   ├── validate.py      # Contrato de datos e informes
│   │   └── clean.py         # Duplicados, nulos, tipos y target
│   └── training/
│       ├── train.py         # CV, candidatos y tracking MLflow
│       └── registry.py      # Registro, quality gate y alias champion
├── notebooks/
│   ├── Proyecto_MLOps.ipynb # Fase exploratoria y mapeo al repositorio
│   └── Colab.ipynb          # Ejecución end-to-end en Google Colab
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_registry.py
│   └── test_api.py
├── data/
│   ├── raw/                 # CSV extraído
│   └── processed/           # CSV limpio (generado)
├── reports/                 # Informes JSON generados
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── requirements-dev.txt
├── Makefile
├── pytest.ini
└── README.md
```

El flujo nativo con `uv` es la vía principal de desarrollo. Docker se incluye
como capa de despliegue opcional (sección 7) y no es necesario para ejecutar
`make setup`, `make pipeline` ni `make api`.

---

## 3. Instalación

Requisitos:

- Python 3.13.
- [`uv`](https://docs.astral.sh/uv/) para crear el entorno.
- Git, opcional para clonar/publicar el repositorio.
- Docker y Docker Compose, opcionales para la capa de despliegue de la
  sección 7.

```bash
# Crear el entorno con Python 3.13 e instalar dependencias
make setup
```

El comando anterior ejecuta, equivalentemente:

```bash
uv venv --python 3.13 --seed .venv
uv pip install --python .venv/bin/python -r requirements.txt -r requirements-dev.txt
```

Activa el entorno si quieres ejecutar comandos manualmente:

```bash
source .venv/bin/activate
```

Versiones principales fijadas en `requirements.txt`: MLflow 3.17.2,
scikit-learn 1.9.1, FastAPI 0.162.0, pandas 3.1.0 y NumPy 2.5.4.

### Google Colab

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tulipon/mlops-diabetes/blob/main/notebooks/Colab.ipynb)

Para ejecutar el proyecto sin Docker, abre `notebooks/Colab.ipynb` y ejecuta
las celdas en orden. El notebook clona el repositorio, instala `uv`, lanza los
tests y el pipeline, arranca la API en segundo plano y la publica con
`google.colab.output.serve_kernel_port_as_iframe`. Incluye celdas opcionales
para la interfaz de MLflow y para guardar `mlflow.db`, `mlruns/` y `reports/`
en Google Drive.

Colab no ofrece un daemon de Docker fiable y su almacenamiento es efímero: sin
montar Drive, el estado se pierde al desconectar el runtime.

---

## 4. Pipeline de datos y entrenamiento

El pipeline completo se ejecuta con:

```bash
make pipeline
```

o directamente:

```bash
.venv/bin/python -m mlops.pipeline
```

El orquestador ejecuta las siguientes etapas:

1. **Extracción:** carga el dataset de scikit-learn y escribe
   `data/raw/diabetes.csv`. Calcula su SHA-256 para trazabilidad.
2. **Validación de datos crudos:** comprueba columnas obligatorias, target,
   tipos numéricos, valores finitos y tamaño mínimo. Los nulos se registran
   como advertencia porque la limpieza es la etapa que los trata. Genera
   `reports/validation_report.json`.
3. **Limpieza:** elimina duplicados, convierte tipos numéricos, elimina filas
   con nulos o target negativo y escribe `data/processed/diabetes_clean.csv`.
   Genera `reports/cleaning_report.json`.
4. **Validación estricta post-limpieza:** vuelve a validar el dataset ya limpio;
   esta vez los nulos sí son un error. Genera
   `reports/validation_report_clean.json`.
5. **Entrenamiento y registro:** entrena los candidatos con validación cruzada,
   registra los runs en MLflow y registra el mejor modelo en el Model Registry.
6. **Quality gate:** decide si el modelo pasa el umbral y, solo entonces, le
   asigna el alias `champion`. Genera `reports/pipeline_report.json`.

Cada etapa también puede ejecutarse por separado:

```bash
.venv/bin/python -m mlops.data.extract
.venv/bin/python -m mlops.data.validate
.venv/bin/python -m mlops.data.clean
.venv/bin/python -c "from mlops.training.train import train_and_log; train_and_log()"
```

### Modelos candidatos

Se comparan cuatro familias:

- `LinearRegression`
- `Ridge` con búsqueda de `alpha`
- `RandomForestRegressor` con búsqueda de `n_estimators` y `max_depth`
- `GradientBoostingRegressor` con búsqueda de `n_estimators` y
  `learning_rate`

Cada familia se prueba con dos conjuntos de features:

- `raw`: las 10 variables originales.
- `engineered`: las 10 originales más `bmi_age_interaction`,
  `bp_bmi_interaction` y `s1_s2_ratio`, las interacciones exploradas en el
  notebook. La división por `s2` está protegida contra valores cercanos a cero.

### Selección y evaluación

- Split **80 % train / 20 % test**, semilla `42`.
- **5-fold cross-validation** únicamente sobre train.
- El ganador se elige por **RMSE medio de validación cruzada**.
- El test se evalúa **una sola vez** al final, fuera de la selección, para
  evitar leakage.
- Métricas registradas: RMSE, MAE y R² (CV y test).

En la ejecución de referencia, el ganador por CV fue `LinearRegression` con
features `raw`:

| Métrica | Valor |
|---|---:|
| CV RMSE | 55.395 |
| Test RMSE | 53.853 |
| Test MAE | 42.794 |
| Test R² | 0.453 |

Las features engineered se evaluaron, pero no mejoraron el RMSE de CV en esta
partición. Es un ejemplo de por qué no conviene decidir únicamente con el test
set del notebook.

---

## 5. MLflow: tracking, registro y champion

### Tracking local

Por defecto el proyecto usa un backend SQLite local:

```text
sqlite:///mlflow.db
```

Los artefactos se guardan en `mlruns/`. Ambos quedan fuera de git porque son
estado generado.

Para abrir la interfaz de MLflow:

```bash
make mlflow-server
```

y visitar <http://127.0.0.1:5000>. El comando usa el mismo `mlflow.db` y
`mlruns/` que el pipeline. También puedes inspeccionar los runs con:

```bash
.venv/bin/python -c "import mlflow; print(mlflow.search_runs(experiment_names=['diabetes-progression'])[['run_id', 'tags.model_family', 'metrics.cv_rmse', 'metrics.test_rmse']])"
```

### Qué se registra

En cada run candidato:

- Parámetros del modelo, feature set, semilla, tamaño de test y folds.
- Métricas de validación cruzada (`cv_rmse`, `cv_mae`, `cv_r2`,
  `cv_rmse_std`).
- El dataset de entrada mediante `mlflow.log_input`, con su SHA-256.
- Artefactos JSON con los resultados de CV.

En el run ganador, además:

- Métricas finales de test.
- Firma del modelo (`infer_signature`) y ejemplo de entrada.
- El `Pipeline` de scikit-learn, que incluye la transformación de features.
- El artefacto `evaluation.json` con la comparación de candidatos.

El Model Registry contiene el modelo `diabetes-progression-model`. Cada
ejecución crea una versión; solo la que pasa el gate recibe el alias
`champion`.

### Quality gate y selección del champion

El umbral es deliberadamente estricto, tal como se acordó:

- **RMSE de test < 54.0**
- **R² de test > 0.40**

Además, si ya existe un champion, el nuevo modelo debe mejorar su RMSE de test;
en caso contrario se conserva el champion anterior. El motivo de la decisión
queda en `reports/pipeline_report.json` y en las etiquetas de la versión.

El baseline exploratorio ronda RMSE 53.8–53.9, por lo que el umbral 54.0 exige
que el candidato final no empeore el resultado de referencia. El gate se aplica
al holdout final, no al CV: el CV se usa para elegir el candidato y el test para
decidir si se promociona. Si el gate falla, el modelo se registra igualmente
como versión, pero **no** recibe el alias `champion`; la API responderá `503`
hasta que exista un champion válido.

Para consultar el champion actual:

```bash
.venv/bin/python - <<'PY'
from mlflow import MlflowClient

client = MlflowClient("sqlite:///mlflow.db")
version = client.get_model_version_by_alias("diabetes-progression-model", "champion")
print(f"Champion: versión {version.version}, run {version.run_id}")
PY
```

---

## 6. API de serving

Arranca la API:

```bash
make api
```

o, sin recarga automática:

```bash
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Documentación interactiva: <http://127.0.0.1:8000/docs>.

La API carga el modelo con URI `models:/diabetes-progression-model@champion`
durante el arranque. Si el registry o el alias no están disponibles, el proceso
sigue vivo en modo degradado y los endpoints que necesitan el modelo responden
`503`.

### GET `/health`

Comprueba que el servicio está vivo. Devuelve siempre `200` si el proceso está
arriba, e indica si el modelo se cargó:

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### GET `/model-info`

Devuelve el modelo servido y sus metadatos:

```bash
curl http://127.0.0.1:8000/model-info
```

```json
{
  "model_name": "diabetes-progression-model",
  "model_version": "1",
  "champion_alias": "champion",
  "run_id": "e1e1e35011494430aa443e823adf29de",
  "feature_set": "raw",
  "model_family": "linear_regression",
  "registered_at": "2026-10-28T15:58:42.118000+00:00",
  "description": "Candidato linear_regression con features raw. Test RMSE=53.853...",
  "metrics": {
    "cv_rmse": 55.3946,
    "test_rmse": 53.8534,
    "test_mae": 42.7941,
    "test_r2": 0.4526
  }
}
```

### POST `/predict`

Recibe una fila con las 10 features estandarizadas:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 0.038076,
    "sex": 0.050680,
    "bmi": 0.061696,
    "bp": 0.021872,
    "s1": -0.044223,
    "s2": -0.034821,
    "s3": -0.043401,
    "s4": -0.002592,
    "s5": 0.019907,
    "s6": -0.017646
  }'
```

```json
{
  "prediction": 210.74098559942396,
  "model_name": "diabetes-progression-model",
  "model_version": "1",
  "champion_alias": "champion"
}
```

El modelo registrado es un `Pipeline`: recibe las 10 features originales y,
si el champion usa features engineered, aplica la transformación internamente.
Así se evita el skew entre entrenamiento y serving.

---

## 7. Despliegue con Docker

Docker es una **capa de despliegue opcional**. El flujo nativo con `uv`
(`make setup`, `make pipeline`, `make api`) sigue siendo la vía principal para
desarrollo y no requiere Docker. La capa Docker empaqueta el mismo código y
reproduce el ciclo completo: entrenar, registrar y servir.

### Arquitectura

```text
localhost:8000                              localhost:5000
      │                                           │
┌─────▼──────┐        models:/...@champion   ┌───▼────────┐
│    api     │ ─────────────────────────────▶│   mlflow   │
│  FastAPI   │                               │  tracking  │
└─────▲──────┘                               │  registry  │
      │                                      └───┬────┬───┘
      │                                          │    │
┌─────┴────────────┐                    mlflow-data  mlflow-artifacts
│ pipeline         │                         (volúmenes)
│ one-shot         │
│ extract→train→reg│
└──────────────────┘
        │
  pipeline-data / pipeline-reports
```

| Servicio | Tipo | Puerto | Función |
|---|---|---:|---|
| `mlflow` | largo | 5000 | Tracking, Model Registry y proxy de artefactos |
| `pipeline` | one-shot | — | Extrae, valida, limpia, entrena y registra |
| `api` | largo | 8000 | Sirve el champion con FastAPI |

Volúmenes con nombre:

- `mlflow-data`: base SQLite de MLflow.
- `mlflow-artifacts`: artefactos de runs y modelos.
- `pipeline-data`: CSV crudo y limpio.
- `pipeline-reports`: informes JSON del pipeline.

El estado sobrevive a reinicios y a `docker compose down`. Solo se elimina con
`make docker-reset` (`docker compose down -v`).

### Requisitos

- Docker Engine 24 o superior y Docker Compose v2.
- El usuario debe pertenecer al grupo `docker`, o ejecutar los comandos con
  `sudo`.
- Puertos 5000 y 8000 libres.

Comprueba la instalación:

```bash
docker info
docker compose version
```

Si acabas de añadir tu usuario al grupo `docker`, cierra la sesión y vuelve a
entrar para que el cambio surta efecto. Mientras tanto, `docker` puede seguir
pidiendo permisos aunque el daemon esté activo.

### Construcción y arranque

```bash
make docker-up
```

Equivale a:

```bash
docker compose up -d --build
```

Orden de arranque:

1. `mlflow` arranca y espera a que su healthcheck responda.
2. `pipeline` ejecuta `python -m mlops.pipeline` contra `http://mlflow:5000`.
3. `api` arranca cuando el pipeline termina correctamente y carga
   `models:/diabetes-progression-model@champion`.

La primera construcción tarda varios minutos porque instala MLflow,
scikit-learn, pandas y el resto de dependencias. Las siguientes reutilizan la
caché de Docker.

Consulta el estado:

```bash
make docker-ps
```

### URLs y verificación

- API: <http://localhost:8000>
- Swagger: <http://localhost:8000/docs>
- MLflow: <http://localhost:5000>

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model-info
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 0.038076,
    "sex": 0.050680,
    "bmi": 0.061696,
    "bp": 0.021872,
    "s1": -0.044223,
    "s2": -0.034821,
    "s3": -0.043401,
    "s4": -0.002592,
    "s5": 0.019907,
    "s6": -0.017646
  }'
```

La respuesta de `/model-info` debe incluir `champion_alias: "champion"` y una
versión concreta. La predicción de referencia es `210.74098559942396`.

El healthcheck del contenedor `api` consulta `/model-info`, no solo `/health`.
Así, un contenedor que arranca sin champion se marca como `unhealthy` y el
despliegue no se considera correcto. Si el quality gate no promociona ningún
modelo, la API arranca igualmente en modo degradado y responde `503`; el
contenedor queda `unhealthy` de forma intencionada.

### Reentrenar dentro de Docker

```bash
make docker-pipeline
```

Registra una nueva versión en MLflow. El quality gate decide si reemplaza el
alias `champion`; si no mejora el RMSE del champion vigente, se conserva el
anterior y la API sigue sirviendo el modelo existente.

### Logs, parada y limpieza

```bash
make docker-logs    # logs de api y mlflow
make docker-down    # para y elimina contenedores; conserva volúmenes
make docker-reset   # para y elimina también los volúmenes
```

Para recuperar los informes generados por el pipeline:

```bash
docker compose cp pipeline:/app/reports/pipeline_report.json reports/
docker compose cp pipeline:/app/reports/validation_report_clean.json reports/
```

El comando funciona mientras el contenedor `pipeline` siga existiendo, por
ejemplo después de `make docker-up`. Si se ejecutó con `make docker-pipeline`
(`run --rm`), el contenedor se elimina al terminar; en ese caso los informes
quedan en el volumen `pipeline-reports`.

### Limitaciones de la capa Docker

- **SQLite y un solo worker:** MLflow se ejecuta con `--workers 1` porque
  SQLite no soporta bien escrituras concurrentes. Es adecuado para local y
  demostración, no para un equipo grande.
- **Sin autenticación:** MLflow y la API quedan expuestos sin credenciales. No
  publiques estos puertos en Internet sin un reverse proxy con TLS y
  autenticación.
- **Artefactos locales:** el proxy de MLflow sirve los artefactos desde un
  volumen local. En producción se usaría almacenamiento de objetos (S3, GCS,
  Azure Blob) y PostgreSQL.
- **Un solo host:** Compose no proporciona alta disponibilidad ni escalado
  horizontal real.
- **Imagen grande:** MLflow y scikit-learn ocupan bastante; la imagen final
  ronda 1–1,5 GB. La etapa multi-stage evita incluir herramientas de build,
  pero no reduce las dependencias de runtime.

---

## 8. Manejo de errores

| Caso | Código | Comportamiento |
|---|---:|---|
| Petición válida | 200 | Devuelve `prediction`, modelo y versión. |
| Falta una feature, tipo incorrecto, campo extra o valor no finito | 422 | Pydantic rechaza el payload con detalle de validación. |
| Champion no cargado / alias inexistente | 503 | `{"detail": "El modelo champion no está disponible."}` |
| Error durante `model.predict` | 500 | `{"detail": "No se pudo generar la predicción."}`; la traza se registra en logs, no se expone. |
| Error no controlado | 500 | Handler genérico con mensaje genérico. |

`/health` responde `200` aunque el modelo no esté cargado; para saber si se
puede predecir, consulta `model_loaded` o `/model-info`.

Ejemplo de validación:

```bash
curl -i -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 0.038076}'
# HTTP/1.1 422 Unprocessable Content
```

---

## 9. Tests

```bash
make test
```

o:

```bash
.venv/bin/pytest -q
```

La suite cubre:

- Extracción, hash, validación y limpieza de datos.
- Casos de validación inválidos: columna faltante, tipo no numérico, demasiado
  pocas filas, nulos.
- Features raw/engineered y protección de la división por cero.
- Lógica del quality gate y comparación con el champion vigente.
- API: `/health`, `/model-info`, `/predict`, `422` por input inválido, `503`
  sin modelo y `500` cuando el modelo falla.

Estado esperado: **33 tests passed**.

---

## 10. CI/CD con GitHub Actions

El workflow `.github/workflows/ci.yml` se ejecuta en cada push o pull request a
`main`/`master` y contiene dos jobs.

### Job `pipeline` (flujo nativo)

1. Checkout y configuración de Python 3.13 con caché de pip.
2. Instalación de `requirements.txt` y `requirements-dev.txt`.
3. `pytest -q`.
4. Pipeline completo (`python -m mlops.pipeline`) con SQLite local.
5. Arranque de la API y smoke tests con `curl`:
   - `/health`
   - `/model-info`
   - `/predict` con una fila válida
   - `/predict` con una fila incompleta, comprobando `422`
6. Subida de artefactos: `mlflow.db`, `mlruns/`, `reports/`, datos procesados y
   logs de uvicorn.

### Job `docker` (capa de despliegue)

Se ejecuta después del job nativo y valida la imagen y Compose de principio a
fin:

1. `docker compose up -d --build`.
2. Espera a que la API cargue el champion en `/model-info`.
3. Smoke tests: `/health`, `/model-info`, `/predict` y `422`.
4. Reinicia el contenedor `api` y comprueba que se sirve la misma versión del
   champion, verificando la persistencia en los volúmenes.
5. Si algo falla, publica `docker compose ps` y los logs.
6. Siempre ejecuta `docker compose down -v` para limpiar el runner.

El registry de CI es efímero: cada job parte de un runner limpio y reconstruye
el modelo y el alias `champion`. Es suficiente para demostrar la automatización.
Para persistencia entre ejecuciones habría que usar un tracking server remoto
(por ejemplo, un servidor MLflow con PostgreSQL y almacenamiento de objetos),
que queda fuera del alcance de esta entrega.

Si el quality gate no se cumple, el pipeline no asigna champion y el smoke test
de `/model-info`/`/predict` fallará; esto es intencionado: el CI actúa como
bloqueo de promoción.

---

## 11. Notebook exploratorio

El notebook se conserva en
`notebooks/Proyecto_MLOps.ipynb` como evidencia de la fase de ciencia de datos.
Incluye:

- Carga y exploración del dataset.
- Validación y limpieza iniciales.
- Comparación de tres modelos baseline.
- Exploración de interacciones.
- Una sección final que mapea cada paso del notebook con su módulo del
  repositorio.

La conclusión se corrigió para reflejar que GradientBoosting y LinearRegression
estaban prácticamente empatados en el test del notebook, y que la mejora de las
features engineered debía validarse con CV antes de promoverla. En el pipeline
final, la validación cruzada seleccionó LinearRegression con features raw.

Para ejecutar el flujo completo en Colab (tests, pipeline, API y MLflow), usa
`notebooks/Colab.ipynb`; no requiere Docker.

---

## 12. Reproducibilidad

- Semilla fija `42` para split, KFold y modelos.
- Versiones principales fijadas en `requirements.txt`.
- Dataset determinista de scikit-learn, con SHA-256 registrado en MLflow.
- El pipeline se puede repetir desde cero:

```bash
make clean
make pipeline
```

`make clean` elimina `mlruns/`, `mlflow.db`, informes y datos procesados, pero
no el CSV crudo. La siguiente ejecución reconstruye todo.

---

## 13. Limitaciones y siguientes pasos

- **Despliegue local:** tanto el flujo nativo como Compose están pensados para
  un solo host; no hay alta disponibilidad ni escalado horizontal.
- **Registry local/SQLite:** MLflow usa SQLite y artefactos en un volumen; no
  hay autenticación ni escrituras concurrentes. Es una capa de demostración,
  no una plataforma de producción.
- **Dataset pequeño:** 442 filas; las métricas tienen variabilidad y el umbral
  estricto puede fallar con otra partición.
- **Features estandarizadas:** la API espera valores tal como los devuelve
  scikit-learn, no unidades clínicas.
- **Sin monitorización de drift ni reentrenamiento programado.**
- **Sin explicabilidad** (SHAP, importancia de variables) más allá de las
  métricas.

Posibles extensiones: MLflow con PostgreSQL y almacenamiento de objetos,
autenticación de la API, monitorización de drift, tests de contrato más
estrictos y un pipeline de reentrenamiento programado.

---

## 14. Solución de problemas

**`models:/...@champion` no existe o `/model-info` devuelve 503**

Ejecuta el pipeline para registrar y promocionar un modelo:

```bash
make pipeline
```

Comprueba que el alias existe con el snippet de la sección 5.

**El pipeline falla en el quality gate**

El modelo no alcanzó `RMSE < 54.0` o `R² > 0.40`, o no mejoró al champion
vigente. Revisa `reports/pipeline_report.json`. No se asigna champion de forma
automática.

**`ModuleNotFoundError` al ejecutar la API o los tests**

Activa el entorno o usa el intérprete del venv:

```bash
source .venv/bin/activate
# o
.venv/bin/python -m uvicorn api.main:app --port 8000
```

**Puerto 8000 ocupado**

```bash
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8001
```

**Quiero empezar de cero**

```bash
make clean
make pipeline
```

**Docker: `permission denied` al conectar con `/var/run/docker.sock`**

El daemon debe estar activo y tu usuario debe pertenecer al grupo `docker`:

```bash
sudo systemctl start docker
sudo usermod -aG docker "$USER"
# Cierra la sesión y vuelve a entrar
docker info
```

**Docker: el contenedor `api` está `unhealthy`**

Consulta los logs y el estado del pipeline:

```bash
make docker-ps
docker compose logs api pipeline mlflow
```

Si `/model-info` devuelve `503`, el pipeline terminó pero el quality gate no
promocionó ningún modelo. Revisa `reports/pipeline_report.json` dentro del
volumen o los logs de `pipeline`.

**Docker: puerto 5000 u 8000 ocupado**

Para el servicio local que esté usando el puerto (`make api` o
`make mlflow-server`) o cambia la publicación de puertos en
`docker-compose.yml`:

```yaml
ports:
  - "8001:8000"   # API en localhost:8001
  - "5001:5000"   # MLflow en localhost:5001
```

**Docker: quiero borrar también el estado de MLflow**

```bash
make docker-reset   # docker compose down -v
make docker-up
```
