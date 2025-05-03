#  ML Pipeline Container (Training + Tracking)

This containerized pipeline trains and logs an animal outcome prediction model using **MLflow**, storing both metrics and artifacts locally.

---

##  What It Does

* Runs a full **ML training pipeline** using `src.main`
* Logs experiments, metrics, and artifacts to a local **MLflow tracking server**
* Uses Docker + Docker Compose for reproducibility and modularity

---

##  Quick Start

### 1. Build the Container

From your project root:

```bash
docker compose build
```

### 2. Run the Pipeline and MLflow Server

```bash
docker compose up
```

* The **MLflow UI** will be available at [http://localhost:5000](http://localhost:5000)
* The **pipeline container** will execute `src.main` (data prep, training, logging)

---

##  Pipeline Breakdown (`src.main`)

* **Data Preparation**: Loads and preprocesses shelter animal data
* **Modeling**: Trains multiple models and stacks them using Logistic Regression
* **Logging**:

  * Metrics: accuracy, precision, recall, F1
  * Artifacts: model pickle files, encoders
  * Parameters: hyperparameters, timestamps, etc.

All runs are logged to `./mlruns/` and tracked in the UI.

---

##  File Structure

```
project-root/
├── src/
│   └── main.py               # Entrypoint for the pipeline
├── mlruns/                  # MLflow artifacts
├── mlflow.db                # Local tracking DB
├── requirements.txt
├── Dockerfile               # For the pipeline
├── docker-compose.yml       # Manages mlflow + pipeline
```

---

##  Docker Overview

### Dockerfile (Pipeline)

* Uses `python:3.10-slim`
* Copies source code into `/app`
* Installs `build-essential` and Python deps from `requirements.txt`
* Sets `PYTHONPATH=/app`
* Runs `python -m src.main` as default command

### docker-compose.yml

* Defines two services:

  * `mlflow-server`: persistent tracking server
  * `pipeline`: builds and runs model training
* Ensures correct logging via `MLFLOW_TRACKING_URI`

---

##  Useful Commands

```bash
# Rebuild pipeline container
docker compose build pipeline

# View MLflow runs
http://localhost:5000

# Stop everything
docker compose down
```

---

## Extra

* Tracking and artifacts are stored locally for portability
* Works without cloud setup (no S3 or remote DB needed)
* Can be extended to support multiple pipeline variations or schedules
