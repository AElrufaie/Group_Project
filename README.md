# IS2 Group Project: Animal Shelter Data Pipeline & Serving API

 ## Project Overview

This repository extends the previous IS2 group project by not only building a robust end-to-end machine learning pipeline but also containerizing the prediction API using Docker. The project uses animal shelter data to perform data processing, clustering, modeling, explainability, and causal inference. In this version, we've also introduced:

A production-ready FastAPI prediction service.

Dockerized pipeline and server.

docker-compose setup for reproducibility.

GitHub CI/CD checks and Pytest coverage.

 ## Key Features

 ### Pipeline

All steps from data_drafting → preprocessing → clustering → modeling → causal_inference → SHAP_value are orchestrated in src/main.py.

Model training includes ensemble and stacking approaches.

Tracked using MLflow, also launched via docker-compose.

 ### Serving API (FastAPI)

Separate FastAPI app for serving predictions.

/docs Swagger UI gives:

/: Info + lists mappings for categorical inputs.

/predict: Accepts cleaned input and returns prediction.

## Dockerized and runnable via:

# From project root
$ docker build -f serve_model.Dockerfile -t fastapi-serve-app .
$ docker run -p 8000:8000 fastapi-serve-app

Visit http://localhost:8000/docs to try it out!

 Dockerized Services

Dockerfile: For pipeline execution (python -m src.main).

serve_model.Dockerfile: Runs FastAPI app (serve_model.py).

docker-compose.yml:

Launches MLflow tracking server.

Runs pipeline with MLflow URI linked.

docker-compose up --build


 ## Testing and CI/CD

Unit tests in tests/

Enforced with Pytest

GitHub Actions CI pipeline runs tests on push

## Future Enhancements

Add authentication layer to API.

CI for Docker image linting and build.

Deployment to cloud (e.g., Azure Web App or ECS).


## Contributors
- Ashraf Elrufaie
- Boyang Wan
- Clement Orcibal
- Kevin Wang
- Yifei Liu
