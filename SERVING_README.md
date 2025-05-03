# 🐾 FastAPI Model Serving Container (Animal Outcome Predictor)

This container provides a lightweight FastAPI application for serving predictions using a stacked machine learning model trained on animal shelter data. The app exposes two endpoints:

- [`/`](http://localhost:8000/) — Health/info check and mappings for inputs
- [`/predict`](http://localhost:8000/docs) — Accepts animal features and returns a prediction (`Positive`, `Neutral`, or `Negative`)

---

## 🚀 Quick Start

### 1. Build the Docker Image

From your project root:

```bash
docker build -f serve_model.Dockerfile -t fastapi-serve-app .
```

### 2. Run the Container

```bash
docker run -p 8000:8000 fastapi-serve-app
```

Visit the app at: http://localhost:8000/docs

# Notes
1. .pk files of encoders are within src/saved_models
2. The python file that creates these saved models is in the folder called scripts
