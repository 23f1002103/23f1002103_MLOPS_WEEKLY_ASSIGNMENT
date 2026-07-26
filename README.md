# Week 6 - ML Model Deployment & Serving

## Overview
This week focuses on containerizing the IRIS inference API with Docker and deploying it to Kubernetes on GCP, with automated CI/CD via GitHub Actions.

## Architecture
Code Push → GitHub Actions (CD) → Docker Build → Artifact Registry → GKE Deployment → Live API

## Files
| File | Description |
|------|-------------|
| `train.py` | Trains IRIS model and saves as model.joblib |
| `main.py` | FastAPI app serving predictions |
| `Dockerfile` | Containerizes the API |
| `requirements.txt` | Python dependencies |
| `.github/workflows/cd.yml` | CD pipeline via GitHub Actions |

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | POST | Predict IRIS species |

## Sample Request
```json
POST http://34.68.43.2/predict
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
```

## Sample Response
```json
{
    "prediction": 0,
    "species": "setosa"
}
```

## Deployment Details
| Component | Details |
|-----------|---------|
| Docker Image | us-central1-docker.pkg.dev/project-b38b370e-25fb-420a-a54/iris-repo/iris-api:latest |
| Artifact Registry | iris-repo (us-central1) |
| GKE Cluster | iris-cluster (us-central1) |
| External IP | 34.68.43.2 |

## CD Pipeline Steps
1. Authenticate to GCP via Workload Identity Federation
2. Build Docker image
3. Push to Google Artifact Registry
4. Deploy to Google Kubernetes Engine
5. Verify rollout status

## How to Run Locally
```bash
# Train model
python train.py

# Run API
uvicorn main:app --host 0.0.0.0 --port 8000

# Test API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

## Tasks Completed
- [x] Task 1: Pod vs Container explanation (video)
- [x] Task 2: Dockerfile for IRIS API
- [x] Task 3: GCP Service Account + WIF setup
- [x] Task 4: GitHub Actions build & push
- [x] Task 5: Deploy to GKE, live API working
