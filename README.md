# ML Model Deployment Platform

A complete MLOps pipeline for experiment tracking, model registry, serving, canary deployments, and automated monitoring with retraining triggers.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        ML Deployment Platform                    │
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────┐    │
│  │ Training  │───>│  MLflow       │───>│ Model Registry      │    │
│  │ Pipeline  │    │  Tracking     │    │ (Stage Management)  │    │
│  └──────────┘    └──────────────┘    └────────┬────────────┘    │
│                                               │                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                  FastAPI Serving Layer                    │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │    │
│  │  │ /predict │  │ /health  │  │ /metrics │  │/reload │  │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └────────┘  │    │
│  │                                                          │    │
│  │  ┌──────────────────┐    ┌──────────────────────────┐   │    │
│  │  │ Canary Manager   │    │ Prometheus Metrics       │   │    │
│  │  │ (Traffic Split)  │    │ (Latency, Count, Errors) │   │    │
│  │  └──────────────────┘    └──────────────────────────┘   │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────┐    ┌───────────────────────────────┐      │
│  │ Drift Detector   │───>│ Retraining Trigger            │      │
│  │ (Sliding Window) │    │ (Auto-register new model)     │      │
│  └──────────────────┘    └───────────────────────────────┘      │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐                           │
│  │ Prometheus   │───>│ Grafana      │                           │
│  │ (Scraping)   │    │ (Dashboards) │                           │
│  └──────────────┘    └──────────────┘                           │
└──────────────────────────────────────────────────────────────────┘
```

## Tech Stack

- **Python 3.11+** with type hints and Google-style docstrings
- **MLflow** — experiment tracking, model registry, artifact storage
- **FastAPI** — model serving REST API with Pydantic validation
- **Prometheus** — metrics collection (latency, counts, accuracy, errors)
- **Grafana** — dashboards for model performance monitoring
- **Docker / Docker Compose** — containerized full-stack deployment
- **Kubernetes** — manifests with HPA, Ingress, and Kind setup
- **pytest** — test suite with >80% coverage
- **ruff** — linting and formatting
- **GitHub Actions** — CI pipeline

## Features

- **MLflow Experiment Tracking** — log parameters, metrics, artifacts, and run metadata
- **Model Registry** — version management with stage transitions (Staging → Production → Archived)
- **FastAPI Serving** — `/predict`, `/health`, `/metrics`, `/reload` endpoints
- **Canary Deployments** — traffic splitting, per-version metrics, automated rollback
- **Drift Detection** — sliding window accuracy monitoring with configurable thresholds
- **Automated Retraining** — triggers retraining on accuracy degradation, auto-registers new models
- **Prometheus Metrics** — latency histogram, prediction counter, error counter, accuracy gauge
- **Grafana Dashboards** — pre-configured panels for model performance visualization

## Quick Start

### Local Development

```bash
# Clone and install
git clone git@github.com:KarasiewiczStephane/ml-deployment-platform.git
cd ml-deployment-platform
pip install -r requirements.txt

# Train a model
python -c "from src.training.train import train_all_models; train_all_models()"

# Run the API
make run

# Run tests
make test

# Lint
make lint
```

### Docker Compose (Full Stack)

```bash
# Start all services (MLflow, API, Prometheus, Grafana)
docker-compose up -d

# Access services:
# - API:        http://localhost:8000
# - MLflow UI:  http://localhost:5000
# - Prometheus: http://localhost:9090
# - Grafana:    http://localhost:3000 (admin/admin)

# Stop
docker-compose down
```

### Kubernetes (Kind)

```bash
# Build the Docker image
docker build -t ml-deployment-platform:latest .

# Create Kind cluster and deploy
./scripts/kind-setup.sh

# Access via ingress
# Add to /etc/hosts: 127.0.0.1 ml-serving.local
curl http://ml-serving.local/health
```

## API Reference

### POST /predict

Run inference on the loaded model.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 7.0, 3.2, 4.7, 1.4, 6.3]]}'
```

Response:
```json
{
  "predictions": [0],
  "model_name": "ml-deployment-model",
  "model_version": "production",
  "latency_ms": 1.23
}
```

### GET /health

Check API and model status.

```json
{
  "status": "healthy",
  "model_name": "ml-deployment-model",
  "model_version": "production",
  "model_loaded": true
}
```

### GET /metrics

Prometheus-formatted metrics endpoint.

### POST /reload

Trigger a model reload from the MLflow registry.

## Configuration

All settings are in `configs/config.yaml`. Override with environment variables prefixed with `MLP_`:

| Environment Variable | Config Path | Description |
|---|---|---|
| `MLP_MLFLOW_TRACKING_URI` | `mlflow.tracking_uri` | MLflow server URL |
| `MLP_SERVING_PORT` | `serving.port` | API port |
| `MLP_SERVING_MODEL_NAME` | `serving.model_name` | Registered model name |
| `MLP_SERVING_MODEL_STAGE` | `serving.model_stage` | Model stage to load |
| `MLP_LOGGING_LEVEL` | `logging.level` | Log verbosity |

## Project Structure

```
ml-deployment-platform/
├── src/
│   ├── main.py                  # Application entry point
│   ├── training/
│   │   ├── train.py             # Training pipeline with MLflow
│   │   └── evaluate.py          # Model evaluation metrics
│   ├── serving/
│   │   ├── app.py               # FastAPI application
│   │   ├── model_loader.py      # MLflow registry integration
│   │   └── canary.py            # Canary deployment logic
│   ├── monitoring/
│   │   ├── metrics.py           # Prometheus metric collectors
│   │   └── drift_detector.py    # Accuracy drift detection
│   ├── retraining/
│   │   └── trigger.py           # Automated retraining trigger
│   └── utils/
│       ├── config.py            # YAML config loader
│       └── logger.py            # Structured logging
├── k8s/                         # Kubernetes manifests
├── monitoring/
│   ├── prometheus.yml           # Prometheus scrape config
│   └── grafana/                 # Grafana dashboards & provisioning
├── scripts/
│   └── kind-setup.sh            # Kind cluster setup
├── tests/                       # Test suite (>80% coverage)
├── configs/config.yaml          # Application configuration
├── docker-compose.yml           # Full stack orchestration
├── Dockerfile
├── Makefile
├── requirements.txt
└── .github/workflows/ci.yml    # CI pipeline
```

## Monitoring

The Grafana dashboard (`monitoring/grafana/dashboards/model-performance.json`) provides:

- **Model Accuracy** — real-time accuracy gauge with color-coded thresholds
- **Prediction Latency** — p50/p95/p99 percentile time series
- **Request Volume** — predictions per second rate
- **Error Rate** — prediction errors by type
- **Summary Stats** — total predictions, errors, average latency

## License

MIT
