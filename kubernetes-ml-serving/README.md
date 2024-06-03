# Kubernetes ML Serving

Production ML model serving on Kubernetes with FastAPI, Prometheus metrics, HPA autoscaling, and GPU support.

## Features
- FastAPI inference server with async model store
- Prometheus metrics: request count, latency histogram (p50/p95/p99), GPU memory
- Kubernetes Deployment with GPU node selector (A100)
- HPA autoscaling based on CPU and custom latency metrics
- `/healthz` liveness + `/readyz` readiness probes
- Multi-version model registry

## Deploy
```bash
kubectl apply -f k8s/deployment.yaml
kubectl -n ml-serving get hpa ml-server-hpa
```

## Endpoints
| Endpoint | Description |
|----------|-------------|
| `POST /v1/models/{name}/infer` | Run inference |
| `GET /v1/models` | List loaded models |
| `GET /metrics` | Prometheus metrics |
| `GET /healthz` | Liveness probe |
| `GET /readyz` | Readiness probe |
