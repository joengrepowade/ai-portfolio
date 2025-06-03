import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import asyncio
import threading
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Model Server", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Prometheus metrics
REQUEST_COUNT = Counter('ml_requests_total', 'Total requests', ['model', 'status'])
REQUEST_LATENCY = Histogram('ml_request_latency_seconds', 'Request latency', ['model'],
                             buckets=[.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5])
BATCH_SIZE_HIST = Histogram('ml_batch_size', 'Batch sizes', buckets=[1, 2, 4, 8, 16, 32, 64])
MODELS_LOADED = Gauge('ml_models_loaded', 'Number of loaded models')
GPU_MEMORY = Gauge('ml_gpu_memory_used_bytes', 'GPU memory used')


class ModelStore:
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def load(self, name: str, model, version='latest'):
        async with self._lock:
            self._models[f"{name}:{version}"] = model
            self._models[f"{name}:latest"] = model
            MODELS_LOADED.set(len(set(k.split(':')[0] for k in self._models)))
            logger.info(f"Model {name}:{version} loaded")

    async def get(self, name: str, version='latest'):
        key = f"{name}:{version}"
        if key not in self._models:
            raise KeyError(f"Model {name}:{version} not found")
        return self._models[key]

    def list(self):
        return list({k.split(':')[0] for k in self._models})


model_store = ModelStore()


class InferRequest(BaseModel):
    inputs: List[List[float]]
    model_version: Optional[str] = 'latest'
    return_logits: Optional[bool] = False


class InferResponse(BaseModel):
    predictions: List[Any]
    latency_ms: float
    model_version: str
    request_id: str


@app.post("/v1/models/{model_name}/infer", response_model=InferResponse)
async def infer(model_name: str, req: InferRequest, request: Request):
    request_id = request.headers.get('X-Request-ID', str(time.time()))
    start = time.perf_counter()

    try:
        model = await model_store.get(model_name, req.model_version)
    except KeyError as e:
        REQUEST_COUNT.labels(model=model_name, status='404').inc()
        raise HTTPException(status_code=404, detail=str(e))

    inputs = torch.tensor(req.inputs, dtype=torch.float32)
    BATCH_SIZE_HIST.observe(len(inputs))

    try:
        with torch.no_grad():
            outputs = model(inputs)
        preds = outputs.tolist() if req.return_logits else outputs.argmax(1).tolist()
        latency = (time.perf_counter() - start) * 1000
        REQUEST_COUNT.labels(model=model_name, status='200').inc()
        REQUEST_LATENCY.labels(model=model_name).observe(latency / 1000)
        return InferResponse(predictions=preds, latency_ms=round(latency, 2),
                             model_version=req.model_version, request_id=request_id)
    except Exception as e:
        REQUEST_COUNT.labels(model=model_name, status='500').inc()
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    if torch.cuda.is_available():
        GPU_MEMORY.set(torch.cuda.memory_allocated())
    return Response(generate_latest(), media_type="text/plain")


@app.get("/v1/models")
async def list_models():
    return {"models": model_store.list()}


@app.get("/healthz")
async def health():
    return {"status": "ok", "models": len(model_store.list())}


@app.get("/readyz")
async def ready():
    if not model_store.list():
        raise HTTPException(status_code=503, detail="No models loaded")
    return {"status": "ready"}


@app.post("/v1/models/{model_name}/batch_infer")
async def batch_infer(model_name: str, requests: List[InferRequest]):
    """Dynamic batching endpoint - collects multiple requests."""
    results = []
    for req in requests:
        result = await infer(model_name, req, Request(scope={'type': 'http', 'headers': []}))
        results.append(result)
    return results


class AdaptiveBatcher:
    """Adaptive dynamic batching with configurable max latency SLO."""
    def __init__(self, max_batch_size=32, max_wait_ms=20):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue = asyncio.Queue()

    async def add(self, item):
        future = asyncio.get_event_loop().create_future()
        await self._queue.put((item, future))
        return await future

    async def process_loop(self, model):
        while True:
            batch, futures = [], []
            deadline = time.time() + self.max_wait_ms / 1000
            while len(batch) < self.max_batch_size and time.time() < deadline:
                try:
                    item, fut = await asyncio.wait_for(
                        self._queue.get(), timeout=max(0, deadline - time.time()))
                    batch.append(item)
                    futures.append(fut)
                except asyncio.TimeoutError:
                    break
            if batch:
                inputs = torch.tensor(batch)
                with torch.no_grad():
                    results = model(inputs).tolist()
                for fut, res in zip(futures, results):
                    fut.set_result(res)
