# ML Inference Optimization

Production-grade inference optimization toolkit: ONNX export, quantization (INT8/dynamic/static), torch.compile, KV cache, and continuous batching.

## Features
- **ONNX Export**: Dynamic axes, graph optimization, ORT session with CUDA EP
- **Quantization**: Dynamic (INT8), Static (PTQ with calibration), SmoothQuant
- **torch.compile**: `reduce-overhead` and `max-autotune` modes with Inductor backend
- **KV Cache**: Autoregressive inference with cached key/value for LLM decoding
- **Continuous Batching**: vLLM-style batching for higher GPU utilization
- **Latency Benchmarking**: p50/p95/p99 with warmup

## Results (example)
| Method | Latency p50 | Speedup |
|--------|-------------|---------|
| FP32 baseline | 12.4ms | 1.0x |
| torch.compile | 7.1ms | 1.7x |
| ONNX + ORT | 5.8ms | 2.1x |
| INT8 dynamic | 3.9ms | 3.2x |

## Usage
```python
from src.onnx_export import export_to_onnx, ONNXInferenceEngine
from src.quantize import dynamic_quantize, benchmark_latency

# Export + benchmark
export_to_onnx(model, (1, 3, 224, 224), "model.onnx")
engine = ONNXInferenceEngine("model.onnx")
stats = engine.benchmark((1, 3, 224, 224))
```
