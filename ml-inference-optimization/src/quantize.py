import torch
import torch.nn as nn
from torch.quantization import (
    prepare, convert, get_default_qconfig,
    quantize_dynamic, QuantStub, DeQuantStub
)
import numpy as np
from typing import Optional, Callable


class QuantizationWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        return self.dequant(x)


def dynamic_quantize(model: nn.Module, dtype=torch.qint8) -> nn.Module:
    """Apply dynamic quantization to Linear and LSTM layers."""
    return quantize_dynamic(model, {nn.Linear, nn.LSTM}, dtype=dtype)


def static_quantize(model: nn.Module, calibration_loader,
                    backend='fbgemm') -> nn.Module:
    """Apply static (post-training) quantization with calibration data."""
    model.eval()
    model.qconfig = get_default_qconfig(backend)
    wrapped = QuantizationWrapper(model)
    prepare(wrapped, inplace=True)

    with torch.no_grad():
        for batch in calibration_loader:
            if isinstance(batch, (list, tuple)):
                wrapped(batch[0])
            else:
                wrapped(batch)

    convert(wrapped, inplace=True)
    return wrapped


class SmoothQuant:
    """SmoothQuant: Accurate and Efficient Post-Training Quantization."""
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.scales = {}

    def compute_scales(self, model: nn.Module, activation_stats: dict):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if name in activation_stats:
                    act_max = activation_stats[name]
                    w_max = module.weight.abs().max(dim=0)[0]
                    scale = (act_max ** self.alpha) / (w_max ** (1 - self.alpha))
                    scale = scale.clamp(min=1e-5)
                    self.scales[name] = scale
        return self.scales

    def apply(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in self.scales:
                scale = self.scales[name]
                module.weight.data = module.weight.data / scale.unsqueeze(0)
        return model


def benchmark_latency(model: nn.Module, input_shape: tuple,
                       n_warmup=20, n_runs=100, device='cpu') -> dict:
    """Benchmark model inference latency."""
    model.eval().to(device)
    dummy = torch.randn(*input_shape).to(device)
    import time

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device == 'cuda':
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    return {
        'mean_ms': np.mean(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
        'std_ms': np.std(latencies),
    }
