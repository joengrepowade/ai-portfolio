import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from typing import Dict, List, Tuple, Optional


def export_to_onnx(model: nn.Module, input_shape: tuple, output_path: str,
                   opset_version=17, dynamic_axes: Optional[Dict] = None,
                   input_names=None, output_names=None) -> str:
    model.eval()
    dummy_input = torch.randn(*input_shape)
    input_names = input_names or ['input']
    output_names = output_names or ['output']
    dynamic_axes = dynamic_axes or {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
    torch.onnx.export(
        model, dummy_input, output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model exported to {output_path}")
    return output_path


def optimize_onnx(model_path: str, output_path: str) -> str:
    """Apply ONNX graph optimizations."""
    from onnxruntime.transformers import optimizer
    optimized = optimizer.optimize_model(
        model_path,
        model_type='bert',
        num_heads=12,
        hidden_size=768,
        optimization_options=None
    )
    optimized.save_model_to_file(output_path)
    return output_path


class ONNXInferenceEngine:
    def __init__(self, model_path: str, providers=None):
        self.providers = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(model_path, opts, providers=self.providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def run(self, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        return self.session.run(self.output_names, inputs)

    def benchmark(self, input_shape: Tuple, n_warmup=20, n_runs=100) -> Dict:
        import time
        dummy = {self.input_names[0]: np.random.randn(*input_shape).astype(np.float32)}
        for _ in range(n_warmup):
            self.run(dummy)
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.run(dummy)
            latencies.append((time.perf_counter() - t0) * 1000)
        return {
            'mean_ms': np.mean(latencies),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
        }

    def validate(self, torch_model: nn.Module, input_shape: Tuple,
                 rtol=1e-3, atol=1e-5) -> bool:
        dummy = torch.randn(*input_shape)
        torch_model.eval()
        with torch.no_grad():
            torch_out = torch_model(dummy).numpy()
        onnx_out = self.run({self.input_names[0]: dummy.numpy()})[0]
        match = np.allclose(torch_out, onnx_out, rtol=rtol, atol=atol)
        print(f"Output match: {match} (max diff: {np.abs(torch_out - onnx_out).max():.6f})")
        return match
