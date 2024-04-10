import torch
import torch.nn as nn
from typing import Optional
import time
import numpy as np


def compile_model(model: nn.Module, mode='reduce-overhead',
                  backend='inductor', fullgraph=False) -> nn.Module:
    """
    Compile model with torch.compile (PyTorch 2.0+).
    modes: 'default', 'reduce-overhead', 'max-autotune'
    """
    return torch.compile(model, mode=mode, backend=backend, fullgraph=fullgraph)


class KVCacheAttention(nn.Module):
    """Multi-head attention with KV cache for autoregressive inference."""
    def __init__(self, embed_dim, n_heads, max_seq_len=2048):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = embed_dim // n_heads
        self.scale = self.d_k ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.max_seq_len = max_seq_len
        self.k_cache = None
        self.v_cache = None

    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None

    def forward(self, x, use_cache=True):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)

        if use_cache and self.k_cache is not None:
            k = torch.cat([self.k_cache, k], dim=2)
            v = torch.cat([self.v_cache, v], dim=2)

        if use_cache:
            self.k_cache = k
            self.v_cache = v

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B, T, D)
        return self.out_proj(out)


class ContinuousBatcher:
    """Continuous batching for LLM inference (vLLM-style)."""
    def __init__(self, model, max_batch_size=32, max_tokens_per_batch=4096):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens_per_batch
        self.queue = []

    def add_request(self, input_ids, request_id=None):
        self.queue.append({'ids': input_ids, 'id': request_id, 'generated': []})

    def step(self):
        if not self.queue:
            return []
        batch = self.queue[:self.max_batch_size]
        inputs = torch.nn.utils.rnn.pad_sequence(
            [r['ids'] for r in batch], batch_first=True, padding_value=0
        )
        with torch.no_grad():
            logits = self.model(inputs)
        next_tokens = logits[:, -1, :].argmax(dim=-1)
        results = []
        for i, req in enumerate(batch):
            req['generated'].append(next_tokens[i].item())
            req['ids'] = torch.cat([req['ids'], next_tokens[i:i+1]])
            results.append(req)
        return results


def profile_model(model, input_shape, device='cpu', use_cuda_events=False):
    model.eval().to(device)
    x = torch.randn(*input_shape).to(device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA] if device == 'cuda' else
                   [torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True,
        profile_memory=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(10):
                model(x)
    return prof.key_averages().table(sort_by='cpu_time_total', row_limit=20)
