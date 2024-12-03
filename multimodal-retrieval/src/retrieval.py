import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
from typing import List, Tuple, Optional, Dict


class ContrastiveModel(nn.Module):
    """CLIP-style video-text contrastive model."""
    def __init__(self, video_encoder, text_encoder,
                 embed_dim=512, temperature=0.07):
        super().__init__()
        self.video_enc = video_encoder
        self.text_enc = text_encoder
        self.video_proj = nn.Linear(embed_dim, embed_dim)
        self.text_proj = nn.Linear(embed_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def encode_video(self, video):
        return F.normalize(self.video_proj(self.video_enc(video)), dim=-1)

    def encode_text(self, input_ids, attention_mask):
        return F.normalize(self.text_proj(self.text_enc(input_ids, attention_mask)), dim=-1)

    def forward(self, video, input_ids, attention_mask):
        v = self.encode_video(video)
        t = self.encode_text(input_ids, attention_mask)
        scale = self.logit_scale.exp().clamp(max=100)
        logits_v2t = scale * v @ t.T
        logits_t2v = logits_v2t.T
        labels = torch.arange(len(v), device=v.device)
        loss = (F.cross_entropy(logits_v2t, labels) + F.cross_entropy(logits_t2v, labels)) / 2
        return loss, logits_v2t


class FAISSVideoIndex:
    """FAISS-based approximate nearest neighbor index for video embeddings."""
    def __init__(self, embed_dim=512, index_type='IVFFlat', n_lists=100,
                 use_gpu=False):
        self.embed_dim = embed_dim
        self.use_gpu = use_gpu
        self.metadata: List[Dict] = []

        if index_type == 'Flat':
            quantizer = faiss.IndexFlatIP(embed_dim)
            self.index = quantizer
        elif index_type == 'IVFFlat':
            quantizer = faiss.IndexFlatIP(embed_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embed_dim, n_lists,
                                             faiss.METRIC_INNER_PRODUCT)
        elif index_type == 'HNSW':
            self.index = faiss.IndexHNSWFlat(embed_dim, 32,
                                              faiss.METRIC_INNER_PRODUCT)
        elif index_type == 'IVF_PQ':
            quantizer = faiss.IndexFlatIP(embed_dim)
            self.index = faiss.IndexIVFPQ(quantizer, embed_dim, n_lists, 16, 8)

        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def train(self, embeddings: np.ndarray):
        if not self.index.is_trained:
            self.index.train(embeddings.astype(np.float32))

    def add(self, embeddings: np.ndarray, metadata: Optional[List[Dict]] = None):
        if not self.index.is_trained:
            self.train(embeddings)
        self.index.add(embeddings.astype(np.float32))
        if metadata:
            self.metadata.extend(metadata)

    def search(self, query: np.ndarray, k=10, nprobe=10) -> Tuple[np.ndarray, List[Dict]]:
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe
        query = query.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        scores, indices = self.index.search(query, k)
        results = [self.metadata[i] for i in indices[0] if i >= 0 and i < len(self.metadata)]
        return scores[0], results

    def save(self, path: str):
        faiss.write_index(faiss.index_gpu_to_cpu(self.index) if self.use_gpu else self.index, path)

    def load(self, path: str):
        self.index = faiss.read_index(path)


class MultimodalRetriever:
    def __init__(self, model: ContrastiveModel, index: FAISSVideoIndex,
                 tokenizer, device='cuda'):
        self.model = model.eval().to(device)
        self.index = index
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def index_videos(self, videos: List, metadata: List[Dict], batch_size=32):
        embeddings = []
        for i in range(0, len(videos), batch_size):
            batch = torch.stack(videos[i:i+batch_size]).to(self.device)
            emb = self.model.encode_video(batch).cpu().numpy()
            embeddings.append(emb)
        embeddings = np.concatenate(embeddings)
        self.index.add(embeddings, metadata)
        print(f"Indexed {len(embeddings)} videos")

    @torch.no_grad()
    def retrieve_by_text(self, query: str, k=10) -> List[Dict]:
        enc = self.tokenizer(query, return_tensors='pt',
                             padding=True, truncation=True).to(self.device)
        t_emb = self.model.encode_text(enc['input_ids'], enc['attention_mask'])
        scores, results = self.index.search(t_emb.cpu().numpy(), k)
        for r, s in zip(results, scores):
            r['score'] = float(s)
        return results

    @torch.no_grad()
    def retrieve_by_video(self, video: torch.Tensor, k=10) -> List[Dict]:
        v_emb = self.model.encode_video(video.unsqueeze(0).to(self.device))
        scores, results = self.index.search(v_emb.cpu().numpy(), k)
        for r, s in zip(results, scores):
            r['score'] = float(s)
        return results


class ReRanker:
    """Cross-encoder re-ranking for top-k retrieval results."""
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def rerank(self, query: str, candidates: List[Dict], text_key='caption', top_k=5) -> List[Dict]:
        pairs = [(query, c[text_key]) for c in candidates]
        enc = self.tokenizer(pairs, return_tensors='pt', padding=True,
                             truncation=True, max_length=256).to(self.device)
        scores = self.model(**enc).logits[:, 1].cpu().numpy()
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [r for r, _ in ranked[:top_k]]
