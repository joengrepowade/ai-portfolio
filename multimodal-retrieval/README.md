# Multimodal Retrieval

Video-text retrieval system using CLIP-style contrastive learning with FAISS ANN indexing.

## Features
- CLIP-style contrastive model for video-text alignment
- FAISS index types: Flat, IVFFlat, HNSW, IVF-PQ (GPU support)
- Text-to-video and video-to-video similarity search
- Batch video indexing with metadata storage

## Usage
```python
from src.retrieval import MultimodalRetriever, FAISSVideoIndex

index = FAISSVideoIndex(embed_dim=512, index_type='HNSW')
retriever = MultimodalRetriever(model, index, tokenizer)

retriever.index_videos(videos, metadata)
results = retriever.retrieve_by_text("a dog running in the park", k=10)
```
