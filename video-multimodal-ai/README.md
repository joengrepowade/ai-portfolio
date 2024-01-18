# Video & Multimodal AI

Video understanding and multimodal embedding system combining video, text, and audio encoders with contrastive learning.

## Architecture
- **VideoEncoder**: Divided space-time attention (TimeSformer-style) with 3D tubelet patch embedding
- **TextEncoder**: BERT-based with projection head
- **AudioEncoder**: Conv1D stem + Transformer encoder over mel-spectrogram
- **VideoTextContrastive**: CLIP-style contrastive loss across video-text pairs
- **MultimodalFusion**: Late fusion of all three modalities

## Features
- Tubelet embedding (spatial + temporal patches via Conv3D)
- Temporal and spatial divided attention
- Video-text contrastive training
- Flexible frame sampling: uniform, random, dense

## Usage
```python
from src.video_encoder import VideoEncoder
from src.multimodal_model import TextEncoder, VideoTextContrastive

video_enc = VideoEncoder(n_frames=8, embed_dim=768, depth=12, output_dim=512)
text_enc = TextEncoder(output_dim=512)
model = VideoTextContrastive(video_enc, text_enc)

loss = model(video_tensor, input_ids, attention_mask)
```
