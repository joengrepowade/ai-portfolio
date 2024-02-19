import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', output_dim=512, freeze=False):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        self.proj = nn.Linear(self.model.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        return F.normalize(self.proj(pooled), dim=-1)


class AudioEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=512, n_layers=4):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm1d(hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (B, freq_bins, time)
        x = self.conv_stem(x).transpose(1, 2)
        x = self.transformer(x).transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return F.normalize(self.proj(x), dim=-1)


class VideoTextContrastive(nn.Module):
    def __init__(self, video_encoder, text_encoder, temperature=0.07):
        super().__init__()
        self.video_encoder = video_encoder
        self.text_encoder = text_encoder
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, video, input_ids, attention_mask):
        v_emb = self.video_encoder(video)
        t_emb = self.text_encoder(input_ids, attention_mask)
        logits = (v_emb @ t_emb.T) / self.temperature.exp().clamp(min=0.01)
        labels = torch.arange(len(v_emb), device=v_emb.device)
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.T, labels)
        return (loss_v2t + loss_t2v) / 2

    @torch.no_grad()
    def encode_video(self, video):
        return self.video_encoder(video)

    @torch.no_grad()
    def encode_text(self, input_ids, attention_mask):
        return self.text_encoder(input_ids, attention_mask)


class MultimodalFusionModel(nn.Module):
    """Late fusion of video, text, and audio embeddings."""
    def __init__(self, embed_dim=512, n_classes=400, fusion='attention'):
        super().__init__()
        self.fusion = fusion
        if fusion == 'attention':
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
            self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(embed_dim, n_classes)
        )

    def forward(self, v_emb, t_emb, a_emb):
        # Simple concat fusion
        fused = torch.cat([v_emb, t_emb, a_emb], dim=-1)
        return self.classifier(fused)
