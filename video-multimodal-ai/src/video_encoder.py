import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional


class PatchEmbed3D(nn.Module):
    """Tubelet embedding for video transformers (TimeSformer-style)."""
    def __init__(self, img_size=224, patch_size=16, tubelet_size=2,
                 in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size)
        )
        self.n_patches_h = img_size // patch_size
        self.n_patches_w = img_size // patch_size

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        B, D, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, T*H*W, D)
        return x, T, H, W


class TemporalAttention(nn.Module):
    """Divided space-time attention (temporal part)."""
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = embed_dim // n_heads
        self.scale = self.d_k ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, T, H, W):
        B, N, D = x.shape
        residual = x
        x = self.norm(x)
        # reshape to attend over time dimension
        x = rearrange(x, 'b (t h w) d -> (b h w) t d', t=T, h=H, w=W)
        BHW, t, d = x.shape
        qkv = self.qkv(x).reshape(BHW, t, 3, self.n_heads, self.d_k).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(BHW, t, d)
        out = rearrange(out, '(b h w) t d -> b (t h w) d', b=B, h=H, w=W)
        return residual + self.proj(out)


class SpatialAttention(nn.Module):
    """Divided space-time attention (spatial part)."""
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = embed_dim // n_heads
        self.scale = self.d_k ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, T, H, W):
        residual = x
        x = self.norm(x)
        x = rearrange(x, 'b (t h w) d -> (b t) (h w) d', t=T, h=H, w=W)
        BT, hw, d = x.shape
        qkv = self.qkv(x).reshape(BT, hw, 3, self.n_heads, self.d_k).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1,2).reshape(BT, hw, d)
        out = rearrange(out, '(b t) (h w) d -> b (t h w) d', b=x.shape[0]//T, t=T, h=H, w=W)
        return residual + self.proj(out)


class VideoTransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, n_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.temporal_attn = TemporalAttention(embed_dim, n_heads, dropout)
        self.spatial_attn = SpatialAttention(embed_dim, n_heads, dropout)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(mlp_dim, embed_dim), nn.Dropout(dropout)
        )

    def forward(self, x, T, H, W):
        x = self.temporal_attn(x, T, H, W)
        x = self.spatial_attn(x, T, H, W)
        x = x + self.mlp(self.norm(x))
        return x


class VideoEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, tubelet_size=2,
                 n_frames=8, embed_dim=768, depth=12, n_heads=12,
                 mlp_ratio=4.0, dropout=0.1, output_dim=512):
        super().__init__()
        self.patch_embed = PatchEmbed3D(img_size, patch_size, tubelet_size,
                                         embed_dim=embed_dim)
        n_patches = (img_size // patch_size) ** 2 * (n_frames // tubelet_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([
            VideoTransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, output_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: (B, C, T, H, W)
        tokens, T, H, W = self.patch_embed(x)
        B = tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens, T, H, W)
        tokens = self.norm(tokens)
        return F.normalize(self.head(tokens[:, 0]), dim=-1)


def load_pretrained_videomae(n_classes=400, n_frames=16) -> VideoEncoder:
    """Build VideoMAE-style encoder with pretrained config."""
    return VideoEncoder(
        img_size=224, patch_size=16, tubelet_size=2,
        n_frames=n_frames, embed_dim=768, depth=12,
        n_heads=12, output_dim=512
    )
