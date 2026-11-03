"""Tiny vision transformer and DINO projection head used in Lab 4."""

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        # Separate normalization precedes attention and the feed-forward sublayer.
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        # One projection produces queries, keys and values; the other mixes attention heads.
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        # Expand each token's channels fourfold, apply GELU, then return to dim channels.
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        B, T, D = x.shape  # Batch size, token count and embedding dimension.
        q, k, v = (
            self.qkv(self.norm1(x))  # Normalize tokens and project all Q/K/V at once.
            .reshape(B, T, 3, self.heads, D // self.heads)  # Split Q/K/V and head dimensions.
            .permute(2, 0, 3, 1, 4)  # Arrange as (Q/K/V, batch, heads, tokens, head_dim).
        )
        # Attend within each head, concatenate head outputs, project, and add the input.
        x = x + self.proj(
            F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, D)
        )  # residual connection
        return x + self.mlp(self.norm2(x))  # A second residual adds the token-wise MLP output.


class TinyViT(nn.Module):
    def __init__(self, dim=128, depth=4, heads=4, patch=4):
        super().__init__()
        self.dim = dim
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)  # Embed non-overlapping image patches.
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))  # Learned token that summarizes the image.
        self.pos = nn.Parameter(torch.randn(1, dim, 8, 8) * 0.02)  # Learned patch-grid positions.
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, dim))  # Separate position for the CLS token.
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))  # Register all blocks.
        self.norm = nn.LayerNorm(dim)  # Normalize embeddings before extracting CLS features.

    def forward(self, x):
        # (batch, 3, height, width), normalized -> (batch, 128) CLS features
        patches = self.patch_embed(x)  # Shape: (batch, dim, patch_rows, patch_columns).
        # Resize the learned position grid only when the input has a different patch grid.
        pos = (
            self.pos
            if patches.shape[2:] == self.pos.shape[2:]
            else F.interpolate(
                self.pos, size=patches.shape[2:], mode="bicubic", align_corners=False
            )
        )
        x = (patches + pos).flatten(2).transpose(1, 2)  # Convert the grid into a token sequence.
        # Prepend one copy of the shared CLS token per image; expand avoids copying its weights.
        x = torch.cat([self.cls.expand(len(x), -1, -1) + self.cls_pos, x], 1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0]  # Return only the CLS token as each image's representation.


class DINONetwork(nn.Module):
    # The backbone plus the head that produces the K codes: MLP, L2 normalization, weight-normalized linear layer.
    def __init__(self, K):
        super().__init__()
        self.backbone = TinyViT()  # The backbone's CLS feature is used for downstream k-NN.
        # Map backbone features into the smaller space used by the projection head.
        self.mlp = nn.Sequential(
            nn.Linear(self.backbone.dim, 256), nn.GELU(), nn.Linear(256, 64)
        )
        # Produce K code logits with separately parameterized weight directions and magnitudes.
        self.last = nn.utils.parametrizations.weight_norm(nn.Linear(64, K, bias=False))

    def forward(self, x):
        # Extract features, project them, normalize their length, then compute code logits.
        return self.last(F.normalize(self.mlp(self.backbone(x)), dim=-1))
