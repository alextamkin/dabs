'''A Vision Transformer implementation from https://github.com/lucidrains/vit-pytorch'''

from typing import Any, Dict, List

import torch
from einops import rearrange, repeat
from torch import einsum, nn

from src.models.pytorch.base_model import BaseModel


class PreNorm(nn.Module):
    '''Applies pre-layer layer normalization.'''

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    '''Simple feed-forward network with GELU activation.'''

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    '''Vanilla self-attention block.'''

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    '''Transformer encoder model.'''

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class DomainAgnosticTransformer(BaseModel):
    '''Domain-agnostic Transformer architecture.'''

    def __init__(
        self,
        input_specs: List[Dict[str, Any]],
        embed_dim: int = 256,
        dim: int = 256,
        out_dim: int = 128,
        depth: int = 12,
        heads: int = 8,
        mlp_dim: int = 512,
        pool: str = 'mean',
        dim_head: int = 64,
        dropout: float = 0.,
        emb_dropout: float = 0.,
    ):
        assert embed_dim == dim, f'Different embed dim than model dim is currently not allowed'

        # See src/encoders/base.py for embedding modules.
        super().__init__(input_specs=input_specs, embed_dim=embed_dim)

        # Sequence length is stored in each embedding module.
        seq_len = sum(module.length for module in self.embed_modules)

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool

        self.emb_dim = dim
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim),
        )

    def encode(self, x: torch.tensor, prepool=False, prehead=False):
        # Concatenate CLS token and add positional embeddings.
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        # Pass through Transformer.
        x = self.dropout(x)
        x = self.transformer(x)

        if prepool:
            return x
        # Aggregate features and project.
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        if prehead:
            return x

        x = self.to_latent(x)
        return self.mlp_head(x)
