from typing import List, Union

import numpy as np
import torch.nn as nn
from einops.layers.torch import Rearrange


class InputTokensToEmbeddings(nn.Module):
    '''Class that embeds tokens.'''

    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        embed_dim: int,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, input_text):
        assert input_text.dim() == 2  # (batch_size, seq_len)
        assert input_text.shape[1] == self.seq_len
        return self.embed(input_text)

    @property
    def length(self):
        return self.seq_len


class Input1dToEmbeddings(nn.Module):
    '''Class that projects 1d inputs to embedding dimension without grouping into patches.'''

    def __init__(
        self,
        seq_len: int,
        segment_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.segment_size = segment_size
        self.in_channels = in_channels

        assert self.seq_len % self.segment_size == 0

        self.embed = nn.Sequential(
            Rearrange('b (h2 h1) c -> b h2 (h1 c)', h1=segment_size),
            nn.Linear(in_channels * segment_size, embed_dim),
        )

    def forward(self, input_1d):
        assert input_1d.dim() == 3  # (batch_size, seq_len, in_channels)
        assert input_1d.shape[1] == self.seq_len
        assert input_1d.shape[2] == self.in_channels
        return self.embed(input_1d)

    @property
    def length(self):
        return self.seq_len // self.segment_size


class Input2dToEmbeddings(nn.Module):
    '''Class that divides 2d inputs into 2d patches, flattens, and projects to embedding dimension.'''

    def __init__(
        self,
        input_size: Union[int, List[int]],
        patch_size: Union[int, List[int]],
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        # Format and check that sizes are lists.
        if isinstance(input_size, int):
            input_size = [input_size] * 2
        if isinstance(patch_size, int):
            patch_size = [patch_size] * 2
        assert len(input_size) == 2
        assert len(patch_size) == 2

        # Check for even divisibility.
        for i, p in zip(input_size, patch_size):
            assert i % p == 0

        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size[0], p2=patch_size[1]),
            nn.Linear(in_channels * patch_size[0] * patch_size[1], embed_dim),
        )

    def forward(self, input_2d):
        assert input_2d.dim() == 4  # (batch_size, in_channels, input_height, input_width)
        assert input_2d.shape[1] == self.in_channels
        assert all(a == b for a, b in zip(input_2d.shape[2:], self.input_size))
        return self.embed(input_2d)

    @property
    def length(self):
        return np.prod(self.input_size) // np.prod(self.patch_size)


class Input3dToEmbeddings(nn.Module):
    '''Class that divides 3d inputs into 3d patches, flattens, and projects to embedding dimension.'''

    def __init__(
        self,
        input_size: Union[int, List[int]],
        patch_size: Union[int, List[int]],
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        if isinstance(input_size, int):
            input_size = [input_size] * 3
        if isinstance(patch_size, int):
            patch_size = [patch_size] * 3
        assert len(input_size) == 3
        assert len(patch_size) == 3

        # Check for even divisibility.
        for i, p in zip(input_size, patch_size):
            assert i % p == 0

        self.input_size = input_size
        self.in_channels = in_channels
        self.embed = nn.Sequential(
            Rearrange(
                'b c (t p1) (h p2) (w p3) -> b (t h w) (p1 p2 p3 c)', p1=patch_size[0], p2=patch_size[1], p3=patch_size[2]
            ),
            nn.Linear(in_channels * patch_size[0] * patch_size[1] * patch_size[2], embed_dim),
        )

    def forward(self, input_3d):
        assert input_3d.dim() == 5  # (batch_size, in_channels, input_depth, input_height, input_width)
        assert input_3d.shape[1] == self.in_channels
        assert all(a == b for a, b in zip(input_3d.shape[2:], self.input_size))
        return self.embed(input_3d)

    @property
    def length(self):
        return np.prod(self.input_size) // np.prod(self.patch_size)


class InputTabularToEmbeddings(nn.Module):
    '''Class that embeds tabular data'''

    # (batch_size, number_of_features, 1)
    NUM_DIMS = 3

    def __init__(
        self,
        num_features: int,
        embed_dim: int,
    ):
        super().__init__()

        self.num_features = num_features
        self.embed = nn.Linear(1, embed_dim)

    def forward(self, input_tabular):
        assert input_tabular.dim() == self.NUM_DIMS
        assert input_tabular.shape[1] == self.num_features
        assert input_tabular.shape[2] == 1
        return self.embed(input_tabular)

    @property
    def length(self):
        return self.num_features
