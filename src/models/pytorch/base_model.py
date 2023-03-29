from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn

from src.models.pytorch.embedding_modules import (
    Input1dToEmbeddings,
    Input2dToEmbeddings,
    Input3dToEmbeddings,
    InputTabularToEmbeddings,
    InputTokensToEmbeddings,
)


class BaseModel(ABC, nn.Module):
    '''
    Base encoder class. A forward() pass through this module consists of an embed() and an encode().

    The embed() function takes in an arbitrary-lengthed sequence of tensors, embeds each one to a sequence
    of vectors of the same size, and concatenates them. The embedding module used for each input
    is instantiated based on the input type and can be found in src/models/embedding_modules.py.
    NOTE: To maintain compatibility with the paper baselines, this function should not be changed.

    The encode() function takes this concatenated sequence of embeddings and maps it to some feature
    vector (doesn't have to be same size). This defines the feature extraction for both pretraining
    and downstream transfer learning.
    NOTE: This function should be implemented for each subclass architecture. For many self-supervised
    learning algorithms, the feature representation before the final projection into the feature vector
    improves downstream performance - the prehead argument can control for that.
    '''

    def __init__(
        self,
        input_specs: List[Dict[str, Any]],
        embed_dim: int = 256,
    ):
        super().__init__()

        # Initialize list of embedding modules based on input specs and types.
        self.embed_modules = nn.ModuleList()
        for input_spec in input_specs:
            # Retrieve corresponding embedding module and initialize.
            if input_spec.input_type == 'tokens':
                embedding_module = InputTokensToEmbeddings(
                    seq_len=input_spec.seq_len,
                    vocab_size=input_spec.vocab_size,
                    embed_dim=embed_dim,
                )
            elif input_spec.input_type == '1d':
                embedding_module = Input1dToEmbeddings(
                    seq_len=input_spec.seq_len,
                    segment_size=input_spec.segment_size,
                    in_channels=input_spec.in_channels,
                    embed_dim=embed_dim,
                )
            elif input_spec.input_type == '2d':
                embedding_module = Input2dToEmbeddings(
                    input_size=input_spec.input_size,
                    patch_size=input_spec.patch_size,
                    in_channels=input_spec.in_channels,
                    embed_dim=embed_dim,
                )
            elif input_spec.input_type == '3d':  # same as 2d
                embedding_module = Input3dToEmbeddings(
                    input_size=input_spec.input_size,
                    patch_size=input_spec.patch_size,
                    in_channels=input_spec.in_channels,
                    embed_dim=embed_dim,
                )
            elif input_spec.input_type == 'tabular':
                embedding_module = InputTabularToEmbeddings(
                    num_features=input_spec.num_features,
                    embed_dim=embed_dim,
                )
            else:
                raise ValueError(f'Unrecognized dataset input type: {input_spec.input_type}.')
            self.embed_modules += [embedding_module]

    def embed(self, inputs: Sequence[torch.Tensor]):
        assert len(inputs) == len(self.embed_modules), 'Number of inputs is different than expected.'

        # Embed all inputs and concatenate.
        x = []
        for tensor, module in zip(inputs, self.embed_modules):
            x += [module(tensor)]
        x = torch.cat(x, dim=1)  # (batch_size, seq_len, channels)
        return x

    @abstractmethod
    def encode(self, x: torch.Tensor, prehead: bool = False, **kwargs):
        pass

    def forward(self, *inputs: Sequence[torch.Tensor], **kwargs):
        x = self.embed(*inputs)
        x = self.encode(x, **kwargs)
        return x
