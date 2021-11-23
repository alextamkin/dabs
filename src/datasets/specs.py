'''Contains read-only classes that define specs for data formats.'''

from typing import Tuple, Union


class InputTokensSpec(object):
    '''Defines the specs for token inputs.'''

    input_type = 'tokens'

    def __init__(self, seq_len: int, vocab_size: int):
        self.seq_len = seq_len
        self.vocab_size = vocab_size


class Input1dSpec(object):
    '''Defines the specs for 1d inputs.'''

    input_type = '1d'

    def __init__(self, seq_len: int, segment_size: int, in_channels: int):
        self.seq_len = seq_len
        self.segment_size = segment_size
        self.in_channels = in_channels


class Input2dSpec(object):
    '''Defines the specs for 2d inputs.'''

    input_type = '2d'

    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]],
        in_channels: int,
    ):
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels


class Input3dSpec(object):
    '''Defines the specs for 3d inputs.'''

    input_type = '3d'

    def __init__(
        self,
        input_size: Union[int, Tuple[int, int, int]],
        patch_size: Union[int, Tuple[int, int, int]],
        in_channels: int,
    ):
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
