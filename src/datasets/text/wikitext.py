import os

import numpy as np

from src.datasets.specs import InputTokensSpec
from src.datasets.text.utils import EnglishHuggingFaceDataset


class WikiText103(EnglishHuggingFaceDataset):
    '''The WikiText dataset loaded from Hugging Face

    Args:
        root (str): The location of the dataset files
    '''

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        self.root = os.path.join(base_root, 'text')
        super().__init__(
            path='wikitext',
            name='wikitext-103-v1',
            input_keys=['text'],
            output_keys=['input_ids'],
            base_root=self.root,
            train=train,
        )

    def __getitem__(self, index):
        index, text = super().__getitem__(index)
        dummy_label = np.array(0)
        return index, text, dummy_label

    @staticmethod
    def num_classes():
        raise NotImplementedError

    @staticmethod
    def spec():
        return [
            InputTokensSpec(seq_len=WikiText103.SEQ_LEN, vocab_size=WikiText103.VOCAB_SIZE),
        ]
