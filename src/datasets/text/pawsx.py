import os

import datasets as ds
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.datasets.specs import InputTokensSpec


class PawsX(Dataset):
    '''The base PAWS-X dataset loaded from Hugging Face

    Args:
        root (str): The location of the dataset files
    '''

    TOKENIZER = 'xlm-roberta-base'
    SEQ_LEN = 128
    # AutoTokenizer.from_pretrained('xlm-roberta-base').vocab_size
    VOCAB_SIZE = 250002
    NUM_CLASSES = 1

    def __init__(self, base_root: str, language: str, download: bool = True, train: bool = True, **kwargs) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'text', 'paws')
        split = 'train' if train else 'validation'
        self.split = split

        self.input_keys = ['sentence1', 'sentence2']
        self.output_keys = ['input_ids']

        # Load dataset and filter by present fields.
        self.dataset = ds.load_dataset(
            'paws-x',
            language,
            split=split,
            cache_dir=self.root,
        )

        # Initialize tokenizer and target fields.
        self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER, cache_dir=self.root)

    def __getitem__(self, index):
        metadata_dict = self.dataset[index]
        metadata_dict.update(
            self.tokenizer(
                # unpack for sentence pairs
                *[metadata_dict[to_tokenize] for to_tokenize in self.input_keys],
                max_length=self.SEQ_LEN,
                padding='max_length',
                truncation=True,
            )
        )
        return (index, *[torch.tensor(metadata_dict[output_key]) for output_key in self.output_keys], metadata_dict['label'])

    @staticmethod
    def num_classes():
        return PawsX.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            InputTokensSpec(seq_len=PawsX.SEQ_LEN, vocab_size=PawsX.VOCAB_SIZE),
        ]

    def __len__(self):
        return len(self.dataset)


class PawsEN(PawsX):

    def __init__(self, base_root: str, download=True, train=True):
        super().__init__(base_root, 'en', train=train)


class PawsFR(PawsX):

    def __init__(self, base_root: str, download=True, train=True):
        super().__init__(base_root, 'fr', train=train)


class PawsES(PawsX):

    def __init__(self, base_root: str, download=True, train=True):
        super().__init__(base_root, 'es', train=train)


class PawsDE(PawsX):

    def __init__(self, base_root: str, download=True, train=True):
        super().__init__(base_root, 'de', train=train)


class PawsZH(PawsX):

    def __init__(self, base_root: str, download=True, train=True):
        super().__init__(base_root, 'zh', train=train)


class PawsJA(PawsX):

    def __init__(self, base_root: str, download=True, train=True):
        super().__init__(base_root, 'ja', train=train)


class PawsKO(PawsX):

    def __init__(self, base_root: str, download=True, train=True):
        super().__init__(base_root, 'ko', train=train)
