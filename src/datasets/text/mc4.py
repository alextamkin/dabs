import os

import datasets as ds
import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

from src.datasets.specs import InputTokensSpec


class MC4(IterableDataset):
    '''The base GLUE dataset loaded from Hugging Face

    Args:
        root (str): The location of the dataset files
    '''

    TOKENIZER = 'xlm-roberta-base'
    SEQ_LEN = 128
    VOCAB_SIZE = 250002  # AutoTokenizer.from_pretrained('xlm-roberta-base').vocab_size

    def __init__(self, base_root: str, train: bool = True, **kwargs) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'text', 'mc4')
        split = 'train' if train else 'validation'
        self.split = split

        input_keys = ['text']
        self.input_keys = input_keys
        self.output_keys = ['input_ids']

        # Load dataset and filter by present fields.
        data_en = ds.load_dataset(
            'mc4',
            languages=['en'],
            split=split,
            cache_dir=self.root,
            streaming=True,
        )
        data_es = ds.load_dataset(
            'mc4',
            languages=['es'],
            split=split,
            cache_dir=self.root,
            streaming=True,
        )
        data_fr = ds.load_dataset(
            'mc4',
            languages=['fr'],
            split=split,
            cache_dir=self.root,
            streaming=True,
        )
        data_de = ds.load_dataset(
            'mc4',
            languages=['de'],
            split=split,
            cache_dir=self.root,
            streaming=True,
        )
        data_zh = ds.load_dataset(
            'mc4',
            languages=['zh'],
            split=split,
            cache_dir=self.root,
            streaming=True,
        )
        data_ja = ds.load_dataset(
            'mc4',
            languages=['ja'],
            split=split,
            cache_dir=self.root,
            streaming=True,
        )
        data_ko = ds.load_dataset(
            'mc4',
            languages=['ko'],
            split=split,
            cache_dir=self.root,
            streaming=True,
        )

        # Initialize tokenizer and target fields.
        self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER, cache_dir=self.root)
        self.dataset = ds.interleave_datasets([data_en, data_fr, data_de, data_zh, data_es, data_ja, data_ko])

    def __iter__(self):
        if self.split == 'train':
            self.dataset.shuffle(buffer_size=10000, seed=42)

        dummy_label = 0
        index = 0
        for metadata_dict in self.dataset:
            metadata_dict.update(
                self.tokenizer(
                    *[metadata_dict[to_tokenize] for to_tokenize in self.input_keys],  # unpack for sentence pairs
                    max_length=self.SEQ_LEN,
                    padding='max_length',
                    truncation=True,
                )
            )
            yield (index, *[torch.tensor(metadata_dict[output_key]) for output_key in self.output_keys], dummy_label)
            index += 1

    @staticmethod
    def num_classes():
        raise NotImplementedError

    @staticmethod
    def spec():
        return [
            InputTokensSpec(seq_len=MC4.SEQ_LEN, vocab_size=MC4.VOCAB_SIZE),
        ]
