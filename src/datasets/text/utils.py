from typing import Iterable

import datasets as ds
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class EnglishHuggingFaceDataset(Dataset):
    '''A base dataset class used to provide data from Hugging Face datasets.

    Args:
        path (str): Path of the Hugging Face dataset to load
        name (str): Name of the Hugging Face dataset to load
        input_keys (Iterable[str]): The fields of data that contain text to be tokenized
        output_keys (Iterable[str]): The fields of tokenized output to be returned from the dataset
        root (str): Path to the dataset files
        train (bool, optional): Whether to load the train or test split. Defaults to True.
    '''

    TOKENIZER = 'bert-base-uncased'
    SEQ_LEN = 128
    VOCAB_SIZE = 35022  # AutoTokenizer.from_pretrained('bert-base-uncased').vocab_size

    def __init__(
        self,
        path: str,
        name: str,
        input_keys: Iterable[str],
        output_keys: Iterable[str],
        base_root: str,
        train: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        split = 'train' if train else 'validation'

        # Load dataset and filter by present fields.
        self.data = ds.load_dataset(
            path,
            name,
            split=split,
            cache_dir=base_root,
            **kwargs,
        )
        self.data = self.data.filter(lambda ds: all(len(ds[keys]) > 0 for keys in input_keys))

        # Initialize tokenizer and target fields.
        self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER, cache_dir=base_root)
        self.input_keys = input_keys
        self.output_keys = output_keys

    def __getitem__(self, index):
        metadata_dict = self.data[index]
        metadata_dict.update(
            self.tokenizer(
                *[metadata_dict[to_tokenize] for to_tokenize in self.input_keys],  # unpack for sentence pairs
                max_length=self.SEQ_LEN,
                padding='max_length',
                truncation=True,
            )
        )
        return (index, *[torch.tensor(metadata_dict[label]) for label in self.output_keys])

    def __len__(self):
        return len(self.data)
