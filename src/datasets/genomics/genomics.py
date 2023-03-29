import math
import os

import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from torch.utils.data import Dataset

from src.datasets.specs import InputTokensSpec


# Abstract class.
class GenomicsBase(Dataset):
    SEQ_LEN = 250
    VOCAB_SIZE = 4

    # Proportion of data used for training.
    TRAIN_PERCENT = 0.9

    # Proportion of data used for validation.
    VAL_PERCENT = 0.1

    # Token representation of genomic bases.
    BASES = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    # The start (inclusive) and end (exclusive) indices of the genomic sequence.
    SEQUENCE_START_IDX = 2
    SEQUENCE_END_IDX = 252

    # The start (inclusive) and end (exclusive) indices of the label and sequence columns in the dataframe.
    DF_START_IDX = 2
    DF_END_IDX = 4

    RANDOM_SEED = 42

    def __init__(self, base_root: str, data_split: str, download: bool = False, train: bool = True):
        self.download_root = os.path.join(base_root, 'genomics')
        if download:
            self.download_dataset(download, data_split)

        if not self._is_downloaded():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        genomics_df = pd.read_csv(self.csv_root)

        # Shuffle the rows of the dataframe.
        genomics_df = genomics_df.sample(frac=1, random_state=self.RANDOM_SEED).reset_index(drop=True)

        # Slice part of dataframe containing label and genomic sequence.
        genomics_df = genomics_df.iloc[:, self.DF_START_IDX:self.DF_END_IDX]

        len_genomics = len(genomics_df)
        if train:
            self.genomics_df = genomics_df.head(math.ceil(self.TRAIN_PERCENT * len_genomics))
        else:
            self.genomics_df = genomics_df.tail(math.floor(self.VAL_PERCENT * len_genomics))

        self.len = len(self.genomics_df)

    def _is_downloaded(self):
        return os.path.exists(self.csv_root)

    def download_dataset(self, download, data_split):
        if self._is_downloaded():
            return
        print('Downloading...')
        # data_split="train" for pretraining
        # data_split="validation" for in-domain transfer
        # data_split="validation_ood" for out-of-domain transfer
        tfdata = tfds.load('genomics_ood', data_dir=self.download_root, split=data_split, download=download)
        genomics_df = tfds.as_dataframe(tfdata)
        genomics_df.to_csv(self.csv_root)
        print("Done!")
        return

    def __getitem__(self, index):
        assert (index < self.len)
        row = self.genomics_df.iloc[index]
        label = row[0]
        if self.ood:
            # The transfer OOD labels are labeled from 10-69. We subtract 10 to make these labels 0-indexed.
            label -= 10
        seq = row[1]
        seq = seq[self.SEQUENCE_START_IDX:self.SEQUENCE_END_IDX]
        tokens = []
        for base in seq:
            if base not in self.BASES:
                raise RuntimeError('Genomic sequence contains the base "' + base + '" which is not one of: A, C, G, T.')
            tokens.append(self.BASES[base])
        tokens = np.array(tokens)
        return index, tokens, label

    def __len__(self):
        return self.len

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            InputTokensSpec(seq_len=GenomicsBase.SEQ_LEN, vocab_size=GenomicsBase.VOCAB_SIZE),
        ]


class GenomicsPretrain(GenomicsBase):
    NUM_CLASSES = 10

    def __init__(self, base_root: str, download: bool = False, train: bool = True):
        self.ood = False
        self.csv_root = os.path.join(base_root, 'genomics', 'genomicsPretrain.csv')
        super().__init__(base_root, data_split="train", download=download, train=train)

    @staticmethod
    def num_classes():
        return GenomicsPretrain.NUM_CLASSES


class GenomicsTransferID(GenomicsBase):
    NUM_CLASSES = 10

    def __init__(self, base_root: str, download: bool = False, train: bool = True):
        self.ood = False
        self.csv_root = os.path.join(base_root, 'genomics', 'genomicsTransferID.csv')
        super().__init__(base_root, data_split="validation", download=download, train=train)

    @staticmethod
    def num_classes():
        return GenomicsTransferID.NUM_CLASSES


class GenomicsTransferOOD(GenomicsBase):
    NUM_CLASSES = 60

    def __init__(self, base_root: str, download: bool = False, train: bool = True):
        self.ood = True
        self.csv_root = os.path.join(base_root, 'genomics', 'genomicsTransferOOD.csv')
        # Use validation_ood split for out-of-domain transfer task.
        super().__init__(base_root, data_split="validation_ood", download=download, train=train)

    @staticmethod
    def num_classes():
        return GenomicsTransferOOD.NUM_CLASSES
