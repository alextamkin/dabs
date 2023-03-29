import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.datasets.specs import InputTokensSpec
from src.datasets.speech.utils import download_and_extract_archive


# Abstract class.
class ProteinTransfer(Dataset):
    SEQ_LEN = 128
    VOCAB_SIZE = 26
    RANDOM_SEED = 42

    def __init__(self, download: bool = False, train: bool = True):
        if train:
            self.csv_root = os.path.join(self.root, self.DATASET_RESOURCES['directory'] + '_train.csv')
        else:
            self.csv_root = os.path.join(self.root, self.DATASET_RESOURCES['directory'] + '_valid.csv')
        if download:
            self.download_dataset(train)

        if not self._is_downloaded():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.data = pd.read_csv(self.csv_root)

        self.len = len(self.data)

        # Map each amino acid to a numerical value. These values will comprise elements of the feature vectors of a protein.
        amino_acids = "XARNDCQEGHILKMFPSTWYVUOBZJ"
        self.amino_acid_map = {}
        for i in range(len(amino_acids)):
            amino_acid = amino_acids[i]
            self.amino_acid_map[amino_acid] = i

    def _is_downloaded(self):
        return os.path.exists(self.csv_root)

    def download_dataset(self, train):
        if self._is_downloaded():
            return

        # Download SCOP architecture if scop not already downloaded.
        if not (os.path.exists(os.path.join(self.root, self.DATASET_RESOURCES['directory'], self.DATASET_RESOURCES['val_json'])
                              ) and os.path.exists(os.path.join(self.root, self.DATASET_RESOURCES['directory'],
                                                                self.DATASET_RESOURCES['train_json']))):
            print("Downloading raw data.")
            if not (os.path.exists(self.root)):
                os.makedirs(self.root)
            download_and_extract_archive(url=self.DATASET_RESOURCES['raw_data'], download_root=self.root)

        if not (os.path.exists(self.csv_root)):
            if train:
                print("Converting training set json to csv")
                data_split = pd.read_json(
                    os.path.join(self.root, self.DATASET_RESOURCES['directory'], self.DATASET_RESOURCES['train_json'])
                )
                done_message = "Finished converting training set json to csv"
            else:
                print("Converting validation set json to csv")
                data_split = pd.read_json(
                    os.path.join(self.root, self.DATASET_RESOURCES['directory'], self.DATASET_RESOURCES['val_json'])
                )
                done_message = "Finished converting validation set json to csv"
            # Randomize examples in split.
            data_split = data_split.sample(frac=1, random_state=self.RANDOM_SEED).reset_index(drop=True)
            data_split.to_csv(self.csv_root)
            print(done_message)

    def __len__(self):
        return self.len

    def __getrow__(self, index):
        assert (index < self.len)
        row = self.data.iloc[index]
        seq = row['primary']
        # Make all protein sequence lengths equal to 128, and pad sequence with 'X' amimo acid
        # if original protein sequence's length is less than 128 ('X' is essentially a pad token).
        seq = (seq[:self.SEQ_LEN]).ljust(self.SEQ_LEN, 'X')
        tokens = []

        # Build feature vector from amino acid to integer map created in __init__ .
        for amino_acid in seq:
            if amino_acid not in self.amino_acid_map:
                raise Exception("Protein sequence contains an invalid amino acid notation: " + amino_acid)
            tokens.append(self.amino_acid_map[amino_acid])
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        return index, token_tensor, row

    @staticmethod
    def num_classes():
        raise NotImplementedError  # each subclass should overwrite this

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            InputTokensSpec(seq_len=ProteinTransfer.SEQ_LEN, vocab_size=ProteinTransfer.VOCAB_SIZE),
        ]
