import math
import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.datasets.specs import InputTokensSpec
from src.datasets.speech.utils import download_and_extract_archive

PFAM_RESOURCES = {'pfam': 'http://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/pfam.tar.gz'}


class Pfam(Dataset):
    SEQ_LEN = 128
    VOCAB_SIZE = 26

    NUM_CLASSES = 623

    # Number of examples used for transfer.
    NUM_TRANSFER_EXAMPLES = 200000

    # Proportion of data used for training.
    TRAIN_PROPORTION = 0.9

    # Proportion of data used for validation.
    VAL_PROPORTION = 0.1

    # The start (inclusive) index of the label and feature columns of the data frame.
    DF_START_IDX = 1

    RANDOM_SEED = 42

    def __init__(self, base_root: str, download: bool = False, train: bool = True):
        self.root = os.path.join(base_root, 'proteins')
        if download:
            self.download_dataset()

        if not self._is_downloaded():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        # Get proper root for csv file containing examples depending on whether we're pretraining/transfer and train/val.
        csv_root = None
        if self.is_pretraining and train:
            csv_root = self.pretrain_train_root
        elif self.is_pretraining and not train:
            csv_root = self.pretrain_val_root
        else:
            csv_root = self.transfer_root

        self.pfam_df = pd.read_csv(csv_root)
        if not self.is_pretraining:
            if train:
                self.pfam_df = self.pfam_df.head(math.ceil(self.TRAIN_PROPORTION * len(self.pfam_df)))
            else:
                self.pfam_df = self.pfam_df.tail(math.floor(self.VAL_PROPORTION * len(self.pfam_df)))

        # Remove index from data frame.
        self.pfam_df = self.pfam_df.iloc[:, self.DF_START_IDX:]

        self.len = len(self.pfam_df)

        # Map each amino acid to a numerical value. These values will comprise elements of the feature vectors of a protein.
        amino_acids = "XARNDCQEGHILKMFPSTWYVUOBZJ"
        self.amino_acid_map = {}
        for i in range(len(amino_acids)):
            amino_acid = amino_acids[i]
            self.amino_acid_map[amino_acid] = i

        if self.is_pretraining and train:
            pfam_pretrain = self.pfam_df
        else:
            pfam_pretrain = pd.read_csv(self.pretrain_train_root)

        clans = set()
        for clan in pfam_pretrain['clan']:
            clans.add(clan)
        clans = list(clans)
        clans.sort()

        # Renumber the clans to account for missing clans and avoid indexing errors.
        self.label_map = {}
        for i in range(len(clans)):
            self.label_map[clans[i]] = i

    def _is_downloaded(self):
        if self.is_pretraining:
            return os.path.exists(self.pretrain_train_root) and os.path.exists(self.pretrain_val_root)
        else:
            return os.path.exists(self.transfer_root)

    def download_dataset(self):
        if self._is_downloaded():
            return

        # Download pfam dataset if pfam not already downloaded.
        if not (os.path.exists(os.path.join(self.root, 'pfam', 'pfam_valid.json')) and
                os.path.exists(os.path.join(self.root, 'pfam', 'pfam_train.json'))):
            print("Downloading main pfam architecture")
            if not (os.path.exists(self.root)):
                os.makedirs(self.root)
            download_and_extract_archive(url=PFAM_RESOURCES['pfam'], download_root=self.root)

        # Always download the pfam_train set since it's used for both pretraining and transfer in __init__.
        if not os.path.exists(self.pretrain_train_root):
            print("Downloading pfam_train.json")
            pfam_train = pd.read_json(os.path.join(self.root, 'pfam', 'pfam_train.json'))
            # Randomize examples in data.
            pfam_train = pfam_train.sample(frac=1, random_state=self.RANDOM_SEED).reset_index(drop=True)
            pfam_train.to_csv(self.pretrain_train_root)
            print("Finished downloading pfam_train.json")

        # Download the pfam_val data if we're pretraining and don't have the pretraining validation set, or we're transfering and don't have the transfer dataset.
        if not ((self.is_pretraining and os.path.exists(self.pretrain_val_root)) or
                (self.is_pretraining == False and os.path.exists(self.transfer_root))):
            print("Downloading pfam_val.json")
            pfam_val = pd.read_json(os.path.join(self.root, 'pfam', 'pfam_valid.json'))
            # Randomize examples in data.
            pfam_val = pfam_val.sample(frac=1, random_state=self.RANDOM_SEED).reset_index(drop=True)
            if self.is_pretraining:
                pfam_val = pfam_val.head(len(pfam_val) - self.NUM_TRANSFER_EXAMPLES)
                pfam_val.to_csv(self.pretrain_val_root)
                print("Finished downloading pfam_val.json for pretrain")
            else:
                pfam_val = pfam_val.tail(self.NUM_TRANSFER_EXAMPLES)
                pfam_val.to_csv(self.transfer_root)
                print("Finished downloading pfam_val.json for transfer")
        print("Done")

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        assert (index < self.len)
        row = self.pfam_df.iloc[index]
        seq = row['primary']
        # Make all protein sequence length's equal to 128, and pad sequence with 'X' amimo acid if original protein sequence's length is less than 128.
        seq = (seq[:self.SEQ_LEN]).ljust(self.SEQ_LEN, 'X')
        tokens = []

        # Build feature vector from amino acid to integer map created in __init__ .
        for amino_acid in seq:
            if amino_acid not in self.amino_acid_map:
                raise Exception("Protein sequence contains an invalid amino acid notation: " + amino_acid)
            tokens.append(self.amino_acid_map[amino_acid])
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        clan = row['clan']
        if clan not in self.label_map:
            raise Exception("The protein's clan is not represented in the pretraining dataset: " + str(clan))
        label = self.label_map[clan]
        return index, token_tensor, label

    @staticmethod
    def num_classes():
        return Pfam.NUM_CLASSES

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            InputTokensSpec(seq_len=Pfam.SEQ_LEN, vocab_size=Pfam.VOCAB_SIZE),
        ]


class PfamPretrain(Pfam):

    def __init__(self, base_root: str, download: bool = False, train: bool = True):
        self.is_pretraining = True
        self.pretrain_train_root = os.path.join(base_root, 'proteins', 'pfam_pretrain_train.csv')
        self.pretrain_val_root = os.path.join(base_root, 'proteins', 'pfam_pretrain_val.csv')
        super().__init__(base_root, download=download, train=train)


class PfamTransfer(Pfam):

    def __init__(self, base_root: str, download: bool = False, train: bool = True):
        self.is_pretraining = False
        self.pretrain_train_root = os.path.join(base_root, 'proteins', 'pfam_pretrain_train.csv')
        self.transfer_root = os.path.join(base_root, 'proteins', 'pfam_transfer.csv')
        super().__init__(base_root, download=download, train=train)
