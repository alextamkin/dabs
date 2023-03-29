import math
import os

import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import torch
from torch.utils.data import Dataset

from src.datasets.specs import InputTabularSpec


class HiggsParent(Dataset):
    NUM_CLASSES = 1
    NUM_FEATURES = 28

    # Proportion of total data used for pretraining.
    PRETRAIN_PROPORTION = 0.9

    # Proportion of total data used for transfer.
    TRANSFER_PROPORTION = 0.1

    # Proportion of data used for training.
    TRAIN_PROPORTION = 0.9

    # Proportion of data used for validation.
    VAL_PROPORTION = 0.1

    # The start (inclusive) index of the label and feature columns of the data frame.
    DF_START_IDX = 1

    RANDOM_SEED = 42

    def __init__(self, base_root: str, download: bool = False, train: bool = True):
        self.root = os.path.join(base_root, 'particle_physics')
        if download:
            self.download_dataset(download)

        if not self._is_downloaded():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        higgs_df = pd.read_csv(self.csv_root)

        # Shuffle the rows of the dataframe.
        higgs_df = higgs_df.sample(frac=1, random_state=self.RANDOM_SEED).reset_index(drop=True)

        higgs_df = higgs_df.iloc[:, self.DF_START_IDX:]

        # Get first 90% of examples for train split, or last 10% of examples for val split.
        if train:
            self.higgs_df = higgs_df.head(math.ceil(self.TRAIN_PROPORTION * len(higgs_df)))
        else:
            self.higgs_df = higgs_df.tail(math.floor(self.VAL_PROPORTION * len(higgs_df)))

        self.len = len(self.higgs_df)

    def _is_downloaded(self):
        return os.path.exists(self.csv_root)

    def download_dataset(self, download):
        # https://github.com/tensorflow/datasets/issues/1441: limit the number of open file descriptors
        import resource
        _, high = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

        if self._is_downloaded():
            return
        print('Downloading...')
        tfdata = tfds.load('higgs', data_dir=self.root, split='train', download=download)
        higgs = tfds.as_dataframe(tfdata)
        if self.is_pretraining:
            higgs = higgs.head(math.ceil(self.PRETRAIN_PROPORTION * len(higgs)))
        else:
            higgs = higgs.tail(math.floor(self.TRANSFER_PROPORTION * len(higgs)))
        higgs.to_csv(self.csv_root)
        print("Done!")
        return

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        assert (index < self.len)
        row = self.higgs_df.iloc[index]
        label = int(row[0])

        features = torch.tensor(row[1:])
        features = features.view(self.NUM_FEATURES, 1)
        return index, features.float(), np.array([label])

    @staticmethod
    def num_classes():
        return HiggsParent.NUM_CLASSES

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            InputTabularSpec(num_features=HiggsParent.NUM_FEATURES),
        ]


class HiggsPretrain(HiggsParent):

    def __init__(self, base_root: str, download: bool = False, train: bool = True):
        self.is_pretraining = True
        self.csv_root = os.path.join(base_root, 'particle_physics', 'higgsPretrain.csv')
        super().__init__(base_root, download=download, train=train)


class HiggsTransfer(HiggsParent):

    def __init__(self, base_root: str, download: bool = False, train: bool = True):
        self.is_pretraining = False
        self.csv_root = os.path.join(base_root, 'particle_physics', 'higgsTransfer.csv')
        super().__init__(base_root, download=download, train=train)
