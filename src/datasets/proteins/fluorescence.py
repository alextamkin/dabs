import os

import numpy as np

from src.datasets.proteins.utils import ProteinTransfer


class Fluorescence(ProteinTransfer):
    DATASET_RESOURCES = {
        'raw_data': 'http://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/fluorescence.tar.gz',
        'directory': 'fluorescence',
        'train_json': 'fluorescence_train.json',
        'val_json': 'fluorescence_valid.json'
    }

    def __init__(self, base_root: str, download: bool = False, train: bool = True):
        self.root = os.path.join(base_root, 'proteins')
        super().__init__(download=download, train=train)

    def __getitem__(self, index):
        index, token_tensor, row = super().__getrow__(index)
        # Convert log_fluorescence score from string to np float
        score = np.float32(row['log_fluorescence'][1:-1])
        return index, token_tensor, score

    @staticmethod
    def num_classes():
        return None
