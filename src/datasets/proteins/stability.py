import os

import numpy as np

from src.datasets.proteins.utils import ProteinTransfer


class Stability(ProteinTransfer):
    DATASET_RESOURCES = {
        'raw_data': 'http://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/stability.tar.gz',
        'directory': 'stability',
        'train_json': 'stability_train.json',
        'val_json': 'stability_valid.json'
    }

    def __init__(self, base_root: str, download: bool = False, train: bool = True):
        self.root = os.path.join(base_root, 'proteins')
        super().__init__(download=download, train=train)

    def __getitem__(self, index):
        index, token_tensor, row = super().__getrow__(index)

        # Convert stability score from string to numpyfloat.
        score = np.float32(row['stability_score'][1:-1])
        return index, token_tensor, score

    @staticmethod
    def num_classes():
        return None
