import os

from src.datasets.proteins.utils import ProteinTransfer


class SCOP(ProteinTransfer):
    NUM_CLASSES = 1195
    DATASET_RESOURCES = {
        'raw_data': 'http://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/remote_homology.tar.gz',
        'directory': 'remote_homology',
        'train_json': 'remote_homology_valid.json',
        'val_json': 'remote_homology_valid.json'
    }

    def __init__(self, base_root: str, download: bool = False, train: bool = True):
        self.root = os.path.join(base_root, 'proteins')
        super().__init__(download=download, train=train)

    def __getitem__(self, index):
        index, token_tensor, row = super().__getrow__(index)
        return index, token_tensor, row['fold_label']

    @staticmethod
    def num_classes():
        return SCOP.NUM_CLASSES
