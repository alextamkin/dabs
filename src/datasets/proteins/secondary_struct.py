import ast
import os

import torch

from src.datasets.proteins.utils import ProteinTransfer


class SecondaryStructure(ProteinTransfer):
    NUM_CLASSES = 4
    PADDING_IDX = 3
    DATASET_RESOURCES = {
        'raw_data': 'http://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/secondary_structure.tar.gz',
        'directory': 'secondary_structure',
        'train_json': 'secondary_structure_train.json',
        'val_json': 'secondary_structure_valid.json'
    }

    def __init__(self, base_root: str, download: bool = False, train: bool = True):
        self.root = os.path.join(base_root, 'proteins')
        super().__init__(download=download, train=train)

    def __getitem__(self, index):
        index, token_tensor, row = super().__getrow__(index)

        # Convert secondary structure representation from string to list, with max length of SEQ_LEN
        secondary_struct_seq = (ast.literal_eval(row['ss3']))[:self.SEQ_LEN]
        for _ in range(self.SEQ_LEN - len(secondary_struct_seq)):
            secondary_struct_seq.append(self.PADDING_IDX)
        secondary_struct_seq = torch.tensor(secondary_struct_seq)
        return index, token_tensor, secondary_struct_seq

    @staticmethod
    def num_classes():
        return SecondaryStructure.NUM_CLASSES
