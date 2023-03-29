import math
import os

import pandas as pd
import tensorflow_datasets as tfds
from torch.utils.data import Dataset
from torchvision import transforms

from src.datasets.specs import Input2dSpec


class EurosatParent(Dataset):
    NUM_CLASSES = 10
    INPUT_SIZE = (64, 64)
    PATCH_SIZE = (8, 8)
    IN_CHANNELS = 13
    MAE_OUTPUT_SIZE = 832

    # Proportion of total data used for pretraining.
    PRETRAIN_PROPORTION = 0.9

    # Proportion of total data used for transfer.
    TRANSFER_PROPORTION = 0.1

    # Proportion of data used for training.
    TRAIN_PROPORTION = 0.9

    # Proportion of data used for validation.
    VAL_PROPORTION = 0.1

    RANDOM_SEED = 42

    MEAN = [
        1354.3003, 1117.7579, 1042.2800, 947.6443, 1199.6334, 2001.9829, 2372.5579, 2299.6663, 731.0175, 12.0956, 1822.4083,
        1119.5759, 2598.4456
    ]

    STD = [
        244.0469, 323.4128, 385.0928, 584.1638, 566.0543, 858.5753, 1083.6704, 1103.0342, 402.9594, 4.7207, 1002.4071,
        759.6080, 1228.4104
    ]

    def __init__(self, base_root: str, download: bool = False, train: bool = True):
        self.root = os.path.join(base_root, 'satellite_images')
        self.transforms = transforms.Compose([transforms.ToTensor()])

        if download:
            self.download_dataset(download)

        if not self._is_downloaded():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        eurosat_df = pd.read_pickle(self.pkl_root)

        # Get first 90% of examples for train split, or last 10% of examples for val split.
        if train:
            self.eurosat = eurosat_df.head(math.ceil(self.TRAIN_PROPORTION * len(eurosat_df)))
        else:
            self.eurosat = eurosat_df.tail(math.floor(self.VAL_PROPORTION * len(eurosat_df)))

        self.len = len(self.eurosat)

    def _is_downloaded(self):
        return os.path.exists(self.pkl_root)

    def download_dataset(self, download):
        # https://github.com/tensorflow/datasets/issues/1441: limit the number of open file descriptors
        import resource
        _, high = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

        if self._is_downloaded():
            return
        print('Downloading...')
        tfdata = tfds.load('eurosat/all', data_dir=self.root, split='train', download=download)
        eurosat_df = tfds.as_dataframe(tfdata)

        # Randomize the data.
        eurosat_df = eurosat_df.sample(frac=1, random_state=self.RANDOM_SEED).reset_index(drop=True)

        if self.is_pretraining:
            eurosat_df = eurosat_df.head(math.ceil(self.PRETRAIN_PROPORTION * len(eurosat_df)))
        else:
            eurosat_df = eurosat_df.tail(math.floor(self.TRANSFER_PROPORTION * len(eurosat_df)))

        eurosat_df.to_pickle(self.pkl_root)
        print("Done!")
        return

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        assert (index < self.len)
        row = self.eurosat.iloc[index]
        label = row['label']
        img = row['sentinel2']
        img = ((img - self.MEAN) / self.STD).astype(dtype='float32')
        img = self.transforms(img)

        return index, img, label

    @staticmethod
    def num_classes():
        return EurosatParent.NUM_CLASSES

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(
                input_size=EurosatParent.INPUT_SIZE,
                patch_size=EurosatParent.PATCH_SIZE,
                in_channels=EurosatParent.IN_CHANNELS
            ),
        ]


class EurosatPretrain(EurosatParent):

    def __init__(self, base_root: str, download: bool = False, train: bool = True):
        self.is_pretraining = True
        self.pkl_root = os.path.join(base_root, 'satellite_images', 'pretrain.pkl')
        super().__init__(base_root, download=download, train=train)


class EurosatTransfer(EurosatParent):

    def __init__(self, base_root: str, download: bool = False, train: bool = True):
        self.is_pretraining = False
        self.pkl_root = os.path.join(base_root, 'satellite_images', 'transfer.pkl')
        super().__init__(base_root, download=download, train=train)
