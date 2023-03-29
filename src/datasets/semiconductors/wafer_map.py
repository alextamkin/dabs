import math
import os
import pickle
from pathlib import Path

import mat73
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.datasets.natural_images.utils import download_and_extract_archive
from src.datasets.specs import Input2dSpec


class WaferMap(Dataset):
    NUM_CLASSES = 9
    INPUT_SIZE = (32, 32)
    PATCH_SIZE = (4, 4)
    IN_CHANNELS = 3
    MAE_OUTPUT_SIZE = 48

    RANDOM_SEED = 42
    NUM_EXAMPLES = 811457

    # Multiply against all-zero, all-one, and all-two vectors to make white, gray, and black pixel representations.
    GRAYSCALE_MULTIPLIER = 127.5
    TRAIN_PROPORTION = 0.9
    VAL_PROPORTION = 0.1
    WM_RESOURCES = {'url': "http://mirlab.org/dataSet/public/MIR-WM811K.zip"}
    FAILURE_MAP = {
        'unlabeled': 0,
        'none': 0,
        'random': 1,
        'donut': 2,
        'scratch': 3,
        'center': 4,
        'loc': 5,
        'edge-loc': 6,
        'edge-ring': 7,
        'near-full': 8
    }

    def __init__(self, base_root: str, pkl_file: str, download: bool = False, train: bool = True) -> None:
        self.root = os.path.join(base_root, 'semiconductors')
        self.transforms = transforms.Compose([transforms.Resize(self.INPUT_SIZE), transforms.ToTensor()])

        if download:
            self.download_dataset()

        if not self._is_downloaded():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        data = pd.read_pickle(os.path.join(self.root, pkl_file))
        data = data.sample(frac=1, random_state=self.RANDOM_SEED).reset_index(drop=True)
        if train:
            self.data = data.head(math.ceil(self.TRAIN_PROPORTION * len(data)))
        else:
            self.data = data.tail(math.floor(self.VAL_PROPORTION * len(data)))

        self.len = len(self.data)

    def _is_downloaded(self) -> bool:
        return os.path.exists(os.path.join(self.root, 'labeled.pkl')
                             ) and os.path.exists(os.path.join(self.root, 'unlabeled.pkl'))

    def download_dataset(self):

        if self._is_downloaded():
            return
        # download and extract files
        print('Downloading wafer map')
        if not (os.path.exists(self.root)):
            Path(self.root).mkdir(parents=True, exist_ok=True)

        wafer_mat_file = os.path.join(self.root, 'MIR-WM811K', 'MATLAB', 'WM811K.mat')

        if not os.path.exists(wafer_mat_file):
            print('Downloading and extracting file...')
            download_and_extract_archive(url=self.WM_RESOURCES['url'], download_root=self.root)

        print('Finished downloading and extracting file')

        full_pkl_path = os.path.join(self.root, 'WM811.pkl')

        if not os.path.exists(full_pkl_path):
            print("Reading .mat file...")
            data_dict = mat73.loadmat(wafer_mat_file)
            data_dict = data_dict['data']
            file_write = open(full_pkl_path, "wb")
            pickle.dump(data_dict, file_write)

        file_read = open(full_pkl_path, "rb")

        data_dict = pickle.load(file_read)
        print("Finished reading .mat file")
        for key in data_dict.keys():
            assert len(
                data_dict[key]
            ) == self.NUM_EXAMPLES, f'The data dictionary is corrupted. Delete {os.path.join(self.root, "WM811.pkl")} and restart.'

        data = pd.DataFrame.from_dict(data_dict)

        data = data[['failureType', 'waferMap']]

        data_unlabeled = []
        data_labeled = []
        print("Processing data dictionary.")
        for _, row in data.iterrows():
            # Unnest data from within a 1-element list.
            row['waferMap'] = row['waferMap'][0]
            row['failureType'] = row['failureType'][0]

            if type(row['failureType']) != str:
                row['failureType'] = 'unlabeled'
            row['failureType'] = (row['failureType']).lower()

            # Create gray-scale pixel represenation of wafer map.
            pixels = np.expand_dims(row['waferMap'], axis=2)
            pixels = np.repeat(pixels, 3, axis=2) * self.GRAYSCALE_MULTIPLIER
            pixels = pixels.astype(int)
            row['pixels'] = pixels
            if row['failureType'] == 'unlabeled':
                data_unlabeled.append(row)
            else:
                data_labeled.append(row)

        df_unlabeled = pd.DataFrame(data_unlabeled)
        df_labeled = pd.DataFrame(data_labeled)
        df_unlabeled = df_unlabeled[['failureType', 'pixels']]
        df_labeled = df_labeled[['failureType', 'pixels']]
        df_unlabeled.to_pickle(os.path.join(self.root, 'unlabeled.pkl'))
        df_labeled.to_pickle(os.path.join(self.root, 'labeled.pkl'))

        print("Done!")

    def __getitem__(self, index):

        row = self.data.iloc[index]
        img, label = row['pixels'], self.FAILURE_MAP[row['failureType']]
        img = img.astype('uint8')
        img = Image.fromarray(img).convert('RGB')
        img = self.transforms(img)
        return index, img, label

    def __len__(self):
        return self.len

    @staticmethod
    def num_classes():
        return WaferMap.NUM_CLASSES

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(input_size=WaferMap.INPUT_SIZE, patch_size=WaferMap.PATCH_SIZE, in_channels=WaferMap.IN_CHANNELS),
        ]


class WaferMapPretrain(WaferMap):

    def __init__(self, base_root: str, download: bool = False, train: bool = True):
        super().__init__(base_root, pkl_file='unlabeled.pkl', download=download, train=train)


class WaferMapTransfer(WaferMap):

    def __init__(self, base_root: str, download: bool = False, train: bool = True):
        super().__init__(base_root, pkl_file='labeled.pkl', download=download, train=train)
