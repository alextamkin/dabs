import os
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets.utils import extract_archive
from torchvision.datasets.vision import VisionDataset

from src.datasets.specs import Input2dSpec

# From DATASET_ROOT/chexpert/CheXpert-v1.0-small/valid.csv
CHEXPERT_LABELS = {
    'No Finding': 0,
    'Enlarged Cardiomediastinum': 1,
    'Cardiomegaly': 2,
    'Lung Opacity': 3,
    'Lung Lesion': 4,
    'Edema': 5,
    'Consolidation': 6,
    'Pneumonia': 7,
    'Atelectasis': 8,
    'Pneumothorax': 9,
    'Pleural Effusion': 10,
    'Pleural Other': 11,
    'Fracture': 12,
    'Support Devices': 13,
}


def any_exist(files):
    return any(map(os.path.exists, files))


class CheXpert(VisionDataset):
    '''A dataset class for the CheXpert dataset (https://stanfordmlgroup.github.io/competitions/chexpert/).
    Note that you must register and manually download the data to use this dataset.
    '''
    # Dataset information.
    TRANSFORMS = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    LABELS_COL = 5

    # From https://arxiv.org/abs/1901.07031 (Irvin et al. 2019)
    CHEXPERT_LABELS_IDX = np.array(
        [
            CHEXPERT_LABELS['Atelectasis'],
            CHEXPERT_LABELS['Cardiomegaly'],
            CHEXPERT_LABELS['Consolidation'],
            CHEXPERT_LABELS['Edema'],
            CHEXPERT_LABELS['Pleural Effusion'],
        ],
        dtype=np.int32
    )

    NUM_CLASSES = 5  # 14 total, but we select 5: len(self.CHEXPERT_LABELS_IDX)
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 3

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        self.root = os.path.join(base_root, 'medical_images', 'chexpert')
        super().__init__(self.root)
        self.index_location = self.find_data()
        self.split = 'train' if train else 'valid'
        self.build_index()

    def find_data(self):
        os.makedirs(self.root, exist_ok=True)
        components = list(map(lambda x: os.path.join(self.root, 'CheXpert-v1.0' + x), ['', '-small', '.zip', '-small.zip']))
        # if no data is present, prompt the user to download it
        if not any_exist(components):
            raise RuntimeError(
                """
                'Visit https://stanfordmlgroup.github.io/competitions/chexpert/ to download the data'
                'Once you receive the download links, place the zip file in {}'.format(self.root)
                'To maintain compatibility with the paper baselines, download the sampled version (CheXpert-v1.0-small).'
                """
            )

        # if the data has not been extracted, extract the data, prioritizing the full-res dataset
        if not any_exist(components[:2]):
            for i in (2, 3):
                if os.path.exists(components[i]):
                    print('Extracting data...')
                    extract_archive(components[i])
                    print('Done')
                    break

        # return the data folder, prioritizing the full-res dataset
        for i in (0, 1):
            if os.path.exists(components[i]):
                return components[i]
        raise FileNotFoundError('CheXpert data not found')

    def build_index(self):
        print('Building index...')
        index_file = os.path.join(self.index_location, self.split + '.csv')
        self.fnames = np.loadtxt(index_file, dtype=np.str, delimiter=',', skiprows=1, usecols=0)

        end_col = self.LABELS_COL + len(CHEXPERT_LABELS)
        # missing values occur when no comment is made on a particular diagnosis. we treat this as a negative diagnosis
        self.labels = np.genfromtxt(
            index_file,
            dtype=np.float,
            delimiter=',',
            skip_header=1,
            usecols=range(self.LABELS_COL, end_col),
            missing_values='',
            filling_values=0,
        )
        self.labels = np.maximum(self.labels, 0)  # convert -1 (unknown) to 0
        print('Done')

    def __len__(self) -> int:
        return self.fnames.shape[0]

    def __getitem__(self, index: int) -> Any:
        fname = self.fnames[index]
        image = Image.open(os.path.join(self.root, fname)).convert('RGB')
        image = self.TRANSFORMS(image)
        label = torch.tensor(self.labels[index][self.CHEXPERT_LABELS_IDX]).long()
        return index, image.float(), label

    @staticmethod
    def num_classes():
        return CheXpert.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(input_size=CheXpert.INPUT_SIZE, patch_size=CheXpert.PATCH_SIZE, in_channels=CheXpert.IN_CHANNELS),
        ]


def open_folder(path: str):
    '''Opens a folder in the file explorer. Attempts to be platform-independent

    Args:
        path (str): The folder path to be opened in a file explorer
    '''
    try:
        if hasattr(os, 'startfile'):
            os.startfile(path)
        elif sys.platform == 'darwin':
            os.system('open ' + path)
        else:
            os.system('xdg-open ' + path)
    except:  # noqa E722
        pass
