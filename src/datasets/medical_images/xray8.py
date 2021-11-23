import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive, download_url
from torchvision.datasets.vision import VisionDataset

from src.datasets.specs import Input2dSpec

CHEST_XRAY_8_LINKS = [
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
    'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
    'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
    'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
    'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
    'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
    'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
    'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
    'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
    'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
    'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
    'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
]
CHEST_XRAY_8_METADATA_LINK = 'https://nihcc.app.box.com/v/ChestXray-NIHCC/file/219760887468'


class ChestXray8(VisionDataset):
    """A dataset class for ChestX-ray8 (https://arxiv.org/abs/1705.02315).
    """
    TRANSFORMS = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    NUM_CLASSES = 8
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 3

    PATHOLOGIES = [
        'Atelectasis',
        'Cardiomegaly',
        'Effusion',
        'Infiltration',
        'Mass',
        'Nodule',
        'Pneumonia',
        'Pneumothorax',
    ]  # We use the 8 original pathologies from the paper.

    def __init__(self, base_root: str, download: bool = False, train=True, **kwargs) -> None:
        self.root = os.path.join(base_root, 'medical_images', 'xray8')
        super().__init__(self.root)
        self.image_root = os.path.join(self.root, 'images')
        if download:
            self.download()
        self.metadata = pd.read_csv(os.path.join(self.root, 'metadata.csv'))

        def is_png(fname):
            return fname[-4:] == '.png'

        self.fnames = list(filter(is_png, os.listdir(self.image_root)))
        self.fnames = self.random_split(self.fnames, train=train)

    def __len__(self) -> int:
        return len(self.fnames)

    def random_split(self, fnames, train=True, train_frac=0.9):
        fnames = np.array(fnames)
        rs = np.random.RandomState(42)
        rs.shuffle(fnames)
        split_idx = int(len(fnames) * train_frac)
        if train:
            ret = fnames[:split_idx]
        else:
            ret = fnames[split_idx:]
        return ret

    def __getitem__(self, index: int) -> Any:
        '''Images are 2D RGB square images, labels are 8-dimensional binary tensors.'''
        image = Image.open(os.path.join(self.image_root, self.fnames[index])).convert('RGB')
        # Label strings look like "Cardiomegaly|Effusion|Hernia"
        label_str = self.metadata[self.metadata['Image Index'] == self.fnames[index]]['Finding Labels'].item()
        labels = torch.tensor([1 if pathology in label_str else 0 for pathology in self.PATHOLOGIES])
        return index, self.TRANSFORMS(image), labels

    def download(self):
        if os.path.isdir(self.image_root):
            print('Images already downloaded')
            return
        os.mkdir(self.root)
        print('Downloading images...')
        for i, link in enumerate(CHEST_XRAY_8_LINKS):
            download_and_extract_archive(link, download_root=self.root, filename='chestxray8_images_{}.tar.gz'.format(i))
        download_url(CHEST_XRAY_8_METADATA_LINK, self.root, filename='metadata.csv')

    @staticmethod
    def spec():
        return [
            Input2dSpec(
                input_size=ChestXray8.INPUT_SIZE, patch_size=ChestXray8.PATCH_SIZE, in_channels=ChestXray8.IN_CHANNELS
            )
        ]

    @staticmethod
    def num_classes():
        return ChestXray8.NUM_CLASSES
