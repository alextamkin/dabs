import os
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms

from src.datasets.natural_images.utils import download_and_extract_archive
from src.datasets.specs import Input2dSpec

VGGFLOWER_RESOURCES = {
    'vggflower':
        {
            'images': 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz',
            'labels': 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat',
        }
}


class VGGFlower(Dataset):
    # Dataset information.
    NUM_CLASSES = 102
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 3

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.train = train
        self.root = os.path.join(base_root, 'natural_images', 'flowers')
        self.transforms = transforms.Compose(
            [transforms.Resize(self.INPUT_SIZE),
             transforms.CenterCrop(self.INPUT_SIZE),
             transforms.ToTensor()]
        )

        if download:
            self.download_dataset()

        if not self._is_downloaded():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        paths, labels = self.load_images()
        self.paths, self.labels = paths, labels

    def _is_downloaded(self) -> bool:
        return (os.path.exists(self.root))

    def download_dataset(self):
        '''Download the meta dataset if not exists already'''

        if self._is_downloaded():
            return

        # download and extract files
        print('Downloading and Extracting...')

        for _, urls in VGGFLOWER_RESOURCES['vggflower'].items():
            if type(urls) != str and len(urls) > 1:
                for url in urls.values():
                    filename = url.rpartition('/')[2]
                    download_and_extract_archive(url, download_root=self.root, filename=filename)

            else:
                filename = urls.rpartition('/')[2]
                download_and_extract_archive(urls, download_root=self.root, filename=filename)

        print('Done!')

    def load_images(self):
        rs = np.random.RandomState(42)
        imagelabels_path = os.path.join(self.root, 'imagelabels.mat')
        with open(imagelabels_path, 'rb') as f:
            labels = loadmat(f)['labels'][0]

        all_filepaths = defaultdict(list)
        for i, label in enumerate(labels):
            all_filepaths[label].append(os.path.join(self.root, 'jpg', 'image_{:05d}.jpg'.format(i + 1)))
        # train test split
        split_filepaths, split_labels = [], []
        for label, paths in all_filepaths.items():
            num = len(paths)
            paths = np.array(paths)
            indexer = np.arange(num)
            rs.shuffle(indexer)
            paths = paths[indexer].tolist()

            if self.train:
                paths = paths[:int(0.8 * num)]
            else:
                paths = paths[int(0.8 * num):]

            labels = [label] * len(paths)
            split_filepaths.extend(paths)
            split_labels.extend(labels)

        return split_filepaths, split_labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = int(self.labels[index]) - 1
        image = Image.open(path).convert(mode='RGB')
        image = self.transforms(image)
        return index, image, label

    @staticmethod
    def num_classes():
        return VGGFlower.NUM_CLASSES

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(input_size=VGGFlower.INPUT_SIZE, patch_size=VGGFlower.PATCH_SIZE, in_channels=VGGFlower.IN_CHANNELS),
        ]


class VGGFlowerSmall(VGGFlower):
    # Dataset information.
    INPUT_SIZE = (32, 32)
    PATCH_SIZE = (4, 4)

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(
                input_size=VGGFlowerSmall.INPUT_SIZE,
                patch_size=VGGFlowerSmall.PATCH_SIZE,
                in_channels=VGGFlowerSmall.IN_CHANNELS,
            ),
        ]
