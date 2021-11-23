import os
from os.path import join

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.datasets.natural_images.utils import download_and_extract_archive
from src.datasets.specs import Input2dSpec

DTD_RESOURCES = {'dtd': 'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz'}


class DTD(Dataset):
    # Dataset information.
    NUM_CLASSES = 47
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 3

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.train = train
        self.root = os.path.join(base_root, 'natural_images', 'dtd')
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.INPUT_SIZE),
                transforms.CenterCrop(self.INPUT_SIZE),
                transforms.ToTensor(),
            ]
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

        filename = DTD_RESOURCES['dtd'].rpartition('/')[2]
        download_and_extract_archive(DTD_RESOURCES['dtd'], download_root=self.root, filename=filename)

        print('Done!')

    def load_images(self):
        if self.train:
            train_info_path = os.path.join(self.root, 'dtd', 'labels', 'train1.txt')
            with open(train_info_path, 'r') as f:
                train_info = [line.split('\n')[0] for line in f.readlines()]

            val_info_path = os.path.join(self.root, 'dtd', 'labels', 'val1.txt')
            with open(val_info_path, 'r') as f:
                val_info = [line.split('\n')[0] for line in f.readlines()]

            split_info = train_info + val_info
        else:
            test_info_path = os.path.join(self.root, 'dtd', 'labels', 'test1.txt')
            with open(test_info_path, 'r') as f:
                split_info = [line.split('\n')[0] for line in f.readlines()]

        # pull out categoires from paths
        categories = []
        for row in split_info:
            image_path = row
            category = image_path.split('/')[0]
            categories.append(category)
        categories = sorted(list(set(categories)))

        all_paths, all_labels = [], []
        for row in split_info:
            image_path = row
            category = image_path.split('/')[0]
            label = categories.index(category)
            all_paths.append(join(self.root, 'dtd', 'images', image_path))
            all_labels.append(label)

        return all_paths, all_labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]

        image = Image.open(path).convert(mode='RGB')
        image = self.transforms(image)
        return index, image, label

    @staticmethod
    def num_classes():
        return DTD.NUM_CLASSES

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(input_size=DTD.INPUT_SIZE, patch_size=DTD.PATCH_SIZE, in_channels=DTD.IN_CHANNELS),
        ]


class DTDSmall(DTD):
    INPUT_SIZE = (32, 32)
    PATCH_SIZE = (4, 4)

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(
                input_size=DTDSmall.INPUT_SIZE,
                patch_size=DTDSmall.PATCH_SIZE,
                in_channels=DTDSmall.IN_CHANNELS,
            ),
        ]
