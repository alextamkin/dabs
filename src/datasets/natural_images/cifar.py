import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from src.datasets.specs import Input2dSpec


class CIFAR10(Dataset):
    # Dataset information.
    NUM_CLASSES = 10
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 3

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'natural_images', 'cifar10')
        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.INPUT_SIZE),
                transforms.CenterCrop(self.INPUT_SIZE),
                transforms.ToTensor(),
            ]
        )
        self.dataset = datasets.cifar.CIFAR10(
            root=self.root,
            train=train,
            download=download,
        )

    def __getitem__(self, index):
        img, label = self.dataset.data[index], int(self.dataset.targets[index])
        img = Image.fromarray(img).convert('RGB')
        img = self.transforms(img)
        return index, img, label

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def num_classes():
        return CIFAR10.NUM_CLASSES

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(input_size=CIFAR10.INPUT_SIZE, patch_size=CIFAR10.PATCH_SIZE, in_channels=CIFAR10.IN_CHANNELS),
        ]


class CIFAR10Small(CIFAR10):
    INPUT_SIZE = (32, 32)
    PATCH_SIZE = (4, 4)

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(
                input_size=CIFAR10Small.INPUT_SIZE,
                patch_size=CIFAR10Small.PATCH_SIZE,
                in_channels=CIFAR10Small.IN_CHANNELS,
            ),
        ]
