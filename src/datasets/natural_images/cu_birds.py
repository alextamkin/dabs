import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.datasets.specs import Input2dSpec

CUBIRDS_RESOURCES = {
    'cu_birds':
        [
            'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz',
            'https://drive.google.com/u/1/uc?id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45&export=download',
        ],
}


class CUBirds(Dataset):
    # Dataset information.
    NUM_CLASSES = 200
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 3

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'natural_images', 'cu_birds')
        self.download = download
        self.train = train
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.INPUT_SIZE),
                transforms.CenterCrop(self.INPUT_SIZE),
                transforms.ToTensor(),
            ]
        )

        if not self._is_downloaded():
            raise RuntimeError(
                f'''Dataset not found. Please visit the URL: {CUBIRDS_RESOURCES['cu_birds'][0]}
                and download the file with your own Google credentials', ' unzip the file and move
                using mv -t /CUB_200_2011 /DATASETS/natural_images/cu_birds'''
            )

        paths, labels = self.load_images()
        self.paths, self.labels = paths, labels

    def _is_downloaded(self) -> bool:
        return (os.path.exists(self.root))

    def load_images(self):
        # load id to image path information
        image_info_path = os.path.join(self.root, 'CUB_200_2011', 'images.txt')
        with open(image_info_path, 'r') as f:
            image_info = [line.split('\n')[0].split(' ', 1) for line in f.readlines()]
        image_info = dict(image_info)

        # load image to label information
        label_info_path = os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt')
        with open(label_info_path, 'r') as f:
            label_info = [line.split('\n')[0].split(' ', 1) for line in f.readlines()]
        label_info = dict(label_info)

        # load train test split
        train_test_info_path = os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt')
        with open(train_test_info_path, 'r') as f:
            train_test_info = [line.split('\n')[0].split(' ', 1) for line in f.readlines()]
        train_test_info = dict(train_test_info)

        all_paths, all_labels = [], []
        for index, image_path in image_info.items():
            label = label_info[index]
            split = int(train_test_info[index])

            if self.train:
                if split == 1:
                    all_paths.append(image_path)
                    all_labels.append(label)
            else:
                if split == 0:
                    all_paths.append(image_path)
                    all_labels.append(label)

        return all_paths, all_labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.root, 'CUB_200_2011', 'images', self.paths[index])
        label = int(self.labels[index]) - 1

        image = Image.open(path).convert(mode='RGB')
        image = self.transforms(image)

        return index, image, label

    @staticmethod
    def num_classes():
        return CUBirds.NUM_CLASSES

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(input_size=CUBirds.INPUT_SIZE, patch_size=CUBirds.PATCH_SIZE, in_channels=CUBirds.IN_CHANNELS),
        ]


class CUBirdsSmall(CUBirds):
    INPUT_SIZE = (32, 32)
    PATCH_SIZE = (4, 4)

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(
                input_size=CUBirdsSmall.INPUT_SIZE,
                patch_size=CUBirdsSmall.PATCH_SIZE,
                in_channels=CUBirdsSmall.IN_CHANNELS,
            ),
        ]
