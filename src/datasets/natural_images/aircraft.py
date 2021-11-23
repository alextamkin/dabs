import os
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive

from src.datasets.specs import Input2dSpec

AIRCRAFT_RESOURCES = {
    'aircraft': 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz',
}


class Aircraft(Dataset):
    # Dataset information.
    NUM_CLASSES = 102
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 3

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.train = train
        self.root = os.path.join(base_root, 'natural_images', 'aircraft')
        self.transforms = transforms.Compose(
            [transforms.Resize(self.INPUT_SIZE),
             transforms.CenterCrop(self.INPUT_SIZE),
             transforms.ToTensor()]
        )

        if download:
            self.download_dataset()

        if not self._is_downloaded():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        paths, bboxes, labels = self.load_images()
        self.paths = paths
        self.bboxes = bboxes
        self.labels = labels

    def _is_downloaded(self) -> bool:
        return os.path.exists(os.path.join(self.root, 'fgvc-aircraft-2013b'))

    def download_dataset(self):
        '''Download the meta dataset if not exists already'''

        if self._is_downloaded():
            return

        # download and extract files
        print('Downloading and Extracting...')

        filename = AIRCRAFT_RESOURCES['aircraft'].rpartition('/')[2]
        download_and_extract_archive(AIRCRAFT_RESOURCES['aircraft'], download_root=self.root, filename=filename)

        print('Done!')

    def load_images(self):
        split = 'trainval' if self.train else 'test'
        variant_path = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data', f'images_variant_{split}.txt')
        with open(variant_path, 'r') as f:
            names_to_variants = [line.split('\n')[0].split(' ', 1) for line in f.readlines()]
        names_to_variants = dict(names_to_variants)

        # Build mapping from variant to filenames. 'Variant' refers to the aircraft
        # model variant (e.g., A330-200) and is used as the class name in the
        # dataset. The position of the class name in the concatenated list of
        # training, validation, and test class name constitutes its class ID.
        variants_to_names = defaultdict(list)
        for name, variant in names_to_variants.items():
            variants_to_names[variant].append(name)

        names_to_bboxes = self.get_bounding_boxes()

        variants = sorted(list(set(variants_to_names.keys())))
        split_files, split_labels, split_bboxes = [], [], []

        for variant_id, variant in enumerate(variants):
            class_files = [
                os.path.join(self.root, 'fgvc-aircraft-2013b', 'data', 'images', f'{filename}.jpg')
                for filename in sorted(variants_to_names[variant])
            ]
            bboxes = [names_to_bboxes[name] for name in sorted(variants_to_names[variant])]
            labels = list([variant_id] * len(class_files))

            split_files += class_files
            split_labels += labels
            split_bboxes += bboxes

        return split_files, split_bboxes, split_labels

    def get_bounding_boxes(self):
        bboxes_path = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data', 'images_box.txt')
        with open(bboxes_path, 'r') as f:
            names_to_bboxes = [line.split('\n')[0].split(' ') for line in f.readlines()]
            names_to_bboxes = dict(
                (name, list(map(int, (xmin, ymin, xmax, ymax)))) for name, xmin, ymin, xmax, ymax in names_to_bboxes
            )

        return names_to_bboxes

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        bbox = tuple(self.bboxes[index])
        label = self.labels[index]

        image = Image.open(path).convert(mode='RGB')
        image = image.crop(bbox)
        image = self.transforms(image)

        return index, image.float(), label

    @staticmethod
    def num_classes():
        return Aircraft.NUM_CLASSES

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(input_size=Aircraft.INPUT_SIZE, patch_size=Aircraft.PATCH_SIZE, in_channels=Aircraft.IN_CHANNELS),
        ]


class AircraftSmall(Aircraft):
    INPUT_SIZE = (32, 32)
    PATCH_SIZE = (4, 4)

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(
                input_size=AircraftSmall.INPUT_SIZE,
                patch_size=AircraftSmall.PATCH_SIZE,
                in_channels=AircraftSmall.IN_CHANNELS,
            ),
        ]
