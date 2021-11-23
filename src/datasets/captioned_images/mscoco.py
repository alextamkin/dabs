import json
import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive
from transformers import AutoTokenizer

from src.datasets.specs import Input2dSpec, InputTokensSpec


class MSCOCO(Dataset):
    # Dataset information.
    TRANSFORMS = transforms.Compose(
        [
            transforms.Resize((640, 480)),
            transforms.CenterCrop((480, 480)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    NUM_CLASSES = 80
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 3

    SEQ_LEN = 32
    VOCAB_SIZE = 30522  # AutoTokenizer.from_pretrained('bert-base-uncased').vocab_size

    DATASET_RESOURCES = {
        'mscoco':
            {
                'train': 'http://images.cocodataset.org/zips/train2017.zip',
                'val': 'http://images.cocodataset.org/zips/val2017.zip',
                'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
            }
    }

    BOX_SCALE_RATIO = 1.2

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.train = train
        self.root = os.path.join(base_root, 'captioned_images', 'mscoco')

        if not os.path.isdir(self.root):
            os.makedirs(self.root)

        if download:
            self.download_dataset()

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir=self.root)
        assert self.tokenizer.vocab_size == self.VOCAB_SIZE

        if not self._is_downloaded():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        annotations, coco_cat_id_to_label = self.load_coco()
        paths, bboxes, labels, captions = self.load_images(annotations, coco_cat_id_to_label)
        self.paths = paths
        self.bboxes = bboxes
        self.labels = labels
        self.captions = captions

    def _is_downloaded(self) -> bool:
        return (os.path.exists(self.root))

    def download_dataset(self):
        '''Download the meta dataset if not exists already'''

        if self._is_downloaded():
            return

        # download and extract files
        print('Downloading and Extracting...')

        for _, urls in self.DATASET_RESOURCES['mscoco'].items():
            if type(urls) != str and len(urls) > 1:
                for url in urls.values():
                    filename = url.rpartition('/')[2]
                    download_and_extract_archive(url, download_root=self.root, filename=filename)

            else:
                filename = urls.rpartition('/')[2]
                download_and_extract_archive(urls, download_root=self.root, filename=filename)

        print('Done!')

    def load_coco(self):
        annotation_name = ('instances_train2017.json' if self.train else 'instances_val2017.json')
        annotation_path = os.path.join(self.root, 'annotations', annotation_name)

        caption_name = ('captions_train2017.json' if self.train else 'captions_val2017.json')
        caption_path = os.path.join(self.root, 'annotations', caption_name)

        with open(annotation_path, 'r') as json_file:
            annotations = json.load(json_file)
            instance_annotations = annotations['annotations']
            categories = annotations['categories']

        category_ids = [cat['id'] for cat in categories]
        coco_cat_id_to_label = dict(zip(category_ids, range(len(categories))))

        # Add captions to instance annotations.
        with open(caption_path, 'r') as json_file:
            captions = json.load(json_file)['annotations']
            caption_annotations = {c['image_id']: c['caption'] for c in captions}
            for annotation in instance_annotations:
                annotation['caption'] = caption_annotations[annotation['image_id']]

        return instance_annotations, coco_cat_id_to_label

    def load_images(self, annotations, coco_cat_id_to_label):
        image_dir_name = ('train2017' if self.train else 'val2017')
        image_dir = os.path.join(self.root, image_dir_name)
        all_filepaths, all_bboxes, all_labels, all_captions = [], [], [], []
        for anno in annotations:
            image_id = anno['image_id']
            image_path = os.path.join(image_dir, '%012d.jpg' % image_id)
            bbox = anno['bbox']
            coco_class_id = anno['category_id']
            label = coco_cat_id_to_label[coco_class_id]
            caption = anno['caption']
            all_filepaths.append(image_path)
            all_bboxes.append(bbox)
            all_labels.append(label)
            all_captions.append(caption)

        return all_filepaths, all_bboxes, all_labels, all_captions

    def handle_bboxes(self, image: Image, bbox):
        image_w, image_h = image.size

        def scale_box(bbox, scale_ratio):
            x, y, w, h = bbox
            x = x - 0.5 * w * (scale_ratio - 1.0)
            y = y - 0.5 * h * (scale_ratio - 1.0)
            w = w * scale_ratio
            h = h * scale_ratio
            return [x, y, w, h]

        x, y, w, h = scale_box(bbox, self.BOX_SCALE_RATIO)
        # Convert half-integer to full-integer representation.
        # The Python Imaging Library uses a Cartesian pixel coordinate system,
        # with (0,0) in the upper left corner. Note that the coordinates refer
        # to the implied pixel corners; the centre of a pixel addressed as
        # (0, 0) actually lies at (0.5, 0.5). Since COCO uses the later
        # convention and we use PIL to crop the image, we need to convert from
        # half-integer to full-integer representation.
        xmin = max(int(round(x - 0.5)), 0)
        ymin = max(int(round(y - 0.5)), 0)
        xmax = min(int(round(x + w - 0.5)) + 1, image_w)
        ymax = min(int(round(y + h - 0.5)) + 1, image_h)
        image_crop = image.crop((xmin, ymin, xmax, ymax))
        crop_width, crop_height = image_crop.size

        if crop_width <= 0 or crop_height <= 0:
            raise ValueError('crops are not valid.')

        return image_crop

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]

        caption = self.captions[index]
        caption = self.tokenizer.encode(
            caption,
            max_length=self.SEQ_LEN,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        ).squeeze(0)

        image = Image.open(path).convert(mode='RGB')

        if self.TRANSFORMS:
            image = self.TRANSFORMS(image)

        return (index, image.float(), caption.long(), label)

    @staticmethod
    def num_classes():
        return MSCOCO.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(input_size=MSCOCO.INPUT_SIZE, patch_size=MSCOCO.PATCH_SIZE, in_channels=MSCOCO.IN_CHANNELS),
            InputTokensSpec(seq_len=MSCOCO.SEQ_LEN, vocab_size=MSCOCO.VOCAB_SIZE),
        ]


class MismatchedCaption(MSCOCO):
    '''MSCOCO dataset, adapted for transfer learning.

    Each example has its caption swapped with 50% probability (with other examples who are candidates for swapping),
    and is assigned a corresponding label of 1 if the image-caption pair is a match, and 0 if it's a mismatch.
    '''

    NUM_CLASSES = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.static_swap()

    def static_swap(self):
        '''Randomly swaps captions for each image with 50% probability. Revises the labels to reflect swapped or original.'''
        random.seed(42)

        # Accumulate indices to be swapped with 50% probability.
        orig_indices = []
        sames, swaps = 0, 0
        for index in range(len(self)):
            if random.random() < 0.5:
                orig_indices += [index]
                self.labels[index] = 0
                swaps += 1
            else:
                self.labels[index] = 1
                sames += 1

        # Roll indices.
        roll_length = random.randint(1, len(orig_indices))
        new_indices = orig_indices[roll_length:] + orig_indices[:roll_length]

        # Reassign captions.
        captions_copy = self.captions[:]  # copy here to avoid overwriting originals
        for orig_index, new_index in zip(orig_indices, new_indices):
            self.captions[orig_index] = captions_copy[new_index]

        print(f'{sames} examples kept same, {swaps} examples swapped.')

    @staticmethod
    def num_classes():
        return MismatchedCaption.NUM_CLASSES
