import json
import os
import random
from typing import Any, Tuple

import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
from transformers import AutoTokenizer

from src.datasets.specs import Input2dSpec, InputTokensSpec

COMPONENTS = {
    'train2014-images': {
        'result': 'train2014',
        'url': 'http://images.cocodataset.org/zips/train2014.zip'
    },
    'val2014-images': {
        'result': 'val2014',
        'url': 'http://images.cocodataset.org/zips/val2014.zip'
    },
    'annotations2014':
        {
            'result': 'annotations',
            'url': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
        },
    'vqa-train-annotations':
        {
            'result': 'mscoco_train2014_annotations.json',
            'url': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Train_mscoco.zip'
        },
    'vqa-val-annotations':
        {
            'result': 'mscoco_val2014_annotations.json',
            'url': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Val_mscoco.zip'
        },
    'vqa-train-questions':
        {
            'result': 'MultipleChoice_mscoco_train2014_questions',
            'url': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Train_mscoco.zip'
        },
    'vqa-val-questions':
        {
            'result': 'MultipleChoice_mscoco_val2014_questions',
            'url': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Val_mscoco.zip'
        }
}


class VQA(VisionDataset):
    '''A dataset class for the Visual Question Answering dataset

    Has separate `transforms` and `transoform`/`target_transform` for compatibility with torchvision
    `VisionDatasets`.
    '''
    TRANSFORMS = transforms.Compose(
        [
            transforms.Resize((640, 480)),
            transforms.CenterCrop((480, 480)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    NUM_CLASSES = 1
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 3

    SEQ_LEN = 32
    VOCAB_SIZE = 30522  # AutoTokenizer.from_pretrained('bert-base-uncased').vocab_size

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        self.root = os.path.join(base_root, 'captioned_images', 'vqa')
        super().__init__(self.root)
        self.split = 'train' if train else 'val'
        self.data_root = os.path.join(self.root, 'coco-vqa')
        self.image_root = os.path.join(self.data_root, f'{self.split}2014')

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir=self.root)
        assert self.tokenizer.vocab_size == self.VOCAB_SIZE

        if download:
            self.download()

        self.id2image_file = {}
        self.id2question = {}
        self.id2choices = {}
        self.annotations = []

        self._build_index()
        self._static_swap()

    def _build_index(self) -> None:
        '''Builds an index from the Coco annotations, the VQA annotations, and the VQA questions.

        The Coco annotations map image ids to filenames, and the VQA questions file maps question ids to
        the question text. The VQA annotations file maps question ids and answers to image ids.

        The index allows us to pick a question and find both the associated image file and the question text.

        More details on the file formats can be found here: https://visualqa.org/download.html.
        '''
        print('Building index...')
        anns = json.load(open(os.path.join(self.data_root, f'annotations/captions_{self.split}2014.json'), 'r'))
        vqa_anns = json.load(open(os.path.join(self.data_root, f'mscoco_{self.split}2014_annotations.json'), 'r'))
        questions = json.load(
            open(os.path.join(self.data_root, f'MultipleChoice_mscoco_{self.split}2014_questions.json'), 'r')
        )

        for image in anns['images']:
            self.id2image_file[image['id']] = image['file_name']

        for question in questions['questions']:
            self.id2question[question['question_id']] = question['question']
            self.id2choices[question['question_id']] = question['multiple_choices']

        self.annotations = vqa_anns['annotations']
        print('Finished index')

    def _static_swap(self) -> None:
        '''Swaps answers between two images of the same question with 50% probability and
        assigns a 0 or 1 indicating whether the resulting image-question-answer triple is swapped.
        '''
        swaps, sames, trash = 0, 0, 0
        annotations = []
        for index, annotation in enumerate(self.annotations):
            question_id = annotation['question_id']
            answer = annotation['multiple_choice_answer']

            # Make sure the answer is in the list of choices.
            choices = self.id2choices[question_id]
            if answer not in choices:
                trash += 1
                continue

            # Swap in wrong answer with 50% probability.
            if random.random() < 0.5:
                choices = [choice for choice in choices if choice != answer]
                answer = random.choice(choices)

                # Overwrite (in place but we reassign the whole thing later anyways).
                self.annotations[index]['multiple_choice_answer'] = answer
                self.annotations[index]['label'] = 0
                swaps += 1
            else:
                self.annotations[index]['label'] = 1
                sames += 1

            annotations += [self.annotations[index]]

        self.annotations = annotations
        print(f'{swaps} answers swapped, {sames} answers kept same, {trash} answers thrown out.')

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tuple[Tuple[Any, Any], Any]:
        '''Gets a question from the dataset.

        Args:
            index (int): The index of the question

        Returns:
            Tuple[Tuple[Any, Any], Any]: A nested tuple of ((image, question), answers)
        '''
        annotation = self.annotations[index]

        image = Image.open(os.path.join(self.image_root, self.id2image_file[annotation['image_id']])).convert('RGB')
        image = self.TRANSFORMS(image)

        question = self.id2question[annotation['question_id']]
        answer = annotation['multiple_choice_answer']

        label = annotation['label']
        tokens = self.tokenizer.encode(
            question,
            answer,
            max_length=self.SEQ_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        ).squeeze(0)

        return index, image, tokens, torch.tensor(label, dtype=torch.long)

    def download(self) -> None:
        '''Download components from the COMPONENTS dict by name. The function checks if the components
        are already present first.

        Args:
            root (str): The location of the DATASETS directory
            components (Iterable[str]): The names of which components to download
        '''
        components = [
            f'{self.split}2014-images',
            f'vqa-{self.split}-annotations',
            f'vqa-{self.split}-questions',
            'annotations2014',
        ]

        base_path = os.path.join(self.root, 'coco-vqa')
        os.makedirs(base_path, exist_ok=True)
        for name in components:
            comp = COMPONENTS[name]
            if os.path.exists(os.path.join(base_path, comp['result'])):
                print('Component {} already downloaded'.format(name))
                continue
            print('Downloading component {}...'.format(name))
            download_and_extract_archive(comp['url'], self.root, base_path)

    @staticmethod
    def num_classes():
        return VQA.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(input_size=VQA.INPUT_SIZE, patch_size=VQA.PATCH_SIZE, in_channels=VQA.IN_CHANNELS),
            InputTokensSpec(seq_len=VQA.SEQ_LEN, vocab_size=VQA.VOCAB_SIZE),
        ]
