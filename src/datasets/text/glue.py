import os
from typing import Iterable

from src.datasets.specs import InputTokensSpec
from src.datasets.text.utils import EnglishHuggingFaceDataset


class GLUE(EnglishHuggingFaceDataset):
    '''The base GLUE dataset loaded from Hugging Face

    Args:
        root (str): The location of the dataset files
    '''

    def __init__(
        self,
        base_root: str,
        dataset: str,
        input_keys: Iterable[str],
        download: bool = False,
        train: bool = True,
    ) -> None:
        self.root = os.path.join(base_root, 'text')
        super().__init__(
            path='glue',
            name=dataset,
            input_keys=input_keys,
            output_keys=['input_ids', 'label'],
            base_root=self.root,
            train=train,
        )

    @staticmethod
    def num_classes():
        raise NotImplementedError  # each subclass should overwrite this

    @staticmethod
    def spec():
        return [
            InputTokensSpec(seq_len=GLUE.SEQ_LEN, vocab_size=GLUE.VOCAB_SIZE),
        ]


class COLA(GLUE):
    '''The CoLA dataset from Hugging Face.'''

    NUM_CLASSES = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__(base_root, 'cola', ['sentence'], download=download, train=train)

    def __getitem__(self, index):
        index, tokens, label = super().__getitem__(index)
        return index, tokens, label.float()

    @staticmethod
    def num_classes():
        return COLA.NUM_CLASSES


class MNLIMatched(GLUE):
    '''The MNLI matched dataset from Hugging Face.'''

    NUM_CLASSES = 3

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        dataset = 'mnli' if train else 'mnli_matched'
        super().__init__(base_root, dataset, ['premise', 'hypothesis'], download=download, train=train)

    @staticmethod
    def num_classes():
        return MNLIMatched.NUM_CLASSES


class MNLIMismatched(GLUE):
    '''The MNLI mismatched dataset from Hugging Face.'''

    NUM_CLASSES = 3

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        dataset = 'mnli' if train else 'mnli_mismatched'
        super().__init__(base_root, dataset, ['premise', 'hypothesis'], download=download, train=train)

    @staticmethod
    def num_classes():
        return MNLIMismatched.NUM_CLASSES


class MRPC(GLUE):
    '''The MRPC mismatched dataset from Hugging Face.'''

    NUM_CLASSES = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__(base_root, 'mrpc', ['sentence1', 'sentence2'], download=download, train=train)

    @staticmethod
    def num_classes():
        return MRPC.NUM_CLASSES


class QNLI(GLUE):
    '''The QNLI dataset from Hugging Face.'''

    NUM_CLASSES = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__(base_root, 'qnli', ['question', 'sentence'], download=download, train=train)

    @staticmethod
    def num_classes():
        return QNLI.NUM_CLASSES


class QQP(GLUE):
    '''The QQP mismatched dataset from Hugging Face.'''

    NUM_CLASSES = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__(base_root, 'qqp', ['question1', 'question2'], download=download, train=train)

    @staticmethod
    def num_classes():
        return QQP.NUM_CLASSES


class RTE(GLUE):
    '''The RTE dataset from Hugging Face.'''

    NUM_CLASSES = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__(base_root, 'rte', ['sentence1', 'sentence2'], download=download, train=train)

    @staticmethod
    def num_classes():
        return RTE.NUM_CLASSES


class SST2(GLUE):
    '''The SST2 dataset from Hugging Face.'''

    NUM_CLASSES = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__(base_root, 'sst2', ['sentence'], download=download, train=train)

    @staticmethod
    def num_classes():
        return SST2.NUM_CLASSES


class STSB(GLUE):
    '''The STS benchmark dataset from Hugging Face.'''

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__(base_root, 'stsb', ['sentence1', 'sentence2'], download=download, train=train)

    @staticmethod
    def num_classes():
        return None


class WNLI(GLUE):
    '''The WNLI dataset from Hugging Face.'''

    NUM_CLASSES = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__(base_root, 'wnli', ['sentence1', 'sentence2'], download=download, train=train)

    @staticmethod
    def num_classes():
        return WNLI.NUM_CLASSES
