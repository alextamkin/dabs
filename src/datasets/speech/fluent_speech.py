import os
import random

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from src.datasets.specs import Input2dSpec
from src.datasets.speech.utils import download_and_extract_archive

FLUENTSPEECH_RESOURCES = {'fluent': 'http://fluent.ai:2052/jf8398hf30f0381738rucj3828chfdnchs.tar.gz'}
FLUENTSPEECH_MEAN = [-31.809]
FLUENTSPEECH_STDEV = [13.127]

FLUENTSPEECH_ACTIONS = ['change language', 'activate', 'deactivate', 'increase', 'decrease', 'bring']
FLUENTSPEECH_OBJECTS = [
    'none', 'music', 'lights', 'volume', 'heat', 'lamp', 'newspaper', 'juice', 'socks', 'shoes', 'Chinese', 'Korean',
    'English', 'German'
]
FLUENTSPEECH_LOCATIONS = ['none', 'kitchen', 'bedroom', 'washroom']


class FluentSpeechCommandsBase(Dataset):
    MAX_LENGTH = 150526

    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, label_type: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'speech', 'fluent_speech')
        self.label_type = label_type
        assert self.label_type in ['action', 'object', 'location']

        if download:
            self.download_dataset()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

        if train:
            train_path = os.path.join(self.root, 'fluent_speech_commands_dataset', 'data', 'train_data.csv')
            val_path = os.path.join(self.root, 'fluent_speech_commands_dataset', 'data', 'valid_data.csv')
            train_data = pd.read_csv(train_path)
            train_paths = list(train_data['path'])
            train_labels = list(train_data[self.label_type])
            val_data = pd.read_csv(val_path)
            val_paths = list(val_data['path'])
            val_labels = list(val_data[self.label_type])
            wav_paths = train_paths + val_paths
            labels = train_labels + val_labels
        else:
            test_path = os.path.join(self.root, 'fluent_speech_commands_dataset', 'data', 'test_data.csv')
            test_data = pd.read_csv(test_path)
            wav_paths = list(test_data['path'])
            labels = list(test_data[self.label_type])

        self.wav_paths = wav_paths
        self.labels = labels
        self.train = train

    def _check_exists(self) -> bool:
        return (os.path.exists(self.root))

    def download_dataset(self):
        '''Download the meta dataset if not exists already'''

        if self._check_exists():
            return

        # download and extract files
        print('Downloading and Extracting...')
        filename = FLUENTSPEECH_RESOURCES['fluent'].rpartition('/')[2]
        download_and_extract_archive(FLUENTSPEECH_RESOURCES['fluent'], download_root=self.root, filename=filename)

        print('Done!')

    def __getitem__(self, index):
        wav_name = self.wav_paths[index]
        wav_path = os.path.join(self.root, 'fluent_speech_commands_dataset', wav_name)

        label = self.labels[index]

        if self.label_type == 'action':
            label = FLUENTSPEECH_ACTIONS.index(label)
        elif self.label_type == 'object':
            label = FLUENTSPEECH_OBJECTS.index(label)
        elif self.label_type == 'location':
            label = FLUENTSPEECH_LOCATIONS.index(label)

        wavform, sample_rate = torchaudio.load(wav_path)
        wavform = wavform[0].numpy()

        # pad to 150k frames
        if len(wavform) > self.MAX_LENGTH:
            # randomly pick which side to chop off (fix if validation)
            flip = (bool(random.getrandbits(1)) if self.train else True)
            padded = (wavform[:self.MAX_LENGTH] if flip else wavform[-self.MAX_LENGTH:])
        else:
            padded = np.zeros(self.MAX_LENGTH)
            padded[:len(wavform)] = wavform  # pad w/ silence

        hop_length_dict = {224: 672, 112: 1344, 64: 2360, 32: 4800}
        spectrum = librosa.feature.melspectrogram(
            padded,
            sample_rate,
            hop_length=hop_length_dict[self.INPUT_SIZE[0]],
            n_mels=self.INPUT_SIZE[0],
        )

        # log mel-spectrogram
        spectrum = librosa.power_to_db(spectrum**2)
        spectrum = torch.from_numpy(spectrum).float()
        spectrum = spectrum.unsqueeze(0)

        normalize = Normalize(FLUENTSPEECH_MEAN, FLUENTSPEECH_STDEV)
        spectrum = normalize(spectrum)

        return index, spectrum, int(label)

    def __len__(self):
        return len(self.wav_paths)

    @staticmethod
    def num_classes():
        raise NotImplementedError

    @staticmethod
    def spec():
        return [
            Input2dSpec(
                input_size=FluentSpeechCommandsBase.INPUT_SIZE,
                patch_size=FluentSpeechCommandsBase.PATCH_SIZE,
                in_channels=FluentSpeechCommandsBase.IN_CHANNELS,
            )
        ]


class FluentSpeechLocation(FluentSpeechCommandsBase):

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__(base_root=base_root, label_type='location', download=download, train=train)

    @staticmethod
    def num_classes():
        return len(FLUENTSPEECH_LOCATIONS)


class FluentSpeechAction(FluentSpeechCommandsBase):

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__(base_root=base_root, label_type='action', download=download, train=train)

    @staticmethod
    def num_classes():
        return len(FLUENTSPEECH_ACTIONS)


class FluentSpeechObject(FluentSpeechCommandsBase):

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__(base_root=base_root, label_type='object', download=download, train=train)

    @staticmethod
    def num_classes():
        return len(FLUENTSPEECH_OBJECTS)
