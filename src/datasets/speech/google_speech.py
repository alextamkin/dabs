import os
import random

import librosa
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from src.datasets.specs import Input2dSpec
from src.datasets.speech.utils import download_and_extract_archive

GOOGLESPEECH_RESOURCES = {'google_speech': 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'}
GOOGLESPEECH_MEAN = [-46.847]
GOOGLESPEECH_STDEV = [19.151]
GOOGLESPEECH_LABELS = [
    'eight', 'right', 'happy', 'three', 'yes', 'up', 'no', 'stop', 'on', 'four', 'nine', 'zero', 'down', 'go', 'six', 'two',
    'left', 'five', 'off', 'seven', 'one', 'cat', 'bird', 'marvin', 'wow', 'tree', 'dog', 'sheila', 'bed', 'house', 'follow',
    'visual', 'backward', 'forward', 'learn'
]


class GoogleSpeechCommands(Dataset):
    MAX_LENGTH = 150526

    NUM_CLASSES = len(GOOGLESPEECH_LABELS)  # 35
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.train = train
        self.root = os.path.join(base_root, 'speech', 'google_speech')

        if download:
            self.download_dataset()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

        if train:
            train_paths = []
            for path, _, files in os.walk(self.root):
                for name in files:
                    if name.endswith('wav'):
                        train_paths.append(os.path.join(path.split('/')[-1], name))
            val_paths = open(os.path.join(self.root, 'validation_list.txt'), 'r').readlines()
            test_paths = open(os.path.join(self.root, 'testing_list.txt'), 'r').readlines()
            train_paths = (set(train_paths) - set(val_paths) - set(test_paths))
            wav_paths = list(train_paths) + val_paths
        else:
            wav_paths = open(os.path.join(self.root, 'testing_list.txt'), 'r').readlines()

        wav_paths = [path.strip() for path in wav_paths]

        self.wav_paths = wav_paths

    def _check_exists(self) -> bool:
        return (os.path.exists(self.root))

    def download_dataset(self):
        '''Download the meta dataset if not exists already'''

        if self._check_exists():
            return

        # download and extract files
        print('Downloading and Extracting...')

        filename = GOOGLESPEECH_RESOURCES['google_speech'].rpartition('/')[2]
        download_and_extract_archive(GOOGLESPEECH_RESOURCES['google_speech'], download_root=self.root, filename=filename)

        print('Done!')

    def __getitem__(self, index):
        wav_name = self.wav_paths[index]
        label_name = wav_name.split('/')[0].lower()
        label = GOOGLESPEECH_LABELS.index(label_name)
        wav_path = os.path.join(self.root, wav_name)

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

        normalize = Normalize(GOOGLESPEECH_MEAN, GOOGLESPEECH_STDEV)
        spectrum = normalize(spectrum)

        return index, spectrum, int(label)

    def __len__(self):
        return len(self.wav_paths)

    @staticmethod
    def num_classes():
        return GoogleSpeechCommands.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(
                input_size=GoogleSpeechCommands.INPUT_SIZE,
                patch_size=GoogleSpeechCommands.PATCH_SIZE,
                in_channels=GoogleSpeechCommands.IN_CHANNELS,
            )
        ]
