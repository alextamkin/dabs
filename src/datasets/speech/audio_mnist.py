import os
import random
from glob import glob

import librosa
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from src.datasets.specs import Input2dSpec

AUDIOMNIST_MEAN = [-90.293]
AUDIOMNIST_STDEV = [11.799]
AUDIOMNIST_TRAIN_SPK = [
    28, 56, 7, 19, 35, 1, 6, 16, 23, 34, 46, 53, 36, 57, 9, 24, 37, 2, 8, 17, 29, 39, 48, 54, 43, 58, 14, 25, 38, 3, 10, 20,
    30, 40, 49, 55
]
AUDIOMNIST_VAL_SPK = [12, 47, 59, 15, 27, 41, 4, 11, 21, 31, 44, 50]
AUDIOMNIST_TEST_SPK = [26, 52, 60, 18, 32, 42, 5, 13, 22, 33, 45, 51]


class AudioMNIST(Dataset):
    MAX_LENGTH = 150526

    NUM_CLASSES = 10
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'speech', 'AudioMNIST-master')
        self.train = train

        if not self._check_exists():
            raise RuntimeError(
                '''Dataset not found. You can use git clone 'https://github.com/soerenab/AudioMNIST' to download it
                and move using mv -t /AudioMNIST/data /DATASETS/audio_mnist'''
            )

        if train:
            speakers = AUDIOMNIST_TRAIN_SPK + AUDIOMNIST_VAL_SPK
        else:
            speakers = AUDIOMNIST_TEST_SPK
        wav_paths = []
        for spk in speakers:
            spk_paths = glob(os.path.join(self.root, 'data', '{:02d}'.format(spk), '*.wav'))
            wav_paths.extend(spk_paths)
        self.wav_paths = wav_paths

    def _check_exists(self) -> bool:
        return (os.path.exists(self.root))

    def __getitem__(self, index):
        wav_path = self.wav_paths[index]
        label, _, _ = wav_path.rstrip('.wav').split('/')[-1].split('_')

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

        normalize = Normalize(AUDIOMNIST_MEAN, AUDIOMNIST_STDEV)
        spectrum = normalize(spectrum)

        return index, spectrum, int(label)

    def __len__(self):
        return len(self.wav_paths)

    @staticmethod
    def num_classes():
        return AudioMNIST.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(
                input_size=AudioMNIST.INPUT_SIZE,
                patch_size=AudioMNIST.PATCH_SIZE,
                in_channels=AudioMNIST.IN_CHANNELS,
            )
        ]
