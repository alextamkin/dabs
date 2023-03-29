import os
import random
from collections import defaultdict

import librosa
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from src.datasets.specs import Input2dSpec

VOX_CELEB_MEAN = [-37.075]
VOX_CELEB_STDEV = [19.776]


class VoxCeleb1(Dataset):
    MAX_LENGTH = 150526

    NUM_CLASSES = 1251
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'speech', 'voxceleb1')

        if not self._check_exists():
            raise RuntimeError(
                'Dataset not found. You can register and download it at'
                'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html and save it to DATASETS/vox_celeb1'
            )

        wav_paths, speaker_strs = self.get_split(train)
        # change speaker_strs to integers
        unique_speakers = sorted(set(speaker_strs))
        assert VoxCeleb1.NUM_CLASSES == len(unique_speakers)
        speaker_id_map = dict(zip(unique_speakers, range(len(unique_speakers))))
        speaker_ids = [speaker_id_map[sp] for sp in speaker_strs]

        self.train = train
        self.wav_paths = wav_paths
        self.speaker_ids = speaker_ids

    def _check_exists(self) -> bool:
        return (os.path.exists(self.root))

    def get_split(self, train=True):
        split_file = os.path.join(self.root, 'iden_split.txt')
        with open(split_file, 'r') as fp:
            splits = fp.readlines()

        paths = defaultdict(lambda: [])
        for split in splits:
            spl, path = split.strip().split(' ')
            paths[spl].append(path)

        train_paths = paths['1'] + paths['2']
        test_paths = paths['3']

        train_speaker_ids = [p.split('/')[0] for p in train_paths]
        test_speaker_ids = [p.split('/')[0] for p in test_paths]

        if train:
            return train_paths, train_speaker_ids
        else:
            return test_paths, test_speaker_ids

    def __getitem__(self, index):
        wav_path = os.path.join(self.root, 'wav', self.wav_paths[index])
        speaker_id = self.speaker_ids[index]
        wavform, sample_rate = torchaudio.load(wav_path)
        wavform = wavform[0].numpy()

        # pad to 150k frames
        if len(wavform) > self.MAX_LENGTH:
            # randomly pick which side to chop off (fix if validation)
            flip = bool(random.getrandbits(1)) if self.train else True
            padded = (wavform[:self.MAX_LENGTH] if flip else wavform[-self.MAX_LENGTH:])
        else:
            padded = np.zeros(self.MAX_LENGTH)
            padded[:len(wavform)] = wavform  # pad w/ silence

        hop_length_dict = {224: 672, 112: 1344, 64: 2360, 32: 4800}
        spectrum = librosa.feature.melspectrogram(
            y=padded,
            sr=sample_rate,
            hop_length=hop_length_dict[self.INPUT_SIZE[0]],
            n_mels=self.INPUT_SIZE[0],
        )

        # log mel-spectrogram
        spectrum = librosa.power_to_db(spectrum**2)
        spectrum = torch.from_numpy(spectrum).float()
        spectrum = spectrum.unsqueeze(0)

        normalize = Normalize(VOX_CELEB_MEAN, VOX_CELEB_STDEV)
        spectrum = normalize(spectrum)

        return index, spectrum, speaker_id

    def __len__(self):
        return len(self.wav_paths)

    @staticmethod
    def num_classes():
        return VoxCeleb1.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(
                input_size=VoxCeleb1.INPUT_SIZE,
                patch_size=VoxCeleb1.PATCH_SIZE,
                in_channels=VoxCeleb1.IN_CHANNELS,
            )
        ]
