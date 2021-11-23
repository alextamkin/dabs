import os
import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from torchaudio.datasets import LIBRISPEECH
from torchvision.transforms import Normalize

from src.datasets.specs import Input2dSpec

BAD_LIBRISPEECH_INDICES = [60150]
LIBRISPEECH_MEAN = [-22.924]
LIBRISPEECH_STDEV = [12.587]


class LibriSpeech(Dataset):
    MAX_LENGTH = 150526

    ALL_TRAIN_NUM_CLASSES = 2338
    DEV_CLEAN_NUM_CLASSES = 40

    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'speech', 'librispeech')

        self.small = False  # we want to use all LibriSpeech data
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        if train:
            if self.small:
                self.dataset = LIBRISPEECH(
                    self.root, url='train-clean-100', download=download, folder_in_archive='LibriSpeech'
                )
            else:
                self.dataset1 = LIBRISPEECH(
                    self.root, url='train-clean-100', download=download, folder_in_archive='LibriSpeech'
                )
                self.dataset2 = LIBRISPEECH(
                    self.root, url='train-clean-360', download=download, folder_in_archive='LibriSpeech'
                )
                self.dataset3 = LIBRISPEECH(
                    self.root, url='train-other-500', download=download, folder_in_archive='LibriSpeech'
                )
        else:
            self.dataset = LIBRISPEECH(self.root, url='dev-clean', download=download, folder_in_archive='LibriSpeech')

        self.train = train
        all_speaker_ids = self.get_speaker_ids()
        unique_speaker_ids = sorted(list(set(all_speaker_ids)))
        num_classes = self.ALL_TRAIN_NUM_CLASSES if train else self.DEV_CLEAN_NUM_CLASSES
        assert num_classes == len(unique_speaker_ids)
        self.speaker_id_map = dict(zip(unique_speaker_ids, range(num_classes)))
        self.all_speaker_ids = np.array([self.speaker_id_map[sid] for sid in all_speaker_ids])

    def get_speaker_ids(self):
        if self.train and not self.small:
            speaker_ids_1 = self._get_speaker_ids(self.dataset1)
            speaker_ids_2 = self._get_speaker_ids(self.dataset2)
            speaker_ids_3 = self._get_speaker_ids(self.dataset3)
            return np.concatenate([speaker_ids_1, speaker_ids_2, speaker_ids_3])
        else:
            return self._get_speaker_ids(self.dataset)

    def _get_speaker_ids(self, dataset):
        speaker_ids = []
        for i in range(len(dataset)):
            fileid = dataset._walker[i]
            speaker_id = self.load_librispeech_speaker_id(
                fileid,
                dataset._path,
                dataset._ext_audio,
                dataset._ext_txt,
            )
            speaker_ids.append(speaker_id)
        return np.array(speaker_ids)

    def load_librispeech_speaker_id(self, fileid, path, ext_audio, ext_txt):
        speaker_id, _, _ = fileid.split('-')
        return int(speaker_id)

    def __getitem__(self, index):  # noqa: C901
        if self.train and not self.small:
            if index >= (len(self.dataset1) + len(self.dataset2)):
                try:
                    wavform, sample_rate, _, speaker_id, _, _ = \
                        self.dataset3.__getitem__(index - len(self.dataset1) - len(self.dataset2))
                except Exception:
                    index2 = (index - len(self.dataset1) - len(self.dataset2) + 1) % len(self.dataset3)
                    wavform, sample_rate, _, speaker_id, _, _ = self.dataset3(index2)
            elif index >= len(self.dataset1):
                try:
                    wavform, sample_rate, _, speaker_id, _, _ = \
                        self.dataset2.__getitem__(index - len(self.dataset1))
                except Exception:
                    index2 = (index - len(self.dataset1) + 1) % len(self.dataset2)
                    wavform, sample_rate, _, speaker_id, _, _ = self.dataset2.__getitem__(index2)
            else:
                try:
                    wavform, sample_rate, _, speaker_id, _, _ = self.dataset1.__getitem__(index)
                except Exception:
                    index2 = (index + 1) % len(self.dataset)
                    wavform, sample_rate, _, speaker_id, _, _ = self.dataset1.__getitem__(index2)
        else:
            try:
                wavform, sample_rate, _, speaker_id, _, _ = self.dataset.__getitem__(index)
            except Exception:
                index2 = (index + 1) % len(self.dataset)
                wavform, sample_rate, _, speaker_id, _, _ = self.dataset.__getitem__(index2)

        speaker_id = self.speaker_id_map[speaker_id]
        wavform = np.asarray(wavform[0])

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
            hop_length=hop_length_dict[self.INPUT_SIZE[1]],
            n_mels=self.INPUT_SIZE[0],
        )

        # log mel-spectrogram
        spectrum = librosa.power_to_db(spectrum**2)
        spectrum = torch.from_numpy(spectrum).float()
        spectrum = spectrum.unsqueeze(0)

        normalize = Normalize(LIBRISPEECH_MEAN, LIBRISPEECH_STDEV)
        spectrum = normalize(spectrum)

        return index, spectrum, speaker_id

    def __len__(self):
        if self.train and not self.small:
            return len(self.dataset1) + len(self.dataset2) + len(self.dataset3)
        else:
            return len(self.dataset)

    @staticmethod
    def num_classes():
        raise NotImplementedError

    @staticmethod
    def spec():
        return [
            Input2dSpec(
                input_size=LibriSpeech.INPUT_SIZE,
                patch_size=LibriSpeech.PATCH_SIZE,
                in_channels=LibriSpeech.IN_CHANNELS,
            )
        ]


class LibriSpeechTransfer(Dataset):
    '''
    Divide the dev-clean split of LibriSpeech into train and
    test splits by speaker so we can train a logreg fairly.
    '''
    MAX_LENGTH = 150526

    NUM_CLASSES = 40
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'speech', 'librispeech')
        self.dataset = LIBRISPEECH(self.root, url='dev-clean', download=download, folder_in_archive='LibriSpeech')

        all_speaker_ids = self.get_speaker_ids(self.dataset)
        unique_speaker_ids = sorted(list(set(all_speaker_ids)))
        assert self.NUM_CLASSES == len(unique_speaker_ids)
        self.speaker_id_map = dict(zip(unique_speaker_ids, range(self.NUM_CLASSES)))
        self.all_speaker_ids = np.array([self.speaker_id_map[sid] for sid in all_speaker_ids])

        self.indices = self.train_test_split(self.dataset, all_speaker_ids, train=train)
        self.train = train

    def get_speaker_ids(self, dataset):
        speaker_ids = []
        for i in range(len(dataset)):
            fileid = dataset._walker[i]
            speaker_id = self.load_librispeech_speaker_id(
                fileid,
                dataset._path,
                dataset._ext_audio,
                dataset._ext_txt,
            )
            speaker_ids.append(speaker_id)
        return np.array(speaker_ids)

    def train_test_split(self, dataset, speaker_ids, train=True):
        rs = np.random.RandomState(42)  # fix seed so reproducible splitting

        unique_speaker_ids = sorted(set(speaker_ids))
        unique_speaker_ids = np.array(unique_speaker_ids)

        # train test split to ensure the 80/20 splits
        train_indices, test_indices = [], []
        for speaker_id in unique_speaker_ids:
            speaker_indices = np.where(speaker_ids == speaker_id)[0]
            size = len(speaker_indices)
            rs.shuffle(speaker_indices)
            train_size = int(0.8 * size)
            train_indices.extend(speaker_indices[:train_size].tolist())
            test_indices.extend(speaker_indices[train_size:].tolist())

        return train_indices if train else test_indices

    def load_librispeech_speaker_id(self, fileid, path, ext_audio, ext_txt):
        speaker_id, _, _ = fileid.split('-')
        return int(speaker_id)

    def __getitem__(self, index):
        # NOTE: overwrite index with our custom indices mapping exapmles
        #       to the training and test splits
        index = self.indices[index]

        try:
            wavform, sample_rate, _, speaker_id, _, _ = self.dataset.__getitem__(index)
        except Exception:
            index2 = (index + 1) % len(self.dataset)
            wavform, sample_rate, _, speaker_id, _, _ = self.dataset.__getitem__(index2)

        speaker_id = self.speaker_id_map[speaker_id]
        wavform = np.asarray(wavform[0])

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
            hop_length=hop_length_dict[self.INPUT_SIZE[1]],
            n_mels=self.INPUT_SIZE[0],
        )

        # log mel-spectrogram
        spectrum = librosa.power_to_db(spectrum**2)
        spectrum = torch.from_numpy(spectrum).float()
        spectrum = spectrum.unsqueeze(0)

        normalize = Normalize(LIBRISPEECH_MEAN, LIBRISPEECH_STDEV)
        spectrum = normalize(spectrum)

        return index, spectrum, speaker_id

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def num_classes():
        return LibriSpeechTransfer.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(
                input_size=LibriSpeech.INPUT_SIZE,
                patch_size=LibriSpeech.PATCH_SIZE,
                in_channels=LibriSpeech.IN_CHANNELS,
            )
        ]
