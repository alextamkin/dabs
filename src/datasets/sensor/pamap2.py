# Wearable sensor dataset.

import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torchaudio.transforms import Spectrogram
from torchvision.datasets.utils import download_and_extract_archive

from src.datasets.specs import Input1dSpec, Input2dSpec

ACTIVITY_LABELS = [
    1,  # lying
    2,  # sitting
    3,  # standing
    4,  # walking
    5,  # running
    6,  # cycling
    7,  # Nordic walking
    # 9, # watching TV (optional)
    # 10, # computer work  (optional)
    # 11, # car driving (optional)
    12,  # ascending stairs
    13,  # descending stairs
    16,  # vacuum cleaning
    # 17, # ironing (optional)
    # 18, # folding laundry (optional)
    # 19, # house cleaning (optional)
    # 20, # playing soccer (optional)
    24  # rope jumping
]

FEATURE_MEANS = np.array(
    [
        -10.12912214, -11.29261799, 0.67638378, 0.81824769, 0.75297834, -0.35109685, 0.04085698, -0.38876906, -2.48238567,
        -3.41956712, -3.3872513, 1.36282383, 1.55308991, 1.56087922, -10.76128503, -10.35194776, -10.44513743, -10.37285293,
        -11.23690636, -0.20944169, 0.56012058, 0.807821, -1.45611818, -0.35643357, -0.25041446, -2.76965766, -3.24698013,
        -3.85922755, 1.1442057, 1.46386916, 1.51837609, -11.07261072, -11.14997687, -11.13951721, -11.12178224, -11.29449096,
        1.94817929, 2.33591061, 1.97720141, 0.91686234, 1.53700002, 0.88543364, -1.64330728, -2.63160618, -2.51725697,
        1.42671659, 1.6363767, 1.65463002, -10.68715032, -10.14333333, -10.40543887, -10.2161264
    ]
)

FEATURE_STDS = np.array(
    [
        7.52822918, 6.7065013, 3.95108152, 3.95592566, 3.42002526, 4.64231584, 4.44694546, 4.16510321, 3.71419447, 3.21044202,
        3.59042373, 3.39598192, 3.24402304, 3.26736989, 4.84615018, 4.85592083, 4.75026502, 5.0713948, 6.76148597, 3.16121415,
        4.10307909, 3.42466748, 3.91835069, 4.63133192, 4.12213119, 3.21565752, 3.00317751, 3.04138402, 2.96988045, 3.30489875,
        3.05622836, 4.66155384, 4.38560602, 4.45703007, 4.35220719, 6.72132295, 4.49144193, 4.40899389, 3.80700876, 5.15785846,
        4.82611255, 4.45424858, 3.65129909, 3.15463525, 3.965269, 3.46149886, 3.22442971, 3.17674841, 4.71934308, 5.41595717,
        4.97243856, 5.33158206
    ]
)


class PAMAP2(data.Dataset):
    '''Transform and return PAMAP2 sensor dataset. This dataset consists of various sensor readings taken over time.
    Each example contains 320 52-channel measurements.

    '''
    # Dataset information.
    SENSOR_RESOURCES = {'pamap2': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip'}

    TRAIN_EXAMPLES_PER_EPOCH = 50000  # examples are generated stochastically
    VAL_EXAMPLES_PER_EPOCH = 10000
    MEASUREMENTS_PER_EXAMPLE = 320  # measurements used

    NUM_CLASSES = 12  # NOTE: They're not contiguous labels.
    SEGMENT_SIZE = 5
    IN_CHANNELS = 52  # multiple sensor readings from different parts of the body
    MAE_OUTPUT_SIZE = 260

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'sensor', 'pamap2')
        self.mode = 'train' if train else 'val'  # there are more options, just not by default

        if download:
            self.download_dataset()

        if not self._is_downloaded():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.subject_data = self.load_data(self.root)

    def _is_downloaded(self):
        return (os.path.exists(self.root))

    def download_dataset(self):
        '''Download the dataset if not exists already'''

        if self._is_downloaded():
            return

        # download and extract files
        print('Downloading and Extracting...')

        filename = self.SENSOR_RESOURCES['pamap2'].rpartition('/')[2]
        download_and_extract_archive(self.SENSOR_RESOURCES['pamap2'], download_root=self.root, filename=filename)

        print('Done!')

    def get_subject_ids(self, mode):
        if mode == 'train':
            nums = [1, 2, 3, 4, 7, 8, 9]
        elif mode == 'train_small':
            nums = [1]
        elif mode == 'val':
            nums = [5]
        elif mode == 'test':
            nums = [6]
        else:
            raise ValueError(f'mode must be one of [train, train_small, val, test]. got {mode}.')
        return nums

    def get_subject_filenames(self, mode):
        nums = self.get_subject_ids(mode)
        return [f'subject10{num}.dat' for num in nums]  # like 'subject101.dat'

    def load_data(self, root_path):
        subject_data = []  # list of data frames, one for subject
        for subject_filename in self.get_subject_filenames(self.mode):
            columns = ['timestamp', 'activity_id', 'heart_rate']
            for part in ['hand', 'chest', 'ankle']:
                for i in range(17):
                    columns.append(part + str(i))
            subj_path = os.path.join(root_path, 'PAMAP2_Dataset', 'Protocol', subject_filename)
            subj_path_cache = subj_path + '.p'
            if os.path.isfile(subj_path_cache):
                print(f'Loading {subj_path_cache}')
                df = pd.read_pickle(subj_path_cache)
            else:
                df = pd.read_csv(subj_path, names=columns, sep=' ')
                df = df.interpolate()  # Interpolate out NaNs.
                print(f'Saving {subj_path_cache}')
                df.to_pickle(subj_path_cache)
            subject_data.append(df)
        return subject_data

    def load_measurements(self):
        # pick random number
        while True:
            subject_id = np.random.randint(len(self.subject_data))
            activity_id = np.random.randint(len(ACTIVITY_LABELS))
            df = self.subject_data[subject_id]
            activity_data = df[df['activity_id'] == ACTIVITY_LABELS[activity_id]].to_numpy()
            if len(activity_data) > self.MEASUREMENTS_PER_EXAMPLE:
                break
        start_idx = np.random.randint(len(activity_data) - self.MEASUREMENTS_PER_EXAMPLE)

        # Get frame and also truncate off label and timestamp.
        # [self.MEASUREMENTS_PER_EXAMPLE, 52]
        measurements = activity_data[start_idx:start_idx + self.MEASUREMENTS_PER_EXAMPLE, 2:]
        return measurements.astype(np.float32), activity_id

    def __getitem__(self, index):
        measurements, label = self.load_measurements()
        return (index, measurements, label)

    def __len__(self):
        return self.TRAIN_EXAMPLES_PER_EPOCH if self.mode == 'train' else self.VAL_EXAMPLES_PER_EPOCH

    @staticmethod
    def num_classes():
        return PAMAP2.NUM_CLASSES

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input1dSpec(
                seq_len=int(PAMAP2.MEASUREMENTS_PER_EXAMPLE), segment_size=PAMAP2.SEGMENT_SIZE, in_channels=PAMAP2.IN_CHANNELS
            ),
        ]


class SpectrogramPAMAP2(data.Dataset):
    '''Transform and return sensor datasets using spectrogram

    Return the spectrogram of the sensor dataset.

    This implementation is no longer used by the benchmark.
    However, it was featured in the original submission of the paper, so it is provided here.

    '''
    # Dataset information.
    SENSOR_RESOURCES = {'pamap2': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip'}

    TRAIN_EXAMPLES_PER_EPOCH = 50000  # examples are generated stochastically
    VAL_EXAMPLES_PER_EPOCH = 10000
    MEASUREMENTS_PER_EXAMPLE = 1000  # measurements used to make spectrogram

    NUM_CLASSES = 12  # NOTE: They're not contiguous labels.
    INPUT_SIZE = (32, 32)
    PATCH_SIZE = (4, 4)
    IN_CHANNELS = 52  # multiple sensor readings from different parts of the body

    def __init__(self, base_root: str, download: bool = False, train: bool = True) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'sensor', 'pamap2')
        self.mode = 'train' if train else 'val'  # there are more options, just not by default

        if download:
            self.download_dataset()

        if not self._is_downloaded():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.subject_data = self.load_data(self.root)

    def _is_downloaded(self):
        return (os.path.exists(self.root))

    def download_dataset(self):
        '''Download the dataset if not exists already'''

        if self._is_downloaded():
            return

        # download and extract files
        print('Downloading and Extracting...')

        filename = self.SENSOR_RESOURCES['pamap2'].rpartition('/')[2]
        download_and_extract_archive(self.SENSOR_RESOURCES['pamap2'], download_root=self.root, filename=filename)

        print('Done!')

    def get_subject_ids(self, mode):
        if mode == 'train':
            nums = [1, 2, 3, 4, 7, 8, 9]
        elif mode == 'train_small':
            nums = [1]
        elif mode == 'val':
            nums = [5]
        elif mode == 'test':
            nums = [6]
        else:
            raise ValueError(f'mode must be one of [train, train_small, val, test]. got {mode}.')
        return nums

    def get_subject_filenames(self, mode):
        nums = self.get_subject_ids(mode)
        return [f'subject10{num}.dat' for num in nums]  # like 'subject101.dat'

    def load_data(self, root_path):
        subject_data = []  # list of data frames, one for subject
        for subject_filename in self.get_subject_filenames(self.mode):
            columns = ['timestamp', 'activity_id', 'heart_rate']
            for part in ['hand', 'chest', 'ankle']:
                for i in range(17):
                    columns.append(part + str(i))
            subj_path = os.path.join(root_path, 'PAMAP2_Dataset', 'Protocol', subject_filename)
            subj_path_cache = subj_path + '.p'
            if os.path.isfile(subj_path_cache):
                print(f'Loading {subj_path_cache}')
                df = pd.read_pickle(subj_path_cache)
            else:
                df = pd.read_csv(subj_path, names=columns, sep=' ')
                df = df.interpolate()  # Interpolate out NaNs.
                print(f'Saving {subj_path_cache}')
                df.to_pickle(subj_path_cache)
            subject_data.append(df)
        return subject_data

    def load_spectrogram(self, index):
        # pick random number
        while True:
            subject_id = np.random.randint(len(self.subject_data))
            activity_id = np.random.randint(len(ACTIVITY_LABELS))
            df = self.subject_data[subject_id]
            activity_data = df[df['activity_id'] == ACTIVITY_LABELS[activity_id]].to_numpy()
            if len(activity_data) > self.MEASUREMENTS_PER_EXAMPLE:
                break
        start_idx = np.random.randint(len(activity_data) - self.MEASUREMENTS_PER_EXAMPLE)

        # Get frame and also truncate off label and timestamp.
        # [self.MEASUREMENTS_PER_EXAMPLE, 52]
        measurements = activity_data[start_idx:start_idx + self.MEASUREMENTS_PER_EXAMPLE, 2:]

        # Yields spectrograms of shape [52, 32, 32]
        spectrogram_transform = Spectrogram(n_fft=64 - 1, hop_length=32, power=2)
        spectrogram = spectrogram_transform(torch.tensor(measurements.T))
        spectrogram = (spectrogram + 1e-6).log()

        # Normalize.
        spectrogram = (spectrogram - FEATURE_MEANS.reshape(-1, 1, 1)) / FEATURE_STDS.reshape(-1, 1, 1)

        return spectrogram, activity_id

    def __getitem__(self, index):
        # pick random number
        img_data, label = self.load_spectrogram(index)
        return (index, img_data.float(), label)

    def __len__(self):
        return self.TRAIN_EXAMPLES_PER_EPOCH if self.mode == 'train' else self.VAL_EXAMPLES_PER_EPOCH

    @staticmethod
    def num_classes():
        return SpectrogramPAMAP2.NUM_CLASSES

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(
                input_size=SpectrogramPAMAP2.INPUT_SIZE,
                patch_size=SpectrogramPAMAP2.PATCH_SIZE,
                in_channels=SpectrogramPAMAP2.IN_CHANNELS
            ),
        ]
