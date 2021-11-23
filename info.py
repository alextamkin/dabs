'''Simple script that outputs dataset information.'''

from src.datasets.catalog import DATASET_DICT, PRETRAINING_DATASETS, TRANSFER_DATASETS


def bold(string):
    return f'\033[1m{string}\033[0m'


def main():
    print(bold('All supported datasets'))
    max_name_len = max(len(name) for name in DATASET_DICT.keys())
    for name in sorted(DATASET_DICT.keys()):
        class_name = DATASET_DICT[name].__name__
        print(f'{name.ljust(max_name_len)}: {class_name}')

    print()
    print(bold('Datasets for pretraining'))
    print(f'{", ".join(sorted(PRETRAINING_DATASETS))}')

    print()
    print(bold('Datasets for transfer learning'))
    print(f'{", ".join(sorted(TRANSFER_DATASETS))}')

    print()
    print('Pass the dataset name for training from the command line as the flag `dataset=name` (i.e. dataset=cifar10)')


if __name__ == '__main__':
    main()
