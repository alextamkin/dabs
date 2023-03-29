''' Script for training all the transfer datasets for a single pretrained checkpoint

Sample usage:

python3 transfer_single_domain.py \
    --domain=satellite_images --gpus=0 --exp_base_dir=/path/to/base_dir/ \
    --transfer_data_root=/path/to/data  --pretrain_exp_name=shed_librispeech_0.15_pytorch '--ckpt=epoch=293-step=99999.ckpt' 
'''
import argparse
import os
import subprocess

from src.datasets.catalog import PRETRAIN_TO_TRANSFER_DICT


def parse_args():
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument('--domain', type=str, required=True, choices=list(PRETRAIN_TO_TRANSFER_DICT.keys()))
    parser.add_argument('--exp_base_dir', type=str, required=True)
    parser.add_argument('--pretrain_exp_name', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)

    # Optional
    parser.add_argument('--gpus', type=str, required=False, default='0')
    parser.add_argument("--debug", required=False, action='store_true', default=False)
    return parser.parse_args()


def run(command, debug=False):
    '''Runs command with error catching'''
    print(command)
    if debug:
        return

    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as error:
        print(error.output)


def main():
    args = parse_args()
    domain = PRETRAIN_TO_TRANSFER_DICT[args.domain]

    # Print pretty domain summary
    header = f'Spawn script for {args.domain.upper()} domain'
    print(header)
    print('=' * len(header))
    print('Pretrain dataset:')
    print(f'  * {domain.pretrain}')
    print('Transfer datasets:')
    for transfer in domain.transfers:
        print(f'  * {transfer}')
    print('=' * len(header))

    for transfer in domain.transfers:
        ckpt = os.path.join(args.exp_base_dir, args.pretrain_exp_name, args.ckpt)
        # Name like shed_librispeech_0.15_pytorch
        algorithm, _, fraction, framework = args.pretrain_exp_name.split('_')

        transfer_exp_name = f"{algorithm}_{domain.pretrain}_{transfer}_{fraction}_{framework}_transfer"
        print(transfer_exp_name)
        command = (
            'python3 transfer.py '
            f'\'ckpt=\"{ckpt}\"\' '
            f'dataset={transfer} '
            f'exp.name={transfer_exp_name} '
            f'exp.base_dir={args.exp_base_dir} '
            f'data_root={args.data_root} '
            f'gpus={args.gpus} '
            f'framework={framework}'
        )
        run(command, debug=args.debug)


if __name__ == '__main__':
    main()
