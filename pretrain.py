'''Main pretraining script.'''

import os

import flatten_dict
import hydra

from src.datasets.catalog import PRETRAINING_DATASETS, UNLABELED_DATASETS


def run_pytorch(config):
    '''Runs pretraining in PyTorch.'''
    import pytorch_lightning as pl

    from src.evaluators.pytorch import online_evaluator
    from src.systems.pytorch import contpred, emix, mae, shed

    # Check for dataset.
    assert config.dataset.name in PRETRAINING_DATASETS, f'{config.dataset.name} not one of {PRETRAINING_DATASETS}.'

    # Set up config, callbacks, loggers.
    flat_config = flatten_dict.flatten(config, reducer='dot')
    save_dir = os.path.join(config.exp.base_dir, config.exp.name)

    # Set RNG.
    pl.seed_everything(config.trainer.seed)
    wandb_logger = pl.loggers.WandbLogger(project='domain-agnostic', name=config.exp.name)
    wandb_logger.log_hyperparams(flat_config)
    callbacks = [
        pl.callbacks.ModelCheckpoint(dirpath=save_dir, every_n_train_steps=config.trainer.ckpt_every_n_steps, save_top_k=-1)
    ]

    # Initialize training module.
    if config.algorithm == 'emix':
        system = emix.EMixSystem(config)
    elif config.algorithm == 'shed':
        system = shed.ShEDSystem(config)
    elif config.algorithm == 'capri':
        system = contpred.ContpredSystem(config, negatives='sequence')
    elif config.algorithm == 'mae':
        system = mae.MAESystem(config)
    else:
        raise ValueError(f'Unimplemented algorithm config.algorithm={config.algorithm}.')

    # Online evaluator for labeled datasets.
    if config.dataset.name not in UNLABELED_DATASETS:
        ssl_online_evaluator = online_evaluator.SSLOnlineEvaluator(
            dataset=config.dataset.name,
            metric=config.dataset.metric,
            loss=config.dataset.loss,
            z_dim=config.model.kwargs.dim,
            num_classes=system.dataset.num_classes()
        )
        callbacks += [ssl_online_evaluator]

    # PyTorch Lightning Trainer.
    trainer = pl.Trainer(
        default_root_dir=save_dir,
        logger=wandb_logger,
        gpus=[config.gpus],  # GPU indices
        max_steps=config.trainer.max_steps,
        min_steps=config.trainer.max_steps,
        resume_from_checkpoint=config.trainer.resume_from_checkpoint,
        val_check_interval=config.trainer.val_check_interval,
        limit_val_batches=config.trainer.limit_val_batches,
        callbacks=callbacks,
        weights_summary=config.trainer.weights_summary,
        gradient_clip_val=config.trainer.gradient_clip_val,
        precision=config.trainer.precision,
    )

    trainer.fit(system)


def print_pytorch_info():
    import torch
    import torchaudio
    import torchvision
    header = '==== Using Framework: PYTORCH ===='
    print(header)
    print(f'   - [torch]       {torch.__version__}')
    print(f'   - [torchvision] {torchvision.__version__}')
    print(f'   - [torchaudio]  {torchaudio.__version__}')
    print('=' * len(header))


@hydra.main(config_path='conf', config_name='pretrain')
def run(config):
    '''Wrapper around actual run functions to import and run for specified framework.'''
    if config.framework == 'pytorch':
        print_pytorch_info()
        run_pytorch(config)
    else:
        raise ValueError(f'Framework {config.framework} not supported.')


if __name__ == '__main__':
    run()
