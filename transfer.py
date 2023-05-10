'''Main transfer script.'''

import os

import flatten_dict
import hydra

from src.datasets.catalog import TRANSFER_DATASETS


@hydra.main(config_path='conf', config_name='transfer')
def run_pytorch(config):
    # Deferred imports for faster tab completion

    import pytorch_lightning as pl

    from src.systems.pytorch import transfer

    pl.seed_everything(config.trainer.seed)

    # Saving checkpoints and logging with wandb.
    flat_config = flatten_dict.flatten(config, reducer='dot')
    save_dir = os.path.join(config.exp.base_dir, config.exp.name)
    wandb_logger = pl.loggers.WandbLogger(project='domain-agnostic', name=config.exp.name)
    wandb_logger.log_hyperparams(flat_config)
    ckpt_callback = pl.callbacks.ModelCheckpoint(dirpath=save_dir)

    assert config.dataset.name in TRANSFER_DATASETS, f'{config.dataset.name} not one of {TRANSFER_DATASETS}.'

    # PyTorch Lightning Trainer.
    trainer = pl.Trainer(
        default_root_dir=save_dir,
        logger=wandb_logger,
        gpus=[config.gpus],
        max_epochs=config.trainer.max_epochs,
        min_epochs=config.trainer.max_epochs,
        val_check_interval=config.trainer.val_check_interval,
        limit_val_batches=config.trainer.limit_val_batches,
        callbacks=[ckpt_callback],
        weights_summary=config.trainer.weights_summary,
        precision=config.trainer.precision
    )

    system = transfer.TransferSystem(config)
    trainer.fit(system)


@hydra.main(config_path='conf', config_name='transfer')
def run(config):
    if config.framework == 'pytorch':
        from pretrain import print_pytorch_info
        print_pytorch_info()
        run_pytorch(config)
    else:
        raise ValueError(f'Framework {config.framework} not supported.')


if __name__ == '__main__':
    run()
