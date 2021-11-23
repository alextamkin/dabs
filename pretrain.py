'''Main pretraining script.'''

import hydra


@hydra.main(config_path='conf', config_name='pretrain')
def run(config):
    # Deferred imports for faster tab completion
    import os

    import flatten_dict
    import pytorch_lightning as pl

    from src import online_evaluator
    from src.datasets.catalog import MULTILABEL_DATASETS, PRETRAINING_DATASETS, UNLABELED_DATASETS
    from src.systems import emix, shed

    pl.seed_everything(config.trainer.seed)

    # Saving checkpoints and logging with wandb.
    flat_config = flatten_dict.flatten(config, reducer='dot')
    save_dir = os.path.join(config.exp.base_dir, config.exp.name)
    wandb_logger = pl.loggers.WandbLogger(project='domain-agnostic', name=config.exp.name)
    wandb_logger.log_hyperparams(flat_config)
    callbacks = [pl.callbacks.ModelCheckpoint(dirpath=save_dir, every_n_train_steps=20000, save_top_k=-1)]

    assert config.dataset.name in PRETRAINING_DATASETS, f'{config.dataset.name} not one of {PRETRAINING_DATASETS}.'

    if config.algorithm == 'emix':
        system = emix.EMixSystem(config)
    elif config.algorithm == 'shed':
        system = shed.ShEDSystem(config)
    else:
        raise ValueError(f'Unimplemented algorithm config.algorithm={config.algorithm}.')

    # Online evaluator for labeled datasets.
    if config.dataset.name not in UNLABELED_DATASETS:
        ssl_online_evaluator = online_evaluator.SSLOnlineEvaluator(
            dataset=config.dataset.name,
            z_dim=config.model.kwargs.dim,
            num_classes=system.dataset.num_classes(),
            multi_label=(config.dataset.name in MULTILABEL_DATASETS),
        )
        callbacks += [ssl_online_evaluator]

    # PyTorch Lightning Trainer.
    trainer = pl.Trainer(
        default_root_dir=save_dir,
        logger=wandb_logger,
        gpus=str(config.gpus),  # GPU indices
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


if __name__ == '__main__':
    run()
