import os
from abc import abstractmethod

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, IterableDataset

from src.datasets.catalog import DATASET_DICT
from src.models.pytorch.transformer import DomainAgnosticTransformer


def get_model(config: DictConfig, dataset_class: Dataset):
    '''Retrieves the specified model class, given the dataset class.'''
    if config.model.name == 'transformer':
        model_class = DomainAgnosticTransformer
    else:
        raise ValueError(f'Encoder {config.model.name} doesn\'t exist.')

    # Retrieve the dataset-specific params.
    return model_class(
        input_specs=dataset_class.spec(),
        **config.model.kwargs,
    )


class BaseSystem(pl.LightningModule):

    def __init__(self, config: DictConfig):
        '''An abstract class that implements some shared functionality for training.

        Args:
            config: a hydra config
        '''
        super().__init__()
        self.config = config
        self.dataset = DATASET_DICT[config.dataset.name]
        self.dataset_name = config.dataset.name
        self.model = get_model(config, self.dataset)

    @abstractmethod
    def objective(self, *args):
        '''Computes the loss and accuracy.'''
        pass

    @abstractmethod
    def forward(self, batch):
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    def setup(self, stage):
        '''Called right after downloading data and before fitting model, initializes datasets with splits.'''
        self.train_dataset = self.dataset(base_root=self.config.data_root, download=True, train=True)
        self.val_dataset = self.dataset(base_root=self.config.data_root, download=True, train=False)
        try:
            print(f'{len(self.train_dataset)} train examples, {len(self.val_dataset)} val examples')
        except TypeError:
            print('Iterable/streaming dataset- undetermined length.')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            shuffle=not isinstance(self.train_dataset, IterableDataset),
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        if not self.val_dataset:
            raise ValueError('Cannot get validation data for this dataset')

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        if self.config.optim.name == 'adam':
            optim = torch.optim.AdamW(params, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.name == 'sgd':
            optim = torch.optim.SGD(
                params,
                lr=self.config.optim.lr,
                weight_decay=self.config.optim.weight_decay,
                momentum=self.config.optim.momentum,
            )
        else:
            raise ValueError(f'{self.config.optim.name} optimizer unrecognized.')
        return optim

    def on_train_end(self):
        model_path = os.path.join(self.trainer.checkpoint_callback.dirpath, 'model.ckpt')
        torch.save(self.state_dict(), model_path)
        print(f'Pretrained model saved to {model_path}')
