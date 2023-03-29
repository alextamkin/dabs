from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import device, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torchmetrics import AUROC
from torchmetrics.functional import accuracy


class SSLEvaluator(nn.Module):

    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if n_hidden is None:
            # use linear classifier
            self.block_forward = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes, bias=True),
            )
        else:
            # use simple MLP classifier
            self.block_forward = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True),
            )

    def forward(self, x):
        logits = self.block_forward(x)
        return logits


class SSLOnlineEvaluator(Callback):  # pragma: no cover
    '''
    Attaches a MLP for fine-tuning using the standard self-supervised protocol.
    Example::
        # your model must have 2 attributes
        model = Model()
        model.z_dim = ... # the representation dim
        model.num_classes = ... # the num of classes in the model
        online_eval = SSLOnlineEvaluator(
            z_dim=model.z_dim,
            num_classes=model.num_classes,
            dataset='imagenet'
        )
    '''

    def __init__(
        self,
        dataset: str,
        metric: str,
        loss: str,
        drop_p: float = 0.2,
        hidden_dim: Optional[int] = None,
        z_dim: int = None,
        num_classes: int = None,
    ):
        '''
        Args:
            dataset: if stl10, need to get the labeled batch
            drop_p: Dropout probability
            hidden_dim: Hidden dimension for the fine-tune MLP
            z_dim: Representation dimension
            num_classes: Number of classes
        '''
        super().__init__()

        self.hidden_dim = hidden_dim
        self.drop_p = drop_p
        self.optimizer: Optimizer

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.dataset = dataset

        self.metric = metric.lower()
        self.loss = loss.lower()
        if self.metric == 'auroc':
            self.auroc = AUROC(num_classes=num_classes)

    def on_pretrain_routine_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.non_linear_evaluator = SSLEvaluator(
            n_input=self.z_dim,
            n_classes=self.num_classes,
            p=self.drop_p,
            n_hidden=self.hidden_dim,
        ).to(pl_module.device)

        self.optimizer = torch.optim.Adam(pl_module.non_linear_evaluator.parameters(), lr=1e-4)

    def get_representations(self, pl_module: LightningModule, x: Sequence[Tensor]) -> Tensor:
        # Default augmentations already applied. Don't normalize or apply views.
        # Also, get representations from prepool layer.
        representations = pl_module(x, prehead=True)
        representations = representations.reshape(representations.size(0), -1)
        return representations

    def to_device(self, batch: Sequence, device: Union[str, device]) -> Tuple[Tensor, Tensor]:
        # Get the labeled batch
        if self.dataset == 'stl10':
            labeled_batch = batch[1]
            batch = labeled_batch

        inputs, y = batch[1:-1], batch[-1]

        # All inputs for online eval
        x = [x.to(device) for x in inputs]
        y = y.to(device)

        return x, y

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()

        # forward pass
        mlp_preds = pl_module.non_linear_evaluator(representations)  # type: ignore[operator]
        if self.loss == 'binary_cross_entropy':
            mlp_loss = F.binary_cross_entropy_with_logits(mlp_preds, y.float())
        elif self.loss == 'cross_entropy':
            mlp_loss = F.cross_entropy(mlp_preds, y)
        else:
            raise Exception(
                "This dataset uses a loss function that isn't one of 'cross_entropy' or 'binary_cross_entropy', which the online evaluator does not support. Check spelling if you believe you received this in error."
            )

        # update finetune weights
        mlp_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # log loss + accuracy per batch, but auroc per epoch to guarantee label instances for each class
        with torch.no_grad():
            if self.metric == 'auroc':
                self.auroc.update(torch.sigmoid(mlp_preds), y)
            elif self.metric == 'accuracy' and self.num_classes == 1:
                train_acc = accuracy(torch.sigmoid(mlp_preds), y)
                pl_module.log('online_train_acc', train_acc, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
            else:
                train_acc = accuracy(F.softmax(mlp_preds, dim=1), y)
                pl_module.log('online_train_acc', train_acc, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

        pl_module.log('online_train_loss', mlp_loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=False)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # log auroc at end of epoch here
        if self.metric == 'auroc':
            pl_module.log(
                'online_train_auroc', self.auroc.compute(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
            )
            self.auroc.reset()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()

        # forward pass
        mlp_preds = pl_module.non_linear_evaluator(representations)  # type: ignore[operator]

        if self.loss == 'binary_cross_entropy':
            mlp_loss = F.binary_cross_entropy_with_logits(mlp_preds, y.float())
        elif self.loss == 'cross_entropy':
            mlp_loss = F.cross_entropy(mlp_preds, y)
        else:
            raise Exception(
                "This dataset uses a loss function that isn't one of 'cross_entropy' or 'binary_cross_entropy', which the online evaluator does not support. Check your spelling if you believe you received this in error."
            )

        # log loss + accuracy per batch, but auroc per epoch to guarantee label instances for each class
        with torch.no_grad():
            if self.metric == 'auroc':
                self.auroc.update(torch.sigmoid(mlp_preds), y)
            elif self.metric == 'accuracy' and self.num_classes == 1:
                val_acc = accuracy(torch.sigmoid(mlp_preds), y)
                pl_module.log('online_val_acc', val_acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
            else:
                val_acc = accuracy(F.softmax(mlp_preds, dim=1), y)
                pl_module.log('online_val_acc', val_acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        pl_module.log('online_val_loss', mlp_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # log auroc at end of epoch here
        if self.metric == 'auroc':
            # we want to run initial validation sanity check, so we try and catch the error here
            try:
                pl_module.log(
                    'online_val_auroc', self.auroc.compute(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
                )
            except ValueError as error:
                pl_module.log('online_val_auroc', 0.0, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
                print(f'Logging `0.0` due to {error}. Is this from sanity check?')
            self.auroc.reset()
