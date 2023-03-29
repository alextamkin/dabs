from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.systems.pytorch.base_system import BaseSystem


class EMixSystem(BaseSystem):
    '''Implements the i-Mix algorithm on embedded inputs defined in https://arxiv.org/abs/2010.08887.

    Because there aren't predefined augmentations, i-Mix is applied to the original embeddings. The
    algorithm under default parameters can be summarized as

    Algorithm 1:
        lambda ~ Beta(1, 1)
        lambda = max(lambda, 1 - lambda)  # assures mixing coefficient >= 0.5

        embs = embed(*x)
        permuted_idxs = permute(arange(embs))
        permuted_embs = stop_gradient[embs][permuted_idx]
        mixed_embs = lambda * embs + (1 - lambda) * permuted_embs

        logits = mixed_embs @ embs.T
        contrastive_loss = cross_entropy(logits, arange(embs))
        mixed_virtual_loss = cross_entropy(logits, permuted_idxs)

        loss = contrastive_loss + mixed_virtual_loss
    '''

    ALPHA = 1.0
    TEMPERATURE = 0.2

    def __init__(self, config):
        super().__init__(config)
        self.beta_distribution = torch.distributions.beta.Beta(self.ALPHA, self.ALPHA)
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def objective(
        self,
        embs_mix: torch.Tensor,
        embs: torch.Tensor,
        mix_coeff: Union[torch.Tensor, float],
        randidx: torch.Tensor,
    ):
        embs_mix = F.normalize(embs_mix, dim=1)
        embs = F.normalize(embs, dim=1)

        logits = embs_mix @ embs.T / self.TEMPERATURE

        labels = torch.arange(logits.shape[0], dtype=torch.long, device=logits.device)

        loss = mix_coeff * self.ce(logits, labels) + (1 - mix_coeff) * self.ce(logits, randidx)
        loss = loss.mean()
        with torch.no_grad():
            acc = (logits.argmax(1) == labels).float().mean()
        return loss, acc

    def ssl_forward(self, batch, prehead=False):
        batch = batch[1:-1]  # could be multiple tensors here

        # Embed first.
        embs = self.model.embed(batch)
        batch_size = embs.shape[0]

        # Sample mixing coefficient from beta distribution.
        mix_coeff = self.beta_distribution.sample([batch_size]).to(embs.device)
        mix_coeff = mix_coeff.view(-1, *[1] * (embs.dim() - 1))
        mix_coeff = torch.max(mix_coeff, 1 - mix_coeff)

        # Generate augmentations.
        randidx = torch.randperm(batch_size, device=embs.device)
        embs_mix = embs[randidx].detach()
        embs_mix = mix_coeff * embs + (1 - mix_coeff) * embs_mix
        embs_mix = self.model.encode(embs_mix, prehead=prehead)

        embs = self.model.encode(embs, prehead=prehead)

        return embs_mix, embs, mix_coeff, randidx

    def forward(self, inputs: Sequence[torch.Tensor], prehead=False):
        return self.model.forward(inputs, prehead=prehead)

    def training_step(self, batch, batch_idx):
        embs_i, embs_j, mix_coeff, randidx = self.ssl_forward(batch, prehead=False)
        loss, acc = self.objective(embs_i, embs_j, mix_coeff, randidx)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log('train_acc', acc.item(), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        embs_i, embs_j, mix_coeff, randidx = self.ssl_forward(batch, prehead=False)
        loss, acc = self.objective(embs_i, embs_j, mix_coeff, randidx)
        return {'loss': loss.item(), 'acc': acc.item()}

    def validation_epoch_end(self, validation_step_outputs):
        loss = [val_dict['loss'] for val_dict in validation_step_outputs]
        acc = [val_dict['acc'] for val_dict in validation_step_outputs]
        self.log('val_loss', sum(loss) / len(loss), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_acc', sum(acc) / len(acc), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
