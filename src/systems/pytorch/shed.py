import numpy as np
import torch
import torchmetrics

from src.systems.pytorch.base_system import BaseSystem


class ShEDSystem(BaseSystem):
    '''System for Shuffled Embedding Detection.

    Permutes 15% of embeddings within an example.
    Objective is to predict which embeddings were replaced.
    '''

    def __init__(self, config):
        super().__init__(config)
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')
        self.predictor = torch.nn.Linear(self.model.emb_dim, 2)  # Predictor: replaced or not.
        self.accuracy = torchmetrics.Accuracy()
        self.permute_frac = config.corruption_rate

    def forward(self, x, prehead=False):
        return self.model.forward(x, prehead=prehead)

    def ssl_forward(self, batch):
        batch = batch[1:-1]  # Remove label.

        # Embed first. [batch_size, seq_len, emb_dim]
        embs = self.model.embed(batch)
        permuted_embs, is_permuted = self.permute_embeddings(embs)

        # We pass prepool=True because we want the embeddings for each token.
        return self.model.encode(permuted_embs, prepool=True), is_permuted

    def training_step(self, batch, batch_idx):
        embs, is_permuted = self.ssl_forward(batch)
        loss, acc = self.objective(embs, is_permuted)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log('train_acc', acc.item(), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        embs, is_permuted = self.ssl_forward(batch)
        loss, acc = self.objective(embs, is_permuted)
        self.log('val_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_acc', acc.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def objective(self, embs, is_permuted):
        preds = self.predictor(embs)[:, 1:]  # Remove [CLS] token.
        loss = self.ce(preds.reshape(-1, 2), is_permuted.reshape(-1)).mean()
        acc = self.accuracy(preds.argmax(-1), is_permuted)
        return loss, acc

    def permute_embeddings(self, embs):
        '''Permutes self.permute_frac of embeddings within each example.

        Args:
            embs: [batch_size, seq_len, emb_size] embeddings to permute

        Returns:
            permuted_embs: [batch_size, seq_len, emb_size] embeddings
                with self.permute_frac permuted
            is_permuted: [batch_size, seq_len] of ints indicating whether each token was permuted or not

        '''
        num_to_permute = int(np.ceil(embs.size(1) * self.permute_frac))

        # [batch_size, num_to_permute]
        indices_to_permute = torch.rand(embs.size(0), embs.size(1)).topk(dim=-1, k=num_to_permute).indices.to(self.device)
        # [batch_size, num_to_permute, emb_size] (repeat along last dimension for torch.scatter)
        indices_to_permute_for_scatter = indices_to_permute.unsqueeze(-1).repeat(1, 1, embs.size(-1))

        # NOTE: we want a "derangement," which is a permutation where no element
        # is left in its original position. torch.roll in an efficient way of doing this,
        # compared to e.g. running torch.randperm multiple times, however
        # this technically does not produce the full distribution of derangements,
        # as it excludes derangements with multiple cycles.
        new_embs_indices = torch.roll(indices_to_permute, shifts=1, dims=-1)
        new_embs_indices_for_gather = new_embs_indices.unsqueeze(-1).repeat(1, 1, embs.size(-1))

        new_embs = torch.gather(embs, -2, new_embs_indices_for_gather)
        permuted_embs = torch.scatter(embs, -2, indices_to_permute_for_scatter, new_embs)

        is_permuted = torch.zeros(embs.size(0), embs.size(1), dtype=int, device=self.device)
        is_permuted = torch.scatter(is_permuted, 1, indices_to_permute, 1)

        return permuted_embs, is_permuted
